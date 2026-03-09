from openai import AzureOpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

import backoff
import requests
import sys


def _get_field(obj, key, default=0):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_reasoning_tokens_from_usage(usage):
    output_details = _get_field(usage, "output_tokens_details", None)
    reasoning_tokens = _get_field(output_details, "reasoning_tokens", 0)
    if reasoning_tokens:
        return int(reasoning_tokens)

    completion_details = _get_field(usage, "completion_tokens_details", None)
    reasoning_tokens = _get_field(completion_details, "reasoning_tokens", 0)
    return int(reasoning_tokens or 0)


def _extract_text_from_responses_json(response_json):
    chunks = []
    for item in response_json.get("output", []):
        for piece in item.get("content", []):
            txt = piece.get("text")
            if txt:
                chunks.append(txt)
    return "\n".join(chunks).strip()


def _extract_usage_from_responses_json(response_json):
    usage = response_json.get("usage", {}) or {}
    prompt_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    completion_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    reasoning_tokens = int(
        _get_field(_get_field(usage, "output_tokens_details", {}), "reasoning_tokens", 0) or 0
    )
    return prompt_tokens, completion_tokens, reasoning_tokens


@backoff.on_exception(backoff.expo, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError))
def azure_chat_engine(
    client,
    engine,
    msg,
    temperature,
    top_p,
    max_tokens=32000,
    enable_reasoning=False,
    reasoning_effort="medium",
):
    kwargs = {
        "model": engine,
        "messages": msg,
        "max_completion_tokens": max_tokens,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if enable_reasoning:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    return client.chat.completions.create(**kwargs)


def azure_responses_http(
    azure_endpoint,
    api_key,
    api_version,
    engine,
    msg,
    temperature,
    top_p,
    max_tokens=32000,
    enable_reasoning=False,
    reasoning_effort="medium",
):
    url = f"{azure_endpoint}/openai/deployments/{engine}/responses?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    payload = {
        "model": engine,
        "input": msg,
        "max_output_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if enable_reasoning:
        payload["reasoning"] = {"effort": reasoning_effort}

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    return resp.json()


@backoff.on_exception(backoff.expo, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError))
def azure_chat_engine_o3(client, engine, msg, temperature, top_p, max_tokens=32000):
    return client.beta.chat.completions.parse(
        model=engine,
        messages=msg,
        max_completion_tokens=max_tokens,
    )


class AzureEngine:

    def __init__(self, llm_engine_name, api_key, api_version, azure_endpoint):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )
        self.llm_engine_name = llm_engine_name
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint

    def respond(
        self,
        user_input,
        temperature,
        top_p,
        max_tokens=32000,
        enable_reasoning=False,
        reasoning_effort="medium",
        use_responses_api=False,
        reasoning_budget_tokens=20000,
    ):
        try:
            if not hasattr(self, "_debug_printed"):
                print(f"DEBUG AzureEngine: Using model name '{self.llm_engine_name}'", file=sys.stderr)
                self._debug_printed = True

            if "o3" in self.llm_engine_name:
                response = azure_chat_engine_o3(
                    self.client,
                    self.llm_engine_name,
                    user_input,
                    temperature,
                    top_p,
                    max_tokens,
                )
                content = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                prompt_tokens = int(_get_field(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(_get_field(usage, "completion_tokens", 0) or 0)
                meta = {
                    "reasoning_tokens": _extract_reasoning_tokens_from_usage(usage),
                    "api_mode": "chat_completions_o3",
                }
                return content, prompt_tokens, completion_tokens, meta

            if use_responses_api:
                response_json = azure_responses_http(
                    self.azure_endpoint,
                    self.api_key,
                    self.api_version,
                    self.llm_engine_name,
                    user_input,
                    temperature,
                    top_p,
                    max_tokens,
                    enable_reasoning=enable_reasoning,
                    reasoning_effort=reasoning_effort,
                )
                content = _extract_text_from_responses_json(response_json)
                prompt_tokens, completion_tokens, reasoning_tokens = _extract_usage_from_responses_json(response_json)
                meta = {
                    "reasoning_tokens": reasoning_tokens,
                    "api_mode": "responses",
                }
                return content, prompt_tokens, completion_tokens, meta

            try:
                response = azure_chat_engine(
                    self.client,
                    self.llm_engine_name,
                    user_input,
                    temperature,
                    top_p,
                    max_tokens,
                    enable_reasoning=enable_reasoning,
                    reasoning_effort=reasoning_effort,
                )
                choice = response.choices[0]
                content = choice.message.content or ""

                usage = getattr(response, "usage", None)
                prompt_tokens = int(_get_field(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(_get_field(usage, "completion_tokens", 0) or 0)
                meta = {
                    "reasoning_tokens": _extract_reasoning_tokens_from_usage(usage),
                    "api_mode": "chat_completions",
                }
                return content, prompt_tokens, completion_tokens, meta

            except Exception as e:
                error_msg = str(e)
                if "Unsupported parameter: 'messages'" in error_msg or "Responses API" in error_msg:
                    response_json = azure_responses_http(
                        self.azure_endpoint,
                        self.api_key,
                        self.api_version,
                        self.llm_engine_name,
                        user_input,
                        temperature,
                        top_p,
                        max_tokens,
                        enable_reasoning=enable_reasoning,
                        reasoning_effort=reasoning_effort,
                    )
                    content = _extract_text_from_responses_json(response_json)
                    prompt_tokens, completion_tokens, reasoning_tokens = _extract_usage_from_responses_json(response_json)
                    meta = {
                        "reasoning_tokens": reasoning_tokens,
                        "api_mode": "responses_fallback",
                    }
                    return content, prompt_tokens, completion_tokens, meta
                raise

        except Exception as e:
            print(f"ERROR: Can't invoke '{self.llm_engine_name}' on Azure. Reason: {e}")
            return "ERROR", 0, 0, {"reasoning_tokens": 0, "api_mode": "error"}
