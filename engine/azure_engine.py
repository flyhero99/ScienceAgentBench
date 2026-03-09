from openai import AzureOpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

import backoff
import requests
import sys


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
    # Azure/OpenAI responses use input_tokens/output_tokens naming.
    prompt_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    completion_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    return prompt_tokens, completion_tokens


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
    # Keep existing o3 parse path untouched in behavior.
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
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                return content, prompt_tokens, completion_tokens

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
                prompt_tokens, completion_tokens = _extract_usage_from_responses_json(response_json)
                return content, prompt_tokens, completion_tokens

            try:
                # Preferred default path: chat completions
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
                content = choice.message.content
                if content is None:
                    content = ""

                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                return content, prompt_tokens, completion_tokens

            except Exception as e:
                # Auto-fallback to responses endpoint for deployments that require it.
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
                    prompt_tokens, completion_tokens = _extract_usage_from_responses_json(response_json)
                    return content, prompt_tokens, completion_tokens
                raise

        except Exception as e:
            print(f"ERROR: Can't invoke '{self.llm_engine_name}' on Azure. Reason: {e}")
            return "ERROR", 0, 0
