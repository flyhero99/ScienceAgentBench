from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

import backoff


def _get_field(obj, key, default=0):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_reasoning_tokens_from_usage(usage):
    # Responses API: usage.output_tokens_details.reasoning_tokens
    output_details = _get_field(usage, "output_tokens_details", None)
    reasoning_tokens = _get_field(output_details, "reasoning_tokens", 0)
    if reasoning_tokens:
        return int(reasoning_tokens)

    # Chat Completions (some models): usage.completion_tokens_details.reasoning_tokens
    completion_details = _get_field(usage, "completion_tokens_details", None)
    reasoning_tokens = _get_field(completion_details, "reasoning_tokens", 0)
    return int(reasoning_tokens or 0)


@backoff.on_exception(backoff.expo, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError))
def openai_chat_engine(
    client,
    engine,
    msg,
    temperature,
    top_p,
    max_completion_tokens,
    enable_reasoning=False,
    reasoning_effort="medium",
):
    kwargs = {
        "model": engine,
        "messages": msg,
    }

    if engine.startswith("gpt") or engine.startswith("o"):
        kwargs.update(
            {
                "temperature": temperature,
                "max_completion_tokens": max_completion_tokens,
                "top_p": top_p,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }
        )

    if enable_reasoning:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    return client.chat.completions.create(**kwargs)


@backoff.on_exception(backoff.expo, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError))
def openai_responses_engine(
    client,
    engine,
    msg,
    temperature,
    top_p,
    max_tokens,
    enable_reasoning=False,
    reasoning_effort="medium",
):
    kwargs = {
        "model": engine,
        "input": msg,
        "max_output_tokens": max_tokens,
    }

    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if enable_reasoning:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    return client.responses.create(**kwargs)


class OpenaiEngine:

    def __init__(self, llm_engine_name):
        self.client = OpenAI(max_retries=10, timeout=120.0)
        self.llm_engine_name = llm_engine_name

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
        # Default stays Chat Completions for backward compatibility.
        # When requested, or for GPT-5 family, Responses API can be used.
        should_use_responses = use_responses_api or self.llm_engine_name.startswith("gpt-5")

        if should_use_responses:
            response = openai_responses_engine(
                self.client,
                self.llm_engine_name,
                user_input,
                temperature,
                top_p,
                max_tokens,
                enable_reasoning=enable_reasoning,
                reasoning_effort=reasoning_effort,
            )

            content = response.output_text or ""
            usage = getattr(response, "usage", None)
            prompt_tokens = int(_get_field(usage, "input_tokens", 0) or 0)
            completion_tokens = int(_get_field(usage, "output_tokens", 0) or 0)
            meta = {
                "reasoning_tokens": _extract_reasoning_tokens_from_usage(usage),
                "api_mode": "responses",
            }
            return content, prompt_tokens, completion_tokens, meta

        response = openai_chat_engine(
            self.client,
            self.llm_engine_name,
            user_input,
            temperature,
            top_p,
            max_tokens,
            enable_reasoning=enable_reasoning,
            reasoning_effort=reasoning_effort,
        )

        usage = getattr(response, "usage", None)
        prompt_tokens = int(_get_field(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(_get_field(usage, "completion_tokens", 0) or 0)
        meta = {
            "reasoning_tokens": _extract_reasoning_tokens_from_usage(usage),
            "api_mode": "chat_completions",
        }
        return response.choices[0].message.content, prompt_tokens, completion_tokens, meta
