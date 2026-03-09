from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

import backoff


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
    # Keep legacy behavior by default. If reasoning is enabled, pass reasoning
    # hints when supported by the model/API.
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

    response = client.chat.completions.create(**kwargs)
    return response


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
    # Responses API is preferred for reasoning models (e.g., GPT-5 family)
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

    response = client.responses.create(**kwargs)
    return response


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
            prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
            return content, prompt_tokens, completion_tokens

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

        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
