from openai import AzureOpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
import backoff
import os
import sys

# @backoff.on_exception(backoff.expo, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError))
def azure_chat_engine(client, engine, msg, temperature, top_p, max_tokens=32000):
    """
    Preferred Azure Chat Completions call.
    Newer Azure models (e.g., gpt-5.1) expect max_completion_tokens.
    Supports both text and multimodal (text + image) messages.
    Messages can be in format:
    - Text-only: [{"role": "user", "content": "text string"}]
    - Multimodal: [{"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]}]
    """
    response = client.chat.completions.create(
        model=engine,
        messages=msg,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response


def azure_responses_http(azure_endpoint, api_key, api_version, engine, msg, temperature, top_p, max_tokens=32000):
    """
    Fallback: direct HTTP call to Azure Responses API when chat endpoint rejects 'messages'.
    """
    import requests

    url = f"{azure_endpoint}/openai/deployments/{engine}/responses?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    payload = {
        "model": engine,
        "input": msg,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "max_output_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    return resp.json()

# @backoff.on_exception(backoff.expo, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError))
def azure_chat_engine_o3(client, engine, msg, struct_format, temperature, top_p, max_tokens=32000):
    # print("azure engine o3")
    response = client.beta.chat.completions.parse(
        model=engine,
        messages=msg,
        #temperature=temperature,
        max_completion_tokens=max_tokens,
        #top_p=top_p,Í›
        #frequency_penalty=0,
        #presence_penalty=0
    )
    return response

class AzureEngine():

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

    def respond(self, user_input, temperature, top_p, max_tokens=32000):
        try:
            # Debug: print model name being used
            import sys
            if hasattr(self, '_debug_printed'):
                pass
            else:
                print(f"DEBUG AzureEngine: Using model name '{self.llm_engine_name}'", file=sys.stderr)
                self._debug_printed = True
            
            if "o3" in self.llm_engine_name:
                response = azure_chat_engine_o3(
                    self.client,
                    self.llm_engine_name,
                    user_input,
                    temperature,
                    top_p,
                    max_tokens
                )
                content = response.choices[0].message.content
                if content is None:
                    print(f"WARNING: Azure '{self.llm_engine_name}' (o3) returned None content.", file=sys.stderr)
                    return "", response.usage.prompt_tokens if hasattr(response, 'usage') else 0, response.usage.completion_tokens if hasattr(response, 'usage') else 0
                return content, response.usage.prompt_tokens, response.usage.completion_tokens
            else:
                try:
                    # Preferred: chat completions (messages)
                    response = azure_chat_engine(
                        self.client,
                        self.llm_engine_name,
                        user_input,
                        temperature,
                        top_p,
                        max_tokens
                    )
                    choice = response.choices[0]
                    content = choice.message.content
                    finish_reason = getattr(choice, 'finish_reason', "unknown")
                    
                    # Debug: Print full response structure for empty content cases
                    if content is None or (isinstance(content, str) and content.strip() == ""):
                        print(f"WARNING: Azure '{self.llm_engine_name}' returned empty content.", file=sys.stderr)
                        print(f"Finish reason: {finish_reason}", file=sys.stderr)
                        print(f"Output tokens: {response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0}", file=sys.stderr)
                        print(f"Max tokens: {max_tokens}", file=sys.stderr)
                        
                        # Print full choice object for debugging
                        print(f"DEBUG: Full choice object: {choice}", file=sys.stderr)
                        print(f"DEBUG: Choice message type: {type(choice.message)}", file=sys.stderr)
                        print(f"DEBUG: Choice message attributes: {dir(choice.message)}", file=sys.stderr)
                        
                        # Check if there's content in other fields
                        if hasattr(choice.message, 'content') and choice.message.content:
                            print(f"DEBUG: Content exists but is empty/None: {repr(choice.message.content)}", file=sys.stderr)
                        
                        # Try to get any alternative content from the response
                        if hasattr(choice.message, 'refusal') and choice.message.refusal:
                            print(f"Model refusal: {choice.message.refusal}", file=sys.stderr)
                        
                        # If finish_reason is 'length' and we have tokens, this is very strange
                        # This appears to be a known issue with Azure GPT-5.1 on multimodal requests
                        # when hitting the token limit - it returns empty content instead of partial content
                        if finish_reason == 'length' and response.usage.completion_tokens > 0:
                            print(f"ERROR: finish_reason is 'length' with {response.usage.completion_tokens} tokens but content is empty!", file=sys.stderr)
                            print(f"This appears to be a known issue with Azure GPT-5.1 on multimodal requests.", file=sys.stderr)
                            print(f"Suggestions: 1) Increase max_tokens significantly (e.g., 8000-16000)", file=sys.stderr)
                            print(f"             2) Try without images (text-only) to see if issue persists", file=sys.stderr)
                            print(f"             3) Check Azure Portal for model-specific limitations", file=sys.stderr)
                            
                            # Check content_filter_results for any filtering issues
                            if hasattr(choice, 'content_filter_results') and choice.content_filter_results:
                                print(f"Content filter results: {choice.content_filter_results}", file=sys.stderr)
                            
                            # Return error message instead of empty string so we can track this issue
                            error_msg = f"[ERROR: Azure GPT-5.1 returned empty content despite {response.usage.completion_tokens} output tokens. This may be a model limitation with multimodal requests at token limits.]"
                            return error_msg, response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0, response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0
                        
                        # Return empty string to trigger retry
                        return "", response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0, response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0
                    
                    # Log finish reason for debugging
                    if finish_reason == 'length':
                        print(f"INFO: Response completed but may be truncated (finish_reason: {finish_reason})", file=sys.stderr)
                    
                    return content, response.usage.prompt_tokens, response.usage.completion_tokens
                except Exception as e:
                    # If chat API complains about 'messages', fall back to Responses API (input)
                    error_msg = str(e)
                    if "Unsupported parameter: 'messages'" in error_msg or "Responses API" in error_msg:
                        try:
                            response_json = azure_responses_http(
                                self.azure_endpoint,
                                self.api_key,
                                self.api_version,
                                self.llm_engine_name,
                                user_input,
                                temperature,
                                top_p,
                                max_tokens
                            )
                            # Parse HTTP response format
                            content = ""
                            usage_prompt = 0
                            usage_completion = 0
                            output_items = response_json.get("output", [])
                            chunks = []
                            for item in output_items:
                                if "content" in item:
                                    for piece in item["content"]:
                                        if "text" in piece:
                                            chunks.append(piece["text"])
                            content = "\n".join(chunks).strip()
                            usage = response_json.get("usage", {})
                            usage_prompt = usage.get("prompt_tokens", 0)
                            usage_completion = usage.get("completion_tokens", 0)
                            return content, usage_prompt, usage_completion
                        except Exception as e2:
                            print(f"ERROR: Can't invoke '{self.llm_engine_name}' on Azure (responses). Reason: {e2}")
                            return "ERROR", 0, 0
                    else:
                        print(f"ERROR: Can't invoke '{self.llm_engine_name}' on Azure. Reason: {e}")
                        return "ERROR", 0, 0
        
        except Exception as e:
            print(f"ERROR: Can't invoke '{self.llm_engine_name}' on Azure. Reason: {e}")
            return "ERROR", 0, 0