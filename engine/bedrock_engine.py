from botocore.exceptions import ClientError, ReadTimeoutError
from botocore.client import Config

import boto3
import base64

def bedrock_converse_engine(
    client,
    engine,
    msg,
    temperature,
    top_p,
    maxTokens=32000,
    enable_thinking=False,
    budget_tokens=20000,
):
    """
    Bedrock Anthropic models forbid specifying both temperature and topP together.
    Build inferenceConfig accordingly:
      - if both provided, drop topP and keep temperature (default path)
      - if temperature is None, use topP only
    """
    reasoning_config = None
    if enable_thinking:
        reasoning_config = {
            "thinking": {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }
        }

    if temperature is not None and top_p is not None:
        # Drop topP to satisfy Anthropic constraint
        inference_cfg = {"maxTokens": maxTokens, "temperature": temperature}
    elif temperature is None and top_p is not None:
        inference_cfg = {"maxTokens": maxTokens, "topP": top_p}
    else:
        # Fallback to only temperature (or neither)
        inference_cfg = {"maxTokens": maxTokens}
        if temperature is not None:
            inference_cfg["temperature"] = temperature
        if top_p is not None:
            inference_cfg["topP"] = top_p

    converse_kwargs = dict(
        modelId=engine,
        messages=msg,
        inferenceConfig=inference_cfg,
    )
    if reasoning_config is not None:
        converse_kwargs["additionalModelRequestFields"] = reasoning_config

    response = client.converse(**converse_kwargs)
    return response

class BedrockEngine():

    def __init__(self, llm_engine_name):
        self.client = boto3.client(
            "bedrock-runtime", 
            region_name="us-east-2", 
            config=Config(retries={"total_max_attempts": 3}, read_timeout=1200)
        )
        self.llm_engine_name = llm_engine_name

    def respond(self, user_input, temperature, top_p, max_tokens=32000, enable_thinking=False, budget_tokens=20000):
        """
        Process user input and get response from Bedrock API.
        Supports both text and multimodal (text + image) messages.
        For multimodal messages, content should be a list with text and image_url items.
        """
        conversation = []
        for turn in user_input:
            content = turn.get("content")
            # If content is already a list (multimodal format), convert to Bedrock format
            if isinstance(content, list):
                # Convert OpenAI-style multimodal format to Bedrock format
                bedrock_content = []
                for item in content:
                    if item.get("type") == "text":
                        bedrock_content.append({"text": item.get("text", "")})
                    elif item.get("type") == "image_url":
                        # Bedrock uses "image" type with "source" containing "bytes" or "url"
                        image_url = item.get("image_url", {}).get("url", "")
                        # If it's a data URL (base64), extract the base64 part
                        if image_url.startswith("data:image"):
                            # Extract base64 data from data URL
                            header, encoded = image_url.split(",", 1)
                            # Determine image format from header
                            if "png" in header:
                                image_format = "png"
                            elif "jpeg" in header or "jpg" in header:
                                image_format = "jpeg"
                            else:
                                image_format = "png"  # default
                            
                            bedrock_content.append({
                                "image": {
                                    "format": image_format,
                                    "source": {
                                        "bytes": base64.b64decode(encoded)
                                    }
                                }
                            })
                        else:
                            # Regular URL
                            bedrock_content.append({
                                "image": {
                                    "source": {
                                        "url": image_url
                                    }
                                }
                            })
                conversation.append({"role": turn["role"], "content": bedrock_content})
            else:
                # Text-only content
                conversation.append({"role": turn["role"], "content": [{"text": str(content)}]})

        try:
            response = bedrock_converse_engine(
                self.client, 
                self.llm_engine_name, 
                conversation,
                temperature,
                top_p,
                max_tokens,
                enable_thinking,
                budget_tokens,
            )
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{self.llm_engine_name}'. Reason: {e}")
            return "ERROR", 0, 0

        return response["output"]["message"]["content"][0]["text"], response["usage"]["inputTokens"], response["usage"]["outputTokens"]
    

if __name__ == "__main__":
    # Simple local test for thinking mode
    # Choose a recent Claude Sonnet deployment; adjust if your deployment name differs.
    eng = BedrockEngine("us.anthropic.claude-sonnet-4-5-20250929-v1_0")
    import time
    prompt = "Explain briefly how a rocket works."
    start_time = time.time()
    
    resp = eng.respond(
        [{"role": "user", "content": prompt}],
        temperature=1,
        top_p=None,
        max_tokens=32000,
        enable_thinking=True,
        budget_tokens=20000,
    )
    import pdb; pdb.set_trace()
    print("Response:", resp[0])
    print("Input tokens:", resp[1], "Output tokens:", resp[2])
    print("Elapsed:", time.time() - start_time, "seconds")