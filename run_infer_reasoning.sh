#!/usr/bin/env bash
set -euo pipefail

# Reasoning-enabled inference examples.
# By default, run_infer.py keeps old behavior. Add these flags to enable
# thinking/reasoning for supported models:
#   --enable_reasoning
#   --reasoning_effort {low|medium|high}
#   --use_responses_api        (recommended for GPT-5/Azure reasoning models)
#   --reasoning_budget_tokens  (used by Bedrock thinking mode)

# Example 1: Bedrock Claude Sonnet 4.6 with thinking mode
python -u run_infer.py \
  --llm_engine_name us.anthropic.claude-sonnet-4-6 \
  --log_fname claude-sonnet-4-6_sd_ot_oe_reasoning_run1.jsonl \
  --use_self_debug \
  --enable_reasoning \
  --reasoning_budget_tokens 20000

# Example 2: Azure GPT-5.2 with Responses API + reasoning effort
# python -u run_infer.py \
#   --llm_engine_name azure_gpt-5.2 \
#   --log_fname gpt-5.2_sd_ot_oe_reasoning_run1.jsonl \
#   --use_self_debug \
#   --enable_reasoning \
#   --reasoning_effort medium \
#   --use_responses_api

# Example 3: OpenAI GPT-5.2 with Responses API + reasoning effort
# python -u run_infer.py \
#   --llm_engine_name gpt-5.2 \
#   --log_fname gpt-5.2_sd_ot_oe_reasoning_run1.jsonl \
#   --use_self_debug \
#   --enable_reasoning \
#   --reasoning_effort medium \
#   --use_responses_api
