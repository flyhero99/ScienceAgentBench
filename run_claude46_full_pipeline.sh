#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ec2-user/ScienceAgentBench"
cd "$ROOT"

MODEL="us.anthropic.claude-sonnet-4-6"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT}/logs/nightly_pipeline"
mkdir -p "$LOG_DIR"

# You can flip to false if needed.
USE_SELF_DEBUG=true

INFER_LOG="claude-sonnet-4-6_sd_ot_oe_${RUN_TAG}.jsonl"
INFER_VERIFIED_LOG="claude-sonnet-4-6_sd_ot_oe_verified_${RUN_TAG}.jsonl"
PIPELINE_LOG="${LOG_DIR}/claude46_pipeline_${RUN_TAG}.log"

echo "Pipeline log: ${PIPELINE_LOG}"
echo "Run tag: ${RUN_TAG}"
echo "Model: ${MODEL}"

{
  echo "===== [1/4] run_infer.py start $(date -u) ====="
  if [ "$USE_SELF_DEBUG" = true ]; then
    python -u run_infer.py \
      --llm_engine_name "$MODEL" \
      --log_fname "$INFER_LOG" \
      --use_self_debug
  else
    python -u run_infer.py \
      --llm_engine_name "$MODEL" \
      --log_fname "$INFER_LOG"
  fi
  echo "===== [1/4] run_infer.py done $(date -u) ====="

  echo "===== [2/4] run_evaluation.sh ${INFER_LOG} start $(date -u) ====="
  bash run_evaluation.sh "$INFER_LOG"
  echo "===== [2/4] run_evaluation.sh ${INFER_LOG} done $(date -u) ====="

  echo "===== [3/4] run_infer_verified.py start $(date -u) ====="
  if [ "$USE_SELF_DEBUG" = true ]; then
    python -u run_infer_verified.py \
      --llm_engine_name "$MODEL" \
      --log_fname "$INFER_VERIFIED_LOG" \
      --use_self_debug
  else
    python -u run_infer_verified.py \
      --llm_engine_name "$MODEL" \
      --log_fname "$INFER_VERIFIED_LOG"
  fi
  echo "===== [3/4] run_infer_verified.py done $(date -u) ====="

  echo "===== [4/4] run_evaluation.sh ${INFER_VERIFIED_LOG} start $(date -u) ====="
  bash run_evaluation.sh "$INFER_VERIFIED_LOG"
  echo "===== [4/4] run_evaluation.sh ${INFER_VERIFIED_LOG} done $(date -u) ====="

  echo "===== Pipeline finished $(date -u) ====="
  echo "Inference logs:"
  echo "  - ${INFER_LOG}"
  echo "  - ${INFER_VERIFIED_LOG}"
  echo "Evaluation logs:"
  echo "  - eval_${INFER_LOG}"
  echo "  - eval_${INFER_VERIFIED_LOG}"
} | tee "$PIPELINE_LOG"

