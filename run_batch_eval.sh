#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ec2-user/ScienceAgentBench"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
BASE_OUTPUT_DIR="${ROOT}/batch_eval_outputs/${RUN_TAG}"

EXP_DIRS=(
  "${ROOT}/claude_code_outputs_opus46"
  "${ROOT}/claude_code_outputs_sonnet46"
  "${ROOT}/codex_cli_outputs_gpt52"
)

RUN_ID_START=300

for i in "${!EXP_DIRS[@]}"; do
  EXP_DIR="${EXP_DIRS[$i]}"
  EXP_NAME="$(basename "${EXP_DIR}")"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}"
  RUN_ID=$((RUN_ID_START + i))

  echo "==== [${EXP_NAME}] docker prune ===="
  docker system prune -a -f

  echo "==== [${EXP_NAME}] start eval ===="
  python run_batch_pred_eval.py \
    --exp_dirs "${EXP_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_id_start "${RUN_ID}" \
    --max_workers 8 \
    --force_rebuild True \
    --cache_level base
done

echo "All done. Results under: ${BASE_OUTPUT_DIR}"
