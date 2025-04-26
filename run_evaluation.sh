# python recover_pred_from_log_gpt4o-mini.py \
# --log_fname qwen2.5-coder-14b_instruct_sft_0425.jsonl

python -m evaluation.harness.run_evaluation \
--benchmark_path benchmark \
--pred_program_path pred_programs \
--log_fname eval_qwen2.5-coder-14b_instruct_sft_0425_4omini.jsonl \
--run_id 28 \
--force_rebuild True \
--cache_level base \
--max_workers 8