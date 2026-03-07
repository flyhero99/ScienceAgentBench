python run_infer.py --llm_engine_name us.anthropic.claude-sonnet-4-5-20250929-v1:0 --log_fname claude-sonnet-4-5_sd_ot_oe_run1.jsonl --use_self_debug

python recover_pred_from_log.py --log_fname claude-sonnet-4-5_sd_ot_oe_run1.jsonl

python run_eval.py --log_fname eval_claude-sonnet-4-5_sd_ot_oe_run1.jsonl
