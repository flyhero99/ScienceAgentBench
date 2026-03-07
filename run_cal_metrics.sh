EVAL_LOGS="eval_claude-sonnet-4-6_sd_vt_oe_run1.jsonl"

python calculate_metrics.py \
--run_logs claude-sonnet-4-6_sd_vt_oe_run1.jsonl \
--eval_logs $EVAL_LOGS

python -c 'import json;print(",".join(str(i) for i,l in enumerate(open("'"$EVAL_LOGS"'")) if json.loads(l).get("success_rate")==1))'
