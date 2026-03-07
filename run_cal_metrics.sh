EVAL_LOGS="eval_claude-sonnet-4-6_sd_ot_oe_run1.jsonl"

python calculate_metrics.py \
--run_logs claude-sonnet-4-6_sd_ot_oe_run1.jsonl \
--eval_logs $EVAL_LOGS

python - << EOF
import json

indices=[]
for i,l in enumerate(open("$EVAL_LOGS")):
    if json.loads(l).get("success_rate")==1:
        indices.append(i)

print(",".join(map(str,indices)))
print("success_count:", len(indices))
EOF