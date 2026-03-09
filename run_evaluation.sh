#!/bin/bash

# 检查是否提供了参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <log_filename>"
    exit 1
fi

# 获取命令行第一个参数
LOG_FNAME="$1"

# 执行 recover_pred_from_log.py
echo "Running: python recover_pred_from_log.py --log_fname $LOG_FNAME"
python recover_pred_from_log.py --log_fname "$LOG_FNAME"

# 为 run_evaluation 计算 log_fname
EVAL_LOG_FNAME="eval_$LOG_FNAME"

# 计算 run_id：检测当前最大数字并加一
# 首先检查 logs 目录是否存在
if [ ! -d "logs" ]; then
    echo "Directory 'logs' does not exist. Starting with run_id = 0."
    NEXT_RUN_ID=0
else
    # 然后检查 logs/run_evaluation 目录是否存在
    if [ ! -d "logs/run_evaluation" ]; then
        echo "Directory 'logs/run_evaluation' does not exist. Starting with run_id = 0."
        NEXT_RUN_ID=0
    else
        # 改进的方法，使用数值排序而不是字符串排序
        # 首先找出所有数字命名的目录
        # 然后用 sort -n 确保正确的数值排序
        # 最后取最大值
        NUMERIC_DIRS=$(find "logs/run_evaluation" -maxdepth 1 -type d -name "[0-9]*" | xargs -I{} basename {} 2>/dev/null)

        if [ -z "$NUMERIC_DIRS" ]; then
            echo "No numeric directories found in 'logs/run_evaluation'. Starting with run_id = 0."
            NEXT_RUN_ID=0
        else
            # 使用 sort -n 确保正确的数值排序，找出最大值
            LATEST_RUN_ID=$(echo "$NUMERIC_DIRS" | sort -n | tail -n 1)
            NEXT_RUN_ID=$((LATEST_RUN_ID + 1))
            echo "Found latest run_id: $LATEST_RUN_ID. Next run_id will be: $NEXT_RUN_ID."
        fi
    fi
fi

# 执行 run_evaluation
echo "Running: python -m evaluation.harness.run_evaluation --benchmark_path benchmark --pred_program_path pred_programs --log_fname $EVAL_LOG_FNAME --run_id $NEXT_RUN_ID --force_rebuild True --cache_level base --max_workers 8"
python -m evaluation.harness.run_evaluation \
    --benchmark_path benchmark \
    --pred_program_path pred_programs \
    --log_fname "$EVAL_LOG_FNAME" \
    --run_id "$NEXT_RUN_ID" \
    --force_rebuild False \
    --cache_level base \
    --max_workers 8


