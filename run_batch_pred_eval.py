#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch evaluate multiple experiment folders that contain pred_programs, "
            "then calculate metrics with run logs + eval logs."
        )
    )
    parser.add_argument(
        "--exp_dirs",
        nargs="+",
        required=True,
        help=(
            "Experiment directories. Each directory should contain pred_programs/. "
            "Example: /path/to/claude_code_outputs_opus46 /path/to/another_run"
        ),
    )
    parser.add_argument(
        "--run_logs",
        nargs="*",
        default=None,
        help=(
            "Optional inference/run jsonl paths aligned with --exp_dirs. "
            "If omitted for an exp dir, the script tries to recover cost logs from "
            "<exp_dir>/logs/tasks/*/conversation.log."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="batch_eval_outputs",
        help="Directory for generated eval logs and recovered run logs.",
    )
    parser.add_argument(
        "--benchmark_path",
        type=str,
        default="benchmark",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="osunlp/ScienceAgentBench",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--force_rebuild",
        type=str,
        default="True",
        help="Passed to run_evaluation as-is (True/False).",
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        default="base",
        choices=["none", "base", "instance"],
    )
    parser.add_argument(
        "--run_id_start",
        type=int,
        default=0,
        help="Start run_id for this batch. Each experiment increments by 1.",
    )
    parser.add_argument(
        "--task_id_start",
        type=int,
        default=1,
        help="Mapping for recovered costs: line_idx 0 -> task id task_id_start.",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation and only run metric calculation.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned commands and file paths without executing.",
    )
    return parser.parse_args()


def _normalize_name(path: Path) -> str:
    return path.name.replace(" ", "_")


def _ensure_run_log_mapping(exp_dirs: List[Path], run_logs: Optional[List[str]]) -> Dict[Path, Optional[Path]]:
    if run_logs is None:
        return {d: None for d in exp_dirs}
    if len(run_logs) > len(exp_dirs):
        raise ValueError("--run_logs cannot be longer than --exp_dirs.")
    mapping: Dict[Path, Optional[Path]] = {d: None for d in exp_dirs}
    for idx, run_log in enumerate(run_logs):
        mapping[exp_dirs[idx]] = Path(run_log)
    return mapping


def _extract_total_cost_from_conversation_log(conversation_log: Path) -> Optional[float]:
    if not conversation_log.exists():
        return None
    cost = None
    with conversation_log.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "result":
                value = obj.get("total_cost_usd")
                if value is not None:
                    try:
                        cost = float(value)
                    except (TypeError, ValueError):
                        pass
    return cost


def recover_run_log_from_tasks(
    exp_dir: Path,
    expected_rows: int,
    task_id_start: int,
    out_path: Path,
) -> Path:
    tasks_dir = exp_dir / "logs" / "tasks"
    if not tasks_dir.exists():
        raise FileNotFoundError(
            f"Cannot recover run log for {exp_dir}: {tasks_dir} does not exist."
        )

    rows = []
    for row_idx in range(expected_rows):
        task_id = str(task_id_start + row_idx)
        conversation_log = tasks_dir / task_id / "conversation.log"
        cost = _extract_total_cost_from_conversation_log(conversation_log)
        if cost is None:
            cost = 0.0
        rows.append({"cost": cost})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out_path


def line_count(jsonl_path: Path) -> int:
    with jsonl_path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def run_evaluation(
    pred_program_dir: Path,
    eval_log_path: Path,
    run_id: int,
    args: argparse.Namespace,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "evaluation.harness.run_evaluation",
        "--benchmark_path",
        args.benchmark_path,
        "--pred_program_path",
        str(pred_program_dir),
        "--log_fname",
        str(eval_log_path),
        "--run_id",
        str(run_id),
        # "--instance_ids",
        # str("9 12 21 26 29 31 32 34 35 67 78 92"),
        "--force_rebuild",
        args.force_rebuild,
        "--cache_level",
        args.cache_level,
        "--max_workers",
        str(args.max_workers),
        "--dataset_name",
        args.dataset_name,
        "--split",
        args.split,
    ]
    print("Running eval:", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def calculate_metrics(run_log_path: Path, eval_log_path: Path) -> Dict[str, float]:
    from calculate_metrics import main as calc_main

    return calc_main([str(run_log_path)], [str(eval_log_path)])


def main() -> None:
    args = parse_args()

    exp_dirs = [Path(d).resolve() for d in args.exp_dirs]
    run_log_mapping = _ensure_run_log_mapping(exp_dirs, args.run_logs)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for idx, exp_dir in enumerate(exp_dirs):
        pred_program_dir = exp_dir / "pred_programs"
        if not pred_program_dir.exists():
            raise FileNotFoundError(f"{pred_program_dir} does not exist.")

        exp_name = _normalize_name(exp_dir)
        eval_log_path = output_dir / f"eval_{exp_name}.jsonl"

        if not args.skip_eval:
            run_id = args.run_id_start + idx
            run_evaluation(
                pred_program_dir=pred_program_dir,
                eval_log_path=eval_log_path,
                run_id=run_id,
                args=args,
                dry_run=args.dry_run,
            )
        elif not eval_log_path.exists():
            raise FileNotFoundError(
                f"--skip_eval is set but eval log does not exist: {eval_log_path}"
            )

        maybe_run_log = run_log_mapping[exp_dir]
        if maybe_run_log is not None:
            run_log_path = maybe_run_log.resolve()
            if not run_log_path.exists():
                raise FileNotFoundError(f"Provided run log not found: {run_log_path}")
        else:
            recovered_run_log = output_dir / f"run_recovered_{exp_name}.jsonl"
            if args.dry_run:
                print(
                    f"[dry-run] recover run log from {exp_dir / 'logs' / 'tasks'} "
                    f"-> {recovered_run_log} (rows=<from eval log>)"
                )
                run_log_path = recovered_run_log
            else:
                expected_rows = line_count(eval_log_path)
                run_log_path = recover_run_log_from_tasks(
                    exp_dir=exp_dir,
                    expected_rows=expected_rows,
                    task_id_start=args.task_id_start,
                    out_path=recovered_run_log,
                )

        if args.dry_run:
            print(f"[dry-run] calculate_metrics with run={run_log_path} eval={eval_log_path}")
            metrics = {
                "success_rate": -1.0,
                "codebert_score": -1.0,
                "valid_program_rate": -1.0,
                "cost": -1.0,
            }
        else:
            metrics = calculate_metrics(run_log_path, eval_log_path)

        summary.append(
            {
                "experiment_dir": str(exp_dir),
                "pred_program_dir": str(pred_program_dir),
                "run_log": str(run_log_path),
                "eval_log": str(eval_log_path),
                **metrics,
            }
        )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print("\n=== Batch Summary ===")
    for row in summary:
        print(
            f"- {Path(row['experiment_dir']).name}: "
            f"SR={row['success_rate']:.4f}, "
            f"CBS={row['codebert_score']:.4f}, "
            f"VPR={row['valid_program_rate']:.4f}, "
            f"Cost={row['cost']:.4f}"
        )
        print(f"  run_log: {row['run_log']}")
        print(f"  eval_log: {row['eval_log']}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
