#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import List


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".svg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find visualization tasks from ScienceAgentBench_verified.csv and "
            "copy images from logs/run_evaluation/<run_id>/<instance_id>/pred_results "
            "to a new folder."
        )
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/ec2-user/ScienceAgentBench/benchmark/ScienceAgentBench_verified.csv",
    )
    parser.add_argument(
        "--run_eval_root",
        type=str,
        default="/home/ec2-user/ScienceAgentBench/logs/run_evaluation",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="302",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ec2-user/ScienceAgentBench/new_folder",
    )
    parser.add_argument(
        "--category_keyword",
        type=str,
        default="visualization",
        help="Case-insensitive keyword for matching subtask_categories.",
    )
    return parser.parse_args()


def get_visualization_instance_ids(csv_path: Path, keyword: str) -> List[str]:
    ids: List[str] = []
    keyword_lower = keyword.lower()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            categories = (row.get("subtask_categories") or "").lower()
            if keyword_lower in categories:
                ids.append(str(row["instance_id"]))
    return ids


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    run_root = Path(args.run_eval_root) / str(args.run_id)
    output_dir = Path(args.output_dir)

    ids = get_visualization_instance_ids(csv_path, args.category_keyword)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    missing_pred_results = []
    instance_with_images = []

    for instance_id in ids:
        pred_results_dir = run_root / instance_id / "pred_results"
        if not pred_results_dir.exists():
            missing_pred_results.append(instance_id)
            continue

        image_files = [p for p in pred_results_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not image_files:
            continue

        instance_with_images.append(instance_id)
        dst_dir = output_dir / instance_id
        dst_dir.mkdir(parents=True, exist_ok=True)

        for src in image_files:
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            copied_files.append(str(dst))

    ids_csv = ",".join(ids)
    print(ids_csv)
    print("instance_count:", len(ids))
    print("instance_with_images:", len(instance_with_images))
    print("copied_image_count:", len(copied_files))
    print("missing_pred_results_count:", len(missing_pred_results))
    if missing_pred_results:
        print("missing_pred_results_ids:", ",".join(missing_pred_results))

    report = {
        "run_id": str(args.run_id),
        "keyword": args.category_keyword,
        "instance_ids": ids,
        "instance_count": len(ids),
        "instance_with_images": instance_with_images,
        "instance_with_images_count": len(instance_with_images),
        "copied_image_count": len(copied_files),
        "missing_pred_results_ids": missing_pred_results,
        "missing_pred_results_count": len(missing_pred_results),
        "copied_files": copied_files,
    }
    (output_dir / "visualization_copy_report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
