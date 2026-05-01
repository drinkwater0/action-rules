"""Empirically sweep CandidateGenerator._gpu_kernel_min_work on real datasets.

This script reuses the existing benchmark_runner.run_profile infrastructure and
varies only the intra-GPU dispatch threshold that decides when to launch the
custom CUDA support kernel.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Iterable, Optional


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from action_rules.candidates.candidate_generator import CandidateGenerator
from benchmark_runner import list_dataset_presets, run_profile


def _parse_csv_values(raw: str) -> list[str]:
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            values.append(token)
    return values


def _parse_thresholds(raw: str) -> list[int]:
    values = [int(value) for value in _parse_csv_values(raw)]
    if not values:
        raise ValueError("Expected at least one threshold value.")
    parsed = sorted({max(0, int(value)) for value in values})
    if not parsed:
        raise ValueError("Expected at least one non-negative threshold value.")
    return parsed


def _parse_datasets(raw: str) -> list[str]:
    values = [value.lower() for value in _parse_csv_values(raw)]
    if not values:
        raise ValueError("Expected at least one dataset preset.")
    return values


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_records(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in records:
        if row.get("actual_backend") != "gpu":
            continue
        elapsed = row.get("elapsed_seconds")
        if elapsed is None:
            continue
        key = (str(row["dataset_key"]), int(row["gpu_kernel_min_work"]))
        grouped.setdefault(key, []).append(float(elapsed))

    summary = []
    for (dataset_key, threshold), values in grouped.items():
        values_sorted = sorted(values)
        summary.append(
            {
                "dataset_key": dataset_key,
                "gpu_kernel_min_work": int(threshold),
                "runs": int(len(values_sorted)),
                "mean_elapsed_seconds": float(statistics.fmean(values_sorted)),
                "median_elapsed_seconds": float(statistics.median(values_sorted)),
                "min_elapsed_seconds": float(values_sorted[0]),
                "max_elapsed_seconds": float(values_sorted[-1]),
                "stdev_elapsed_seconds": (
                    float(statistics.pstdev(values_sorted))
                    if len(values_sorted) > 1
                    else 0.0
                ),
            }
        )
    summary.sort(key=lambda row: (row["dataset_key"], row["gpu_kernel_min_work"]))
    return summary


def _best_thresholds(summary: list[dict]) -> list[dict]:
    by_dataset: dict[str, list[dict]] = {}
    for row in summary:
        by_dataset.setdefault(str(row["dataset_key"]), []).append(row)

    winners = []
    for dataset_key, rows in by_dataset.items():
        best = min(
            rows,
            key=lambda row: (
                float(row["mean_elapsed_seconds"]),
                float(row["median_elapsed_seconds"]),
                int(row["gpu_kernel_min_work"]),
            ),
        )
        winners.append(
            {
                "dataset_key": dataset_key,
                "best_gpu_kernel_min_work": int(best["gpu_kernel_min_work"]),
                "best_mean_elapsed_seconds": float(best["mean_elapsed_seconds"]),
                "best_median_elapsed_seconds": float(best["median_elapsed_seconds"]),
                "runs": int(best["runs"]),
            }
        )
    winners.sort(key=lambda row: row["dataset_key"])
    return winners


def run_sweep(
    *,
    datasets: list[str],
    thresholds: list[int],
    runs: int,
    max_gpu_mem_mb: Optional[int],
    gpu_node_batch_size: Optional[int],
    min_support_count: Optional[int],
    min_confidence: Optional[float],
    include_dataset_profile: bool,
    verbose: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    records = []
    original_threshold = int(CandidateGenerator._gpu_kernel_min_work)
    run_count = max(1, int(runs))
    try:
        for dataset in datasets:
            for threshold in thresholds:
                CandidateGenerator._gpu_kernel_min_work = int(threshold)
                for run_index in range(run_count):
                    result = run_profile(
                        use_gpu=True,
                        dataset=dataset,
                        repeat_factor=1,
                        max_gpu_mem_mb=max_gpu_mem_mb,
                        gpu_node_batch_size=gpu_node_batch_size,
                        min_support_count=min_support_count,
                        min_confidence=min_confidence,
                        verbose=verbose,
                        include_dataset_profile=include_dataset_profile,
                        autotune=False,
                    )
                    row = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "dataset_key": dataset,
                        "gpu_kernel_min_work": int(threshold),
                        "run_index": int(run_index),
                        "status": str(result.get("status", "")),
                        "actual_backend": str(result.get("actual_backend", "")),
                        "elapsed_seconds": float(result.get("elapsed_seconds", 0.0)),
                        "rule_count": int(result.get("rule_count", 0)),
                        "gpu_node_batch_size": result.get("gpu_node_batch_size"),
                        "max_gpu_mem_mb": result.get("max_gpu_mem_mb"),
                    }
                    records.append(row)
                    print(
                        f"[dataset={dataset}] threshold={threshold} run={run_index + 1}/{run_count} "
                        f"backend={row['actual_backend']} elapsed={row['elapsed_seconds']:.6f}s"
                    )
    finally:
        CandidateGenerator._gpu_kernel_min_work = original_threshold

    summary = _summarize_records(records)
    winners = _best_thresholds(summary)
    return records, summary, winners


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep GPU kernel dispatch threshold (_gpu_kernel_min_work) using benchmark_runner."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="telco,adult,census_income",
        help="Comma-separated dataset presets (use --list-datasets to view choices).",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0,128,256,512,1024,2048",
        help="Comma-separated non-negative thresholds to test.",
    )
    parser.add_argument(
        
        "--runs",
        type=int,
        default=3,
        help="Repetitions per (dataset, threshold) pair.",
    )
    parser.add_argument(
        "--max-gpu-mem-mb",
        type=int,
        default=None,
        help="Optional CuPy pool cap in MB.",
    )
    parser.add_argument(
        "--gpu-node-batch-size",
        type=int,
        default=None,
        help="Optional fixed BFS node-batch size.",
    )
    parser.add_argument(
        "--min-support-count",
        type=int,
        default=None,
        help="Optional support threshold override (absolute count).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Optional confidence threshold override.",
    )
    parser.add_argument(
        "--include-dataset-profile",
        action="store_true",
        help="Also compute dataset profile in each run (slower).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_DIR / "comparison_suites" / "data",
        help="Directory where sweep CSV/JSONL outputs are saved.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional filename tag.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print available dataset presets and exit.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mining output.")
    args = parser.parse_args()

    if args.list_datasets:
        print("Available datasets:", ", ".join(list_dataset_presets()))
        return

    datasets = _parse_datasets(args.datasets)
    thresholds = _parse_thresholds(args.thresholds)
    runs = max(1, int(args.runs))

    records, summary, winners = run_sweep(
        datasets=datasets,
        thresholds=thresholds,
        runs=runs,
        max_gpu_mem_mb=args.max_gpu_mem_mb,
        gpu_node_batch_size=args.gpu_node_batch_size,
        min_support_count=args.min_support_count,
        min_confidence=args.min_confidence,
        include_dataset_profile=bool(args.include_dataset_profile),
        verbose=bool(args.verbose),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""

    records_jsonl = output_dir / f"gpu_kernel_threshold_records{tag}_{ts}.jsonl"
    records_csv = output_dir / f"gpu_kernel_threshold_records{tag}_{ts}.csv"
    summary_csv = output_dir / f"gpu_kernel_threshold_summary{tag}_{ts}.csv"
    winners_json = output_dir / f"gpu_kernel_threshold_best{tag}_{ts}.json"

    _write_jsonl(records_jsonl, records)
    _write_csv(records_csv, records)
    _write_csv(summary_csv, summary)
    winners_json.write_text(json.dumps(winners, indent=2), encoding="utf-8")

    print("")
    print(f"Saved records JSONL: {records_jsonl}")
    print(f"Saved records CSV:   {records_csv}")
    print(f"Saved summary CSV:   {summary_csv}")
    print(f"Saved best JSON:     {winners_json}")
    if winners:
        print("")
        print("Best threshold per dataset (by mean elapsed seconds):")
        for row in winners:
            print(
                f"  {row['dataset_key']}: {row['best_gpu_kernel_min_work']} "
                f"(mean={row['best_mean_elapsed_seconds']:.6f}s, runs={row['runs']})"
            )
    else:
        print("")
        print("No GPU runs were recorded; check CuPy availability and run status.")


if __name__ == "__main__":
    main()
