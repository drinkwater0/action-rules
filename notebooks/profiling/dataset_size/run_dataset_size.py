"""Benchmark CPU/GPU bitset runtime across dataset-size repeat factors."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Iterable


CURRENT_DIR = Path(__file__).resolve().parent
PROFILING_DIR = CURRENT_DIR.parent
if str(PROFILING_DIR) not in sys.path:
    sys.path.insert(0, str(PROFILING_DIR))

from profile_telco_bitset import run_profile  # noqa: E402


def _parse_int_list(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_modes(raw: str) -> list[str]:
    allowed = {"cpu", "gpu"}
    modes = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in allowed:
            raise ValueError(f"Unsupported mode '{token}'. Allowed: cpu,gpu")
        modes.append(token)
    if not modes:
        raise ValueError("Expected at least one mode.")
    return modes


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataset-size benchmark for CPU/GPU bitset modes.")
    parser.add_argument(
        "--repeat-factors",
        type=str,
        default="20,200",
        help="Comma-separated repeat factors (e.g. 1,20,100,200).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs per (mode, repeat_factor).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="cpu,gpu",
        help="Comma-separated modes: cpu,gpu",
    )
    parser.add_argument(
        "--max-gpu-mem-mb",
        type=int,
        default=None,
        help="Optional CuPy memory-pool limit in MB for GPU mode.",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=None,
        help="Optional candidate batch size for GPU mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_DIR / "data",
        help="Output directory for benchmark data files.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag added to output file names.",
    )
    args = parser.parse_args()

    repeat_factors = _parse_int_list(args.repeat_factors)
    modes = _parse_modes(args.modes)
    run_count = max(1, int(args.runs))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    jsonl_path = args.output_dir / f"dataset_size_runs{suffix}_{ts}.jsonl"
    csv_path = args.output_dir / f"dataset_size_runs{suffix}_{ts}.csv"
    latest_jsonl = args.output_dir / "dataset_size_runs_latest.jsonl"
    latest_csv = args.output_dir / "dataset_size_runs_latest.csv"

    all_rows = []
    for mode in modes:
        use_gpu = mode == "gpu"
        for repeat_factor in repeat_factors:
            for run_index in range(run_count):
                print(f"mode={mode} repeat={repeat_factor} run={run_index + 1}/{run_count}")
                result = run_profile(
                    use_gpu=use_gpu,
                    repeat_factor=repeat_factor,
                    max_gpu_mem_mb=args.max_gpu_mem_mb,
                    gpu_batch_size=args.gpu_batch_size,
                    verbose=False,
                )
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "mode_key": mode,
                    "run_index": run_index,
                    **result,
                }
                all_rows.append(row)

    _write_jsonl(jsonl_path, all_rows)
    _write_csv(csv_path, all_rows)
    _write_jsonl(latest_jsonl, all_rows)
    _write_csv(latest_csv, all_rows)

    print(f"\nSaved JSONL: {jsonl_path}")
    print(f"Saved CSV:   {csv_path}")
    print(f"Updated:     {latest_jsonl}")
    print(f"Updated:     {latest_csv}")


if __name__ == "__main__":
    main()
