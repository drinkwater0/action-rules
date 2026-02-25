"""Profile ActionRules on Telco data using the bitset-only implementation."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Optional
from datetime import datetime, timezone
import sys

import pandas as pd

# Allow running directly from repository without package install.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from action_rules import ActionRules


def run_profile(
    use_gpu: bool,
    repeat_factor: int = 20,
    max_gpu_mem_mb: Optional[int] = None,
    gpu_batch_size: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    stable_attributes = ["gender", "SeniorCitizen", "Partner"]
    flexible_attributes = [
        "PhoneService",
        "InternetService",
        "OnlineSecurity",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
    ]
    target = "Churn"
    min_stable_attributes = 2
    min_flexible_attributes = 1
    min_undesired_support = 80
    min_undesired_confidence = 0.6
    min_desired_support = 80
    min_desired_confidence = 0.6
    undesired_state = "Yes"
    desired_state = "No"

    pd.set_option("display.max_columns", None)
    data_path = Path(__file__).resolve().parent.parent / "data" / "telco.csv"
    data_frame = pd.read_csv(data_path, sep=";")
    data_frame = pd.concat([data_frame] * repeat_factor, ignore_index=True)

    action_rules = ActionRules(
        min_stable_attributes=min_stable_attributes,
        min_flexible_attributes=min_flexible_attributes,
        min_undesired_support=min_undesired_support,
        min_undesired_confidence=min_undesired_confidence,
        min_desired_support=min_desired_support,
        min_desired_confidence=min_desired_confidence,
        verbose=verbose,
    )

    started = perf_counter()
    action_rules.fit(
        data=data_frame,
        stable_attributes=stable_attributes,
        flexible_attributes=flexible_attributes,
        target=target,
        target_undesired_state=undesired_state,
        target_desired_state=desired_state,
        use_gpu=use_gpu,
        max_gpu_mem_mb=max_gpu_mem_mb,
        gpu_batch_size=gpu_batch_size,
    )
    elapsed = perf_counter() - started

    rule_count = len(action_rules.get_rules().action_rules)
    mode = "GPU bitset" if use_gpu else "CPU bitset"
    print(f"Mode: {mode}")
    print(f"Rows: {len(data_frame)}")
    print(f"Number of action rules: {rule_count}")
    print(f"Elapsed seconds: {elapsed:.6f}")
    return {
        "mode": mode,
        "use_gpu": bool(use_gpu),
        "repeat_factor": int(repeat_factor),
        "rows": int(len(data_frame)),
        "rule_count": int(rule_count),
        "elapsed_seconds": float(elapsed),
        "max_gpu_mem_mb": None if max_gpu_mem_mb is None else int(max_gpu_mem_mb),
        "gpu_batch_size": None if gpu_batch_size is None else int(gpu_batch_size),
    }


def _write_metrics(result: dict, metrics_dir: Path, metrics_tag: str = "") -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    mode_slug = "gpu" if result.get("use_gpu") else "cpu"
    repeat = result.get("repeat_factor", "x")
    tag = f"_{metrics_tag}" if metrics_tag else ""
    out_path = metrics_dir / f"{mode_slug}_repeat{repeat}{tag}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile ActionRules bitset path on Telco dataset.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (CuPy) for the bitset path.")
    parser.add_argument(
        "--repeat-factor",
        type=int,
        default=20,
        help="How many times to concatenate the source dataset.",
    )
    parser.add_argument(
        "--max-gpu-mem-mb",
        type=int,
        default=None,
        help="Optional CuPy memory-pool limit in MB.",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=None,
        help="Optional candidate batch size for GPU batch processing.",
    )
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Save run metrics JSON into --metrics-dir.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_runs",
        help="Directory for saved run metrics JSON files.",
    )
    parser.add_argument(
        "--metrics-tag",
        type=str,
        default="",
        help="Optional tag appended to metrics filename.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mining output.")
    args = parser.parse_args()

    result = run_profile(
        use_gpu=args.gpu,
        repeat_factor=args.repeat_factor,
        max_gpu_mem_mb=args.max_gpu_mem_mb,
        gpu_batch_size=args.gpu_batch_size,
        verbose=args.verbose,
    )
    if args.save_metrics:
        out_path = _write_metrics(result, args.metrics_dir, args.metrics_tag)
        print(f"Saved run metrics: {out_path}")


if __name__ == "__main__":
    main()
