"""Profile ActionRules on Telco data using the bitset-only implementation."""

import argparse
from pathlib import Path
from time import perf_counter
from typing import Optional

import pandas as pd

from action_rules import ActionRules


def run_profile(
    use_gpu: bool,
    repeat_factor: int = 20,
    max_gpu_mem_mb: Optional[int] = None,
    gpu_batch_size: Optional[int] = None,
    verbose: bool = False,
) -> int:
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
    print(f"Mode: {'GPU bitset' if use_gpu else 'CPU bitset'}")
    print(f"Rows: {len(data_frame)}")
    print(f"Number of action rules: {rule_count}")
    print(f"Elapsed seconds: {elapsed:.6f}")
    return rule_count


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
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mining output.")
    args = parser.parse_args()

    run_profile(
        use_gpu=args.gpu,
        repeat_factor=args.repeat_factor,
        max_gpu_mem_mb=args.max_gpu_mem_mb,
        gpu_batch_size=args.gpu_batch_size,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
