"""Run ActionRules on CPU bitset path for core-usage profiling."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

# Allow running directly from the repository without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from action_rules import ActionRules


def run_cpu_bitset(repeat_factor: int = 20, verbose: bool = False) -> tuple[int, float]:
    """Execute one CPU bitset mining run and return (rule_count, elapsed_seconds)."""
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

    data_path = Path(__file__).resolve().parents[1] / "data" / "telco.csv"
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
        use_gpu=False,
    )
    elapsed = perf_counter() - started

    rule_count = len(action_rules.get_rules().action_rules)
    print("Mode: CPU bitset")
    print(f"Rows: {len(data_frame)}")
    print(f"Number of action rules: {rule_count}")
    print(f"Elapsed seconds: {elapsed:.6f}")
    return rule_count, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU bitset profiling workload.")
    parser.add_argument(
        "--repeat-factor",
        type=int,
        default=20,
        help="How many times to concatenate the source telco dataset.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mining output.")
    args = parser.parse_args()

    run_cpu_bitset(repeat_factor=args.repeat_factor, verbose=args.verbose)


if __name__ == "__main__":
    main()
