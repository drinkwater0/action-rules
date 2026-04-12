"""CPU bitset profiling wrapper with optional metrics export."""

import argparse
from pathlib import Path

from profile_telco_bitset import _write_metrics, run_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU bitset profiling workload.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="telco",
        help="Dataset preset: telco,adult,census_income",
    )
    parser.add_argument(
        "--max-gpu-mem-mb",
        type=int,
        default=None,
        help="Accepted for CLI compatibility, ignored on CPU mode.",
    )
    parser.add_argument(
        "--gpu-node-batch-size",
        type=int,
        default=None,
        help="Accepted for CLI compatibility, ignored on CPU mode.",
    )
    parser.add_argument(
        "--gpu-batch-size",
        dest="gpu_node_batch_size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
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
        use_gpu=False,
        dataset=args.dataset,
        repeat_factor=1,
        max_gpu_mem_mb=args.max_gpu_mem_mb,
        gpu_node_batch_size=args.gpu_node_batch_size,
        verbose=args.verbose,
    )
    if args.save_metrics:
        out_path = _write_metrics(result, args.metrics_dir, args.metrics_tag)
        print(f"Saved run metrics: {out_path}")


if __name__ == "__main__":
    main()
