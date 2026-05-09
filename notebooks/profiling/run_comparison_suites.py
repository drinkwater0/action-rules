"""Run the FIM/itemset benchmark suite on benchmark datasets.

This used to host three suites (FIM, action-rules vs Sykora, ActionRules
CPU-vs-GPU bitset). The action-rules-vs-Sykora and CPU-vs-GPU comparisons have
both moved to ``commit_benchmarks/`` so all action-rules timings come out of
one isolated-subprocess framework. What's left here is the itemset-mining
comparison that produces the FIM headline figure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
FIM_COMPARE_DIR = CURRENT_DIR / "fim_compare"

if str(FIM_COMPARE_DIR) not in sys.path:
    sys.path.insert(0, str(FIM_COMPARE_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

DEFAULT_DATASET_PRESETS = ["telco", "adult", "census_income"]

from fim_comparison import (  # noqa: E402
    ALGORITHM_ALIASES,
    SUPPORTED_ALGORITHMS,
    run_benchmark,
    summarize_records,
)
from profile_telco_bitset import DATASET_PRESETS  # noqa: E402


DEFAULT_DATASET_PATHS = [DATASET_PRESETS[key].path.resolve() for key in DEFAULT_DATASET_PRESETS]

FIM_ALGORITHMS = [
    "bitset_fim_cpu",
    "bitset_fim_gpu",
    "pyfim_apriori",
    "pyfim_eclat",
    "mlxtend_apriori",
    "mlxtend_fpgrowth",
    "spmf_fpgrowth",
    "spmf_eclat",
]


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    algorithms: list[str]
    runs: int
    warmup_runs: int
    min_support_ratio: float
    min_confidence: float
    max_len: int
    max_apyori_records: int


def _parse_path_list(raw: str) -> list[Path]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(Path(token).expanduser().resolve())
    if not values:
        raise ValueError("Expected at least one dataset path.")
    return values


def _parse_canonical_list(
    raw: str,
    default_values: list[str],
    aliases: dict[str, str],
    supported_values: set[str],
) -> list[str]:
    value = raw.strip()
    if not value:
        return list(default_values)

    values = []
    seen = set()
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        canonical = aliases.get(token, token)
        if canonical not in supported_values:
            allowed = sorted(set(supported_values).union(set(aliases)))
            raise ValueError(f"Unsupported algorithm '{token}'. Allowed: {allowed}")
        if canonical in seen:
            continue
        seen.add(canonical)
        values.append(canonical)

    if not values:
        raise ValueError("Expected at least one algorithm.")
    return values


def _parse_fim_algorithm_list(raw: str, default_values: list[str]) -> list[str]:
    return _parse_canonical_list(
        raw=raw,
        default_values=default_values,
        aliases=ALGORITHM_ALIASES,
        supported_values=SUPPORTED_ALGORITHMS,
    )


def _save_summary(records: list[dict], output_dir: Path, summary_stem: str) -> Path:
    summary = summarize_records(records)
    summary_path = output_dir / f"{summary_stem}.csv"
    if summary.empty:
        pd.DataFrame(
            columns=["dataset_path", "algorithm", "runs", "mean_s", "median_s", "std_s"]
        ).to_csv(summary_path, index=False)
    else:
        summary.to_csv(summary_path, index=False)
    return summary_path


def _run_fim_suite(
    *,
    config: SuiteConfig,
    dataset_paths: list[Path],
    output_dir: Path,
    tag: str,
    spmf_jar: Path | None,
    spmf_timeout_sec: int,
    python_algo_timeout_sec: int | None,
) -> dict:
    suite_tag = "_".join(x for x in [tag, config.name] if x)
    records, output_paths = run_benchmark(
        repeat_factors=[1],
        runs=int(config.runs),
        warmup_runs=int(config.warmup_runs),
        algorithms=list(config.algorithms),
        min_support_count=1,
        min_support_ratio=float(config.min_support_ratio),
        min_confidence=float(config.min_confidence),
        max_len=int(config.max_len),
        max_apyori_records=int(config.max_apyori_records),
        output_dir=output_dir,
        tag=suite_tag,
        spmf_jar=spmf_jar,
        spmf_timeout_sec=int(spmf_timeout_sec),
        python_algo_timeout_sec=python_algo_timeout_sec,
        dataset_paths=dataset_paths,
        dataset_sep="auto",
        tx_columns=None,
    )
    summary_path = _save_summary(records, output_dir, f"{suite_tag}_summary")
    return {
        "suite": config.name,
        "algorithms": list(config.algorithms),
        "runs": int(config.runs),
        "warmup_runs": int(config.warmup_runs),
        "min_support_ratio": float(config.min_support_ratio),
        "min_confidence": float(config.min_confidence),
        "max_len": int(config.max_len),
        "output_paths": {k: str(v) for k, v in output_paths.items()},
        "summary_path": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the FIM/itemset benchmark suite. "
            "(Action-rules and CPU-vs-GPU comparisons live in commit_benchmarks/.)"
        )
    )
    parser.add_argument(
        "--dataset-paths",
        type=str,
        default=",".join(str(p) for p in DEFAULT_DATASET_PATHS),
        help="Comma-separated dataset paths for FIM suite.",
    )
    parser.add_argument(
        "--fim-algorithms",
        type=str,
        default="",
        help=(
            "Optional comma-separated algorithm override for FIM suite. "
            f"Defaults to: {','.join(FIM_ALGORITHMS)}"
        ),
    )
    parser.add_argument("--runs-fim", type=int, default=10, help="Measured runs per dataset/algorithm for FIM suite.")
    parser.add_argument("--warmup-fim", type=int, default=2, help="Warmup runs per dataset for FIM suite.")
    parser.add_argument(
        "--min-support-ratio-fim",
        type=float,
        default=0.05,
        help="Minimum support ratio for FIM itemset suite.",
    )
    parser.add_argument("--max-len", type=int, default=3, help="Maximum itemset/rule length for FIM suite.")
    parser.add_argument(
        "--max-apyori-records",
        type=int,
        default=200000,
        help="Safety cap for apyori output records in FIM suite when enabled.",
    )
    parser.add_argument(
        "--spmf-jar",
        type=Path,
        default=(CURRENT_DIR / "fim_compare" / "spmf.jar"),
        help="Path to SPMF JAR for FIM suite.",
    )
    parser.add_argument("--spmf-timeout-sec", type=int, default=300, help="Timeout per SPMF invocation.")
    parser.add_argument(
        "--python-algo-timeout-sec",
        type=int,
        default=600,
        help=(
            "Wall-clock cap (seconds) per in-process Python FIM algorithm call "
            "(apyori, pyfim_*, mlxtend_*). Use 0 to disable. Default: 600."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_DIR / "comparison_suites" / "data",
        help="Output directory for suite manifests and summaries.",
    )
    parser.add_argument("--tag", type=str, default="", help="Optional tag suffix for output files.")
    args = parser.parse_args()

    dataset_paths = _parse_path_list(args.dataset_paths)
    for path in dataset_paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    manifest_path = args.output_dir / f"comparison_suites_manifest{suffix}_{ts}.json"

    spmf_jar = args.spmf_jar if args.spmf_jar and args.spmf_jar.exists() else None

    fim_algorithms = _parse_fim_algorithm_list(args.fim_algorithms, FIM_ALGORITHMS)
    if spmf_jar is None:
        filtered = [a for a in fim_algorithms if not a.startswith("spmf_")]
        removed = sorted(set(fim_algorithms) - set(filtered))
        fim_algorithms = filtered
        if removed:
            print(
                "SPMF jar not found; removing SPMF algorithms from FIM suite:",
                ", ".join(removed),
            )
    fim_config = SuiteConfig(
        name="fim_itemsets",
        algorithms=fim_algorithms,
        runs=max(1, int(args.runs_fim)),
        warmup_runs=max(0, int(args.warmup_fim)),
        min_support_ratio=float(args.min_support_ratio_fim),
        min_confidence=0.6,
        max_len=max(1, int(args.max_len)),
        max_apyori_records=max(1, int(args.max_apyori_records)),
    )
    fim_result = _run_fim_suite(
        config=fim_config,
        dataset_paths=dataset_paths,
        output_dir=args.output_dir,
        tag=args.tag,
        spmf_jar=spmf_jar,
        spmf_timeout_sec=max(1, int(args.spmf_timeout_sec)),
        python_algo_timeout_sec=(
            None if int(args.python_algo_timeout_sec) <= 0 else int(args.python_algo_timeout_sec)
        ),
    )

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_paths": [str(x) for x in dataset_paths],
        "results": [fim_result],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nSuite manifest:")
    print(manifest_path)
    print(f"\n- {fim_result['suite']}")
    print(f"  summary: {fim_result['summary_path']}")
    print("\nPlot command:")
    print(
        "python notebooks/profiling/plot_comparison_suites.py "
        f"--manifest \"{manifest_path}\""
    )


if __name__ == "__main__":
    main()
