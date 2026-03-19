"""Profile ActionRules bitset path on benchmark datasets (no synthetic repetition)."""

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Optional
import sys

import pandas as pd

# Allow running directly from repository without package install.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from action_rules import ActionRules


DATA_DIR = REPO_ROOT / "notebooks" / "data"


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    path: Path
    sep: str
    stable_attributes: tuple[str, ...]
    flexible_attributes: tuple[str, ...]
    target: str
    undesired_state: str
    desired_state: str
    min_stable_attributes: int
    min_flexible_attributes: int
    min_undesired_support: int
    min_undesired_confidence: float
    min_desired_support: int
    min_desired_confidence: float


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "bank": DatasetPreset(
        name="bank",
        path=DATA_DIR / "bank-full.csv",
        sep=";",
        stable_attributes=("marital", "education", "default"),
        flexible_attributes=("job", "housing", "loan", "contact", "poutcome"),
        target="y",
        undesired_state="yes",
        desired_state="no",
        min_stable_attributes=2,
        min_flexible_attributes=1,
        min_undesired_support=600,
        min_undesired_confidence=0.6,
        min_desired_support=600,
        min_desired_confidence=0.6,
    ),
    "german": DatasetPreset(
        name="german",
        path=DATA_DIR / "german.csv",
        sep=",",
        stable_attributes=("status", "personal_status_sex", "housing"),
        flexible_attributes=("credit_history", "purpose", "savings", "employment_since", "property"),
        target="class",
        undesired_state="2",
        desired_state="1",
        min_stable_attributes=2,
        min_flexible_attributes=1,
        min_undesired_support=30,
        min_undesired_confidence=0.6,
        min_desired_support=30,
        min_desired_confidence=0.6,
    ),
    "covtype": DatasetPreset(
        name="covtype",
        path=DATA_DIR / "covtype.csv",
        sep=",",
        stable_attributes=("wilderness_area_1", "wilderness_area_2", "wilderness_area_3", "wilderness_area_4"),
        flexible_attributes=(
            "soil_type_1",
            "soil_type_2",
            "soil_type_3",
            "soil_type_4",
            "soil_type_5",
            "soil_type_6",
            "soil_type_7",
            "soil_type_8",
        ),
        target="cover_type",
        undesired_state="2",
        desired_state="1",
        min_stable_attributes=2,
        min_flexible_attributes=1,
        min_undesired_support=5000,
        min_undesired_confidence=0.6,
        min_desired_support=5000,
        min_desired_confidence=0.6,
    ),
    "adult": DatasetPreset(
        name="adult",
        path=DATA_DIR / "adult.csv",
        sep=",",
        stable_attributes=("sex", "race", "native_country"),
        flexible_attributes=("workclass", "education", "occupation", "marital_status", "relationship"),
        target="income",
        undesired_state="<=50K",
        desired_state=">50K",
        min_stable_attributes=2,
        min_flexible_attributes=1,
        min_undesired_support=500,
        min_undesired_confidence=0.6,
        min_desired_support=500,
        min_desired_confidence=0.6,
    ),
    "census_income": DatasetPreset(
        name="census_income",
        path=DATA_DIR / "census_income.csv",
        sep=",",
        stable_attributes=("sex", "race", "citizenship", "year"),
        flexible_attributes=(
            "class_of_worker",
            "education",
            "marital_stat",
            "major_industry_code",
            "major_occupation_code",
            "full_or_part_time_employment_stat",
        ),
        target="income",
        undesired_state="<=50K",
        desired_state=">50K",
        min_stable_attributes=2,
        min_flexible_attributes=1,
        min_undesired_support=1000,
        min_undesired_confidence=0.6,
        min_desired_support=1000,
        min_desired_confidence=0.6,
    ),
    "telco": DatasetPreset(
        name="telco",
        path=DATA_DIR / "telco.csv",
        sep=";",
        stable_attributes=("gender", "SeniorCitizen", "Partner"),
        flexible_attributes=(
            "PhoneService",
            "InternetService",
            "OnlineSecurity",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
        ),
        target="Churn",
        undesired_state="Yes",
        desired_state="No",
        min_stable_attributes=2,
        min_flexible_attributes=1,
        min_undesired_support=80,
        min_undesired_confidence=0.6,
        min_desired_support=80,
        min_desired_confidence=0.6,
    ),
}

DATASET_ALIASES = {
    "bank-full": "bank",
    "bank-full.csv": "bank",
    "german.csv": "german",
    "covtype.csv": "covtype",
    "adult.csv": "adult",
    "census_income.csv": "census_income",
    "census-income.csv": "census_income",
    "census": "census_income",
    "adult_income": "adult",
    "telco.csv": "telco",
}


def list_dataset_presets() -> list[str]:
    return sorted(DATASET_PRESETS.keys())


def _normalize_dataset_key(dataset: str) -> str:
    key = dataset.strip().lower()
    key = DATASET_ALIASES.get(key, key)
    if key not in DATASET_PRESETS:
        allowed = ", ".join(list_dataset_presets())
        raise ValueError(f"Unsupported dataset '{dataset}'. Allowed: {allowed}")
    return key


def _detect_separator(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:65536]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        if path.name.lower() == "telco.csv":
            return ";"
        return ","


def _load_frame(path: Path, sep: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    real_sep = _detect_separator(path) if sep == "auto" else sep
    return pd.read_csv(path, sep=real_sep)


def run_profile(
    use_gpu: bool,
    dataset: str = "bank",
    repeat_factor: int = 1,
    max_gpu_mem_mb: Optional[int] = None,
    gpu_batch_size: Optional[int] = None,
    min_support_count: Optional[int] = None,
    min_confidence: Optional[float] = None,
    verbose: bool = False,
) -> dict:
    if int(repeat_factor) != 1:
        raise ValueError(
            "Synthetic dataset multiplication is disabled. "
            "Use real datasets (bank/german/covtype/adult/census_income/telco) without --repeat-factor > 1."
        )

    preset = DATASET_PRESETS[_normalize_dataset_key(dataset)]
    data_frame = _load_frame(preset.path, preset.sep)

    missing = [
        col
        for col in [*preset.stable_attributes, *preset.flexible_attributes, preset.target]
        if col not in data_frame.columns
    ]
    if missing:
        raise ValueError(f"Dataset '{preset.name}' is missing required columns: {missing}")

    effective_support = (
        int(preset.min_desired_support)
        if min_support_count is None
        else max(1, int(min_support_count))
    )
    effective_confidence = (
        float(preset.min_desired_confidence)
        if min_confidence is None
        else float(min_confidence)
    )

    action_rules = ActionRules(
        min_stable_attributes=preset.min_stable_attributes,
        min_flexible_attributes=preset.min_flexible_attributes,
        min_undesired_support=effective_support,
        min_undesired_confidence=effective_confidence,
        min_desired_support=effective_support,
        min_desired_confidence=effective_confidence,
        verbose=verbose,
    )

    started = perf_counter()
    action_rules.fit(
        data=data_frame,
        stable_attributes=list(preset.stable_attributes),
        flexible_attributes=list(preset.flexible_attributes),
        target=preset.target,
        target_undesired_state=preset.undesired_state,
        target_desired_state=preset.desired_state,
        use_gpu=use_gpu,
        max_gpu_mem_mb=max_gpu_mem_mb,
        gpu_batch_size=gpu_batch_size,
    )
    elapsed = perf_counter() - started

    rule_count = len(action_rules.get_rules().action_rules)
    requested_backend = "gpu" if use_gpu else "cpu"
    actual_backend = "gpu" if action_rules.is_gpu_np else "cpu"
    status = "ok"
    note = ""
    if use_gpu and actual_backend != "gpu":
        status = "fallback_cpu"
        note = "GPU requested, but CuPy is unavailable. The benchmark ran on CPU."

    mode = "GPU bitset" if actual_backend == "gpu" else "CPU bitset"
    if note:
        mode += " [fallback]"
    print(f"Mode: {mode}")
    print(f"Dataset: {preset.name}")
    print(f"Rows: {len(data_frame)}")
    print(f"Number of action rules: {rule_count}")
    print(f"Elapsed seconds: {elapsed:.6f}")
    if note:
        print(f"Note: {note}")
    return {
        "status": status,
        "note": note,
        "mode": mode,
        "use_gpu": bool(use_gpu),
        "requested_backend": requested_backend,
        "actual_backend": actual_backend,
        "gpu_acceleration_active": bool(action_rules.is_gpu_np),
        "gpu_dataframe_active": bool(action_rules.is_gpu_pd),
        "dataset_key": preset.name,
        "dataset_path": str(preset.path),
        "rows": int(len(data_frame)),
        "rule_count": int(rule_count),
        "elapsed_seconds": float(elapsed),
        "max_gpu_mem_mb": None if max_gpu_mem_mb is None else int(max_gpu_mem_mb),
        "gpu_batch_size": None if gpu_batch_size is None else int(gpu_batch_size),
        "min_support_count_effective": int(effective_support),
        "min_confidence_effective": float(effective_confidence),
    }


def _write_metrics(result: dict, metrics_dir: Path, metrics_tag: str = "") -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    mode_slug = str(result.get("actual_backend") or ("gpu" if result.get("use_gpu") else "cpu"))
    dataset_slug = str(result.get("dataset_key", "dataset"))
    tag = f"_{metrics_tag}" if metrics_tag else ""
    out_path = metrics_dir / f"{mode_slug}_{dataset_slug}{tag}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile ActionRules bitset path on benchmark datasets.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (CuPy) for the bitset path.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="bank",
        help="Dataset preset: bank,german,covtype,adult,census_income,telco",
    )
    parser.add_argument(
        "--repeat-factor",
        type=int,
        default=1,
        help="Deprecated. Must stay 1 (synthetic row multiplication is disabled).",
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
        "--min-support-count",
        type=int,
        default=None,
        help="Optional override for support threshold (absolute count).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Optional override for confidence threshold.",
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

    result = run_profile(
        use_gpu=args.gpu,
        dataset=args.dataset,
        repeat_factor=args.repeat_factor,
        max_gpu_mem_mb=args.max_gpu_mem_mb,
        gpu_batch_size=args.gpu_batch_size,
        min_support_count=args.min_support_count,
        min_confidence=args.min_confidence,
        verbose=args.verbose,
    )
    if args.save_metrics:
        out_path = _write_metrics(result, args.metrics_dir, args.metrics_tag)
        print(f"Saved run metrics: {out_path}")


if __name__ == "__main__":
    main()
