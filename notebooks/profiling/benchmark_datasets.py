"""Shared dataset presets for thesis benchmark scripts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
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
    "adult.csv": "adult",
    "census_income.csv": "census_income",
    "census-income.csv": "census_income",
    "census": "census_income",
    "adult_income": "adult",
    "telco.csv": "telco",
}


def list_dataset_presets() -> list[str]:
    return sorted(DATASET_PRESETS.keys())


def normalize_dataset_key(dataset: str) -> str:
    key = dataset.strip().lower()
    key = DATASET_ALIASES.get(key, key)
    if key not in DATASET_PRESETS:
        allowed = ", ".join(list_dataset_presets())
        raise ValueError(f"Unsupported dataset '{dataset}'. Allowed: {allowed}")
    return key


def detect_separator(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:65536]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        if path.name.lower() == "telco.csv":
            return ";"
        return ","


def load_frame(path: Path, sep: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    real_sep = detect_separator(path) if sep == "auto" else sep
    return pd.read_csv(path, sep=real_sep)
