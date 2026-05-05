"""Run thesis-ready benchmark suites on benchmark datasets.

This orchestrates three comparisons:
1) FIM comparison (itemset-mining programs where comparison is meaningful),
2) Action-rules comparison (programs that generate action rules),
3) ActionRules CPU vs GPU bitset comparison.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
FIM_COMPARE_DIR = CURRENT_DIR / "fim_compare"

if str(FIM_COMPARE_DIR) not in sys.path:
    sys.path.insert(0, str(FIM_COMPARE_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Defaults stay aligned across FIM and action-rules suites.
DEFAULT_DATASET_PRESETS = ["telco", "adult", "census_income"]

from fim_comparison import (  # noqa: E402
    ALGORITHM_ALIASES,
    SUPPORTED_ALGORITHMS,
    run_benchmark,
    summarize_records,
)
from profile_telco_bitset import DATASET_PRESETS, list_dataset_presets, run_profile  # noqa: E402


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

RULE_ALGORITHMS = [
    "action_rules_cpu",
    "action_rules_gpu",
    "action_rules_auto",
    "actionrules_sykora",
]

RULE_ALGORITHM_ALIASES = {
    "cpu": "action_rules_cpu",
    "gpu": "action_rules_gpu",
    "auto": "action_rules_auto",
    "autotune": "action_rules_auto",
    "bitset_fim_cpu": "action_rules_cpu",
    "bitset_fim_gpu": "action_rules_gpu",
    "actionrules": "actionrules_sykora",
    "actionrulesdiscovery": "actionrules_sykora",
    "actionrules_lukassykora": "actionrules_sykora",
    "aras": "actionrules_sykora",
    "sykora": "actionrules_sykora",
}

SUPPORTED_RULE_ALGORITHMS = {
    "action_rules_cpu",
    "action_rules_gpu",
    "action_rules_auto",
    "actionrules_sykora",
}

BITSET_MODES = ["cpu", "gpu"]

BITSET_MODE_ALIASES = {
    "autotune": "auto",
}

SUPPORTED_BITSET_MODES = {
    "cpu",
    "gpu",
    "auto",
}


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


def _parse_dataset_presets(raw: str) -> list[str]:
    values = []
    seen = set()
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in seen:
            continue
        values.append(token)
        seen.add(token)
    if not values:
        raise ValueError("Expected at least one dataset preset.")
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


def _parse_rule_algorithm_list(raw: str, default_values: list[str]) -> list[str]:
    return _parse_canonical_list(
        raw=raw,
        default_values=default_values,
        aliases=RULE_ALGORITHM_ALIASES,
        supported_values=SUPPORTED_RULE_ALGORITHMS,
    )


def _parse_bitset_mode_list(raw: str, default_values: list[str]) -> list[str]:
    return _parse_canonical_list(
        raw=raw,
        default_values=default_values,
        aliases=BITSET_MODE_ALIASES,
        supported_values=SUPPORTED_BITSET_MODES,
    )


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


def _append_jsonl_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        handle.flush()


def _append_csv_row(path: Path, row: dict) -> None:
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


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


def _run_actionrules_sykora(
    *,
    preset_key: str,
    min_support_count: int,
    min_confidence: float,
) -> dict[str, Any]:
    preset = DATASET_PRESETS[preset_key]
    try:
        module = importlib.import_module("actionrules.actionRulesDiscovery")
    except ModuleNotFoundError as exc:
        return {
            "status": "missing_dependency",
            "note": f"{exc.__class__.__name__}: {exc}",
        }

    runner_cls = getattr(module, "ActionRulesDiscovery", None)
    if runner_cls is None:
        return {
            "status": "error",
            "note": "Module 'actionrules.actionRulesDiscovery' has no ActionRulesDiscovery class.",
        }

    started = perf_counter()
    data_frame = pd.read_csv(preset.path, sep=preset.sep)
    runner = runner_cls()
    runner.load_pandas(data_frame)

    fit_variants = [
        {
            "stable_attributes": list(preset.stable_attributes),
            "flexible_attributes": list(preset.flexible_attributes),
            "consequent": preset.target,
            "conf": float(min_confidence) * 100.0,
            "supp": -int(min_support_count),
            "desired_changes": [[preset.undesired_state, preset.desired_state]],
            "is_nan": False,
            "is_reduction": True,
            "is_strict_flexible": False,
            "min_stable_attributes": int(preset.min_stable_attributes),
            "min_flexible_attributes": int(preset.min_flexible_attributes),
            "max_stable_attributes": 100,
            "max_flexible_attributes": 100,
        },
        {
            "stable_attributes": list(preset.stable_attributes),
            "flexible_attributes": list(preset.flexible_attributes),
            "consequent": preset.target,
            "conf": float(min_confidence) * 100.0,
            "supp": -int(min_support_count),
            "desired_classes": [preset.desired_state],
            "is_nan": False,
            "is_reduction": True,
            "is_strict_flexible": False,
            "min_stable_attributes": int(preset.min_stable_attributes),
            "min_flexible_attributes": int(preset.min_flexible_attributes),
            "max_stable_attributes": 100,
            "max_flexible_attributes": 100,
        },
    ]

    fit_error: Exception | None = None
    fit_variant = "unknown"
    for idx, kwargs in enumerate(fit_variants, start=1):
        try:
            runner.fit(**kwargs)
            fit_variant = f"variant_{idx}"
            fit_error = None
            break
        except TypeError as exc:
            fit_error = exc
            continue
        except Exception as exc:  # pragma: no cover - defensive fallback
            # actionrules-lukassykora may raise KeyError if no classification
            # rules include the consequent at the current thresholds.
            if isinstance(exc, KeyError):
                message = str(exc)
                if preset.target in message and "are in the [columns]" in message:
                    elapsed = perf_counter() - started
                    return {
                        "status": "ok",
                        "note": "no_classification_rules_for_consequent",
                        "mode": "ActionRulesDiscovery",
                        "use_gpu": False,
                        "gpu_acceleration_active": False,
                        "gpu_dataframe_active": False,
                        "dataset_key": preset.name,
                        "dataset_path": str(preset.path),
                        "rows": int(len(data_frame)),
                        "rule_count": 0,
                        "elapsed_seconds": float(elapsed),
                        "max_gpu_mem_mb": None,
                        "gpu_node_batch_size": None,
                        "min_support_count_effective": int(min_support_count),
                        "min_confidence_effective": float(min_confidence),
                    }
            fit_error = exc
            break

    if fit_error is not None:
        return {
            "status": "error",
            "note": f"{fit_error.__class__.__name__}: {fit_error}",
        }

    rule_count: int | None = None
    count_note = "get_action_rules"
    try:
        rule_count = int(len(runner.get_action_rules()))
    except Exception:
        try:
            rule_count = int(len(runner.get_action_rules_representation()))
            count_note = "get_action_rules_representation"
        except Exception as exc:
            return {
                "status": "error",
                "note": f"Unable to read rule count: {exc.__class__.__name__}: {exc}",
            }

    elapsed = perf_counter() - started
    return {
        "status": "ok",
        "note": f"fit={fit_variant};count={count_note}",
        "mode": "ActionRulesDiscovery",
        "use_gpu": False,
        "gpu_acceleration_active": False,
        "gpu_dataframe_active": False,
        "dataset_key": preset.name,
        "dataset_path": str(preset.path),
        "rows": int(len(data_frame)),
        "rule_count": int(rule_count),
        "elapsed_seconds": float(elapsed),
        "max_gpu_mem_mb": None,
        "gpu_node_batch_size": None,
        "min_support_count_effective": int(min_support_count),
        "min_confidence_effective": float(min_confidence),
    }


def _run_action_rules_algorithm(
    *,
    algorithm: str,
    preset_key: str,
    min_support_count: int,
    min_confidence: float,
    max_gpu_mem_mb: int | None,
    gpu_node_batch_size: int | None,
    autotune_sample_frac: float,
    autotune_sample_min_rows: int,
    autotune_sample_max_rows: int,
    autotune_random_state: int,
) -> dict[str, Any]:
    if algorithm == "action_rules_cpu":
        result = run_profile(
            use_gpu=False,
            dataset=preset_key,
            repeat_factor=1,
            max_gpu_mem_mb=max_gpu_mem_mb,
            gpu_node_batch_size=gpu_node_batch_size,
            min_support_count=min_support_count,
            min_confidence=min_confidence,
            verbose=False,
        )
        return {
            "status": str(result.get("status", "ok")),
            "note": str(result.get("note", "")),
            **result,
        }

    if algorithm == "action_rules_gpu":
        result = run_profile(
            use_gpu=True,
            dataset=preset_key,
            repeat_factor=1,
            max_gpu_mem_mb=max_gpu_mem_mb,
            gpu_node_batch_size=gpu_node_batch_size,
            min_support_count=min_support_count,
            min_confidence=min_confidence,
            verbose=False,
        )
        return {
            "status": str(result.get("status", "ok")),
            "note": str(result.get("note", "")),
            **result,
        }

    if algorithm == "action_rules_auto":
        result = run_profile(
            use_gpu=False,
            dataset=preset_key,
            repeat_factor=1,
            max_gpu_mem_mb=max_gpu_mem_mb,
            gpu_node_batch_size=gpu_node_batch_size,
            min_support_count=min_support_count,
            min_confidence=min_confidence,
            autotune=True,
            autotune_sample_frac=autotune_sample_frac,
            autotune_sample_min_rows=autotune_sample_min_rows,
            autotune_sample_max_rows=autotune_sample_max_rows,
            autotune_random_state=autotune_random_state,
            verbose=False,
        )
        return {
            "status": str(result.get("status", "ok")),
            "note": str(result.get("note", "")),
            **result,
        }

    if algorithm == "actionrules_sykora":
        return _run_actionrules_sykora(
            preset_key=preset_key,
            min_support_count=min_support_count,
            min_confidence=min_confidence,
        )

    return {
        "status": "error",
        "note": f"Unsupported action-rules algorithm '{algorithm}'.",
    }


def _normalize_profile_result(
    result: dict[str, Any],
    *,
    fallback_min_support_count: int | None = None,
    fallback_min_confidence: float | None = None,
) -> dict[str, Any]:
    autotune = result.get("autotune")
    autotune_summary = autotune if isinstance(autotune, dict) else {}

    return {
        "status": str(result.get("status", "error")),
        "note": str(result.get("note", "")),
        "mode": result.get("mode"),
        "use_gpu": result.get("use_gpu"),
        "requested_backend": result.get("requested_backend"),
        "actual_backend": result.get("actual_backend"),
        "gpu_acceleration_active": result.get("gpu_acceleration_active"),
        "gpu_dataframe_active": result.get("gpu_dataframe_active"),
        "dataset_path": result.get("dataset_path"),
        "rows": result.get("rows"),
        "rule_count": result.get("rule_count"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "max_gpu_mem_mb": result.get("max_gpu_mem_mb"),
        "gpu_node_batch_size": result.get("gpu_node_batch_size"),
        "min_support_count_effective": result.get(
            "min_support_count_effective", fallback_min_support_count
        ),
        "min_confidence_effective": result.get(
            "min_confidence_effective", fallback_min_confidence
        ),
        "autotune_enabled": bool(autotune_summary.get("enabled", False)),
        "autotune_sample_rows": autotune_summary.get("sample_rows"),
        "autotune_sample_support_count_effective": autotune_summary.get(
            "sample_support_count_effective"
        ),
        "autotune_candidate_count": autotune_summary.get("candidate_count"),
        "autotune_selected_use_gpu": autotune_summary.get("selected_use_gpu"),
        "autotune_selected_gpu_node_batch_size": autotune_summary.get(
            "selected_gpu_node_batch_size"
        ),
        "autotune_selected_actual_backend_on_sample": autotune_summary.get(
            "selected_actual_backend_on_sample"
        ),
    }


def _summarize_action_rules(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset_key",
                "algorithm",
                "runs",
                "mean_s",
                "median_s",
                "std_s",
                "rule_count_min",
                "rule_count_max",
            ]
        )

    df = pd.DataFrame(rows)
    ok_df = df[(df["status"] == "ok") & df["elapsed_seconds"].notna()].copy()
    if ok_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset_key",
                "algorithm",
                "runs",
                "mean_s",
                "median_s",
                "std_s",
                "rule_count_min",
                "rule_count_max",
            ]
        )

    summary = (
        ok_df.groupby(["dataset_key", "algorithm"], as_index=False)
        .agg(
            runs=("elapsed_seconds", "count"),
            mean_s=("elapsed_seconds", "mean"),
            median_s=("elapsed_seconds", "median"),
            std_s=("elapsed_seconds", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
            rule_count_min=("rule_count", "min"),
            rule_count_max=("rule_count", "max"),
        )
        .sort_values(["dataset_key", "mean_s", "algorithm"])
    )
    return summary


def _status_summary(rows: list[dict], key_cols: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[*key_cols, "status", "count"])
    df = pd.DataFrame(rows)
    if "status" not in df.columns:
        return pd.DataFrame(columns=[*key_cols, "status", "count"])
    return (
        df.groupby([*key_cols, "status"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values([*key_cols, "status"])
    )


def _run_action_rules_suite(
    *,
    algorithms: list[str],
    dataset_presets: list[str],
    runs: int,
    warmup_runs: int,
    min_support_ratio: float | None,
    min_confidence: float,
    output_dir: Path,
    tag: str,
    max_gpu_mem_mb: int | None,
    gpu_node_batch_size: int | None,
    autotune_sample_frac: float,
    autotune_sample_min_rows: int,
    autotune_sample_max_rows: int,
    autotune_random_state: int,
) -> dict:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    jsonl_path = output_dir / f"action_rules_compare_runs{suffix}_{ts}.jsonl"
    csv_path = output_dir / f"action_rules_compare_runs{suffix}_{ts}.csv"
    latest_jsonl = output_dir / "action_rules_compare_runs_latest.jsonl"
    latest_csv = output_dir / "action_rules_compare_runs_latest.csv"

    for path in (jsonl_path, csv_path, latest_jsonl, latest_csv):
        if path.exists():
            path.unlink()

    rows: list[dict] = []

    for preset_key in dataset_presets:
        preset = DATASET_PRESETS[preset_key]
        frame = pd.read_csv(preset.path, sep=preset.sep)
        row_count = int(len(frame))
        if min_support_ratio is None:
            min_support_count = int(preset.min_desired_support)
            effective_min_support_ratio = float(min_support_count) / float(max(1, row_count))
        else:
            min_support_count = max(1, int(math.ceil(float(min_support_ratio) * row_count)))
            effective_min_support_ratio = float(min_support_ratio)

        for algorithm in algorithms:
            for warmup_index in range(max(0, int(warmup_runs))):
                print(
                    f"rules warmup dataset={preset_key} algorithm={algorithm} "
                    f"{warmup_index + 1}/{int(warmup_runs)}"
                )
                _ = _run_action_rules_algorithm(
                    algorithm=algorithm,
                    preset_key=preset_key,
                    min_support_count=min_support_count,
                    min_confidence=min_confidence,
                    max_gpu_mem_mb=max_gpu_mem_mb,
                    gpu_node_batch_size=gpu_node_batch_size,
                    autotune_sample_frac=autotune_sample_frac,
                    autotune_sample_min_rows=autotune_sample_min_rows,
                    autotune_sample_max_rows=autotune_sample_max_rows,
                    autotune_random_state=autotune_random_state,
                )

            for run_index in range(max(1, int(runs))):
                print(
                    f"rules run dataset={preset_key} algorithm={algorithm} "
                    f"{run_index + 1}/{int(runs)}"
                )
                try:
                    result = _run_action_rules_algorithm(
                        algorithm=algorithm,
                        preset_key=preset_key,
                        min_support_count=min_support_count,
                        min_confidence=min_confidence,
                        max_gpu_mem_mb=max_gpu_mem_mb,
                        gpu_node_batch_size=gpu_node_batch_size,
                        autotune_sample_frac=autotune_sample_frac,
                        autotune_sample_min_rows=autotune_sample_min_rows,
                        autotune_sample_max_rows=autotune_sample_max_rows,
                        autotune_random_state=autotune_random_state,
                    )
                except Exception as exc:  # pragma: no cover - defensive fallback
                    result = {
                        "status": "error",
                        "note": f"{exc.__class__.__name__}: {exc}",
                    }

                normalized = _normalize_profile_result(
                    result,
                    fallback_min_support_count=min_support_count,
                    fallback_min_confidence=min_confidence,
                )

                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "suite": "rule_search",
                    "dataset_key": preset_key,
                    "algorithm": algorithm,
                    "run_index": int(run_index),
                    "min_support_ratio": (
                        None if min_support_ratio is None else float(min_support_ratio)
                    ),
                    "min_support_ratio_effective": float(effective_min_support_ratio),
                    "min_confidence": float(min_confidence),
                    **normalized,
                }
                rows.append(row)
                _append_jsonl_row(jsonl_path, row)
                _append_csv_row(csv_path, row)
                _append_jsonl_row(latest_jsonl, row)
                _append_csv_row(latest_csv, row)

    summary = _summarize_action_rules(rows)
    summary_path = output_dir / f"action_rules_compare_summary{suffix}_{ts}.csv"
    summary.to_csv(summary_path, index=False)
    status_summary = _status_summary(rows, ["dataset_key", "algorithm"])
    status_summary_path = output_dir / f"action_rules_compare_status{suffix}_{ts}.csv"
    status_summary.to_csv(status_summary_path, index=False)

    return {
        "suite": "rule_search",
        "algorithms": list(algorithms),
        "runs": int(runs),
        "warmup_runs": int(warmup_runs),
        "min_support_ratio": None if min_support_ratio is None else float(min_support_ratio),
        "min_confidence": float(min_confidence),
        "output_paths": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "latest_jsonl": str(latest_jsonl),
            "latest_csv": str(latest_csv),
        },
        "summary_path": str(summary_path),
        "status_summary_path": str(status_summary_path),
    }


def _run_bitset_cpu_gpu_suite(
    *,
    dataset_presets: list[str],
    modes: list[str],
    runs: int,
    warmup_runs: int,
    output_dir: Path,
    tag: str,
    max_gpu_mem_mb: int | None,
    gpu_node_batch_size: int | None,
    autotune_sample_frac: float,
    autotune_sample_min_rows: int,
    autotune_sample_max_rows: int,
    autotune_random_state: int,
) -> dict:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    jsonl_path = output_dir / f"action_rules_bitset_runs{suffix}_{ts}.jsonl"
    csv_path = output_dir / f"action_rules_bitset_runs{suffix}_{ts}.csv"
    latest_jsonl = output_dir / "action_rules_bitset_runs_latest.jsonl"
    latest_csv = output_dir / "action_rules_bitset_runs_latest.csv"

    for path in (jsonl_path, csv_path, latest_jsonl, latest_csv):
        if path.exists():
            path.unlink()

    rows: list[dict] = []

    for dataset in dataset_presets:
        for mode in modes:
            use_gpu = mode == "gpu"
            autotune = mode == "auto"
            for warmup_index in range(max(0, int(warmup_runs))):
                print(
                    f"bitset warmup dataset={dataset} mode={mode} "
                    f"{warmup_index + 1}/{int(warmup_runs)}"
                )
                run_profile(
                    use_gpu=use_gpu,
                    dataset=dataset,
                    repeat_factor=1,
                    max_gpu_mem_mb=max_gpu_mem_mb,
                    gpu_node_batch_size=gpu_node_batch_size,
                    autotune=autotune,
                    autotune_sample_frac=autotune_sample_frac,
                    autotune_sample_min_rows=autotune_sample_min_rows,
                    autotune_sample_max_rows=autotune_sample_max_rows,
                    autotune_random_state=autotune_random_state,
                    verbose=False,
                )

            for run_index in range(max(1, int(runs))):
                print(f"bitset run dataset={dataset} mode={mode} {run_index + 1}/{int(runs)}")
                result = run_profile(
                    use_gpu=use_gpu,
                    dataset=dataset,
                    repeat_factor=1,
                    max_gpu_mem_mb=max_gpu_mem_mb,
                    gpu_node_batch_size=gpu_node_batch_size,
                    autotune=autotune,
                    autotune_sample_frac=autotune_sample_frac,
                    autotune_sample_min_rows=autotune_sample_min_rows,
                    autotune_sample_max_rows=autotune_sample_max_rows,
                    autotune_random_state=autotune_random_state,
                    verbose=False,
                )
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "suite": "bitset_cpu_vs_gpu",
                    "dataset_key": dataset,
                    "mode_key": mode,
                    "run_index": int(run_index),
                    **_normalize_profile_result(result),
                }
                rows.append(row)
                _append_jsonl_row(jsonl_path, row)
                _append_csv_row(csv_path, row)
                _append_jsonl_row(latest_jsonl, row)
                _append_csv_row(latest_csv, row)

    df = pd.DataFrame(rows)
    ok_df = df[(df["status"] == "ok") & df["elapsed_seconds"].notna()].copy() if not df.empty else pd.DataFrame()
    if ok_df.empty:
        summary = pd.DataFrame(
            columns=[
                "dataset_key",
                "mode_key",
                "runs",
                "mean_s",
                "median_s",
                "std_s",
                "rule_count_min",
                "rule_count_max",
                "gpu_acceleration_active_any",
            ]
        )
    else:
        summary = (
            ok_df.groupby(["dataset_key", "mode_key"], as_index=False)
            .agg(
                runs=("elapsed_seconds", "count"),
                mean_s=("elapsed_seconds", "mean"),
                median_s=("elapsed_seconds", "median"),
                std_s=("elapsed_seconds", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
                rule_count_min=("rule_count", "min"),
                rule_count_max=("rule_count", "max"),
                gpu_acceleration_active_any=("gpu_acceleration_active", "max"),
            )
            .sort_values(["dataset_key", "mode_key"])
        )
    summary_path = output_dir / f"action_rules_bitset_summary{suffix}_{ts}.csv"
    summary.to_csv(summary_path, index=False)
    status_summary = _status_summary(rows, ["dataset_key", "mode_key"])
    status_summary_path = output_dir / f"action_rules_bitset_status{suffix}_{ts}.csv"
    status_summary.to_csv(status_summary_path, index=False)

    return {
        "suite": "bitset_cpu_vs_gpu",
        "modes": list(modes),
        "runs": int(runs),
        "warmup_runs": int(warmup_runs),
        "output_paths": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "latest_jsonl": str(latest_jsonl),
            "latest_csv": str(latest_csv),
        },
        "summary_path": str(summary_path),
        "status_summary_path": str(status_summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run FIM/itemset, action-rules, and ActionRules CPU-vs-GPU benchmark suites."
        )
    )
    parser.add_argument(
        "--dataset-paths",
        type=str,
        default=",".join(str(p) for p in DEFAULT_DATASET_PATHS),
        help="Comma-separated dataset paths for FIM suites.",
    )
    parser.add_argument(
        "--dataset-presets",
        type=str,
        default=",".join(DEFAULT_DATASET_PRESETS),
        help="Comma-separated dataset presets for action-rules and ActionRules CPU/GPU suites.",
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
    parser.add_argument(
        "--rule-algorithms",
        type=str,
        default="",
        help=(
            "Optional comma-separated algorithm override for action-rules suite. "
            f"Defaults to: {','.join(RULE_ALGORITHMS)}"
        ),
    )
    parser.add_argument(
        "--bitset-modes",
        type=str,
        default="",
        help=(
            "Optional comma-separated mode override for ActionRules bitset suite. "
            f"Defaults to: {','.join(BITSET_MODES)}"
        ),
    )
    parser.add_argument("--runs-fim", type=int, default=10, help="Measured runs per dataset/algorithm for FIM suite.")
    parser.add_argument("--warmup-fim", type=int, default=2, help="Warmup runs per dataset for FIM suite.")
    parser.add_argument(
        "--runs-rules",
        type=int,
        default=10,
        help="Measured runs per dataset/algorithm for action-rules suite.",
    )
    parser.add_argument("--warmup-rules", type=int, default=2, help="Warmup runs per dataset for action-rules suite.")
    parser.add_argument(
        "--runs-bitset",
        type=int,
        default=10,
        help="Measured runs per dataset/mode for ActionRules CPU-vs-GPU suite.",
    )
    parser.add_argument(
        "--warmup-bitset",
        type=int,
        default=2,
        help="Warmup runs per dataset/mode for ActionRules CPU-vs-GPU suite.",
    )
    parser.add_argument(
        "--min-support-ratio-fim",
        type=float,
        default=0.05,
        help="Minimum support ratio for FIM itemset suite.",
    )
    parser.add_argument(
        "--min-support-ratio-rules",
        type=float,
        default=None,
        help=(
            "Optional minimum support ratio override for action-rules suites. "
            "If omitted, dataset preset support counts are used."
        ),
    )
    parser.add_argument(
        "--min-confidence-rules",
        type=float,
        default=0.6,
        help="Minimum confidence for action-rules suite.",
    )
    parser.add_argument("--max-len", type=int, default=3, help="Maximum itemset/rule length for FIM suites.")
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
    parser.add_argument("--max-gpu-mem-mb", type=int, default=None, help="Optional GPU memory cap for bitset suite.")
    parser.add_argument(
        "--gpu-node-batch-size",
        type=int,
        default=None,
        help="Optional number of BFS queue nodes grouped before one GPU expansion pass.",
    )
    parser.add_argument(
        "--gpu-batch-size",
        dest="gpu_node_batch_size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--autotune-sample-frac",
        type=float,
        default=0.05,
        help="Requested sample fraction used during autotuning.",
    )
    parser.add_argument(
        "--autotune-sample-min-rows",
        type=int,
        default=5000,
        help="Minimum sample size used during autotuning.",
    )
    parser.add_argument(
        "--autotune-sample-max-rows",
        type=int,
        default=50000,
        help="Maximum sample size used during autotuning.",
    )
    parser.add_argument(
        "--autotune-random-state",
        type=int,
        default=42,
        help="Random seed used during autotuning sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_DIR / "comparison_suites" / "data",
        help="Output directory for suite manifests and summaries.",
    )
    parser.add_argument("--tag", type=str, default="", help="Optional tag suffix for output files.")
    parser.add_argument("--skip-fim", action="store_true", help="Skip FIM itemset suite.")
    parser.add_argument("--skip-rules", action="store_true", help="Skip action-rules suite.")
    parser.add_argument("--skip-bitset", action="store_true", help="Skip ActionRules CPU-vs-GPU suite.")
    args = parser.parse_args()

    dataset_paths = _parse_path_list(args.dataset_paths)
    for path in dataset_paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")

    available_presets = set(list_dataset_presets())
    dataset_presets = _parse_dataset_presets(args.dataset_presets)
    unknown = [x for x in dataset_presets if x not in available_presets]
    if unknown:
        raise ValueError(f"Unsupported dataset presets: {unknown}. Allowed: {sorted(available_presets)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    manifest_path = args.output_dir / f"comparison_suites_manifest{suffix}_{ts}.json"

    results: list[dict] = []
    spmf_jar = args.spmf_jar if args.spmf_jar and args.spmf_jar.exists() else None

    if not args.skip_fim:
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
        results.append(
            _run_fim_suite(
                config=fim_config,
                dataset_paths=dataset_paths,
                output_dir=args.output_dir,
                tag=args.tag,
                spmf_jar=spmf_jar,
                spmf_timeout_sec=max(1, int(args.spmf_timeout_sec)),
            )
        )

    if not args.skip_rules:
        rule_algorithms = _parse_rule_algorithm_list(args.rule_algorithms, RULE_ALGORITHMS)
        results.append(
            _run_action_rules_suite(
                algorithms=rule_algorithms,
                dataset_presets=dataset_presets,
                runs=max(1, int(args.runs_rules)),
                warmup_runs=max(0, int(args.warmup_rules)),
                min_support_ratio=(
                    None if args.min_support_ratio_rules is None else float(args.min_support_ratio_rules)
                ),
                min_confidence=float(args.min_confidence_rules),
                output_dir=args.output_dir,
                tag=args.tag,
                max_gpu_mem_mb=args.max_gpu_mem_mb,
                gpu_node_batch_size=args.gpu_node_batch_size,
                autotune_sample_frac=float(args.autotune_sample_frac),
                autotune_sample_min_rows=max(1, int(args.autotune_sample_min_rows)),
                autotune_sample_max_rows=max(1, int(args.autotune_sample_max_rows)),
                autotune_random_state=int(args.autotune_random_state),
            )
        )

    if not args.skip_bitset:
        bitset_modes = _parse_bitset_mode_list(args.bitset_modes, BITSET_MODES)
        results.append(
            _run_bitset_cpu_gpu_suite(
                dataset_presets=dataset_presets,
                modes=bitset_modes,
                runs=max(1, int(args.runs_bitset)),
                warmup_runs=max(0, int(args.warmup_bitset)),
                output_dir=args.output_dir,
                tag=args.tag,
                max_gpu_mem_mb=args.max_gpu_mem_mb,
                gpu_node_batch_size=args.gpu_node_batch_size,
                autotune_sample_frac=float(args.autotune_sample_frac),
                autotune_sample_min_rows=max(1, int(args.autotune_sample_min_rows)),
                autotune_sample_max_rows=max(1, int(args.autotune_sample_max_rows)),
                autotune_random_state=int(args.autotune_random_state),
            )
        )

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_paths": [str(x) for x in dataset_paths],
        "dataset_presets": dataset_presets,
        "results": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nSuite manifest:")
    print(manifest_path)
    print("\nCompleted suites:")
    for entry in results:
        print(f"- {entry['suite']}")
        print(f"  summary: {entry['summary_path']}")
    print("\nPlot command:")
    print(
        "python notebooks/profiling/plot_comparison_suites.py "
        f"--manifest \"{manifest_path}\""
    )


if __name__ == "__main__":
    main()
