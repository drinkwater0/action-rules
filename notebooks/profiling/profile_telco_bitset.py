"""Profile ActionRules bitset path on benchmark datasets (no synthetic repetition)."""

import argparse
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Optional
import sys

import numpy as np

# Allow running directly from repository without package install.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from action_rules import ActionRules
from benchmark_datasets import DATASET_PRESETS, list_dataset_presets, load_frame, normalize_dataset_key


def _load_preset_and_frame(dataset: str):
    preset = DATASET_PRESETS[normalize_dataset_key(dataset)]
    data_frame = load_frame(preset.path, preset.sep)

    missing = [
        col
        for col in [*preset.stable_attributes, *preset.flexible_attributes, preset.target]
        if col not in data_frame.columns
    ]
    if missing:
        raise ValueError(f"Dataset '{preset.name}' is missing required columns: {missing}")
    return preset, data_frame


def _make_action_rules_helper(preset, verbose: bool = False) -> ActionRules:
    return ActionRules(
        min_stable_attributes=preset.min_stable_attributes,
        min_flexible_attributes=preset.min_flexible_attributes,
        min_undesired_support=preset.min_undesired_support,
        min_undesired_confidence=preset.min_undesired_confidence,
        min_desired_support=preset.min_desired_support,
        min_desired_confidence=preset.min_desired_confidence,
        verbose=verbose,
    )


def _manual_popcount_uint64(array: np.ndarray) -> np.ndarray:
    x = array.astype(np.uint64, copy=True)
    x -= (x >> 1) & np.uint64(0x5555555555555555)
    x = (x & np.uint64(0x3333333333333333)) + ((x >> 2) & np.uint64(0x3333333333333333))
    x = (x + (x >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x += x >> 8
    x += x >> 16
    x += x >> 32
    return x & np.uint64(0x7F)


def _popcount_uint64(mask: np.ndarray) -> int:
    array = np.asarray(mask, dtype=np.uint64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if hasattr(np, "bitwise_count"):
        counts = np.bitwise_count(array).sum(axis=1)  # type: ignore[attr-defined]
        return int(counts[0])
    if hasattr(array, "bit_count"):
        counts = array.bit_count().sum(axis=1)  # type: ignore[call-arg]
        return int(counts[0])
    return int(_manual_popcount_uint64(array).sum(axis=1)[0])


def _column_role(column_name: str) -> str:
    if "_<item_stable>_" in column_name:
        return "stable"
    if "_<item_flexible>_" in column_name:
        return "flexible"
    if "_<item_target>_" in column_name:
        return "target"
    return "unknown"


def _density_hint(
    *,
    zero_word_fraction: float,
    top_pair_mean_jaccard: float,
    top_pair_mean_word_equal_fraction: float,
) -> str:
    if zero_word_fraction >= 0.80:
        return "sparse_like"
    if zero_word_fraction <= 0.35 and (
        top_pair_mean_jaccard >= 0.20 or top_pair_mean_word_equal_fraction >= 0.70
    ):
        return "dense_like"
    return "mixed"


def profile_dataset_frame(data_frame, preset, top_k: int = 8) -> dict:
    helper = _make_action_rules_helper(preset)
    helper.set_array_library(use_gpu=False, df=data_frame)

    encoded = helper.one_hot_encode(
        data=data_frame,
        stable_attributes=list(preset.stable_attributes),
        flexible_attributes=list(preset.flexible_attributes),
        target=preset.target,
    )
    binary_data, columns = helper.df_to_array(encoded)
    binary_data = np.asarray(binary_data, dtype=np.uint8)
    bit_masks = np.asarray(helper.build_bit_masks(binary_data), dtype=np.uint64)
    stable_items_binding, flexible_items_binding, target_items_binding, column_values = helper.get_bindings(
        columns,
        list(preset.stable_attributes),
        list(preset.flexible_attributes),
        preset.target,
    )

    row_count = int(len(data_frame))
    item_count = int(binary_data.shape[0])
    words_per_item = int(bit_masks.shape[1]) if bit_masks.ndim == 2 else 0
    padding_transactions = max(0, (words_per_item * 64) - row_count)
    support_counts = np.asarray(binary_data.sum(axis=1), dtype=np.int64)
    support_ratios = (
        support_counts.astype(np.float64, copy=False) / float(row_count)
        if row_count > 0
        else np.zeros(item_count, dtype=np.float64)
    )
    zero_word_fraction = float(np.mean(bit_masks == 0)) if bit_masks.size > 0 else 0.0
    onehot_matrix_density = float(np.mean(binary_data)) if binary_data.size > 0 else 0.0

    top_item_count = min(max(0, int(top_k)), item_count)
    sorted_indices = np.argsort(-support_counts, kind="stable")[:top_item_count].tolist()
    top_items = []
    for idx in sorted_indices:
        column_name = columns[idx]
        attribute, value = column_values.get(idx, ("", column_name))
        top_items.append(
            {
                "item_index": int(idx),
                "role": _column_role(column_name),
                "attribute": str(attribute),
                "value": str(value),
                "encoded_column": str(column_name),
                "support_count": int(support_counts[idx]),
                "support_ratio": float(support_ratios[idx]),
            }
        )

    condition_indices = [
        idx for idx in np.argsort(-support_counts, kind="stable").tolist() if _column_role(columns[idx]) != "target"
    ][:top_item_count]
    pair_count = 0
    pair_jaccard_sum = 0.0
    pair_containment_sum = 0.0
    pair_equal_word_sum = 0.0
    for left_pos, left_idx in enumerate(condition_indices):
        left_mask = bit_masks[left_idx]
        left_support = int(support_counts[left_idx])
        for right_idx in condition_indices[left_pos + 1 :]:
            right_mask = bit_masks[right_idx]
            right_support = int(support_counts[right_idx])
            intersection_support = _popcount_uint64(left_mask & right_mask)
            union_support = left_support + right_support - intersection_support
            pair_jaccard_sum += (
                float(intersection_support) / float(union_support) if union_support > 0 else 1.0
            )
            pair_containment_sum += (
                float(intersection_support) / float(min(left_support, right_support))
                if min(left_support, right_support) > 0
                else 0.0
            )
            pair_equal_word_sum += float(np.mean(left_mask == right_mask))
            pair_count += 1

    top_pair_mean_jaccard = pair_jaccard_sum / float(pair_count) if pair_count > 0 else 0.0
    top_pair_mean_containment = pair_containment_sum / float(pair_count) if pair_count > 0 else 0.0
    top_pair_mean_word_equal_fraction = pair_equal_word_sum / float(pair_count) if pair_count > 0 else 0.0

    return {
        "profile_kind": "dataset_characteristics",
        "dataset_key": preset.name,
        "dataset_path": str(preset.path),
        "rows": row_count,
        "input_column_count": int(data_frame.shape[1]),
        "selected_attribute_count": int(len(preset.stable_attributes) + len(preset.flexible_attributes) + 1),
        "stable_attribute_count": int(len(preset.stable_attributes)),
        "flexible_attribute_count": int(len(preset.flexible_attributes)),
        "condition_item_count": int(sum(len(items) for items in stable_items_binding.values()))
        + int(sum(len(items) for items in flexible_items_binding.values())),
        "target_state_count": int(sum(len(items) for items in target_items_binding.values())),
        "onehot_column_count": item_count,
        "onehot_matrix_density": onehot_matrix_density,
        "bitmask_words_per_item": words_per_item,
        "bitmask_total_bytes": int(bit_masks.nbytes),
        "bitmask_padding_transactions": int(padding_transactions),
        "bitmask_zero_word_fraction": zero_word_fraction,
        "bitmask_nonzero_word_fraction": float(1.0 - zero_word_fraction),
        "item_support_count_min": int(support_counts.min()) if item_count > 0 else 0,
        "item_support_count_median": float(np.median(support_counts)) if item_count > 0 else 0.0,
        "item_support_count_mean": float(np.mean(support_counts)) if item_count > 0 else 0.0,
        "item_support_count_max": int(support_counts.max()) if item_count > 0 else 0,
        "item_support_ratio_min": float(support_ratios.min()) if item_count > 0 else 0.0,
        "item_support_ratio_median": float(np.median(support_ratios)) if item_count > 0 else 0.0,
        "item_support_ratio_mean": float(np.mean(support_ratios)) if item_count > 0 else 0.0,
        "item_support_ratio_max": float(support_ratios.max()) if item_count > 0 else 0.0,
        "top_item_count_considered": int(top_item_count),
        "top_condition_pair_count": int(pair_count),
        "top_condition_pair_mean_jaccard": float(top_pair_mean_jaccard),
        "top_condition_pair_mean_containment": float(top_pair_mean_containment),
        "top_condition_pair_mean_word_equal_fraction": float(top_pair_mean_word_equal_fraction),
        "density_hint": _density_hint(
            zero_word_fraction=zero_word_fraction,
            top_pair_mean_jaccard=top_pair_mean_jaccard,
            top_pair_mean_word_equal_fraction=top_pair_mean_word_equal_fraction,
        ),
        "top_items": top_items,
    }


def profile_dataset(dataset: str = "bank", top_k: int = 8) -> dict:
    preset, data_frame = _load_preset_and_frame(dataset)
    return profile_dataset_frame(data_frame, preset, top_k=top_k)


def _compute_effective_thresholds(
    *,
    preset,
    row_count: int,
    min_support_count: Optional[int],
    min_confidence: Optional[float],
) -> tuple[int, float]:
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
    if row_count <= 0:
        return 1, effective_confidence
    return effective_support, effective_confidence


def _fit_profile_once(
    *,
    data_frame,
    preset,
    use_gpu: bool,
    max_gpu_mem_mb: Optional[int],
    gpu_node_batch_size: Optional[int],
    effective_support: int,
    effective_confidence: float,
    verbose: bool,
    requested_backend: str,
    emit_summary: bool,
) -> dict:
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
        gpu_node_batch_size=gpu_node_batch_size,
    )
    elapsed = perf_counter() - started

    rule_count = len(action_rules.get_rules().action_rules)
    actual_backend = "gpu" if action_rules.is_gpu_np else "cpu"
    status = "ok"
    note = ""
    if use_gpu and actual_backend != "gpu":
        status = "fallback_cpu"
        note = "GPU requested, but CuPy is unavailable. The benchmark ran on CPU."

    mode = "GPU bitset" if actual_backend == "gpu" else "CPU bitset"
    if note:
        mode += " [fallback]"

    result = {
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
        "gpu_node_batch_size": None if gpu_node_batch_size is None else int(gpu_node_batch_size),
        "min_support_count_effective": int(effective_support),
        "min_confidence_effective": float(effective_confidence),
    }

    if emit_summary:
        print(f"Mode: {mode}")
        print(f"Dataset: {preset.name}")
        print(f"Rows: {len(data_frame)}")
        print(f"Number of action rules: {rule_count}")
        print(f"Elapsed seconds: {elapsed:.6f}")
        if note:
            print(f"Note: {note}")
    return result


def _is_cupy_available() -> bool:
    return importlib.util.find_spec("cupy") is not None


def _choose_gpu_batch_candidates(
    *,
    dataset_profile: dict,
    max_gpu_mem_mb: Optional[int],
    explicit_gpu_node_batch_size: Optional[int],
) -> list[int]:
    if explicit_gpu_node_batch_size is not None:
        return [max(1, int(explicit_gpu_node_batch_size))]

    density_hint = str(dataset_profile.get("density_hint", "mixed"))
    rows = int(dataset_profile.get("rows", 0))
    words_per_item = int(dataset_profile.get("bitmask_words_per_item", 0))

    if density_hint == "sparse_like":
        candidates = [16, 32, 64]
    elif density_hint == "dense_like":
        candidates = [32, 64, 128, 256]
    else:
        candidates = [16, 32, 64, 128]

    if rows < 5000 or words_per_item < 32:
        candidates = [batch for batch in candidates if batch <= 64]
    if max_gpu_mem_mb is not None and max_gpu_mem_mb <= 512:
        candidates = [batch for batch in candidates if batch <= 64]

    return sorted(set(candidates))


def _pick_autotune_configs(
    *,
    dataset_profile: dict,
    max_gpu_mem_mb: Optional[int],
    explicit_gpu_node_batch_size: Optional[int],
) -> list[dict]:
    configs = [{"use_gpu": False, "gpu_node_batch_size": None}]
    if not _is_cupy_available():
        return configs

    for batch_size in _choose_gpu_batch_candidates(
        dataset_profile=dataset_profile,
        max_gpu_mem_mb=max_gpu_mem_mb,
        explicit_gpu_node_batch_size=explicit_gpu_node_batch_size,
    ):
        configs.append({"use_gpu": True, "gpu_node_batch_size": int(batch_size)})
    return configs


def _build_sample_frame(
    *,
    data_frame,
    target_column: str,
    sample_frac: float,
    min_rows: int,
    max_rows: int,
    random_state: int,
):
    import pandas as pd

    row_count = int(len(data_frame))
    if row_count <= 0:
        return data_frame.copy()

    requested_rows = int(math.ceil(float(sample_frac) * row_count))
    requested_rows = max(1, requested_rows)
    requested_rows = max(int(min_rows), requested_rows)
    requested_rows = min(row_count, int(max_rows), requested_rows)
    if requested_rows >= row_count:
        return data_frame.copy()

    if target_column not in data_frame.columns:
        return data_frame.sample(n=requested_rows, random_state=random_state).reset_index(drop=True)

    groups = []
    remainders = []
    grouped = list(data_frame.groupby(target_column, sort=False, dropna=False))
    total_rows = float(row_count)
    samples_so_far = 0
    reserved_groups = len(grouped) if requested_rows >= len(grouped) else 0

    for position, (group_value, group_frame) in enumerate(grouped):
        ideal = requested_rows * (len(group_frame) / total_rows)
        base = int(math.floor(ideal))
        if reserved_groups > 0 and base == 0:
            base = 1
        base = min(base, len(group_frame))
        groups.append([position, group_value, group_frame, base])
        remainders.append((ideal - base, position))
        samples_so_far += base

    remaining = requested_rows - samples_so_far
    for _, position in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        group_frame = groups[position][2]
        current_n = groups[position][3]
        if current_n < len(group_frame):
            groups[position][3] = current_n + 1
            remaining -= 1

    sampled_parts = []
    for position, _, group_frame, sample_n in groups:
        if sample_n <= 0:
            continue
        sampled_parts.append(group_frame.sample(n=sample_n, random_state=random_state + position))

    sampled_index = set()
    if sampled_parts:
        for part in sampled_parts:
            sampled_index.update(part.index.tolist())
    if len(sampled_index) < requested_rows:
        remaining_frame = data_frame.loc[[idx for idx in data_frame.index if idx not in sampled_index]]
        top_up = requested_rows - len(sampled_index)
        if top_up > 0 and len(remaining_frame) > 0:
            sampled_parts.append(remaining_frame.sample(n=min(top_up, len(remaining_frame)), random_state=random_state + 999))

    if not sampled_parts:
        return data_frame.sample(n=requested_rows, random_state=random_state).reset_index(drop=True)
    return pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _autotune_run_profile(
    *,
    data_frame,
    preset,
    dataset_profile: dict,
    max_gpu_mem_mb: Optional[int],
    gpu_node_batch_size: Optional[int],
    min_support_count: Optional[int],
    min_confidence: Optional[float],
    verbose: bool,
    sample_frac: float,
    sample_min_rows: int,
    sample_max_rows: int,
    sample_random_state: int,
) -> tuple[dict, dict]:
    full_support, full_confidence = _compute_effective_thresholds(
        preset=preset,
        row_count=len(data_frame),
        min_support_count=min_support_count,
        min_confidence=min_confidence,
    )

    sample_frame = _build_sample_frame(
        data_frame=data_frame,
        target_column=preset.target,
        sample_frac=sample_frac,
        min_rows=sample_min_rows,
        max_rows=sample_max_rows,
        random_state=sample_random_state,
    )
    sample_support = max(1, int(math.ceil(full_support * (len(sample_frame) / max(1, len(data_frame))))))
    configs = _pick_autotune_configs(
        dataset_profile=dataset_profile,
        max_gpu_mem_mb=max_gpu_mem_mb,
        explicit_gpu_node_batch_size=gpu_node_batch_size,
    )

    trials = []
    for config in configs:
        trial_result = _fit_profile_once(
            data_frame=sample_frame,
            preset=preset,
            use_gpu=bool(config["use_gpu"]),
            max_gpu_mem_mb=max_gpu_mem_mb,
            gpu_node_batch_size=config["gpu_node_batch_size"],
            effective_support=sample_support,
            effective_confidence=full_confidence,
            verbose=False,
            requested_backend="auto",
            emit_summary=False,
        )
        trials.append(
            {
                "candidate_use_gpu": bool(config["use_gpu"]),
                "candidate_gpu_node_batch_size": config["gpu_node_batch_size"],
                **trial_result,
            }
        )

    chosen_trial = min(
        trials,
        key=lambda trial: (
            float(trial.get("elapsed_seconds", float("inf"))),
            0 if trial.get("actual_backend") == "gpu" else 1,
            int(trial.get("candidate_gpu_node_batch_size") or 0),
        ),
    )
    final_result = _fit_profile_once(
        data_frame=data_frame,
        preset=preset,
        use_gpu=bool(chosen_trial["candidate_use_gpu"]),
        max_gpu_mem_mb=max_gpu_mem_mb,
        gpu_node_batch_size=chosen_trial["candidate_gpu_node_batch_size"],
        effective_support=full_support,
        effective_confidence=full_confidence,
        verbose=verbose,
        requested_backend="auto",
        emit_summary=True,
    )
    autotune_summary = {
        "enabled": True,
        "sample_fraction_requested": float(sample_frac),
        "sample_rows": int(len(sample_frame)),
        "sample_support_count_effective": int(sample_support),
        "candidate_count": int(len(trials)),
        "selected_use_gpu": bool(chosen_trial["candidate_use_gpu"]),
        "selected_gpu_node_batch_size": chosen_trial["candidate_gpu_node_batch_size"],
        "selected_actual_backend_on_sample": str(chosen_trial.get("actual_backend")),
        "trials": trials,
    }
    return final_result, autotune_summary


def run_profile(
    use_gpu: bool,
    dataset: str = "bank",
    repeat_factor: int = 1,
    max_gpu_mem_mb: Optional[int] = None,
    gpu_node_batch_size: Optional[int] = None,
    gpu_batch_size: Optional[int] = None,
    min_support_count: Optional[int] = None,
    min_confidence: Optional[float] = None,
    verbose: bool = False,
    include_dataset_profile: bool = False,
    autotune: bool = False,
    autotune_sample_frac: float = 0.05,
    autotune_sample_min_rows: int = 5000,
    autotune_sample_max_rows: int = 50000,
    autotune_random_state: int = 42,
) -> dict:
    if int(repeat_factor) != 1:
        raise ValueError(
            "Synthetic dataset multiplication is disabled. "
            "Use real datasets (bank/german/covtype/adult/census_income/telco) without --repeat-factor > 1."
        )

    if gpu_node_batch_size is None:
        gpu_node_batch_size = gpu_batch_size
    elif gpu_batch_size is not None and int(gpu_node_batch_size) != int(gpu_batch_size):
        raise ValueError("gpu_node_batch_size and gpu_batch_size must match when both are provided.")

    preset, data_frame = _load_preset_and_frame(dataset)
    dataset_profile = profile_dataset_frame(data_frame, preset) if (include_dataset_profile or autotune) else None

    if autotune:
        if dataset_profile is None:
            dataset_profile = profile_dataset_frame(data_frame, preset)
        result, autotune_summary = _autotune_run_profile(
            data_frame=data_frame,
            preset=preset,
            dataset_profile=dataset_profile,
            max_gpu_mem_mb=max_gpu_mem_mb,
            gpu_node_batch_size=gpu_node_batch_size,
            min_support_count=min_support_count,
            min_confidence=min_confidence,
            verbose=verbose,
            sample_frac=autotune_sample_frac,
            sample_min_rows=autotune_sample_min_rows,
            sample_max_rows=autotune_sample_max_rows,
            sample_random_state=autotune_random_state,
        )
        result["autotune"] = autotune_summary
        if dataset_profile is not None:
            result["dataset_profile"] = dataset_profile
        print(
            "Autotune selected:",
            f"backend={'gpu' if autotune_summary['selected_use_gpu'] else 'cpu'}",
            f"gpu_node_batch_size={autotune_summary['selected_gpu_node_batch_size']}",
            f"sample_rows={autotune_summary['sample_rows']}",
        )
        return result

    effective_support, effective_confidence = _compute_effective_thresholds(
        preset=preset,
        row_count=len(data_frame),
        min_support_count=min_support_count,
        min_confidence=min_confidence,
    )
    result = _fit_profile_once(
        data_frame=data_frame,
        preset=preset,
        use_gpu=use_gpu,
        max_gpu_mem_mb=max_gpu_mem_mb,
        gpu_node_batch_size=gpu_node_batch_size,
        effective_support=effective_support,
        effective_confidence=effective_confidence,
        verbose=verbose,
        requested_backend="gpu" if use_gpu else "cpu",
        emit_summary=True,
    )
    if dataset_profile is not None:
        result["dataset_profile"] = dataset_profile
    return result


def _write_metrics(result: dict, metrics_dir: Path, metrics_tag: str = "") -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    mode_slug = str(
        result.get("profile_kind")
        or result.get("actual_backend")
        or ("gpu" if result.get("use_gpu") else "cpu")
    )
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
        "--dataset-profile-only",
        action="store_true",
        help="Compute dataset characteristics without running the miner.",
    )
    parser.add_argument(
        "--include-dataset-profile",
        action="store_true",
        help="Attach dataset characteristics to the saved run metrics JSON.",
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="Use a sampled pre-run plus dataset profile to choose CPU/GPU and GPU node batch size automatically.",
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

    if args.dataset_profile_only:
        result = profile_dataset(dataset=args.dataset)
        print(json.dumps(result, indent=2))
        if args.save_metrics:
            out_path = _write_metrics(result, args.metrics_dir, args.metrics_tag)
            print(f"Saved run metrics: {out_path}")
        return

    result = run_profile(
        use_gpu=args.gpu,
        dataset=args.dataset,
        repeat_factor=args.repeat_factor,
        max_gpu_mem_mb=args.max_gpu_mem_mb,
        gpu_node_batch_size=args.gpu_node_batch_size,
        min_support_count=args.min_support_count,
        min_confidence=args.min_confidence,
        verbose=args.verbose,
        include_dataset_profile=args.include_dataset_profile,
        autotune=args.autotune,
        autotune_sample_frac=args.autotune_sample_frac,
        autotune_sample_min_rows=args.autotune_sample_min_rows,
        autotune_sample_max_rows=args.autotune_sample_max_rows,
    )
    if args.save_metrics:
        out_path = _write_metrics(result, args.metrics_dir, args.metrics_tag)
        print(f"Saved run metrics: {out_path}")


if __name__ == "__main__":
    main()
