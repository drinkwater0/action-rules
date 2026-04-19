"""Sampled autotuning for backend and batch-size selection."""

from __future__ import annotations

import importlib.util
import math
from time import perf_counter
from typing import Optional


def _is_cupy_available() -> bool:
    return importlib.util.find_spec("cupy") is not None


def _choose_gpu_batch_candidates(
    *,
    dataset_profile: dict,
    max_gpu_mem_mb: Optional[int],
    explicit_gpu_node_batch_size: Optional[int],
) -> list[int]:
    """Select candidate GPU node-batch sizes based on the dataset profile.

    Sparse datasets get smaller batches; dense datasets get larger ones.
    Memory constraints and small-dataset guards further filter the list.
    """
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
        candidates = [b for b in candidates if b <= 64]
    if max_gpu_mem_mb is not None and max_gpu_mem_mb <= 512:
        candidates = [b for b in candidates if b <= 64]

    return sorted(set(candidates))


def _build_sample_frame(
    *,
    data_frame,
    target_column: str,
    sample_frac: float = 0.05,
    min_rows: int = 5000,
    max_rows: int = 50000,
    random_state: int = 42,
):
    """Create a stratified sample of *data_frame* for autotuning trials.

    The sample preserves the class distribution of *target_column* when
    possible.  The effective size is clamped between *min_rows* and
    *max_rows*.
    """
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

    sampled_index: set = set()
    if sampled_parts:
        for part in sampled_parts:
            sampled_index.update(part.index.tolist())
    if len(sampled_index) < requested_rows:
        remaining_frame = data_frame.loc[[idx for idx in data_frame.index if idx not in sampled_index]]
        top_up = requested_rows - len(sampled_index)
        if top_up > 0 and len(remaining_frame) > 0:
            sampled_parts.append(
                remaining_frame.sample(n=min(top_up, len(remaining_frame)), random_state=random_state + 999)
            )

    if not sampled_parts:
        return data_frame.sample(n=requested_rows, random_state=random_state).reset_index(drop=True)
    return pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def autotune(
    *,
    action_rules_cls,
    data_frame,
    stable_attributes: list,
    flexible_attributes: list,
    target: str,
    target_undesired_state: str,
    target_desired_state: str,
    min_stable_attributes: int,
    min_flexible_attributes: int,
    min_undesired_support: int,
    min_undesired_confidence: float,
    min_desired_support: int,
    min_desired_confidence: float,
    dataset_profile: dict,
    max_gpu_mem_mb: Optional[int] = None,
    gpu_node_batch_size: Optional[int] = None,
    sample_frac: float = 0.05,
    sample_min_rows: int = 5000,
    sample_max_rows: int = 50000,
    sample_random_state: int = 42,
    verbose: bool = False,
) -> dict:
    """Run sampled trials and return the best backend configuration.

    Returns a dictionary with keys ``use_gpu`` (bool) and
    ``gpu_node_batch_size`` (int or None) representing the fastest
    configuration found on a stratified sample of the input data.
    A ``trials`` key contains timing details for every candidate.
    """
    row_count = len(data_frame)

    sample_frame = _build_sample_frame(
        data_frame=data_frame,
        target_column=target,
        sample_frac=sample_frac,
        min_rows=sample_min_rows,
        max_rows=sample_max_rows,
        random_state=sample_random_state,
    )
    sample_support = max(1, int(math.ceil(min_desired_support * (len(sample_frame) / max(1, row_count)))))

    configs: list[dict] = [{"use_gpu": False, "gpu_node_batch_size": None}]
    if _is_cupy_available():
        for batch_size in _choose_gpu_batch_candidates(
            dataset_profile=dataset_profile,
            max_gpu_mem_mb=max_gpu_mem_mb,
            explicit_gpu_node_batch_size=gpu_node_batch_size,
        ):
            configs.append({"use_gpu": True, "gpu_node_batch_size": int(batch_size)})

    trials = []
    for config in configs:
        ar = action_rules_cls(
            min_stable_attributes=min_stable_attributes,
            min_flexible_attributes=min_flexible_attributes,
            min_undesired_support=sample_support,
            min_undesired_confidence=min_undesired_confidence,
            min_desired_support=sample_support,
            min_desired_confidence=min_desired_confidence,
            verbose=False,
        )
        started = perf_counter()
        ar.fit(
            data=sample_frame,
            stable_attributes=list(stable_attributes),
            flexible_attributes=list(flexible_attributes),
            target=target,
            target_undesired_state=target_undesired_state,
            target_desired_state=target_desired_state,
            use_gpu=bool(config["use_gpu"]),
            max_gpu_mem_mb=max_gpu_mem_mb,
            gpu_node_batch_size=config["gpu_node_batch_size"],
        )
        elapsed = perf_counter() - started
        actual_backend = "gpu" if ar.is_gpu_np else "cpu"
        trials.append(
            {
                "candidate_use_gpu": bool(config["use_gpu"]),
                "candidate_gpu_node_batch_size": config["gpu_node_batch_size"],
                "actual_backend": actual_backend,
                "elapsed_seconds": float(elapsed),
                "rule_count": int(len(ar.get_rules().action_rules)),
            }
        )

    chosen = min(
        trials,
        key=lambda t: (
            float(t.get("elapsed_seconds", float("inf"))),
            0 if t.get("actual_backend") == "gpu" else 1,
            int(t.get("candidate_gpu_node_batch_size") or 0),
        ),
    )

    return {
        "use_gpu": bool(chosen["candidate_use_gpu"]),
        "gpu_node_batch_size": chosen["candidate_gpu_node_batch_size"],
        "sample_rows": int(len(sample_frame)),
        "sample_support_count_effective": int(sample_support),
        "candidate_count": int(len(trials)),
        "trials": trials,
    }
