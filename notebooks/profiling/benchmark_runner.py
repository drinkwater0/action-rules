"""Benchmark runner for ActionRules bitset path on benchmark datasets."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Optional
import sys

# Allow running directly from repository without package install.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from action_rules import ActionRules
from action_rules.profiling import profile_dataset as _profile_dataset_core
from action_rules.autotuning import autotune as _autotune_core
from benchmark_datasets import DATASET_PRESETS, list_dataset_presets, load_frame, normalize_dataset_key


# ------------------------------------------------------------------
# Warmup + VRAM measurement helpers
# ------------------------------------------------------------------

_warmup_done = False


def _ensure_gpu_warmup(data_frame, preset, use_gpu: bool) -> None:
    """Run one untimed GPU fit per process to pay NVRTC compile + CuPy init.

    The first GPU fit in a process pays one-time costs that distort timed
    measurements: NVRTC compiles the RawKernel, CuPy initializes its memory
    pool, the driver allocates contexts. Doing one throwaway fit before the
    real measurement isolates those costs from `elapsed_seconds`.
    """
    global _warmup_done
    if _warmup_done or not use_gpu:
        _warmup_done = True
        return
    try:
        sample = data_frame.head(min(200, len(data_frame)))
        helper = ActionRules(
            min_stable_attributes=preset.min_stable_attributes,
            min_flexible_attributes=preset.min_flexible_attributes,
            min_undesired_support=1,
            min_undesired_confidence=0.0,
            min_desired_support=1,
            min_desired_confidence=0.0,
            verbose=False,
        )
        helper.fit(
            data=sample,
            stable_attributes=list(preset.stable_attributes),
            flexible_attributes=list(preset.flexible_attributes),
            target=preset.target,
            target_undesired_state=preset.undesired_state,
            target_desired_state=preset.desired_state,
            use_gpu=True,
        )
    except Exception:
        pass
    finally:
        _warmup_done = True


def _reset_cupy_pool_baseline() -> tuple[object, int]:
    """Free cached pool blocks and return the post-reset baseline allocation."""
    try:
        import cupy as cp
    except ImportError:
        return None, 0
    mempool = cp.get_default_memory_pool()
    try:
        mempool.free_all_blocks()
    except Exception:
        pass
    try:
        baseline_bytes = int(mempool.total_bytes())
    except Exception:
        baseline_bytes = 0
    return mempool, baseline_bytes


def _measure_peak_vram_mb(mempool: object, baseline_bytes: int) -> Optional[float]:
    """Compute peak VRAM used by CuPy during the timed window, in MB."""
    if mempool is None:
        return None
    try:
        delta_bytes = max(0, int(mempool.total_bytes()) - int(baseline_bytes))
    except Exception:
        return None
    return float(delta_bytes) / (1024.0 * 1024.0)


# ------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Profiling wrapper (adds preset-specific fields to the core profile)
# ------------------------------------------------------------------

def profile_dataset_frame(data_frame, preset, top_k: int = 8) -> dict:
    """Profile a dataset using the package-level profiler, adding preset metadata."""
    helper = _make_action_rules_helper(preset)
    result = _profile_dataset_core(
        action_rules=helper,
        data_frame=data_frame,
        stable_attributes=list(preset.stable_attributes),
        flexible_attributes=list(preset.flexible_attributes),
        target=preset.target,
        top_k=top_k,
    )
    result["dataset_key"] = preset.name
    result["dataset_path"] = str(preset.path)
    return result


def profile_dataset(dataset: str = "telco", top_k: int = 8) -> dict:
    preset, data_frame = _load_preset_and_frame(dataset)
    return profile_dataset_frame(data_frame, preset, top_k=top_k)


# ------------------------------------------------------------------
# Threshold helpers
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Single benchmark run
# ------------------------------------------------------------------

def _fit_profile_once(
    *,
    data_frame,
    preset,
    use_gpu,
    max_gpu_mem_mb: Optional[int],
    gpu_node_batch_size: Optional[int],
    effective_support: int,
    effective_confidence: float,
    verbose: bool,
    requested_backend: str,
    emit_summary: bool,
) -> dict:
    _ensure_gpu_warmup(data_frame, preset, bool(use_gpu))

    action_rules = ActionRules(
        min_stable_attributes=preset.min_stable_attributes,
        min_flexible_attributes=preset.min_flexible_attributes,
        min_undesired_support=effective_support,
        min_undesired_confidence=effective_confidence,
        min_desired_support=effective_support,
        min_desired_confidence=effective_confidence,
        verbose=verbose,
    )

    mempool, baseline_pool_bytes = (
        _reset_cupy_pool_baseline() if use_gpu else (None, 0)
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

    peak_vram_mb = _measure_peak_vram_mb(mempool, baseline_pool_bytes)

    rule_count = len(action_rules.get_rules().action_rules)
    actual_backend = "gpu" if action_rules.is_gpu_np else "cpu"
    status = "ok"
    note = ""
    if use_gpu is True and actual_backend != "gpu":
        status = "fallback_cpu"
        note = "GPU requested, but CuPy is unavailable. The benchmark ran on CPU."

    mode = "GPU bitset" if actual_backend == "gpu" else "CPU bitset"
    if note:
        mode += " [fallback]"

    result = {
        "status": status,
        "note": note,
        "mode": mode,
        "use_gpu": bool(use_gpu) if isinstance(use_gpu, bool) else str(use_gpu),
        "requested_backend": requested_backend,
        "actual_backend": actual_backend,
        "gpu_acceleration_active": bool(action_rules.is_gpu_np),
        "gpu_dataframe_active": bool(action_rules.is_gpu_pd),
        "dataset_key": preset.name,
        "dataset_path": str(preset.path),
        "rows": int(len(data_frame)),
        "rule_count": int(rule_count),
        "elapsed_seconds": float(elapsed),
        "peak_vram_mb": None if peak_vram_mb is None else float(peak_vram_mb),
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
        if peak_vram_mb is not None:
            print(f"Peak VRAM (CuPy pool): {peak_vram_mb:.2f} MB")
        if note:
            print(f"Note: {note}")
    return result


# ------------------------------------------------------------------
# Autotune benchmark (uses package-level autotune, then runs full)
# ------------------------------------------------------------------

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

    best = _autotune_core(
        action_rules_cls=ActionRules,
        data_frame=data_frame,
        stable_attributes=list(preset.stable_attributes),
        flexible_attributes=list(preset.flexible_attributes),
        target=preset.target,
        target_undesired_state=preset.undesired_state,
        target_desired_state=preset.desired_state,
        min_stable_attributes=preset.min_stable_attributes,
        min_flexible_attributes=preset.min_flexible_attributes,
        min_undesired_support=full_support,
        min_undesired_confidence=full_confidence,
        min_desired_support=full_support,
        min_desired_confidence=full_confidence,
        dataset_profile=dataset_profile,
        max_gpu_mem_mb=max_gpu_mem_mb,
        gpu_node_batch_size=gpu_node_batch_size,
        sample_frac=sample_frac,
        sample_min_rows=sample_min_rows,
        sample_max_rows=sample_max_rows,
        sample_random_state=sample_random_state,
    )

    final_result = _fit_profile_once(
        data_frame=data_frame,
        preset=preset,
        use_gpu=best["use_gpu"],
        max_gpu_mem_mb=max_gpu_mem_mb,
        gpu_node_batch_size=best["gpu_node_batch_size"],
        effective_support=full_support,
        effective_confidence=full_confidence,
        verbose=verbose,
        requested_backend="auto",
        emit_summary=True,
    )
    autotune_summary = {
        "enabled": True,
        "sample_fraction_requested": float(sample_frac),
        "sample_rows": best["sample_rows"],
        "sample_support_count_effective": best["sample_support_count_effective"],
        "candidate_count": best["candidate_count"],
        "selected_use_gpu": best["use_gpu"],
        "selected_gpu_node_batch_size": best["gpu_node_batch_size"],
        "trials": best["trials"],
    }
    return final_result, autotune_summary


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def run_profile(
    use_gpu: bool,
    dataset: str = "telco",
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
            "Use real datasets (telco/adult/census_income) without --repeat-factor > 1."
        )

    if gpu_node_batch_size is None:
        gpu_node_batch_size = gpu_batch_size
    elif gpu_batch_size is not None and int(gpu_node_batch_size) != int(gpu_batch_size):
        raise ValueError("gpu_node_batch_size and gpu_batch_size must match when both are provided.")

    preset, data_frame = _load_preset_and_frame(dataset)
    dataset_prof = profile_dataset_frame(data_frame, preset) if (include_dataset_profile or autotune) else None

    if autotune:
        if dataset_prof is None:
            dataset_prof = profile_dataset_frame(data_frame, preset)
        result, autotune_summary = _autotune_run_profile(
            data_frame=data_frame,
            preset=preset,
            dataset_profile=dataset_prof,
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
        if dataset_prof is not None:
            result["dataset_profile"] = dataset_prof
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
    if dataset_prof is not None:
        result["dataset_profile"] = dataset_prof
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
    parser = argparse.ArgumentParser(description="Benchmark ActionRules bitset path on datasets.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (CuPy) for the bitset path.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="telco",
        help="Dataset preset: telco,adult,census_income",
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
