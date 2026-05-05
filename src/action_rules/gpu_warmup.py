"""Background GPU warmup that overlaps CuPy import + RawKernel JIT with CPU prep.

The first GPU `fit()` in a process pays one-time costs that distort latency:
- importing CuPy,
- creating the CUDA driver context and CuPy memory pool,
- NVRTC-compiling the bitset RawKernel.

This module spawns a daemon thread early in `fit()` that runs an untimed dummy
fit on a tiny synthetic frame. The dummy fit triggers all of the cold-start
work in parallel with the main thread's CPU preprocessing (one-hot encoding,
bit-mask construction, etc.). By the time the main fit needs CuPy, the kernel
is already compiled and the context is already up.

Disable via ``disable()`` (e.g. to A/B-test the cold-start cost).
"""

from __future__ import annotations

import threading
from typing import Optional


_warmup_done = False
_warmup_thread: Optional[threading.Thread] = None
_warmup_lock = threading.Lock()
_disabled = False
_thread_local = threading.local()


def disable() -> None:
    """Turn off the async warmup. New fit() calls then pay full cold-start cost."""
    global _disabled
    _disabled = True


def enable() -> None:
    """Re-enable the async warmup after a previous ``disable()`` call."""
    global _disabled
    _disabled = False


def is_done() -> bool:
    return _warmup_done


def _is_inside_warmup() -> bool:
    return bool(getattr(_thread_local, "inside_warmup", False))


def _build_synthetic_warmup_frame(
    stable_attributes,
    flexible_attributes,
    target: str,
    target_undesired_state: str,
    target_desired_state: str,
):
    import pandas as pd

    rows = 200
    columns: dict = {}
    for col in stable_attributes:
        columns[col] = (["s0", "s1"] * ((rows + 1) // 2))[:rows]
    for col in flexible_attributes:
        columns[col] = (["a", "b"] * ((rows + 1) // 2))[:rows]
    columns[target] = (
        [target_undesired_state, target_desired_state] * ((rows + 1) // 2)
    )[:rows]
    return pd.DataFrame(columns)


def _run_dummy_fit(kwargs: dict) -> None:
    from .action_rules import ActionRules

    sample = _build_synthetic_warmup_frame(
        kwargs["stable_attributes"],
        kwargs["flexible_attributes"],
        kwargs["target"],
        kwargs["target_undesired_state"],
        kwargs["target_desired_state"],
    )
    helper = ActionRules(
        min_stable_attributes=kwargs["min_stable_attributes"],
        min_flexible_attributes=kwargs["min_flexible_attributes"],
        min_undesired_support=1,
        min_undesired_confidence=0.0,
        min_desired_support=1,
        min_desired_confidence=0.0,
        verbose=False,
    )
    helper.fit(
        data=sample,
        stable_attributes=list(kwargs["stable_attributes"]),
        flexible_attributes=list(kwargs["flexible_attributes"]),
        target=kwargs["target"],
        target_undesired_state=kwargs["target_undesired_state"],
        target_desired_state=kwargs["target_desired_state"],
        use_gpu=True,
    )


def _async_warmup_body(kwargs: dict) -> None:
    _thread_local.inside_warmup = True
    try:
        _run_dummy_fit(kwargs)
    except Exception:
        pass
    finally:
        _thread_local.inside_warmup = False


def start_gpu_warmup_async(
    *,
    min_stable_attributes: int,
    min_flexible_attributes: int,
    stable_attributes,
    flexible_attributes,
    target: str,
    target_undesired_state: str,
    target_desired_state: str,
    use_gpu: bool,
) -> None:
    """Spawn the warmup daemon thread if appropriate. No-op otherwise."""
    global _warmup_thread
    if _disabled or not use_gpu or _warmup_done or _is_inside_warmup():
        return
    with _warmup_lock:
        if _warmup_thread is not None or _warmup_done:
            return
        kwargs = dict(
            min_stable_attributes=min_stable_attributes,
            min_flexible_attributes=min_flexible_attributes,
            stable_attributes=list(stable_attributes),
            flexible_attributes=list(flexible_attributes),
            target=target,
            target_undesired_state=target_undesired_state,
            target_desired_state=target_desired_state,
        )
        _warmup_thread = threading.Thread(
            target=_async_warmup_body,
            args=(kwargs,),
            name="action-rules-gpu-warmup",
            daemon=True,
        )
        _warmup_thread.start()


def ensure_gpu_warmup_done(use_gpu: bool) -> None:
    """Block until warmup is finished. CPU-only callers and the daemon are a no-op."""
    global _warmup_done, _warmup_thread
    if not use_gpu or _warmup_done or _is_inside_warmup():
        return
    if _warmup_thread is not None:
        _warmup_thread.join()
        _warmup_thread = None
    _warmup_done = True
