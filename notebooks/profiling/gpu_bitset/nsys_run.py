"""Run one measured GPU mining call under Nsight Systems, after warmup.

The driver does N untimed warmup runs (JIT compile, allocator warm, kernel
cache hot) and then a single measured run bracketed by cudaProfilerStart/Stop
so nsys captures only that region. Pair with:

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \\
        --trace=cuda,nvtx,osrt --cuda-memory-usage=true \\
        --output=<tag> --force-overwrite=true \\
        python notebooks/profiling/gpu_bitset/nsys_run.py [args]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"
PROFILING_DIR = REPO_ROOT / "notebooks" / "profiling"

for path in (SRC_DIR, PROFILING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_runner import run_profile  # noqa: E402

import cupy as cp  # noqa: E402
from cupy.cuda import runtime as cuda_rt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="census_income")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Override CandidateGenerator._gpu_kernel_min_work for this run.",
    )
    parser.add_argument(
        "--max-gpu-mem-mb",
        type=int,
        default=None,
        help="Optional CuPy pool cap in MB.",
    )
    parser.add_argument(
        "--gpu-node-batch-size",
        type=int,
        default=None,
        help="Optional fixed BFS node-batch size.",
    )
    args = parser.parse_args()

    if args.threshold is not None:
        from action_rules.candidates.candidate_generator import CandidateGenerator
        CandidateGenerator._gpu_kernel_min_work = int(args.threshold)

    common = dict(
        use_gpu=True,
        dataset=args.dataset,
        repeat_factor=1,
        autotune=False,
        verbose=False,
        max_gpu_mem_mb=args.max_gpu_mem_mb,
        gpu_node_batch_size=args.gpu_node_batch_size,
    )

    for i in range(max(0, int(args.warmup))):
        run_profile(**common)
        cp.cuda.Stream.null.synchronize()
        print(f"[warmup {i + 1}/{args.warmup}] done", flush=True)

    cuda_rt.profilerStart()
    try:
        result = run_profile(**common)
        cp.cuda.Stream.null.synchronize()
    finally:
        cuda_rt.profilerStop()

    print(
        f"[measured] dataset={args.dataset} "
        f"threshold={args.threshold} "
        f"backend={result.get('actual_backend')} "
        f"elapsed={float(result.get('elapsed_seconds', 0.0)):.6f}s"
    )


if __name__ == "__main__":
    main()
