"""Warmup overlap A/B: cold-start subprocess timings with warmup on vs off.

Each measurement is a fresh ``python benchmark_runner.py`` subprocess so the
GPU starts cold every time. We time the wall clock of each subprocess and
also pull ``total_seconds`` (load + warmup join + fit) out of the saved
metrics JSON. Output: per-run JSONL + summary CSV with mean/std for both
metrics, per mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


CURRENT_DIR = Path(__file__).resolve().parent
BENCHMARK_RUNNER = CURRENT_DIR / "benchmark_runner.py"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "warmup_ab" / "data"


def _run_one(dataset: str, mode: str, metrics_dir: Path, tag: str) -> dict:
    """Run one cold-start subprocess and parse the resulting metrics JSON."""
    cmd = [
        sys.executable,
        str(BENCHMARK_RUNNER),
        "--gpu",
        "--dataset", dataset,
        "--save-metrics",
        "--metrics-dir", str(metrics_dir),
        "--metrics-tag", tag,
    ]
    if mode == "off":
        cmd.append("--no-warmup")

    metrics_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in metrics_dir.glob("*.json")}
    wall_started = perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_seconds = perf_counter() - wall_started

    if proc.returncode != 0:
        return {
            "dataset": dataset,
            "warmup_mode": mode,
            "status": "error",
            "wall_seconds": float(wall_seconds),
            "total_seconds": None,
            "elapsed_seconds": None,
            "stderr_tail": proc.stderr[-400:],
        }

    after = {p.name for p in metrics_dir.glob("*.json")}
    new_files = sorted(after - before)
    if not new_files:
        return {
            "dataset": dataset,
            "warmup_mode": mode,
            "status": "no_metrics",
            "wall_seconds": float(wall_seconds),
            "total_seconds": None,
            "elapsed_seconds": None,
            "stderr_tail": "",
        }

    latest = metrics_dir / new_files[-1]
    payload = json.loads(latest.read_text(encoding="utf-8"))
    return {
        "dataset": dataset,
        "warmup_mode": mode,
        "status": str(payload.get("status", "ok")),
        "wall_seconds": float(wall_seconds),
        "total_seconds": (
            float(payload["total_seconds"]) if payload.get("total_seconds") is not None else None
        ),
        "elapsed_seconds": (
            float(payload["elapsed_seconds"]) if payload.get("elapsed_seconds") is not None else None
        ),
        "stderr_tail": "",
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row["dataset"], row["warmup_mode"])
        bucket = grouped.setdefault(key, {"wall": [], "total": [], "elapsed": []})
        bucket["wall"].append(float(row["wall_seconds"]))
        if row.get("total_seconds") is not None:
            bucket["total"].append(float(row["total_seconds"]))
        if row.get("elapsed_seconds") is not None:
            bucket["elapsed"].append(float(row["elapsed_seconds"]))

    summary: list[dict] = []
    for (dataset, mode), values in grouped.items():
        def stats(xs: list[float]) -> tuple[float | None, float | None, int]:
            if not xs:
                return None, None, 0
            mean_v = float(statistics.mean(xs))
            std_v = float(statistics.stdev(xs)) if len(xs) > 1 else 0.0
            return mean_v, std_v, len(xs)

        wall_mean, wall_std, n = stats(values["wall"])
        total_mean, total_std, _ = stats(values["total"])
        elapsed_mean, elapsed_std, _ = stats(values["elapsed"])
        summary.append({
            "dataset": dataset,
            "warmup_mode": mode,
            "runs": n,
            "wall_mean_s": wall_mean,
            "wall_std_s": wall_std,
            "total_mean_s": total_mean,
            "total_std_s": total_std,
            "elapsed_mean_s": elapsed_mean,
            "elapsed_std_s": elapsed_std,
        })
    summary.sort(key=lambda r: (r["dataset"], r["warmup_mode"]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Warmup overlap A/B (cold-start, fresh subprocess each run).")
    parser.add_argument(
        "--datasets",
        type=str,
        default="census_income",
        help="Comma-separated dataset presets. Default: census_income (largest effect).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Subprocess runs per (dataset, mode) cell.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="on,off",
        help="Comma-separated warmup modes. Choices: on,off.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-run JSONL and summary CSV.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag suffix for output filenames.",
    )
    args = parser.parse_args()

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    modes = [s.strip() for s in args.modes.split(",") if s.strip() in {"on", "off"}]
    if not datasets or not modes:
        raise ValueError("Need at least one dataset and one valid mode.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.output_dir / "raw_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    runs_jsonl = args.output_dir / f"warmup_ab_runs{suffix}_{ts}.jsonl"
    summary_csv = args.output_dir / f"warmup_ab_summary{suffix}_{ts}.csv"

    rows: list[dict] = []
    total_runs = len(datasets) * len(modes) * int(args.runs)
    counter = 0

    with runs_jsonl.open("w", encoding="utf-8") as jf:
        for dataset in datasets:
            for mode in modes:
                tag = f"warmup_{mode}_{dataset}{suffix}"
                for run_index in range(int(args.runs)):
                    counter += 1
                    print(
                        f"[{counter}/{total_runs}] dataset={dataset} mode={mode} "
                        f"run={run_index + 1}/{int(args.runs)}",
                        flush=True,
                    )
                    row = _run_one(dataset, mode, metrics_dir, tag)
                    row["run_index"] = int(run_index)
                    row["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    rows.append(row)
                    jf.write(json.dumps(row) + "\n")
                    jf.flush()
                    if row["status"] == "ok":
                        print(
                            f"    wall={row['wall_seconds']:.3f}s  "
                            f"total={row['total_seconds']:.3f}s  "
                            f"elapsed={row['elapsed_seconds']:.3f}s",
                            flush=True,
                        )
                    else:
                        print(f"    {row['status']}: {row.get('stderr_tail', '')[:200]}", flush=True)

    summary = _aggregate(rows)
    if summary:
        with summary_csv.open("w", encoding="utf-8", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    print()
    print(f"Per-run JSONL: {runs_jsonl}")
    print(f"Summary CSV:   {summary_csv}")
    if summary:
        print()
        for row in summary:
            wall = f"{row['wall_mean_s']:.3f}±{row['wall_std_s']:.3f}s" if row["wall_mean_s"] is not None else "n/a"
            total = f"{row['total_mean_s']:.3f}±{row['total_std_s']:.3f}s" if row["total_mean_s"] is not None else "n/a"
            print(
                f"  {row['dataset']:<14} mode={row['warmup_mode']:<3} runs={row['runs']:<3} "
                f"wall={wall}  total={total}"
            )


if __name__ == "__main__":
    main()
