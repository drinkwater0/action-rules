"""Plot benchmark suite outputs produced by run_comparison_suites.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = CURRENT_DIR / "comparison_suites" / "data"


def _latest_manifest(data_dir: Path) -> Path:
    candidates = sorted(
        data_dir.glob("comparison_suites_manifest*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No manifest found in {data_dir}. Run run_comparison_suites.py first."
        )
    return candidates[0]


def _dataset_label(value: str) -> str:
    text = str(value)
    path = Path(text)
    if path.suffix.lower() == ".csv":
        return path.stem
    return text


def _short_status(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "no_data"
    if len(text) <= 16:
        return text
    return text[:15] + "..."


def _plot_algorithm_suite(
    summary_path: Path,
    suite_name: str,
    out_dir: Path,
    raw_runs_path: Path | None = None,
) -> list[Path]:
    summary_df = pd.read_csv(summary_path)
    raw_df = pd.read_csv(raw_runs_path) if raw_runs_path is not None and raw_runs_path.exists() else pd.DataFrame()

    if summary_df.empty and raw_df.empty:
        return []

    if not summary_df.empty:
        if "dataset_key" in summary_df.columns:
            summary_df["dataset_label"] = summary_df["dataset_key"].astype(str)
        elif "dataset_path" in summary_df.columns:
            summary_df["dataset_label"] = summary_df["dataset_path"].map(_dataset_label)
        else:
            summary_df["dataset_label"] = "dataset"

    if not raw_df.empty:
        if "dataset_key" in raw_df.columns:
            raw_df["dataset_label"] = raw_df["dataset_key"].astype(str)
        elif "dataset_path" in raw_df.columns:
            raw_df["dataset_label"] = raw_df["dataset_path"].map(_dataset_label)
        else:
            raw_df["dataset_label"] = "dataset"

    dataset_labels = sorted(
        set(summary_df.get("dataset_label", pd.Series(dtype=str)).dropna().astype(str)).union(
            set(raw_df.get("dataset_label", pd.Series(dtype=str)).dropna().astype(str))
        )
    )

    outputs: list[Path] = []
    for dataset_label in dataset_labels:
        summary_group = (
            summary_df[summary_df["dataset_label"] == dataset_label].copy()
            if "dataset_label" in summary_df.columns
            else pd.DataFrame()
        )
        raw_group = (
            raw_df[raw_df["dataset_label"] == dataset_label].copy()
            if "dataset_label" in raw_df.columns
            else pd.DataFrame()
        )

        if not raw_group.empty and "algorithm" in raw_group.columns:
            algorithm_order = list(dict.fromkeys(raw_group["algorithm"].astype(str).tolist()))
        elif not summary_group.empty and "algorithm" in summary_group.columns:
            algorithm_order = list(summary_group["algorithm"].astype(str).tolist())
        else:
            continue

        plot_df = pd.DataFrame({"algorithm": algorithm_order})
        if not summary_group.empty:
            cols = [c for c in ["algorithm", "mean_s", "std_s", "runs"] if c in summary_group.columns]
            plot_df = plot_df.merge(summary_group[cols], on="algorithm", how="left")
        if "mean_s" not in plot_df.columns:
            plot_df["mean_s"] = np.nan
        if "std_s" not in plot_df.columns:
            plot_df["std_s"] = 0.0

        if not raw_group.empty and "status" in raw_group.columns:
            status_map = {}
            for algo, grp in raw_group.groupby("algorithm"):
                counts = grp["status"].astype(str).value_counts()
                total = int(counts.sum())
                ok_count = int(counts.get("ok", 0))
                if ok_count == total:
                    status_map[str(algo)] = "ok"
                elif ok_count > 0:
                    status_map[str(algo)] = f"partial {ok_count}/{total}"
                else:
                    status_map[str(algo)] = str(counts.index[0])
            plot_df["status_label"] = plot_df["algorithm"].map(status_map).fillna("no_data")
        else:
            plot_df["status_label"] = np.where(plot_df["mean_s"].notna(), "ok", "no_data")

        plot_df["is_ok"] = plot_df["mean_s"].notna()
        ok_df = plot_df[plot_df["is_ok"]].sort_values("mean_s")
        fail_df = plot_df[~plot_df["is_ok"]]
        plot_df = pd.concat([ok_df, fail_df], ignore_index=True)

        plot_df["plot_mean_s"] = plot_df["mean_s"].fillna(0.0)
        plot_df["plot_std_s"] = plot_df["std_s"].fillna(0.0)
        fig_w = max(8, 0.7 * len(plot_df))
        fig, ax = plt.subplots(figsize=(fig_w, 5))
        bars = ax.bar(
            plot_df["algorithm"],
            plot_df["plot_mean_s"],
            yerr=plot_df["plot_std_s"],
            capsize=4,
        )
        ax.set_title(f"{suite_name} runtime ({dataset_label})")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Mean elapsed seconds")
        ax.grid(axis="y", alpha=0.3)

        for bar, row in zip(bars, plot_df.to_dict(orient="records")):
            is_ok = bool(row["is_ok"])
            if not is_ok:
                bar.set_color("#BDBDBD")
                bar.set_hatch("//")
            mean_s = float(row["plot_mean_s"])
            std_s = float(row["plot_std_s"])
            text = (
                f"{mean_s:.3f}s +/- {std_s:.3f}s"
                if is_ok
                else _short_status(str(row.get("status_label", "failed")))
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=8,
            )

        if (~plot_df["is_ok"]).any():
            ax.text(
                0.99,
                0.98,
                "Failed/missing runs shown as hatched zero bars.",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        out_path = out_dir / f"{suite_name}_{dataset_label}.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def _plot_mode_summary(summary_path: Path, out_dir: Path, suite_name: str) -> list[Path]:
    df = pd.read_csv(summary_path)
    if df.empty:
        return []

    required = {"dataset_key", "mode_key", "mean_s"}
    if not required.issubset(df.columns):
        return []

    outputs: list[Path] = []
    pivot_mean = (
        df.pivot_table(index="dataset_key", columns="mode_key", values="mean_s", aggfunc="mean")
        .sort_index()
    )
    pivot_std = (
        df.pivot_table(index="dataset_key", columns="mode_key", values="std_s", aggfunc="mean")
        .reindex(index=pivot_mean.index, columns=pivot_mean.columns)
        .fillna(0.0)
    )

    modes = list(pivot_mean.columns)
    if len(modes) >= 2:
        x = np.arange(len(pivot_mean.index))
        width = min(0.36, 0.8 / len(modes))
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_groups = []
        for idx, mode in enumerate(modes):
            offset = (idx - (len(modes) - 1) / 2.0) * width
            means = pivot_mean[mode].fillna(0.0).to_numpy()
            stds = pivot_std[mode].fillna(0.0).to_numpy()
            bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4, label=str(mode))
            bar_groups.append((bars, means, stds))

        title = "ActionRules CPU vs GPU runtime by dataset"
        file_name = "bitset_cpu_vs_gpu_overview.png"

        ax.set_title(title)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Mean elapsed seconds")
        ax.set_xticks(x)
        ax.set_xticklabels(list(pivot_mean.index), rotation=15)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        for bars, means, stds in bar_groups:
            for bar, mean_s, std_s in zip(bars, means, stds):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{mean_s:.3f}s +/- {std_s:.3f}s",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()
        out_path = out_dir / file_name
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        outputs.append(out_path)

        speed = pivot_mean.reset_index()
        if {"cpu", "gpu"}.issubset(speed.columns):
            speed["cpu_over_gpu_speedup"] = speed["cpu"] / speed["gpu"]
            speed_path = out_dir / "bitset_cpu_vs_gpu_speedup.csv"
            speed.to_csv(speed_path, index=False)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot comparison suite outputs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to comparison_suites_manifest*.json. If omitted, latest in --data-dir is used.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory where run_comparison_suites.py stores manifests and summaries.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNG plots. Defaults to <data-dir>/plots.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else _latest_manifest(data_dir)
    out_dir = args.out_dir.resolve() if args.out_dir else (data_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print("Using manifest:", manifest_path)

    produced: list[Path] = []
    for entry in manifest.get("results", []):
        suite = str(entry.get("suite", "suite"))
        summary_path = Path(str(entry.get("summary_path", "")))
        output_paths = entry.get("output_paths", {})
        raw_csv_path = None
        if isinstance(output_paths, dict) and output_paths.get("csv"):
            raw_csv_path = Path(str(output_paths.get("csv")))
        if not summary_path.exists():
            print(f"Skipping {suite}: missing summary {summary_path}")
            continue
        if suite in {"fim_itemsets", "rule_search"}:
            produced.extend(_plot_algorithm_suite(summary_path, suite, out_dir, raw_runs_path=raw_csv_path))
        elif suite == "bitset_cpu_vs_gpu":
            produced.extend(_plot_mode_summary(summary_path, out_dir, suite_name=suite))

    if not produced:
        print("No plots produced.")
        return
    print("Produced plots:")
    for path in produced:
        print("-", path)


if __name__ == "__main__":
    main()
