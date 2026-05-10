"""Build the three figures used in the thesis Results chapter.

Outputs three PDF files into Thesis/TemplateBT-DT/img/:
- results-kernel-threshold.pdf  (was tab:kernel-threshold-sweep)
- results-fim-headline.pdf      (was tab:fim-headline)
- results-gpu-time-breakdown.pdf (was tab:gpu-time-breakdown)

Run from the repo root:
    .venv/Scripts/python notebooks/profiling/make_thesis_figures.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "Thesis" / "TemplateBT-DT" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
})

DATASET_LABEL = {
    "telco": "telco (7 043)",
    "adult": "adult (48 842)",
    "census_income": "census_income (299 285)",
}
DATASET_ORDER = ["telco", "adult", "census_income"]
DATASET_COLOR = {"telco": "#1f77b4", "adult": "#ff7f0e", "census_income": "#2ca02c"}


def kernel_threshold_figure() -> Path:
    src = REPO_ROOT / "notebooks/profiling/kernel_cutoff_sweep/comparison_suites/data/gpu_kernel_threshold_summary_20260506_120242.csv"
    df = pd.read_csv(src)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))

    threshold_order = sorted(df["gpu_kernel_min_work"].unique())
    x_positions = list(range(len(threshold_order)))
    x_labels = []
    for t in threshold_order:
        if t == 0:
            x_labels.append("0")
        elif t >= 1_000_000:
            exp = int(math.log10(t))
            x_labels.append(f"$10^{{{exp}}}$")
        else:
            x_labels.append(str(t))

    for ds in DATASET_ORDER:
        sub = df[df["dataset_key"] == ds].set_index("gpu_kernel_min_work").loc[threshold_order]
        ax.errorbar(
            x_positions,
            sub["mean_elapsed_seconds"] * 1000,
            yerr=sub["stdev_elapsed_seconds"] * 1000,
            marker="o",
            label=DATASET_LABEL[ds],
            color=DATASET_COLOR[ds],
            capsize=3,
            linewidth=1.4,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Dispatch threshold (gpu_kernel_min_work)")
    ax.set_ylabel("Mean fit time (ms)")
    ax.grid(axis="y", which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.axvspan(len(threshold_order) - 1.5, len(threshold_order) - 0.5,
               alpha=0.08, color="red")
    ax.annotate(
        "always-CuPy\n(never use kernel)",
        xy=(len(threshold_order) - 1, df["mean_elapsed_seconds"].max() * 1000 * 0.45),
        ha="center", va="center", fontsize=7, color="#a00",
    )

    fig.tight_layout()
    out = OUT_DIR / "results-kernel-threshold.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fim_headline_figure() -> Path:
    src = REPO_ROOT / "notebooks/profiling/comparison_suites/data/thesis_fim_v2_fim_itemsets_summary.csv"
    df = pd.read_csv(src)
    df["dataset_key"] = df["dataset_path"].str.extract(r"/([^/]+)\.csv$")[0]

    algorithms = [
        "bitset_fim_gpu",
        "bitset_fim_cpu",
        "pyfim_eclat",
        "pyfim_apriori",
        "mlxtend_apriori",
        "mlxtend_fpgrowth",
    ]
    algo_labels = {
        "bitset_fim_gpu": "bitset (GPU)",
        "bitset_fim_cpu": "bitset (CPU)",
        "pyfim_eclat": "pyfim Eclat",
        "pyfim_apriori": "pyfim Apriori",
        "mlxtend_apriori": "mlxtend Apriori",
        "mlxtend_fpgrowth": "mlxtend FpGrowth",
    }

    # Per-panel x-axis caps based on the slowest *competitive* algorithm
    # (mlxtend_apriori on each dataset). FpGrowth and timeouts go off-scale
    # and are drawn as hatched clipped bars with "->" annotations.
    panel_xmax = {"telco": 0.026, "adult": 0.19, "census_income": 20.0}
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.4))
    for ax, ds in zip(axes, DATASET_ORDER):
        sub = df[df["dataset_key"] == ds].set_index("algorithm")
        x_max = panel_xmax[ds]
        y = list(range(len(algorithms)))[::-1]
        for yi, algo in zip(y, algorithms):
            if algo in sub.index:
                t = sub.loc[algo, "mean_s"]
                if t <= x_max:
                    ax.barh(yi, t, color="#4477aa",
                            edgecolor="black", linewidth=0.4)
                    label = f" {_fmt_seconds(t)}"
                    ax.text(t, yi, label, va="center", ha="left", fontsize=7)
                else:
                    ax.barh(yi, x_max, color="white", edgecolor="#a00",
                            hatch="///", linewidth=0.6)
                    ax.text(x_max, yi, f" $\\rightarrow$ {_fmt_seconds(t)}",
                            va="center", ha="left", fontsize=7, color="#a00")
            else:
                ax.barh(yi, x_max, color="white", edgecolor="#a00",
                        hatch="///", linewidth=0.6)
                ax.text(x_max, yi, " $\\rightarrow$ timeout",
                        va="center", ha="left", fontsize=7, color="#a00")
        ax.set_yticks(y)
        ax.set_yticklabels([algo_labels[a] for a in algorithms])
        ax.set_xlim(0, x_max * 1.75)
        ax.set_title(DATASET_LABEL[ds])
        ax.set_xlabel("Fit time (s, linear)")
        ax.grid(axis="x", which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    out = OUT_DIR / "results-fim-headline.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _fmt_seconds(t: float) -> str:
    if t >= 1.0:
        return f"{t:.2f} s"
    if t >= 0.01:
        return f"{t*1000:.0f} ms"
    return f"{t*1000:.1f} ms"


def gpu_breakdown_figure() -> Path:
    # Numbers come straight from tab:gpu-time-breakdown in results.tex
    # (nsys stats --report=cuda_kern_exec_sum on a warm-start census_income fit).
    wall_clock_ms = 919.0
    parts = [
        ("CuPy bitmask AND", 25.83, "#4477aa"),
        ("Custom bitmask AND + popcount", 1.81, "#ee6677"),
        ("Kernel input assembly from queue", 1.49, "#228833"),
        ("Bitmask upload + kernel JIT", 1.32, "#ccbb44"),
    ]
    labels = [p[0] for p in parts]
    sizes = [p[1] for p in parts]
    colors = [p[2] for p in parts]
    total = sum(sizes)

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.2),
    )

    for w, (label, size, _c) in zip(wedges, parts):
        theta = math.radians((w.theta2 + w.theta1) / 2.0)
        share = size / total
        text = f"{size:.2f} ms\n({share*100:.1f} %)"
        if share > 0.20:
            x = 0.55 * math.cos(theta)
            y = 0.55 * math.sin(theta)
            ax.text(x, y, text, ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        else:
            x_in = math.cos(theta)
            y_in = math.sin(theta)
            x_out = 1.55 * x_in
            y_out = 1.35 * y_in
            ha = "left" if x_in >= 0 else "right"
            ax.annotate(
                text,
                xy=(x_in, y_in),
                xytext=(x_out, y_out),
                ha=ha, va="center",
                fontsize=8,
                arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.6),
            )

    ax.legend(
        wedges, labels,
        loc="center left",
        bbox_to_anchor=(1.18, 0.3),
        frameon=False,
        fontsize=8,
    )
    context_ax = ax.inset_axes([1.2, 0.5, 0.34, 0.34])
    context_sizes = [total, wall_clock_ms - total]
    context_ax.pie(
        context_sizes,
        colors=["#222222", "#dddddd"],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=0.5),
    )
    
    context_ax.set_title(
        f"GPU time in fit()\n"
        f"(total $\\approx$ {total:.1f} ms of a {wall_clock_ms:.0f} ms fit)", fontsize=7, pad=1
    )
    context_ax.text(
        0, 0, f"{total:.0f} ms\n{total / wall_clock_ms * 100:.1f} %",
        ha="center", va="center", fontsize=6,
    )
    context_ax.set_aspect("equal")
    ax.set_title(
        f"GPU time per kernel category\n"
        
    )
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.45, 1.4)

    out = OUT_DIR / "results-gpu-time-breakdown.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def commit_evolution_figure() -> Path:
    src = REPO_ROOT / "notebooks/profiling/commit_benchmarks/results/commit_benchmarks_20260509T202029Z.csv"
    df = pd.read_csv(src)

    snapshot_order = [
        "baseline",
        "first_bitset_cpu",
        "batch_support_counting",
        "final",
        "gpu_prefix_only",
        "gpu_carryover",
    ]
    snapshot_short = {
        "baseline": "baseline",
        "first_bitset_cpu": "first bitset",
        "batch_support_counting": "batch support",
        "final": "final CPU",
        "gpu_prefix_only": "GPU prefix-\nonly",
        "gpu_carryover": "GPU carryover\n(current)",
    }
    backend_style = {"cpu": "-", "gpu": "--"}

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x_positions = list(range(len(snapshot_order)))

    timeout_points = []  # (snapshot_idx, dataset, backend)
    for ds in DATASET_ORDER:
        for backend in ("cpu", "gpu"):
            sub = (df[(df["dataset"] == ds) & (df["backend"] == backend)]
                   .set_index("snapshot").reindex(snapshot_order))
            ys, yerrs = [], []
            for i, snap in enumerate(snapshot_order):
                row = sub.loc[snap]
                if row["status"] == "ok":
                    ys.append(row["fit_mean_s"] * 1000)
                    yerrs.append(row["fit_std_s"] * 1000)
                else:
                    ys.append(float("nan"))
                    yerrs.append(float("nan"))
                    timeout_points.append((i, ds, backend))
            ax.errorbar(
                x_positions, ys, yerr=yerrs,
                marker="o", color=DATASET_COLOR[ds],
                linestyle=backend_style[backend],
                capsize=3, linewidth=1.4,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([snapshot_short[s] for s in snapshot_order], fontsize=8)
    ax.set_ylabel("Mean fit time (ms)")
    ax.set_xlabel("Implementation snapshot")
    ax.grid(axis="y", which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_ylim(0, 7500)

    # Two legends: dataset (color) and backend (linestyle).
    from matplotlib.lines import Line2D
    dataset_handles = [
        Line2D([0], [0], color=DATASET_COLOR[ds], linewidth=1.6,
               marker="o", label=DATASET_LABEL[ds])
        for ds in DATASET_ORDER
    ]
    backend_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.4, label="CPU"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.4, label="GPU"),
    ]
    leg1 = ax.legend(handles=dataset_handles, loc="upper right", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=backend_handles, loc="upper right",
              bbox_to_anchor=(1.0, 0.78), framealpha=0.9, fontsize=8)

    # Mark timeouts at top of axis with the dataset's colour. No red edge.
    y_to = ax.get_ylim()[1] * 0.95
    for x, ds, _backend in timeout_points:
        ax.scatter(x, y_to, marker="X", color=DATASET_COLOR[ds],
                   s=70, linewidth=0, zorder=5)
    if timeout_points:
        ax.text(timeout_points[0][0], y_to * 0.92,
                "timeout (900 s cap)",
                ha="left", va="top", fontsize=7, color="#444")

    fig.tight_layout()
    out = OUT_DIR / "results-commit-evolution.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def commit_memory_bars_figure() -> Path | None:
    # Memory is measured in a separate run from speed (instrumentation slows
    # the fit too much to share a CSV). Update this path when the new peak-
    # memory CSV is dropped into results/.
    src = REPO_ROOT / "notebooks/profiling/commit_benchmarks/results/commit_benchmarks_20260509T232723Z_peak_memory.csv"
    if not src.exists():
        print(f"skipped commit_memory_bars_figure: {src.name} not found")
        return None
    df = pd.read_csv(src)

    # Per-snapshot backend pick: CPU snapshots have no VRAM column populated,
    # so we read their host-RSS row from the CPU run; GPU snapshots come from
    # the GPU run so peak_used_mb is meaningful.
    snapshot_specs = [
        ("baseline",               "cpu", "baseline"),
        ("first_bitset_cpu",       "cpu", "first bitset"),
        ("batch_support_counting", "cpu", "batch support"),
        ("gpu_prefix_only",        "gpu", "GPU prefix-only"),
        ("gpu_carryover",          "gpu", "GPU carryover"),
    ]
    snapshots = [s for s, _, _ in snapshot_specs]
    snapshot_short = {s: short for s, _, short in snapshot_specs}
    dataset = "adult"

    keyed = df.set_index(["snapshot", "backend", "dataset"])

    def _row(snap: str, backend: str):
        try:
            return keyed.loc[(snap, backend, dataset)]
        except KeyError:
            return None

    rss: list[float] = []
    vram: list[float] = []
    for snap, backend, _ in snapshot_specs:
        row = _row(snap, backend)
        if row is None or row["status"] != "ok":
            rss.append(0.0)
            vram.append(0.0)
            continue
        rss.append(float(row["peak_host_rss_delta_mb"]))
        v = row["peak_used_mb"]
        vram.append(float(v) if pd.notna(v) else 0.0)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(6.4, 3.8),
        gridspec_kw={"height_ratios": [1, 4], "hspace": 0.06},
    )
    x = list(range(len(snapshots)))
    width = 0.38

    for ax in (ax_top, ax_bot):
        ax.bar([xi - width / 2 for xi in x], rss, width,
               label="Peak RAM", color="#4477aa", edgecolor="black", linewidth=0.4)
        ax.bar([xi + width / 2 for xi in x], vram, width,
               label="Peak VRAM", color="#ee6677", edgecolor="black", linewidth=0.4)

    bot_top_limit = 380
    ax_top.set_ylim(2250, 2400)
    ax_bot.set_ylim(0, bot_top_limit)
    ax_top.set_yticks([2300, 2400])
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)

    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    d2 = d * (4)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d2, 1 + d2), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d2, 1 + d2), **kwargs)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([snapshot_short[s] for s in snapshots], fontsize=8)
    ax_bot.set_ylabel("Memory (MB)")
    ax_bot.yaxis.set_label_coords(-0.08, 0.6)
    ax_bot.set_xlabel("Implementation snapshot")
    ax_top.set_title(f"Peak RAM vs VRAM ({dataset})")
    for ax in (ax_top, ax_bot):
        ax.grid(axis="y", which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax_bot.legend(loc="upper right", framealpha=0.9, fontsize=8)

    def _label(ax, xpos, value):
        if value <= 0:
            ax.text(xpos, 6, "0", ha="center", va="bottom", fontsize=7, color="#666")
            return
        text = f"{value:.0f}" if value >= 10 else f"{value:.1f}"
        ax.text(xpos, value + 6, f"{text} MB", ha="center", va="bottom", fontsize=7)

    for i, (r, v) in enumerate(zip(rss, vram)):
        rss_x = x[i] - width / 2
        vram_x = x[i] + width / 2
        ax_for_rss = ax_top if r > bot_top_limit else ax_bot
        ax_for_rss.text(rss_x, min(r, 2380) + (4 if ax_for_rss is ax_top else 6),
                        f"{r:.0f} MB" if r >= 10 else (f"{r:.1f} MB" if r > 0 else "0"),
                        ha="center", va="bottom", fontsize=7)
        _label(ax_bot, vram_x, v)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.16, hspace=0.06)
    out = OUT_DIR / "results-commit-memory-bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    for fn in (kernel_threshold_figure, fim_headline_figure,
               gpu_breakdown_figure, commit_evolution_figure,
               commit_memory_bars_figure):
        out = fn()
        if out is not None:
            print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
