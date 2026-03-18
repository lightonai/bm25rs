"""Generate benchmark bar charts for README — bm25s vs bm25x vs bm25x GPU."""

import matplotlib.pyplot as plt
import numpy as np

# ── Light theme ─────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#ccc",
        "axes.labelcolor": "#333",
        "text.color": "#333",
        "xtick.color": "#555",
        "ytick.color": "#555",
        "grid.color": "#e0e0e0",
        "grid.alpha": 0.8,
        "font.family": "monospace",
        "font.size": 11,
    }
)

BM25S_COLOR = "#bbb"
BM25X_COLOR = "#2d8cf0"
BM25X_GPU_COLOR = "#22c55e"

# ── Benchmark data (BEIR datasets) ─────────────────────────────────
# CPU: measured on Apple M3
# GPU: measured on NVIDIA H100, all datasets
datasets = [
    "NFCorpus\n3.6k docs",
    "SciFact\n5k docs",
    "SciDocs\n26k docs",
    "FiQA\n58k docs",
    "MS MARCO\n8.8M docs",
]

# NDCG@10 — GPU identical to CPU (verified on all datasets)
ndcg_bm25s = [0.3064, 0.6617, 0.1538, 0.2326, 0.2124]
ndcg_bm25x = [0.3287, 0.6904, 0.1600, 0.2514, 0.2186]
ndcg_bm25x_gpu = [0.3287, 0.6904, 0.1600, 0.2514, 0.2240]

# Index throughput (docs/s)
index_tput_bm25s = [13_658, 15_138, 17_567, 23_698, 23_395]
index_tput_bm25x = [70_621, 77_644, 94_390, 144_060, 82_910]
index_tput_bm25x_gpu = [32_551, 57_570, 68_360, 205_115, 295_458]  # H100 (warm CUDA)

# Search throughput (queries/s)
search_tput_bm25s = [36_992, 18_969, 6_543, 2_431, 16]
search_tput_bm25x = [128_245, 25_992, 7_032, 4_760, 65]
search_tput_bm25x_gpu = [6_940, 5_682, 6_197, 5_935, 3_430]  # H100 GPU search (warm)


def _fmt_tput(v):
    if v >= 1_000:
        return f"{v / 1_000:,.0f}k"
    return f"{v:,.0f}"


def grouped_bar_3(
    ax,
    labels,
    vals_s,
    vals_x,
    vals_gpu,
    ylabel,
    title,
    fmt="{:.4f}",
    show_legend=True,
    log_scale=False,
    abs_s=None,
    abs_x=None,
    abs_gpu=None,
    abs_unit="",
):
    x = np.arange(len(labels))
    w = 0.25

    bars_s = ax.bar(
        x - w,
        vals_s,
        w,
        label="bm25s",
        color=BM25S_COLOR,
        edgecolor="#ffffff",
        linewidth=0.5,
        zorder=3,
    )
    bars_x = ax.bar(
        x,
        vals_x,
        w,
        label="bm25x",
        color=BM25X_COLOR,
        edgecolor="#ffffff",
        linewidth=0.5,
        zorder=3,
    )
    bars_gpu = ax.bar(
        x + w,
        vals_gpu,
        w,
        label="bm25x GPU",
        color=BM25X_GPU_COLOR,
        edgecolor="#ffffff",
        linewidth=0.5,
        zorder=3,
    )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", color="#111", pad=12)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Value labels on top of bars
    for i, (bar_s, bar_x, bar_g, vs, vx, vg) in enumerate(
        zip(bars_s, bars_x, bars_gpu, vals_s, vals_x, vals_gpu)
    ):
        for bar, v, color, fw in [
            (bar_s, vs, "#888", "normal"),
            (bar_x, vx, "#2d8cf0", "bold"),
            (bar_g, vg, "#16a34a", "bold"),
        ]:
            label = fmt.format(v)
            y_pos = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                label,
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=color,
                fontweight=fw,
            )

        # Absolute throughput inside bars
        if abs_s is not None and abs_x is not None and abs_gpu is not None:
            for bar, abs_v, color in [
                (bar_s, abs_s[i], "#555"),
                (bar_x, abs_x[i], "#fff"),
                (bar_g, abs_gpu[i], "#fff"),
            ]:
                txt = f"{_fmt_tput(abs_v)} {abs_unit}"
                h = bar.get_height()
                if h > 0.3:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h**0.5 if log_scale else h / 2,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color=color,
                        fontweight="bold",
                        rotation=90,
                    )

    if show_legend:
        ax.legend(
            loc="upper left",
            fontsize=8,
            facecolor="#fff",
            edgecolor="#ccc",
            labelcolor="#333",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Compute speedups (bm25s = 1x baseline) ─────────────────────────
baseline = [1.0] * len(datasets)
index_speedup_x = [x / s for s, x in zip(index_tput_bm25s, index_tput_bm25x)]
index_speedup_gpu = [x / s for s, x in zip(index_tput_bm25s, index_tput_bm25x_gpu)]
search_speedup_x = [x / s for s, x in zip(search_tput_bm25s, search_tput_bm25x)]
search_speedup_gpu = [x / s for s, x in zip(search_tput_bm25s, search_tput_bm25x_gpu)]

# ── Generate figure ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
fig.subplots_adjust(wspace=0.3, left=0.05, right=0.97, top=0.88, bottom=0.15)

grouped_bar_3(
    axes[0],
    datasets,
    ndcg_bm25s,
    ndcg_bm25x,
    ndcg_bm25x_gpu,
    "NDCG@10",
    "Retrieval Quality (NDCG@10)",
    fmt="{:.4f}",
    show_legend=True,
)

grouped_bar_3(
    axes[1],
    datasets,
    baseline,
    index_speedup_x,
    index_speedup_gpu,
    "speedup vs bm25s (log)",
    "Indexing Speedup",
    fmt="{:.1f}x",
    show_legend=False,
    log_scale=True,
    abs_s=index_tput_bm25s,
    abs_x=index_tput_bm25x,
    abs_gpu=index_tput_bm25x_gpu,
    abs_unit="d/s",
)

grouped_bar_3(
    axes[2],
    datasets,
    baseline,
    search_speedup_x,
    search_speedup_gpu,
    "speedup vs bm25s (log)",
    "Search Speedup",
    fmt="{:.1f}x",
    show_legend=False,
    log_scale=True,
    abs_s=search_tput_bm25s,
    abs_x=search_tput_bm25x,
    abs_gpu=search_tput_bm25x_gpu,
    abs_unit="q/s",
)

fig.savefig("assets/benchmarks.png", dpi=200, facecolor="#ffffff")
print("Saved assets/benchmarks.png")
