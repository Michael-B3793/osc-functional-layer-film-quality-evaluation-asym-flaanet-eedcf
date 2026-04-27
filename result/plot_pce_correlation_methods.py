from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, LinearSegmentedColormap

COLORS = {"EEDCF": "#D58AAE", "ASDI": "#9EC4F8", "MMCSO_dB": "#C3BBF5"}
DISPLAY_NAMES = {"EEDCF": "EEDCF", "ASDI": "ASDI", "MMCSO_dB": "MMCSO"}
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "custom_pastel",
    ["#9EC4F8", "#C3BBF5", "#D58AAE"],
)
LINE_WIDTH = 3.0
SCATTER_EDGE_WIDTH = 1.8
MARKER_EDGE_WIDTH = 1.8

FONT_SIZE_BASE = 17
FONT_SIZE_LABEL = 20
FONT_SIZE_TITLE = 22
FONT_SIZE_LEGEND = 16
FONT_SIZE_ANNOTATION = 16
AXES_LINE_WIDTH = 2.2
HEATMAP_AXES_LINE_WIDTH = 1.6


def spearman_corr(x: np.ndarray, y: np.ndarray):
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_tau_a(x: np.ndarray, y: np.ndarray):
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    return float((concordant - discordant) / denom) if denom else np.nan


def pairwise_order_agreement(x: np.ndarray, y: np.ndarray):
    n = len(x)
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 or dy == 0:
                continue
            total += 1
            if dx * dy > 0:
                agree += 1
    return agree / total if total else np.nan, total


def main():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": FONT_SIZE_BASE,
            "axes.labelsize": FONT_SIZE_LABEL,
            "axes.titlesize": FONT_SIZE_TITLE,
            "xtick.labelsize": FONT_SIZE_BASE,
            "ytick.labelsize": FONT_SIZE_BASE,
            "legend.fontsize": FONT_SIZE_LEGEND,
            "axes.linewidth": AXES_LINE_WIDTH,
            "xtick.major.width": AXES_LINE_WIDTH,
            "ytick.major.width": AXES_LINE_WIDTH,
        }
    )

    root = Path(__file__).resolve().parents[1]
    data_path = root / "result" / "data.csv"
    out_dir = root / "result" / "pce_correlation_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    metrics = ["EEDCF", "ASDI", "MMCSO_dB"]

    corr_rows = []
    for m in metrics:
        x = df["PCE"].to_numpy(dtype=float)
        y = df[m].to_numpy(dtype=float)
        pearson = float(np.corrcoef(x, y)[0, 1])
        spearman = spearman_corr(x, y)
        kendall = kendall_tau_a(x, y)
        poa, pairs = pairwise_order_agreement(x, y)
        corr_rows.append(
            {
                "Metric": m,
                "Pearson_r": pearson,
                "Pearson_R2": pearson * pearson,
                "Spearman_rho": spearman,
                "Kendall_tau": kendall,
                "Pairwise_Order_Agreement": poa,
                "Pairs_Used": pairs,
            }
        )
    corr_df = pd.DataFrame(corr_rows)

    x = df["PCE"].to_numpy(dtype=float)
    fig, ax1 = plt.subplots(figsize=(11, 7), dpi=180)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.24))
    for axis in [ax1, ax2, ax3]:
        for spine in axis.spines.values():
            spine.set_color("black")
            spine.set_linewidth(AXES_LINE_WIDTH)
    ax1.set_box_aspect(1)
    axes_map = {"EEDCF": ax1, "ASDI": ax2, "MMCSO_dB": ax3}
    for m in metrics:
        ax = axes_map[m]
        y = df[m].to_numpy(dtype=float)
        color = COLORS[m]
        ax.scatter(
            x,
            y,
            s=42,
            facecolors=[to_rgba(color, 0.35)],
            edgecolors=color,
            linewidths=SCATTER_EDGE_WIDTH,
        )
        coef = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        yy = coef[0] * xx + coef[1]
        ax.plot(xx, yy, linewidth=LINE_WIDTH, color=color)
        ax.set_ylabel(DISPLAY_NAMES[m], color=color, labelpad=16)
        ax.tick_params(axis="y", color="black", labelcolor=color, width=AXES_LINE_WIDTH)
    legend_handles = []
    legend_labels = []
    max_len = max(len(DISPLAY_NAMES[m]) for m in metrics)
    for m in metrics:
        row = corr_df[corr_df["Metric"] == m].iloc[0]
        color = COLORS[m]
        handle = plt.Line2D([0], [0], color=color, linewidth=LINE_WIDTH)
        legend_handles.append(handle)
        m_txt = f"{DISPLAY_NAMES[m]:<{max_len}}"
        legend_labels.append(
            f"{m_txt}  r={row['Pearson_r']:>6.3f}  R\u00b2={row['Pearson_R2']:>6.3f}"
        )
    ax1.set_xlabel("PCE")
    ax1.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        prop={"family": "monospace"},
    )
    fig.subplots_adjust(top=0.93, right=0.80)
    fig.tight_layout()
    fig.savefig(out_dir / "01_scatter_linear_fit.png", transparent=True)
    plt.close(fig)

    rank_df = pd.DataFrame({"PCE": df["PCE"]})
    rank_df["PCE_rank"] = rank_df["PCE"].rank(method="average")
    for m in metrics:
        rank_df[f"{m}_rank"] = df[m].rank(method="average")

    fig, ax = plt.subplots(figsize=(11, 7), dpi=180)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(AXES_LINE_WIDTH)
    ax.set_box_aspect(1)
    max_len_rank = max(len(DISPLAY_NAMES[m]) for m in metrics)
    for m in metrics:
        row = corr_df[corr_df["Metric"] == m].iloc[0]
        rho = row["Spearman_rho"]
        m_txt = f"{DISPLAY_NAMES[m]:<{max_len_rank}}"
        ax.scatter(
            rank_df["PCE_rank"],
            rank_df[f"{m}_rank"],
            s=42,
            facecolors=[to_rgba(COLORS[m], 0.35)],
            edgecolors=COLORS[m],
            linewidths=SCATTER_EDGE_WIDTH,
            label=f"{m_txt}  \u03c1={rho:>6.3f}",
        )
    ax.plot([1, len(df)], [1, len(df)], linestyle="--", linewidth=LINE_WIDTH, color="gray")
    ax.set_xlabel("PCE rank")
    ax.set_ylabel("Metric rank")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        prop={"family": "monospace"},
    )
    fig.subplots_adjust(top=0.93, right=0.96)
    fig.tight_layout()
    fig.savefig(out_dir / "02_rank_consistency_spearman.png", transparent=True)
    plt.close(fig)

    bins = pd.qcut(df["PCE"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    binned = (
        df.assign(PCE_bin=bins)
        .groupby("PCE_bin", as_index=False)[["PCE", "EEDCF", "ASDI", "MMCSO_dB"]]
        .mean()
    )

    fig, ax1 = plt.subplots(figsize=(11, 7), dpi=180)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.24))
    for axis in [ax1, ax2, ax3]:
        for spine in axis.spines.values():
            spine.set_color("black")
            spine.set_linewidth(AXES_LINE_WIDTH)
    ax1.set_box_aspect(1)
    axes_map = {"EEDCF": ax1, "ASDI": ax2, "MMCSO_dB": ax3}
    x_bins = np.arange(len(binned["PCE_bin"]))
    legend_handles = []
    legend_labels = []
    for m in metrics:
        ax = axes_map[m]
        color = COLORS[m]
        line = ax.plot(
            x_bins,
            binned[m].to_numpy(dtype=float),
            marker="o",
            linewidth=LINE_WIDTH,
            color=color,
            markerfacecolor=to_rgba(color, 0.35),
            markeredgecolor=color,
            markeredgewidth=MARKER_EDGE_WIDTH,
        )[0]
        ax.set_ylabel(f"Mean {DISPLAY_NAMES[m]}", color=color, labelpad=16)
        ax.tick_params(axis="y", color="black", labelcolor=color, width=AXES_LINE_WIDTH)
        legend_handles.append(line)
        legend_labels.append(DISPLAY_NAMES[m])
    ax1.set_xticks(x_bins)
    ax1.set_xticklabels(binned["PCE_bin"])
    ax1.set_xlabel("PCE quartile")
    ax1.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
    )
    fig.subplots_adjust(top=0.93, right=0.80)
    fig.tight_layout()
    fig.savefig(out_dir / "03_binned_trend.png", transparent=True)
    plt.close(fig)

    corr_mat = corr_df.set_index("Metric")[["Pearson_r", "Spearman_rho", "Kendall_tau"]]
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=180)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(HEATMAP_AXES_LINE_WIDTH)
    im = ax.imshow(corr_mat.to_numpy(), aspect="auto", cmap=HEATMAP_CMAP)
    x_labels = [c.split("_")[0] for c in corr_mat.columns]
    ax.set_xticks(range(corr_mat.shape[1]))
    ax.set_xticklabels(x_labels, rotation=0, ha="center")
    ax.set_yticks(range(corr_mat.shape[0]))
    ax.set_yticklabels([DISPLAY_NAMES[i] for i in corr_mat.index])
    for i in range(corr_mat.shape[0]):
        for j in range(corr_mat.shape[1]):
            ax.text(
                j,
                i,
                f"{corr_mat.iloc[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=FONT_SIZE_ANNOTATION,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.08)
    cbar.ax.tick_params(labelsize=FONT_SIZE_BASE)
    fig.subplots_adjust(left=0.22, right=0.90, top=0.95, bottom=0.14)
    fig.tight_layout()
    fig.savefig(out_dir / "04_correlation_heatmap.png", transparent=True)
    plt.close(fig)

    print(f"Saved plots to: {out_dir}")
    print("Generated files:")
    for p in sorted(out_dir.glob("*")):
        print(p.name)


if __name__ == "__main__":
    main()
