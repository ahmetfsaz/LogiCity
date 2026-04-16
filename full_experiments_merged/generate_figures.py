"""
Generate two publication-quality figures from experiment results.

Figure 1: 2x3 panel — Subrule DSR and Action DSR vs k for all 3 mode pairs.
Figure 2: Scatter plot — baseline difficulty (k=0 Act DSR) vs semantic advantage.

Usage:
    python generate_figures.py
    # outputs fig1_performance_curves.pdf and fig2_baseline_vs_advantage.pdf
"""

import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV_PATH = "summary.csv"

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

SEM_COLOR = "#2166ac"
RND_COLOR = "#b2182b"

# ── Load data ────────────────────────────────────────────────────────────────
def load_data():
    data = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            data[(r["label"], r["mode"], r["k"])] = r
    return data


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Performance curves (2 rows × 3 columns)
# ══════════════════════════════════════════════════════════════════════════════
def make_figure1(data):
    # "Hypothesis DSR" is the same as "Subrule DSR" — renamed for paper terminology
    rule_sets = [
        ("fov5_20c8p_r70_original",       "#1b9e77", "o", "Original"),
        ("fov5_20c8p_r70_extended",        "#d95f02", "^", "Extended"),
        ("fov5_20c8p_r70_spatial",         "#7570b3", "v", "Spatial"),
        ("fov5_20c8p_r70_discriminative",  "#e7298a", "D", "Discriminative"),
    ]

    mode_pairs = [
        ("semantic",           "semantic_random",           "Sensor GNA"),
        ("semantic_lna",       "semantic_lna_random",       "Multi-Zone LNA"),
        ("semantic_lna_single","semantic_lna_random_single","Single-Zone GNA"),
    ]
    # Columns: Hypothesis DSR (left), Action DSR (right)
    metrics = [
        ("sr_mean",  "sr_std",  "Hypothesis DSR"),
        ("act_mean", "act_std", "Action DSR"),
    ]

    # Layout: 3 rows (modes) × 2 cols (metrics)
    fig, axes = plt.subplots(3, 2, figsize=(5.0, 6.5), sharex=True)

    for row, (sem_mode, rnd_mode, row_title) in enumerate(mode_pairs):
        is_lna = "lna" in sem_mode
        k_labels = [f"k2={i}" for i in range(6)] if is_lna else [str(i) for i in range(6)]
        k_nums = np.arange(6)

        for col, (mean_key, std_key, col_label) in enumerate(metrics):
            ax = axes[row, col]

            for label, color, marker, rs_name in rule_sets:
                for mode_key, is_sem in [(sem_mode, True), (rnd_mode, False)]:
                    means = []
                    for k in k_labels:
                        key = (label, mode_key, k)
                        if key in data and data[key]["sr_mean"] != "FAIL":
                            means.append(float(data[key][mean_key]))
                        else:
                            means.append(np.nan)
                    means = np.array(means)

                    ls = "-" if is_sem else "--"
                    mfc = "white" if is_sem else color
                    lw = 1.6 if is_sem else 1.0

                    ax.plot(k_nums, means, color=color, ls=ls, marker=marker,
                            markerfacecolor=mfc, markeredgecolor=color,
                            markeredgewidth=1.0, markersize=4.5,
                            linewidth=lw, zorder=3 if is_sem else 2)

            ax.set_xlim(-0.3, 5.3)
            ax.set_xticks(k_nums)
            ax.set_xticklabels([str(i) for i in range(6)])
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.tick_params(direction="in", top=True, right=True)

            if row == 0:
                ax.set_title(col_label, fontweight="bold", pad=6)
            if row == 2:
                ax.set_xlabel("$k$")

            if col == 0:
                ax.set_ylabel(row_title)

            # Y-axis ranges per panel
            if row == 0:
                ax.set_ylim(0.78, 1.015)
            elif col == 0:  # Hypothesis DSR for LNA rows
                ax.set_ylim(0.85, 1.005)
            else:           # Action DSR for LNA rows
                ax.set_ylim(0.80, 0.95)

    # Legend
    rs_handles = [
        Line2D([0], [0], color=c, ls="-", marker=m,
               markerfacecolor="white", markeredgecolor=c,
               markeredgewidth=1.0, markersize=5, label=n)
        for _, c, m, n in rule_sets
    ]
    style_handles = [
        Line2D([0], [0], color="0.3", ls="-", lw=1.6, label="Semantic"),
        Line2D([0], [0], color="0.3", ls="--", lw=1.0, label="FOL-baseline"),
    ]
    all_handles = rs_handles + style_handles

    fig.legend(handles=all_handles, loc="lower center", ncol=3,
              frameon=True, fancybox=False, edgecolor="0.7",
              bbox_to_anchor=(0.5, 0.005), fontsize=7.5,
              handletextpad=0.4, columnspacing=1.2)

    fig.align_ylabels(axes[:, 0])
    plt.subplots_adjust(hspace=0.30, wspace=0.30, bottom=0.14)

    out = "fig1_performance_curves.pdf"
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=300)
    print(f"Saved {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Baseline difficulty vs semantic advantage (scatter)
# ══════════════════════════════════════════════════════════════════════════════
def make_figure2(data):
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Marker style per rule set
    rule_styles = {
        "original":       ("D", "#1b9e77", "Original (12)"),
        "extended":       ("^", "#d95f02", "Extended (48)"),
        "spatial":        ("v", "#7570b3", "Spatial (30)"),
        "discriminative": ("o", "#e7298a", "Discrimin. (30)"),
    }

    # Marker size per FOV
    fov_sizes = {"fov5": 40, "fov7": 75, "fov10": 130}

    all_labels = sorted(set(
        key[0] for key in data
        if key[1] == "semantic" and data[key]["sr_mean"] != "FAIL"
    ))

    xs, ys = [], []
    for label in all_labels:
        base_key = (label, "semantic", "0")
        sem_key  = (label, "semantic", "3")
        rnd_key  = (label, "semantic_random", "3")
        if not all(k in data and data[k]["sr_mean"] != "FAIL"
                   for k in [base_key, sem_key, rnd_key]):
            continue

        baseline = float(data[base_key]["act_mean"])
        gap = float(data[sem_key]["act_mean"]) - float(data[rnd_key]["act_mean"])

        # Determine rule set and FOV
        rule_set = None
        for rs in rule_styles:
            if rs in label:
                rule_set = rs
                break
        fov_tag = label.split("_")[0]

        marker, color, _ = rule_styles[rule_set]
        size = fov_sizes.get(fov_tag, 60)

        # Non-default configs (varied cars, peds, region) get a thicker edge
        is_variant = ("10c" in label or "4p" in label or "12p" in label
                      or "r120" in label or "r50" in label)
        edgew = 1.4 if is_variant else 0.5
        edgec = "#333333" if is_variant else "black"

        ax.scatter(baseline, gap, marker=marker, s=size,
                   facecolors=color, edgecolors=edgec, linewidths=edgew,
                   zorder=5, alpha=0.85)
        xs.append(baseline)
        ys.append(gap)

    # Regression line
    xs, ys = np.array(xs), np.array(ys)
    m, b = np.polyfit(xs, ys, 1)
    x_fit = np.linspace(xs.min() - 0.01, xs.max() + 0.01, 100)
    ax.plot(x_fit, m * x_fit + b, color="0.4", ls="--", lw=1.0, zorder=2)

    # Pearson r
    r = np.corrcoef(xs, ys)[0, 1]
    ax.text(0.97, 0.97, f"$r = {r:.3f}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.7", alpha=0.9))

    ax.set_xlabel("Baseline Action DSR ($k = 0$)")
    ax.set_ylabel("Semantic Advantage\n(Action DSR gap at $k = 3$)")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(direction="in", top=True, right=True)

    # Legend — rule sets (shape + color)
    rule_handles = []
    for rs, (marker, color, lbl) in rule_styles.items():
        rule_handles.append(
            Line2D([0], [0], marker=marker, color="none",
                   markerfacecolor=color, markeredgecolor="black",
                   markeredgewidth=0.5, markersize=6, label=lbl)
        )

    # Legend — FOV (size)
    fov_handles = []
    for fov_tag, sz in [("fov5", 40), ("fov7", 75), ("fov10", 130)]:
        fov_handles.append(
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="0.6", markeredgecolor="black",
                   markeredgewidth=0.5,
                   markersize=math.sqrt(sz) * 0.55,
                   label=f"FOV={fov_tag[3:]}")
        )

    leg1 = ax.legend(handles=rule_handles, loc="upper right",
                     title="Rule set", fontsize=7, title_fontsize=7.5,
                     frameon=True, fancybox=False, edgecolor="0.7",
                     bbox_to_anchor=(0.98, 0.88), borderaxespad=0.2,
                     handletextpad=0.3, labelspacing=0.3)
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=fov_handles, loc="center right",
                     title="FOV", fontsize=7, title_fontsize=7.5,
                     frameon=True, fancybox=False, edgecolor="0.7",
                     bbox_to_anchor=(0.98, 0.45), borderaxespad=0.2,
                     handletextpad=0.3, labelspacing=0.3)

    plt.tight_layout()
    out = "fig2_baseline_vs_advantage.pdf"
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=300)
    print(f"Saved {out}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_data()
    make_figure1(data)
    make_figure2(data)
    print("Done.")
