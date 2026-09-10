"""
generate_ijf_figures.py
=======================
Generates three IJF-style evaluation figures:

  Figure 6 — FEI heatmap (supervisor's top recommendation)
  Figure 7 — CRPS horizontal bar chart
  Figure 8 — Reliability (calibration) diagram

All numbers sourced from existing computed results.

Usage:
    python scripts/generate_ijf_figures.py \
        --fei_csv   figures/fei_by_event_type.csv \
        --crps_csv  figures/proper_scores.csv \
        --kde_csv   kde_output_v3/kde_v3_results_h15.csv \
        --garch_csv figures/kde_v3_results_garch_family.csv \
        --output    figures
"""

import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import norm as scipy_norm

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
})

BLUE   = "#1f77b4"
RED    = "#d62728"
GREEN  = "#2ca02c"
ORANGE = "#ff7f0e"


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — FEI Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig6_fei_heatmap(fei_csv: Path, out_dir: Path):
    df = pd.read_csv(fei_csv)

    EVENT_TYPES = [
        "Geopolitical News",
        "Macroeconomic News",
        "Price Movement",
        "Supply Shocks",
    ]
    ET_SHORT = ["Geopolitical", "Macro-\neconomic", "Price\nMovement", "Supply\nShocks"]

    # Row labels: Benchmark × Metric
    rows = [
        ("GARCH",   "garch",  "KL"),
        ("GARCH",   "garch",  "JS"),
        ("GARCH",   "garch",  "Wass"),
        ("EGARCH",  "egarch", "KL"),
        ("EGARCH",  "egarch", "JS"),
        ("EGARCH",  "egarch", "Wass"),
        ("GJR",     "gjr",    "KL"),
        ("GJR",     "gjr",    "JS"),
        ("GJR",     "gjr",    "Wass"),
    ]

    # Build matrix
    matrix = np.zeros((len(rows), len(EVENT_TYPES)))
    for i, (blabel, bkey, mlabel) in enumerate(rows):
        for j, et in enumerate(EVENT_TYPES):
            sub = df[df["event_type"] == et]
            col = f"{bkey}_{mlabel.lower()}"
            val = sub[col].dropna().mean() if col in sub.columns else np.nan
            matrix[i, j] = val

    # Row labels for y-axis
    y_labels = [f"{b} — {m}" for b, _, m in rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(
        "Figure 6: Forecast Efficacy Index Heatmap\n"
        "KDE v3 vs GARCH Family by Event Type and Divergence Metric\n"
        "$h = +15$ days, $N = 377$ events  |  All values $> 1$ indicate KDE v3 wins",
        fontsize=9.5
    )

    # Colormap: white at 1.0, increasingly green above
    # Clip for display at 1.0 to 1.6
    vmin, vmax = 1.0, 1.6
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "fei_cmap",
        [(0.0, "white"), (0.3, "#c8e6c9"), (0.7, "#388e3c"), (1.0, "#1b5e20")]
    )

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # Annotate cells
    for i in range(len(rows)):
        for j in range(len(EVENT_TYPES)):
            val = matrix[i, j]
            color = "black" if val < 1.35 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    # Axis labels
    ax.set_xticks(range(len(EVENT_TYPES)))
    ax.set_xticklabels(ET_SHORT, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(y_labels, fontsize=8.5)

    # Separator lines between benchmark groups
    for sep in [2.5, 5.5]:
        ax.axhline(sep, color="white", lw=2.5)

    # Benchmark group labels on right
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([1, 4, 7])
    ax2.set_yticklabels(["GARCH(1,1)", "EGARCH(1,1)", "GJR-GARCH"],
                         fontsize=9, rotation=270, va="center")
    ax2.tick_params(right=False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.12)
    cbar.set_label("FEI (higher = KDE more accurate)", fontsize=8.5)
    cbar.ax.axhline(1.0, color="red", lw=1.5, ls="--")

    ax.set_xlabel("Event Type", fontsize=10)
    ax.set_title("")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = out_dir / f"fig6_fei_heatmap.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        print(f"[saved] {p}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — CRPS Horizontal Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def fig7_crps_bar(crps_csv: Path, out_dir: Path):
    df = pd.read_csv(crps_csv)

    methods = ["KDE v3\n(proposed)", "GJR-GARCH", "GARCH(1,1)", "EGARCH(1,1)"]
    cols    = ["crps_kde",           "crps_gjr",   "crps_garch",  "crps_egarch"]
    colors  = [BLUE, GREEN, RED, ORANGE]
    means   = [df[c].dropna().mean() for c in cols]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.suptitle(
        "Figure 7: Mean CRPS by Model\n"
        "$h = +15$ days, $N = 406$ out-of-sample events  "
        "|  Lower CRPS = better distributional accuracy",
        fontsize=9.5
    )

    bars = ax.barh(methods, means, color=colors, alpha=0.85, height=0.55)

    # Value labels
    for bar, val in zip(bars, means):
        ax.text(val + 0.03, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold")

    # Improvement annotations vs best
    best = means[0]
    for i, (method, val) in enumerate(zip(methods[1:], means[1:]), 1):
        pct = (val - best) / best * 100
        ax.text(val + 0.25, bars[i].get_y() + bars[i].get_height()/2,
                f"+{pct:.0f}% vs KDE", va="center", fontsize=8,
                color="gray", style="italic")

    ax.set_xlabel("Mean CRPS (lower = better)", fontsize=10)
    ax.set_xlim(0, max(means) * 1.28)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = out_dir / f"fig7_crps_comparison.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        print(f"[saved] {p}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 — Reliability (Calibration) Diagram
# ─────────────────────────────────────────────────────────────────────────────

def fig8_reliability(kde_csv: Path, garch_csv: Path, out_dir: Path):
    kde_df   = pd.read_csv(kde_csv)
    garch_df = pd.read_csv(garch_csv)

    # Merge
    df = kde_df.merge(
        garch_df[["event_id", "egarch_p5", "egarch_p95",
                  "gjr_p5",   "gjr_p95",
                  "egarch_sigma", "gjr_sigma"]],
        on="event_id", how="left"
    )

    valid = df.dropna(subset=["realised_r", "kde_p5", "garch_p5"])
    r = valid["realised_r"].values

    # Nominal coverage levels
    nominals = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]

    def empirical_coverage_kde(nominal):
        alpha = 1 - nominal
        p_lo = valid[f"kde_p{int(alpha/2*100)}"] if f"kde_p{int(alpha/2*100)}" in valid.columns else None
        # Use stored quantiles: p5/p95 for 90%, need to interpolate others
        # Approximate from kde_p5 and kde_std using Gaussian assumption
        mu    = valid["kde_median"].values
        sigma = valid["kde_std"].values
        lo    = mu + scipy_norm.ppf(alpha/2)   * sigma
        hi    = mu + scipy_norm.ppf(1-alpha/2) * sigma
        return np.mean((r >= lo) & (r <= hi))

    def empirical_coverage_gaussian(nominal, p5_col, p95_col, sigma_col):
        alpha = 1 - nominal
        sigma = valid[sigma_col].values
        mu    = np.zeros(len(valid))   # zero mean for GARCH family
        lo    = mu + scipy_norm.ppf(alpha/2)   * sigma
        hi    = mu + scipy_norm.ppf(1-alpha/2) * sigma
        return np.mean((r >= lo) & (r <= hi))

    kde_cov   = [empirical_coverage_kde(n)   for n in nominals]
    garch_cov = [empirical_coverage_gaussian(n, "garch_p5", "garch_p95", "garch_sigma")
                 for n in nominals]
    egarch_cov= [empirical_coverage_gaussian(n, "egarch_p5", "egarch_p95", "egarch_sigma")
                 for n in nominals]
    gjr_cov   = [empirical_coverage_gaussian(n, "gjr_p5", "gjr_p95", "gjr_sigma")
                 for n in nominals]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    fig.suptitle(
        "Figure 8: Reliability (Calibration) Diagram\n"
        "$h = +15$ days, $N = 406$ out-of-sample events",
        fontsize=9.5
    )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--",
            label="Perfect calibration", zorder=5)

    # Model curves
    ax.plot(nominals, kde_cov,    color=BLUE,   lw=2.2, marker="o",
            ms=5, label="KDE v3 (proposed)", zorder=4)
    ax.plot(nominals, garch_cov,  color=RED,    lw=1.5, marker="s",
            ms=4, ls="--", label="GARCH(1,1)")
    ax.plot(nominals, egarch_cov, color=ORANGE, lw=1.5, marker="^",
            ms=4, ls="-.", alpha=0.8, label="EGARCH(1,1)")
    ax.plot(nominals, gjr_cov,    color=GREEN,  lw=1.5, marker="D",
            ms=4, ls=":",  label="GJR-GARCH")

    # Shade overconfident / underconfident regions
    ax.fill_between([0, 1], [0, 1], [1, 1],
                    alpha=0.04, color="gray",
                    label="Conservative region (over-coverage)")
    ax.fill_between([0, 1], [0, 0], [0, 1],
                    alpha=0.04, color="red",
                    label="Overconfident region (under-coverage)")

    ax.set_xlabel("Nominal coverage", fontsize=10)
    ax.set_ylabel("Empirical coverage", fontsize=10)
    ax.set_xlim(0.45, 1.01)
    ax.set_ylim(0.45, 1.01)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_aspect("equal")

    # Annotation at 90%
    ax.axvline(0.90, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax.axhline(0.90, color="gray", lw=0.8, ls=":", alpha=0.6)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = out_dir / f"fig8_reliability_diagram.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        print(f"[saved] {p}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fei_csv",   required=True)
    ap.add_argument("--crps_csv",  required=True)
    ap.add_argument("--kde_csv",   required=True)
    ap.add_argument("--garch_csv", required=True)
    ap.add_argument("--output",    default="figures")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n== Figure 6: FEI Heatmap ==")
    fig6_fei_heatmap(Path(args.fei_csv), out_dir)

    print("\n== Figure 7: CRPS Bar Chart ==")
    fig7_crps_bar(Path(args.crps_csv), out_dir)

    print("\n== Figure 8: Reliability Diagram ==")
    fig8_reliability(Path(args.kde_csv), Path(args.garch_csv), out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()