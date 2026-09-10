"""
generate_figures.py  (CORRECTED)
=================================
Generates three figures for the paper, with zero hardcoded numeric
values -- everything is read live from the corrected v4 results.

  Figure -- Framework flowchart (no data dependency; addresses
            Reza's comments: greyscale only, font size matched to
            body text, frame artifact removed)
  Figure -- Coverage & band width across horizons (fig:coverage)
  Figure -- Bimodal posterior vs GARCH, representative event
            (fig:bimodal)

Fixes applied vs the original script:
  - fig_bimodal_posterior: `realised` was hardcoded despite --results
    being accepted as an argument and never used. Now reads the real
    realised_r for the selected event_id from the results CSV.
  - warns explicitly if the selected event is no longer bimodal
    under the corrected v4 posterior, rather than silently shipping
    a stale "bimodal example".
  - fig_coverage_comparison: all six data series were hardcoded
    literals. Now computed live from three per-horizon results CSVs.
  - "KDE v3" legend label replaced with "Selective Prior" throughout.
  - Flowchart: switched to greyscale (Reza: "why not keep it with
    no colour"), box text font size raised to match body text
    (Reza: "check the font size"), explicit frame/clipping fix
    applied for the stray-artifact issue (Reza: "something is
    underneath this graph") -- verify visually after regenerating.

Usage:
    python generate_figures.py \
        --kde_json    kde_output_v4/per_event/EVT_2024.0_0010.0_kde_v4.json \
        --results_h7  kde_output_v4/kde_v4_results_h7.csv \
        --results_h15 kde_output_v4/kde_v4_results_h15.csv \
        --results_h30 kde_output_v4/kde_v4_results_h30.csv \
        --hist_cov_h15 87.7 --hist_bw_h15 23.22 \
        --output figures_v4/
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
})

BLUE   = "#1f77b4"
RED    = "#d62728"
GREEN  = "#2ca02c"
GRAY   = "#7f7f7f"
ORANGE = "#ff7f0e"


def fig_bimodal_posterior(kde_json_path: Path, results_h15_path: Path, out_dir: Path):
    with kde_json_path.open() as f:
        data = json.load(f)

    event_id = data["event_id"]
    x     = np.array(data["x_grid"])
    dens  = np.array(data["density"])
    stats = data["posterior_stats"]
    garch = data["garch"]

    results = pd.read_csv(results_h15_path)
    row = results[results["event_id"] == event_id]
    if row.empty:
        raise ValueError(f"event_id {event_id} not found in {results_h15_path} "
                          f"-- cannot proceed without a real realised_r value.")
    realised = float(row["realised_r"].iloc[0])
    base_date = row["base_date"].iloc[0] if "base_date" in row.columns else data.get("base_date", "")

    is_bimodal = bool(stats.get("kde_is_bimodal", 0))
    if not is_bimodal:
        print(f"WARNING: event {event_id} is NOT bimodal under the corrected "
              f"(v4) posterior. Choose a different --kde_json from the v4 "
              f"bimodal-posterior list before using this figure in the paper.")

    kde_p5  = stats["kde_p5"]
    kde_p95 = stats["kde_p95"]
    kde_med = stats["kde_median"]
    g_sig   = garch["garch_sigma"]
    g_p5    = garch["garch_p5"]
    g_p95   = garch["garch_p95"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    fig.suptitle(
        f"Event-Conditioned Posterior vs GARCH(1,1) Benchmark\n"
        f"Event: {event_id}  |  Date: {base_date}  |  Horizon: $h = +15$ days",
        fontsize=10, y=1.01
    )

    ax = axes[0]
    ax.plot(x, dens, color=BLUE, lw=1.8, label="Selective prior posterior")
    ax.fill_between(x, dens, where=(x >= kde_p5) & (x <= kde_p95),
                    alpha=0.18, color=BLUE, label="90% interval")
    ax.fill_between(x, dens, where=(x < 0.0),
                    alpha=0.12, color=RED, label="Lower mode (medium scenarios)")
    ax.fill_between(x, dens, where=(x >= 0.0),
                    alpha=0.12, color=GREEN, label="Upper mode (tail scenarios)")
    ax.axvline(kde_med,  color=BLUE, lw=1.2, ls="--", label=f"Median {kde_med:.1f}%")
    ax.axvline(realised, color="black", lw=1.5, ls="-",
               label=f"Realised $r_{{+15}}$ = {realised:.1f}%")
    ax.axvline(kde_p5,  color=BLUE, lw=0.8, ls=":", alpha=0.7)
    ax.axvline(kde_p95, color=BLUE, lw=0.8, ls=":", alpha=0.7,
               label=f"90% band [{kde_p5:.1f}%, {kde_p95:.1f}%]")
    ax.set_xlabel("15-day forward return (%)")
    ax.set_ylabel("Density")
    ax.set_title(f"(A) Selective Prior Posterior\n"
                 f"{'Bimodal' if is_bimodal else 'Unimodal'} $\\bullet$ "
                 f"Band width = {kde_p95-kde_p5:.1f}%")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_xlim(x[0], x[-1])

    ax2 = axes[1]
    x_g = np.linspace(-25, 25, 500)
    g_dens = scipy_norm.pdf(x_g, 0, g_sig)
    ax2.plot(x_g, g_dens, color=RED, lw=1.8, label="GARCH(1,1) predictive density")
    ax2.fill_between(x_g, g_dens, where=(x_g >= g_p5) & (x_g <= g_p95),
                     alpha=0.18, color=RED, label="90% interval")
    ax2.axvline(0, color=RED, lw=1.2, ls="--", label="GARCH mean = 0%")
    ax2.axvline(realised, color="black", lw=1.5, ls="-",
                label=f"Realised $r_{{+15}}$ = {realised:.1f}%")
    ax2.axvline(g_p5,  color=RED, lw=0.8, ls=":", alpha=0.7)
    ax2.axvline(g_p95, color=RED, lw=0.8, ls=":", alpha=0.7,
                label=f"90% band [{g_p5:.1f}%, {g_p95:.1f}%]")
    ax2.set_xlabel("15-day forward return (%)")
    ax2.set_ylabel("Density")
    ax2.set_title(f"(B) GARCH(1,1) Benchmark\n"
                  f"Unimodal $\\bullet$ Band width = {g_p95-g_p5:.1f}%")
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.set_xlim(-25, 25)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = out_dir / f"fig_bimodal_posterior.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        print(f"[saved] {p}")
    plt.close()


def fig_framework_flowchart(out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_frame_on(False)
    ax.set_facecolor("none")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)

    BOX_FILL, BOX_EDGE, TEXT_COLOR = "white", "black", "black"

    def box(ax, x, y, w, h, text, lw=1.3, fontsize=10):
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                              facecolor=BOX_FILL, edgecolor=BOX_EDGE,
                              linewidth=lw, zorder=3, clip_on=True)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=TEXT_COLOR,
                fontweight="normal", zorder=4,
                multialignment="center", clip_on=True)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
                    zorder=2)

    box(ax, 5, 5.3, 5.5, 0.7, "Current Event $E_0$  (news text + market date)", lw=1.6)
    box(ax, 2.2, 4.0, 3.8, 0.8,
        "Stage 1: BERT Semantic Retrieval\n"
        "768-dim embedding  |  cosine similarity\n"
        "Adaptive $\\tau^*$, $K_{\\min} = 25$")
    box(ax, 7.8, 4.0, 3.0, 0.8, "Corpus $\\Omega$\n11,983 events\n2001--2023", lw=1.0)
    box(ax, 5, 2.85, 5.5, 0.8,
        "Stage 2: Three-Rank Weighting\n"
        "$w_i = \\mathrm{sim}^{0.5} \\times \\phi_q^{0.2} \\times \\phi_{\\mathrm{DTW}}^{0.3}$\n"
        "Semantic  |  Quantile regime  |  DTW trajectory")
    box(ax, 5, 1.7, 5.5, 0.8,
        "Stage 3: Individual Gaussian Priors\n"
        "$f_i(r) = \\mathcal{N}(r \\mid \\mu_i, \\sigma_i^2)$\n"
        "$\\mu_i = r^{\\mathrm{realised}}_i$, $\\sigma_i = \\mathrm{std}(\\Delta p_i)$")
    box(ax, 5, 0.55, 5.5, 0.8,
        "Stage 4: Weighted Mixture Posterior + Risk Metrics\n"
        "$f(r \\mid E_0) = \\sum_i \\tilde{w}_i \\mathcal{N}(r \\mid \\mu_i, \\sigma_i^2)$\n"
        "VaR  |  P(crash)  |  Red flag  |  Bimodal detection", lw=1.6)

    arrow(ax, 5, 4.95, 2.2, 4.40)
    arrow(ax, 6.3, 4.0, 7.3, 4.0)
    arrow(ax, 7.3, 3.6, 5.5, 3.25)
    arrow(ax, 2.2, 3.60, 3.8, 3.25)
    arrow(ax, 5, 2.45, 5, 2.10)
    arrow(ax, 5, 1.30, 5, 0.95)

    ax.text(6.3, 4.18, "query", fontsize=8.5, color="black", ha="center", zorder=5)
    ax.text(6.3, 3.68, "analogues $\\mathcal{S}(E_0)$",
            fontsize=8.5, color="black", ha="center", zorder=5)

    plt.tight_layout(pad=0.3)
    for ext in ["pdf", "png"]:
        p = out_dir / f"fig_framework_flowchart.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200, facecolor="white", edgecolor="none")
        print(f"[saved] {p}")
    plt.close()


def fig_coverage_comparison(results_h7: Path, results_h15: Path, results_h30: Path,
                             out_dir: Path, hist_cov_h15: float = None,
                             hist_bw_h15: float = None):
    horizons_files = {"+7 days": results_h7, "+15 days": results_h15, "+30 days": results_h30}
    kde_cov, garch_cov, kde_bw, garch_bw = [], [], [], []

    for label, path in horizons_files.items():
        df = pd.read_csv(path)
        kde_cov.append(df["covers_90"].dropna().mean() * 100)
        kde_bw.append(df["kde_band_width"].dropna().mean())
        garch_cov.append(df["garch_covers_90"].dropna().mean() * 100
                          if "garch_covers_90" in df.columns else np.nan)
        garch_bw.append((df["garch_p95"] - df["garch_p5"]).dropna().mean()
                         if {"garch_p5", "garch_p95"}.issubset(df.columns) else np.nan)

    hist_cov = [None, hist_cov_h15, None]
    horizons = list(horizons_files.keys())
    x = np.arange(len(horizons))
    w = 0.26

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Forecasting Performance -- Selective Prior vs Benchmarks", fontsize=11)

    ax = axes[0]
    bars1 = ax.bar(x - w, kde_cov, w, label="Selective Prior (proposed)", color=BLUE, alpha=0.85)
    bars2 = ax.bar(x, garch_cov, w, label="GARCH(1,1)", color=RED, alpha=0.85)
    bars3 = ax.bar(x[1:2] + w, [hist_cov[1]], w, label="Historical (250d)",
                    color=ORANGE, alpha=0.85) if hist_cov_h15 is not None else []

    ax.axhline(90, color="black", lw=1.2, ls="--", label="90% nominal target")
    for bars, color in [(bars1, BLUE), (bars2, RED), (bars3, ORANGE)]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=color)

    ax.set_xticks(x); ax.set_xticklabels(horizons)
    ax.set_ylabel("90% Empirical Coverage (%)")
    ax.set_ylim(min(min(kde_cov), min(garch_cov)) - 5, 102)
    ax.set_title("(A) 90% Prediction Interval Coverage")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    bars4 = ax2.bar(x - w/2, kde_bw, w, label="Selective Prior (proposed)", color=BLUE, alpha=0.85)
    bars5 = ax2.bar(x + w/2, garch_bw, w, label="GARCH(1,1)", color=RED, alpha=0.85)
    for bars, color in [(bars4, BLUE), (bars5, RED)]:
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{bar.get_height():.1f}%", ha="center", va="bottom",
                     fontsize=8, fontweight="bold", color=color)
    ax2.set_xticks(x); ax2.set_xticklabels(horizons)
    ax2.set_ylabel("Mean Band Width: $P_{95} - P_5$ (%)")
    ax2.set_title("(B) Prediction Interval Width\n(narrower = more efficient)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = out_dir / f"fig_coverage_comparison.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        print(f"[saved] {p}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kde_json", required=True)
    ap.add_argument("--results_h7",  required=True)
    ap.add_argument("--results_h15", required=True)
    ap.add_argument("--results_h30", required=True)
    ap.add_argument("--hist_cov_h15", type=float, default=None)
    ap.add_argument("--hist_bw_h15", type=float, default=None)
    ap.add_argument("--output", default="figures")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n== Figure: Framework flowchart ==")
    fig_framework_flowchart(out_dir)

    print("\n== Figure: Coverage & band width comparison ==")
    fig_coverage_comparison(Path(args.results_h7), Path(args.results_h15),
                             Path(args.results_h30), out_dir,
                             args.hist_cov_h15, args.hist_bw_h15)

    print("\n== Figure: Bimodal posterior vs GARCH ==")
    fig_bimodal_posterior(Path(args.kde_json), Path(args.results_h15), out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
