"""
ablation_study.py
=================
Ablation analysis for IJF Paper 1, Section 5.3.

Runs 5 configurations of kde_engine_v3 on the 406 out-of-sample
test events and reports 90% coverage and band width for each:

  Config 0 — FULL model        (sim^0.5 × quantile^0.2 × dtw^0.3)
  Config 1 — No semantic       (sim^0.0 × quantile^0.2 × dtw^0.3)
  Config 2 — No quantile       (sim^0.5 × quantile^0.0 × dtw^0.3)
  Config 3 — No DTW            (sim^0.5 × quantile^0.2 × dtw^0.0)
  Config 4 — Uniform weights   (sim^0.0 × quantile^0.0 × dtw^0.0)
              (= historical simulation baseline)

Usage (from your project root, conda env bayesian_env_rf):
    conda activate bayesian_env_rf
    python scripts/ablation_study.py \
        --analogues retrieval_output_v3/per_event \
        --prices    "data/oil_prices_full.xlsx" \
        --ovx       "crude_oil_data/oil_price_data/CBOE Crude Oil Volatility Historical Data.csv" \
        --horizon   15

Output:
    ablation_results.csv   — full event-level results for all configs
    ablation_summary.csv   — one row per config (for LaTeX table)
    ablation_summary.txt   — formatted table ready to paste into paper
"""

import argparse
import json
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Import the engine ──────────────────────────────────────────────────────
# Add scripts/ to path so we can import kde_engine_v3
sys.path.insert(0, str(Path(__file__).resolve().parent))
import Kde_engine_v3 as eng

# ── Ablation configurations ────────────────────────────────────────────────
CONFIGS = [
    {
        "name":        "Full model",
        "label":       "Full (sim^0.5 × q^0.2 × dtw^0.3)",
        "W_SIM":       0.50,
        "W_QUANTILE":  0.20,
        "W_DTW":       0.30,
    },
    {
        "name":        "No semantic",
        "label":       "−Semantic (sim^0.0 × q^0.2 × dtw^0.3)",
        "W_SIM":       0.00,
        "W_QUANTILE":  0.20,
        "W_DTW":       0.30,
    },
    {
        "name":        "No quantile",
        "label":       "−Quantile (sim^0.5 × q^0.0 × dtw^0.3)",
        "W_SIM":       0.50,
        "W_QUANTILE":  0.00,
        "W_DTW":       0.30,
    },
    {
        "name":        "No DTW",
        "label":       "−DTW (sim^0.5 × q^0.2 × dtw^0.0)",
        "W_SIM":       0.50,
        "W_QUANTILE":  0.20,
        "W_DTW":       0.00,
    },
    {
        "name":        "Uniform weights",
        "label":       "Uniform (sim^0 × q^0 × dtw^0)",
        "W_SIM":       0.00,
        "W_QUANTILE":  0.00,
        "W_DTW":       0.00,
    },
]


def run_one_config(
    config: Dict,
    json_files: List[Path],
    prices: eng.PriceData,
    horizon: int,
) -> pd.DataFrame:
    """
    Run all events under one ablation configuration.
    Patches the engine globals for the duration of this config.
    """
    # Patch the engine's weight globals
    eng.W_SIM      = config["W_SIM"]
    eng.W_QUANTILE = config["W_QUANTILE"]
    eng.W_DTW      = config["W_DTW"]

    rows = []
    for jf in tqdm(json_files,
                   desc=f"  {config['name']:<20}",
                   unit=" ev",
                   leave=False):
        try:
            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.warning(f"SKIP {jf.name}: {e}")
            continue

        event_id  = data.get("base_event_id", jf.stem.replace("_analogues", ""))
        base_date = pd.to_datetime(data.get("base_date", ""), errors="coerce")
        analogues = data.get("analogues", [])

        if pd.isna(base_date) or not analogues:
            continue

        out = eng.run_event_v3(
            event_id      = event_id,
            base_date     = base_date,
            analogues     = analogues,
            prices        = prices,
            horizon       = horizon,
            min_analogues = eng.MIN_ANALOGUES,
        )

        if isinstance(out, tuple):
            result = out[0]
        else:
            result = out

        if "error" in result:
            continue

        rows.append({
            "config":       config["name"],
            "event_id":     event_id,
            "base_date":    result.get("base_date"),
            "realised_r":   result.get("realised_r"),
            "covers_90":    result.get("covers_90"),
            "covers_50":    result.get("covers_50"),
            "band_width":   result.get("kde_band_width"),
            "n_analogues":  result.get("n_analogues"),
            "is_bimodal":   result.get("kde_is_bimodal"),
            "tail_weight":  result.get("kde_tail_weight"),
            "p_down10":     result.get("kde_p_down10"),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analogues", required=True,
                    help="Directory containing *_analogues.json files")
    ap.add_argument("--prices",    required=True,
                    help="Path to oil_prices_full.xlsx")
    ap.add_argument("--ovx",       default=None,
                    help="Path to OVX CSV (optional)")
    ap.add_argument("--horizon",   type=int, default=15,
                    help="Forecast horizon in days (default: 15)")
    ap.add_argument("--output",    default=".",
                    help="Output directory for CSV/TXT results")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load prices once
    prices = eng.PriceData(
        Path(args.prices),
        Path(args.ovx) if args.ovx else None,
    )

    analogues_dir = Path(args.analogues)
    json_files = sorted(analogues_dir.glob("*_analogues.json"))
    if not json_files:
        log.error(f"No *_analogues.json files found in {analogues_dir}")
        sys.exit(1)

    log.info(f"\n{'='*65}")
    log.info(f"ABLATION STUDY — KDE v3")
    log.info(f"  {len(json_files)} events  horizon=+{args.horizon}d")
    log.info(f"  5 configurations")
    log.info(f"{'='*65}\n")

    # ── Run all configs ───────────────────────────────────────────────────
    all_dfs = []
    for cfg in CONFIGS:
        log.info(f"Running: {cfg['name']}")
        df_cfg = run_one_config(cfg, json_files, prices, args.horizon)
        all_dfs.append(df_cfg)
        n = len(df_cfg)
        cov = df_cfg["covers_90"].dropna().mean() * 100
        bw  = df_cfg["band_width"].dropna().mean()
        log.info(f"  → {n} events  coverage={cov:.1f}%  band_width={bw:.2f}%")

    # Restore engine defaults
    eng.W_SIM      = 0.50
    eng.W_QUANTILE = 0.20
    eng.W_DTW      = 0.30

    # ── Combine and save event-level CSV ─────────────────────────────────
    combined = pd.concat(all_dfs, ignore_index=True)
    event_csv = out_dir / "ablation_results.csv"
    combined.to_csv(event_csv, index=False)
    log.info(f"\n[save] {event_csv}")

    # ── Build summary table ───────────────────────────────────────────────
    summary_rows = []
    full_cov = None

    for cfg in CONFIGS:
        sub = combined[combined["config"] == cfg["name"]]
        n   = len(sub)

        cov_90 = sub["covers_90"].dropna().mean() * 100
        bw     = sub["band_width"].dropna().mean()
        n_bi   = sub["is_bimodal"].sum() if "is_bimodal" in sub.columns else 0

        if cfg["name"] == "Full model":
            full_cov = cov_90

        delta_cov = cov_90 - full_cov if full_cov is not None else 0.0

        summary_rows.append({
            "Configuration":     cfg["label"],
            "N":                 n,
            "90% Coverage (%)":  round(cov_90, 1),
            "Δ Coverage (pp)":   round(delta_cov, 1),
            "Band Width (%)":    round(bw, 2),
            "Bimodal (N)":       int(n_bi),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    log.info(f"[save] {summary_csv}")

    # ── Print and save formatted table ───────────────────────────────────
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  ABLATION ANALYSIS — KDE v3  "
                 f"(horizon=+{args.horizon}d, N={len(json_files)} events)")
    lines.append("=" * 80)
    lines.append(
        f"  {'Configuration':<45} {'N':>4}  "
        f"{'Cov90%':>7}  {'Δpp':>6}  {'BandW%':>7}  {'Bimodal':>7}"
    )
    lines.append("  " + "-" * 76)

    for row in summary_rows:
        marker = " ◄" if row["Configuration"].startswith("Full") else ""
        lines.append(
            f"  {row['Configuration']:<45} "
            f"{row['N']:>4}  "
            f"{row['90% Coverage (%)']:>7.1f}  "
            f"{row['Δ Coverage (pp)']:>+6.1f}  "
            f"{row['Band Width (%)']:>7.2f}  "
            f"{row['Bimodal (N)']:>7}"
            f"{marker}"
        )

    lines.append("  " + "-" * 76)
    lines.append("  Δpp = percentage-point difference vs Full model.")
    lines.append("  Uniform weights = historical simulation baseline.")
    lines.append("=" * 80)
    lines.append("")

    table_str = "\n".join(lines)
    print(table_str)

    txt_path = out_dir / "ablation_summary.txt"
    txt_path.write_text(table_str)
    log.info(f"[save] {txt_path}")

    # ── LaTeX snippet ─────────────────────────────────────────────────────
    latex_lines = [
        "",
        "% ── LaTeX table snippet (paste into paper1_methodology.tex) ──",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Ablation analysis: 90\% prediction interval coverage "
        r"and band width by weighting configuration, "
        f"$h=+{args.horizon}$ days, {len(json_files)} out-of-sample events.}}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Configuration & 90\% Coverage & $\Delta$ (pp) & Band Width \\",
        r"\midrule",
    ]
    for row in summary_rows:
        name = row["Configuration"].split("(")[0].strip()
        bold_start = r"\textbf{" if row["Configuration"].startswith("Full") else ""
        bold_end   = "}"         if row["Configuration"].startswith("Full") else ""
        sign = "+" if row["Δ Coverage (pp)"] >= 0 else ""
        latex_lines.append(
            f"  {bold_start}{name}{bold_end} & "
            f"{bold_start}{row['90% Coverage (%)']:.1f}\\%{bold_end} & "
            f"{bold_start}{sign}{row['Δ Coverage (pp)']:.1f}{bold_end} & "
            f"{bold_start}{row['Band Width (%)']:.2f}\\%{bold_end} \\\\"
        )
    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    latex_str = "\n".join(latex_lines)
    print(latex_str)

    latex_path = out_dir / "ablation_table.tex"
    latex_path.write_text(latex_str)
    log.info(f"[save] {latex_path}")

    log.info("\nDone. Copy ablation_table.tex content into your paper.\n")


if __name__ == "__main__":
    main()