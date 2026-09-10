"""
ablation_study_combined.py
===========================
Extends ablation_study.py: adds the tail-restricted coverage metric
(|r| > 10%) that the original script never computed, and loops all
three forecast horizons (7/15/30 days) into ONE combined table --
eliminating the manual three-way merge that most likely produced the
duplicate-row issue found in the paper's current Table 6 (the --DTW
row was identical to Full, and Uniform was identical to --Quantile,
across all six reported statistics -- a pattern far more consistent
with a copy-paste error during manual table assembly than a genuine
tied result).

Usage (from your project root):
    conda activate bayesian_env_rf
    python scripts/ablation_study_combined.py \
        --analogues retrieval_output_v3/per_event \
        --prices    "data/oil_prices_full.xlsx" \
        --ovx       "crude_oil_data/oil_price_data/CBOE Crude Oil Volatility Historical Data.csv"

Output:
    ablation_combined_results.csv   -- every event x config x horizon
    ablation_combined_summary.csv   -- one row per config, all horizons
    ablation_combined_table.tex     -- ready to paste into paper1_elsarticle.tex,
                                        replacing the current Table 6 verbatim

Runtime note: this runs 5 configs x 3 horizons x 406 events = 6,090
event evaluations, roughly 3x the original single-horizon script.
"""

import argparse
import json
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kde_engine_v4 as eng

HORIZONS = [7, 15, 30]
TAIL_THRESHOLD = 10.0  # |r| > 10% -- matches eng.TAIL_THRESHOLD and the paper's definition

CONFIGS = [
    {"name": "Full model",    "label": r"\textbf{Full} ($0.50,\,0.20,\,0.30$)",
     "W_SIM": 0.50, "W_QUANTILE": 0.20, "W_DTW": 0.30, "bold": True},
    {"name": "No semantic",   "label": r"$-$Semantic ($\alpha_1 = 0$)",
     "W_SIM": 0.00, "W_QUANTILE": 0.20, "W_DTW": 0.30, "bold": False},
    {"name": "No quantile",   "label": r"$-$Quantile ($\alpha_2 = 0$)",
     "W_SIM": 0.50, "W_QUANTILE": 0.00, "W_DTW": 0.30, "bold": False},
    {"name": "No DTW",        "label": r"$-$DTW ($\alpha_3 = 0$)",
     "W_SIM": 0.50, "W_QUANTILE": 0.20, "W_DTW": 0.00, "bold": False},
    {"name": "Uniform weights", "label": r"Uniform (hist.\ simulation)",
     "W_SIM": 0.00, "W_QUANTILE": 0.00, "W_DTW": 0.00, "bold": False},
]


def run_one_config_horizon(config: Dict, json_files: List[Path],
                            prices: "eng.PriceData", horizon: int) -> pd.DataFrame:
    eng.W_SIM      = config["W_SIM"]
    eng.W_QUANTILE = config["W_QUANTILE"]
    eng.W_DTW      = config["W_DTW"]

    rows = []
    for jf in tqdm(json_files,
                   desc=f"  {config['name']:<18} h=+{horizon}d",
                   unit=" ev", leave=False):
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
            event_id=event_id, base_date=base_date, analogues=analogues,
            prices=prices, horizon=horizon, min_analogues=eng.MIN_ANALOGUES,
        )
        result = out[0] if isinstance(out, tuple) else out
        if "error" in result:
            continue

        rows.append({
            "config":     config["name"],
            "horizon":    horizon,
            "event_id":   event_id,
            "realised_r": result.get("realised_r"),
            "covers_90":  result.get("covers_90"),
            "band_width": result.get("kde_band_width"),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analogues", required=True)
    ap.add_argument("--prices",    required=True)
    ap.add_argument("--ovx",       default=None)
    ap.add_argument("--output",    default=".")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = eng.PriceData(Path(args.prices), Path(args.ovx) if args.ovx else None)

    analogues_dir = Path(args.analogues)
    json_files = sorted(analogues_dir.glob("*_analogues.json"))
    if not json_files:
        log.error(f"No *_analogues.json files found in {analogues_dir}")
        sys.exit(1)

    log.info(f"\n{'='*70}")
    log.info(f"COMBINED ABLATION STUDY -- {len(json_files)} events, "
             f"{len(CONFIGS)} configs x {len(HORIZONS)} horizons "
             f"= {len(CONFIGS)*len(HORIZONS)} runs")
    log.info(f"{'='*70}\n")

    all_dfs = []
    for cfg in CONFIGS:
        for h in HORIZONS:
            log.info(f"Running: {cfg['name']}  h=+{h}d")
            df = run_one_config_horizon(cfg, json_files, prices, h)
            all_dfs.append(df)

    eng.W_SIM, eng.W_QUANTILE, eng.W_DTW = 0.50, 0.20, 0.30  # restore defaults

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(out_dir / "ablation_combined_results.csv", index=False)
    log.info(f"[save] {out_dir / 'ablation_combined_results.csv'}")

    # ── summary: one row per config, columns per horizon ──
    summary_rows = []
    for cfg in CONFIGS:
        row = {"Configuration": cfg["label"]}
        for h in HORIZONS:
            sub = combined[(combined["config"] == cfg["name"]) & (combined["horizon"] == h)]
            sub = sub.dropna(subset=["covers_90", "realised_r"])
            n = len(sub)

            cov_all = sub["covers_90"].mean() * 100 if n else np.nan

            tail_mask = sub["realised_r"].abs() > TAIL_THRESHOLD
            n_tail = tail_mask.sum()
            cov_tail = sub.loc[tail_mask, "covers_90"].mean() * 100 if n_tail > 0 else np.nan

            row[f"h{h}_n"]        = n
            row[f"h{h}_cov"]      = round(cov_all, 1) if pd.notna(cov_all) else None
            row[f"h{h}_n_tail"]   = int(n_tail)
            row[f"h{h}_cov_tail"] = round(cov_tail, 1) if pd.notna(cov_tail) else None
        row["_bold"] = cfg["bold"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "ablation_combined_summary.csv", index=False)
    log.info(f"[save] {out_dir / 'ablation_combined_summary.csv'}")

    # ── sanity check: flag any two configs with identical values ──
    log.info("\n" + "=" * 70)
    log.info("DUPLICATE-ROW CHECK (the issue that motivated this rerun)")
    log.info("=" * 70)
    metric_cols = [c for c in summary_df.columns if c not in ("Configuration", "_bold")]
    n_dupes = 0
    for i in range(len(summary_df)):
        for j in range(i + 1, len(summary_df)):
            row_i = summary_df.iloc[i][metric_cols]
            row_j = summary_df.iloc[j][metric_cols]
            if row_i.equals(row_j):
                n_dupes += 1
                log.warning(f"  IDENTICAL across all metrics: "
                            f"'{summary_df.iloc[i]['Configuration']}' == "
                            f"'{summary_df.iloc[j]['Configuration']}'")
    if n_dupes == 0:
        log.info("  Clean -- no two configurations produced identical results "
                 "across all horizons/metrics.")
    else:
        log.warning(f"  {n_dupes} duplicate pair(s) found -- inspect "
                     f"ablation_combined_results.csv before trusting these rows.")
    log.info("=" * 70 + "\n")

    # ── LaTeX table, matching the paper's Table 6 layout exactly ──
    lines = [
        "",
        "% ── Combined ablation table -- paste in place of the current Table 6 ──",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation analysis: 90\% prediction interval coverage",
        r"for all events and for tail events ($|r| > 10\%$) by weighting",
        r"configuration, across three forecast horizons.",
        r"Uniform weights ($\alpha_1 = \alpha_2 = \alpha_3 = 0$) collapse",
        r"to historical simulation.}",
        r"\label{tab:ablation}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{$h = +7$ days}",
        r" & \multicolumn{2}{c}{$h = +15$ days}",
        r" & \multicolumn{2}{c}{$h = +30$ days} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Configuration & Cov. & Tail & Cov. & Tail & Cov. & Tail \\",
        r"\midrule",
    ]
    for row in summary_rows:
        b0 = r"\textbf{" if row["_bold"] else ""
        b1 = "}" if row["_bold"] else ""
        vals = []
        for h in HORIZONS:
            cov = row[f"h{h}_cov"]
            tail = row[f"h{h}_cov_tail"]
            cov_s = f"{cov:.1f}\\%" if cov is not None else "---"
            tail_s = f"{tail:.1f}\\%" if tail is not None else "---"
            vals += [f"{b0}{cov_s}{b1}", f"{b0}{tail_s}{b1}"]
        lines.append(f"{row['Configuration']} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    latex_str = "\n".join(lines)
    (out_dir / "ablation_combined_table.tex").write_text(latex_str)
    log.info(f"[save] {out_dir / 'ablation_combined_table.tex'}")
    print(latex_str)

    log.info("\nDone. Check the DUPLICATE-ROW CHECK output above before pasting "
             "the table into the paper.\n")


if __name__ == "__main__":
    main()