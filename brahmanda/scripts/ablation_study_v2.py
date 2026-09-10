"""
ablation_study_v2.py
====================
Extended ablation analysis for IJF Paper 1.

Runs 5 configurations × 3 horizons (h=7, 15, 30) on the 406
out-of-sample test events, reporting:
  - 90% coverage (all events)
  - 90% coverage for TAIL events only (|realised_r| > 10%, N~66)
  - Band width (P95 - P5)

Configurations:
  Full     — sim^0.5 × quantile^0.2 × dtw^0.3
  -Semantic — sim^0.0 × quantile^0.2 × dtw^0.3
  -Quantile — sim^0.5 × quantile^0.0 × dtw^0.3
  -DTW      — sim^0.5 × quantile^0.2 × dtw^0.0
  Uniform  — sim^0.0 × quantile^0.0 × dtw^0.0 (historical simulation)

Usage (from project root):
    PYTHONPATH=scripts python scripts/ablation_study_v2.py \
        --analogues retrieval_output_v3/per_event \
        --prices    data/oil_prices_full.xlsx \
        --ovx       "crude_oil_data/oil_price_data/CBOE Crude Oil Volatility Historical Data.csv" \
        --output    ablation_output_v2
"""

import argparse, json, sys, logging, warnings
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import Kde_engine_v3 as eng

TAIL_THRESHOLD = 10.0  # |r| > 10% = tail event

CONFIGS = [
    {"name": "Full model",      "W_SIM": 0.50, "W_QUANTILE": 0.20, "W_DTW": 0.30},
    {"name": "No semantic",     "W_SIM": 0.00, "W_QUANTILE": 0.20, "W_DTW": 0.30},
    {"name": "No quantile",     "W_SIM": 0.50, "W_QUANTILE": 0.00, "W_DTW": 0.30},
    {"name": "No DTW",          "W_SIM": 0.50, "W_QUANTILE": 0.20, "W_DTW": 0.00},
    {"name": "Uniform",         "W_SIM": 0.00, "W_QUANTILE": 0.00, "W_DTW": 0.00},
]

HORIZONS = [7, 15, 30]


def run_one_config(config, json_files, prices, horizon):
    eng.W_SIM      = config["W_SIM"]
    eng.W_QUANTILE = config["W_QUANTILE"]
    eng.W_DTW      = config["W_DTW"]

    rows = []
    for jf in tqdm(json_files,
                   desc=f"  h={horizon:2d}d {config['name']:<14}",
                   unit=" ev", leave=False):
        try:
            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        event_id  = data.get("base_event_id", jf.stem.replace("_analogues",""))
        base_date = pd.to_datetime(data.get("base_date",""), errors="coerce")
        analogues = data.get("analogues", [])
        if pd.isna(base_date) or not analogues:
            continue

        out = eng.run_event_v3(event_id, base_date, analogues,
                               prices, horizon=horizon,
                               min_analogues=eng.MIN_ANALOGUES)
        result = out[0] if isinstance(out, tuple) else out
        if "error" in result:
            continue

        realised = result.get("realised_r")
        rows.append({
            "config":      config["name"],
            "horizon":     horizon,
            "event_id":    event_id,
            "realised_r":  realised,
            "covers_90":   result.get("covers_90"),
            "band_width":  result.get("kde_band_width"),
            "is_tail":     int(abs(realised) > TAIL_THRESHOLD)
                           if realised is not None else None,
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analogues", required=True)
    ap.add_argument("--prices",    required=True)
    ap.add_argument("--ovx",       default=None)
    ap.add_argument("--output",    default="ablation_output_v2")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = eng.PriceData(Path(args.prices),
                           Path(args.ovx) if args.ovx else None)

    json_files = sorted(Path(args.analogues).glob("*_analogues.json"))
    if not json_files:
        log.error(f"No JSON files in {args.analogues}")
        sys.exit(1)

    log.info(f"\n{'='*65}")
    log.info(f"ABLATION v2 — {len(json_files)} events × "
             f"{len(CONFIGS)} configs × {len(HORIZONS)} horizons")
    log.info(f"{'='*65}\n")

    # ── Run all configs × horizons ────────────────────────────────────────
    all_dfs = []
    for horizon in HORIZONS:
        for cfg in CONFIGS:
            df = run_one_config(cfg, json_files, prices, horizon)
            all_dfs.append(df)

    # Restore defaults
    eng.W_SIM = 0.50; eng.W_QUANTILE = 0.20; eng.W_DTW = 0.30

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(out_dir / "ablation_v2_results.csv", index=False)

    # ── Build summary ─────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  ABLATION ANALYSIS v2 — All horizons + tail coverage")
    print(f"{'='*90}")

    latex_rows = []

    for horizon in HORIZONS:
        print(f"\n  h = +{horizon} days")
        print(f"  {'Configuration':<22} {'N':>4}  {'Cov90%':>7}  "
              f"{'Δpp':>5}  {'TailCov':>8}  {'BandW%':>7}")
        print("  " + "-"*65)

        full_cov = None
        for cfg in CONFIGS:
            sub = combined[(combined["config"] == cfg["name"]) &
                           (combined["horizon"] == horizon)]
            n        = len(sub)
            cov      = sub["covers_90"].dropna().mean() * 100
            bw       = sub["band_width"].dropna().mean()
            tail_sub = sub[sub["is_tail"] == 1]
            tail_cov = tail_sub["covers_90"].dropna().mean() * 100 \
                       if len(tail_sub) > 0 else float("nan")
            n_tail   = len(tail_sub)

            if cfg["name"] == "Full model":
                full_cov = cov

            delta = cov - full_cov if full_cov is not None else 0.0
            sign  = "+" if delta >= 0 else ""

            marker = " ◄" if cfg["name"] == "Full model" else ""
            print(f"  {cfg['name']:<22} {n:>4}  "
                  f"{cov:>7.1f}  {sign}{delta:>4.1f}  "
                  f"{tail_cov:>7.1f}% (N={n_tail})  "
                  f"{bw:>7.2f}{marker}")

            latex_rows.append({
                "horizon":  horizon,
                "config":   cfg["name"],
                "n":        n,
                "cov":      cov,
                "delta":    delta,
                "tail_cov": tail_cov,
                "n_tail":   n_tail,
                "bw":       bw,
            })

    print(f"\n{'='*90}\n")

    # ── LaTeX table ───────────────────────────────────────────────────────
    latex = []
    latex.append("% ── Ablation Table v2 (paste into paper) ──────────────────────────────")
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Ablation analysis: 90\% coverage (all events and tail events")
    latex.append(r"where $|r| > 10\%$) and band width by weighting configuration,")
    latex.append(r"across three forecast horizons. $\Delta$ is the percentage-point")
    latex.append(r"difference relative to the full model.}")
    latex.append(r"\label{tab:ablation}")
    latex.append(r"\begin{tabular}{lcccccc}")
    latex.append(r"\toprule")
    latex.append(r" & \multicolumn{2}{c}{$h = +7$ days} & "
                 r"\multicolumn{2}{c}{$h = +15$ days} & "
                 r"\multicolumn{2}{c}{$h = +30$ days} \\")
    latex.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    latex.append(r"Configuration & Cov & TailCov & Cov & TailCov & Cov & TailCov \\")
    latex.append(r"\midrule")

    config_names = [c["name"] for c in CONFIGS]
    ldf = pd.DataFrame(latex_rows)

    for cfg_name in config_names:
        bold = cfg_name == "Full model"
        row_parts = []
        for h in HORIZONS:
            r = ldf[(ldf["config"] == cfg_name) & (ldf["horizon"] == h)]
            if len(r) == 0:
                row_parts += ["---", "---"]
                continue
            r = r.iloc[0]
            cov_str  = f"{r['cov']:.1f}\\%"
            tail_str = f"{r['tail_cov']:.1f}\\%" \
                       if not np.isnan(r["tail_cov"]) else "---"
            if bold:
                row_parts += [f"\\textbf{{{cov_str}}}",
                               f"\\textbf{{{tail_str}}}"]
            else:
                row_parts += [cov_str, tail_str]

        label = cfg_name
        if bold:
            label = f"\\textbf{{{label}}}"
        latex.append(f"  {label} & " + " & ".join(row_parts) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    latex_str = "\n".join(latex)
    print(latex_str)

    (out_dir / "ablation_table_v2.tex").write_text(latex_str)
    log.info(f"\n[save] {out_dir / 'ablation_table_v2.tex'}")
    log.info("Done.\n")


if __name__ == "__main__":
    main()