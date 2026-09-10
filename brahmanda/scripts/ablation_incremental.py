"""
ablation_incremental.py
=======================
Incremental ablation showing build-up from simple to full model.

Config 1: Historical KDE      (uniform weights — baseline)
Config 2: KDE + DTW only      (trajectory similarity only)
Config 3: KDE + Semantic only (BERT retrieval only)
Config 4: Full model          (Semantic + DTW + Quantile)

Usage:
    PYTHONPATH=scripts python scripts/ablation_incremental.py \
        --analogues retrieval_output_v3/per_event \
        --prices    data/oil_prices_full.xlsx \
        --ovx       "crude_oil_data/oil_price_data/CBOE Crude Oil Volatility Historical Data.csv" \
        --horizon   15 \
        --output    ablation_output_incremental
"""

import argparse, json, sys, warnings, logging
from pathlib import Path
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

CONFIGS = [
    {
        "name":   "Historical KDE",
        "label":  "Uniform weights (historical simulation)",
        "W_SIM":      0.00,
        "W_QUANTILE": 0.00,
        "W_DTW":      0.00,
    },
    {
        "name":   "KDE + DTW",
        "label":  "DTW trajectory only ($\\alpha_3=1$)",
        "W_SIM":      0.00,
        "W_QUANTILE": 0.00,
        "W_DTW":      1.00,
    },
    {
        "name":   "KDE + Semantic",
        "label":  "Semantic retrieval only ($\\alpha_1=1$)",
        "W_SIM":      1.00,
        "W_QUANTILE": 0.00,
        "W_DTW":      0.00,
    },
    {
        "name":   "Full model",
        "label":  "Full model ($\\alpha_1=0.50,\\alpha_2=0.20,\\alpha_3=0.30$)",
        "W_SIM":      0.50,
        "W_QUANTILE": 0.20,
        "W_DTW":      0.30,
    },
]

TAIL_THRESHOLD = 10.0


def run_one_config(config, json_files, prices, horizon):
    eng.W_SIM      = config["W_SIM"]
    eng.W_QUANTILE = config["W_QUANTILE"]
    eng.W_DTW      = config["W_DTW"]

    rows = []
    for jf in tqdm(json_files,
                   desc=f"  {config['name']:<22}",
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
            "event_id":    event_id,
            "realised_r":  realised,
            "covers_90":   result.get("covers_90"),
            "band_width":  result.get("kde_band_width"),
            "is_bimodal":  result.get("kde_is_bimodal"),
            "is_tail":     int(abs(realised) > TAIL_THRESHOLD)
                           if realised is not None else None,
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analogues", required=True)
    ap.add_argument("--prices",    required=True)
    ap.add_argument("--ovx",       default=None)
    ap.add_argument("--horizon",   type=int, default=15)
    ap.add_argument("--output",    default="ablation_output_incremental")
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
    log.info(f"INCREMENTAL ABLATION  (h=+{args.horizon}d, N={len(json_files)} events)")
    log.info(f"{'='*65}\n")

    all_dfs = []
    for cfg in CONFIGS:
        log.info(f"Running: {cfg['name']}")
        df_cfg = run_one_config(cfg, json_files, prices, args.horizon)
        all_dfs.append(df_cfg)
        n     = len(df_cfg)
        cov   = df_cfg["covers_90"].dropna().mean() * 100
        bw    = df_cfg["band_width"].dropna().mean()
        bim   = df_cfg["is_bimodal"].sum()
        tail  = df_cfg[df_cfg["is_tail"]==1]["covers_90"].dropna().mean()*100
        log.info(f"  → N={n}  coverage={cov:.1f}%  tail_cov={tail:.1f}%  "
                 f"band_width={bw:.2f}%  bimodal={bim}")

    # Restore defaults
    eng.W_SIM=0.50; eng.W_QUANTILE=0.20; eng.W_DTW=0.30

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(out_dir / "ablation_incremental.csv", index=False)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  INCREMENTAL ABLATION — h=+{args.horizon}d")
    print(f"{'='*80}")
    print(f"  {'Config':<25} {'N':>4}  {'Cov90%':>7}  {'TailCov':>8}  "
          f"{'BandW%':>7}  {'Bimodal':>7}")
    print("  " + "-"*70)

    full_cov = None
    for cfg in CONFIGS:
        sub  = combined[combined["config"] == cfg["name"]]
        n    = len(sub)
        cov  = sub["covers_90"].dropna().mean() * 100
        bw   = sub["band_width"].dropna().mean()
        bim  = sub["is_bimodal"].sum()
        tail_sub = sub[sub["is_tail"]==1]
        tail = tail_sub["covers_90"].dropna().mean() * 100 \
               if len(tail_sub) > 0 else float("nan")
        if cfg["name"] == "Full model":
            full_cov = cov
        delta = cov - full_cov if full_cov is not None else 0.0
        marker = " ◄" if cfg["name"] == "Full model" else ""
        print(f"  {cfg['name']:<25} {n:>4}  {cov:>7.1f}  "
              f"{tail:>7.1f}%  {bw:>7.2f}  {bim:>7}{marker}")

    print(f"\n{'='*80}\n")

    # ── LaTeX table ───────────────────────────────────────────────────────
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Incremental ablation: 90\% prediction interval")
    latex.append(r"coverage, tail event coverage ($|r|>10\%$), and band width")
    latex.append(r"for four weighting configurations, $h = +15$ days,")
    latex.append(r"$N = 406$ out-of-sample events. Each row adds one")
    latex.append(r"component to the baseline historical KDE.}")
    latex.append(r"\label{tab:ablation_incremental}")
    latex.append(r"\begin{tabular}{lccc}")
    latex.append(r"\toprule")
    latex.append(r"Configuration & 90\% Coverage & Tail Coverage & Band Width \\")
    latex.append(r"\midrule")

    for cfg in CONFIGS:
        sub  = combined[combined["config"] == cfg["name"]]
        cov  = sub["covers_90"].dropna().mean() * 100
        bw   = sub["band_width"].dropna().mean()
        tail_sub = sub[sub["is_tail"]==1]
        tail = tail_sub["covers_90"].dropna().mean() * 100 \
               if len(tail_sub) > 0 else float("nan")
        bold = cfg["name"] == "Full model"
        label = cfg["label"]
        tail_str = f"{tail:.1f}\\%" if not np.isnan(tail) else "---"
        if bold:
            latex.append(f"  \\textbf{{{label}}} & "
                         f"\\textbf{{{cov:.1f}\\%}} & "
                         f"\\textbf{{{tail_str}}} & "
                         f"\\textbf{{{bw:.2f}\\%}} \\\\")
        else:
            latex.append(f"  {label} & {cov:.1f}\\% & {tail_str} & {bw:.2f}\\% \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)
    print(latex_str)
    (out_dir / "ablation_incremental.tex").write_text(latex_str)
    log.info(f"[save] {out_dir}/ablation_incremental.tex")
    log.info("Done.\n")


if __name__ == "__main__":
    main()
    