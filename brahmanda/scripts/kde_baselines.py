"""
kde_baselines.py
================
Computes three density-based baseline benchmarks for comparison
against the selective prior KDE v3 framework.

Baseline 1: Plain KDE
    Uses ALL historical returns (unconditional KDE) with
    Silverman bandwidth. No retrieval, no conditioning.

Baseline 2: Rolling KDE
    Uses the most recent 250 trading day returns.
    Time-windowed analogue of historical simulation.

Baseline 3: Filtered Historical Simulation (FHS)
    GARCH(1,1) standardised residuals + KDE on residuals.
    Industry standard for VaR estimation.

Usage:
    python scripts/kde_baselines.py \
        --results  kde_output_v3/kde_v3_results_h15.csv \
        --prices   data/oil_prices_full.xlsx \
        --output   figures
"""

import argparse, warnings, sys, logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm, gaussian_kde
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def load_prices(prices_path):
    df = pd.read_excel(prices_path)
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    col_map = {"Crude Oil-WTI Spot Cushing U$/BBL": "WTI",
               "Crude Oil WTI NYMEX Close M U$/BBL": "WTI"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    col = "WTI" if "WTI" in df.columns else df.columns[0]
    return df[[col]].ffill(limit=5), col


def get_returns(prices_df, col, date, horizon,
                lookback_days=None, max_offset=5):
    """Get forward h-day returns up to 'date' for baseline estimation."""
    idx = prices_df.index
    cands = idx[abs(idx - date) <= pd.Timedelta(days=max_offset)]
    if len(cands) == 0:
        return None, None
    t0 = cands[abs(cands - date).argmin()]
    p0 = prices_df.loc[t0, col]
    if isinstance(p0, pd.Series):
        p0 = p0.iloc[0]
    p0 = float(p0)
    if not np.isfinite(p0) or p0 == 0:
        return None, None

    # Forward return at horizon
    t1_cands = idx[abs(idx - (date + pd.Timedelta(days=horizon)))
                   <= pd.Timedelta(days=10)]
    if len(t1_cands) == 0:
        return None, None
    t1 = t1_cands[abs(t1_cands - (date + pd.Timedelta(days=horizon))).argmin()]
    p1 = prices_df.loc[t1, col]
    if isinstance(p1, pd.Series):
        p1 = p1.iloc[0]
    p1 = float(p1)
    realised = float((p1 - p0) / abs(p0) * 100) if np.isfinite(p1) else None

    # Historical returns for KDE
    if lookback_days:
        start = t0 - pd.Timedelta(days=lookback_days + horizon + 30)
    else:
        start = prices_df.index[0]

    hist = prices_df.loc[start:t0, col].dropna()
    if len(hist) < 30:
        return None, realised

    # Compute h-day forward returns from history
    hday_rets = []
    prices_arr = hist.values.flatten()
    for i in range(0, len(prices_arr) - horizon, 1):
        p_start = float(prices_arr[i])
        p_end   = float(prices_arr[min(i + horizon, len(prices_arr) - 1)])
        if p_start > 0 and np.isfinite(p_end) and np.isfinite(p_start):
            hday_rets.append((p_end - p_start) / abs(p_start) * 100)

    return np.array(hday_rets) if hday_rets else None, realised


def coverage_from_kde(kde_obj, realised, alpha=0.10):
    """Compute 90% interval coverage from a scipy KDE object."""
    x = np.linspace(kde_obj.dataset.min() - 5,
                    kde_obj.dataset.max() + 5, 1000)
    density = kde_obj(x)
    dx = x[1] - x[0]
    cdf = np.cumsum(density) * dx
    cdf = cdf / max(cdf[-1], 1e-10)
    p5  = float(x[np.searchsorted(cdf, alpha / 2)])
    p95 = float(x[np.searchsorted(cdf, 1 - alpha / 2)])
    bw  = p95 - p5
    cov = int(p5 <= realised <= p95) if np.isfinite(realised) else None
    return cov, bw, p5, p95


def fhs_benchmark(prices_df, col, date, horizon, max_offset=5):
    """Filtered Historical Simulation: GARCH residuals + KDE."""
    try:
        from arch import arch_model
    except ImportError:
        return None

    idx = prices_df.index
    cands = idx[abs(idx - date) <= pd.Timedelta(days=max_offset)]
    if len(cands) == 0:
        return None
    t0 = cands[abs(cands - date).argmin()]

    start = t0 - pd.Timedelta(days=500)
    series = prices_df.loc[start:t0, col].dropna()
    if len(series) < 100:
        return None

    rets = series.pct_change().dropna() * 100
    try:
        res = arch_model(rets, vol="Garch", p=1, q=1).fit(
            disp="off", show_warning=False)
        std_resid = res.resid / res.conditional_volatility
        std_resid = std_resid.dropna()
        fc_var = res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
        fc_sigma = float(np.sqrt(fc_var * horizon))
        # Scale standardised residuals by forecast sigma
        sim_rets = std_resid.values * fc_sigma
        if len(sim_rets) < 20:
            return None
        kde = gaussian_kde(sim_rets, bw_method="silverman")
        return kde
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",  required=True)
    ap.add_argument("--prices",   required=True)
    ap.add_argument("--horizon",  type=int, default=15)
    ap.add_argument("--output",   default="figures")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(args.results)
    results_df["base_date"] = pd.to_datetime(results_df["base_date"])
    log.info(f"Loaded {len(results_df)} events")

    prices_df, col = load_prices(Path(args.prices))
    log.info(f"Prices: {len(prices_df)} rows, col={col}")

    rows = []

    for _, row in tqdm(results_df.iterrows(), total=len(results_df),
                       desc="Baselines", unit=" ev"):
        date     = row["base_date"]
        realised = row.get("realised_r")
        event_id = row["event_id"]

        if not pd.notna(realised):
            continue

        # ── Baseline 1: Plain KDE (all history) ──────────────────────────
        hist_all, _ = get_returns(prices_df, col, date, args.horizon,
                                  lookback_days=None)
        plain_cov = plain_bw = np.nan
        if hist_all is not None and len(hist_all) >= 30:
            try:
                kde_plain = gaussian_kde(hist_all, bw_method="silverman")
                c, b, _, _ = coverage_from_kde(kde_plain, realised)
                plain_cov, plain_bw = c, b
            except Exception:
                pass

        # ── Baseline 2: Rolling KDE (250 days) ───────────────────────────
        hist_roll, _ = get_returns(prices_df, col, date, args.horizon,
                                   lookback_days=250 + args.horizon)
        roll_cov = roll_bw = np.nan
        if hist_roll is not None and len(hist_roll) >= 20:
            try:
                kde_roll = gaussian_kde(hist_roll, bw_method="silverman")
                c, b, _, _ = coverage_from_kde(kde_roll, realised)
                roll_cov, roll_bw = c, b
            except Exception:
                pass

        # ── Baseline 3: FHS ──────────────────────────────────────────────
        fhs_cov = fhs_bw = np.nan
        try:
            kde_fhs = fhs_benchmark(prices_df, col, date, args.horizon)
            if kde_fhs is not None:
                c, b, _, _ = coverage_from_kde(kde_fhs, realised)
                fhs_cov, fhs_bw = c, b
        except Exception:
            pass

        rows.append({
            "event_id":    event_id,
            "base_date":   row["base_date"],
            "realised_r":  realised,
            "kde_covers":  row.get("covers_90"),
            "kde_bw":      row.get("kde_band_width"),
            "plain_cov":   plain_cov,
            "plain_bw":    plain_bw,
            "roll_cov":    roll_cov,
            "roll_bw":     roll_bw,
            "fhs_cov":     fhs_cov,
            "fhs_bw":      fhs_bw,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "kde_baselines.csv", index=False)
    log.info(f"[save] {out_dir}/kde_baselines.csv  ({len(df)} events)")

    # ── Summary ───────────────────────────────────────────────────────────
    valid = df.dropna(subset=["kde_covers", "plain_cov", "roll_cov", "fhs_cov"])
    n = len(valid)

    print(f"\n{'='*70}")
    print(f"  DENSITY-BASED BASELINE COMPARISON  (h=+{args.horizon}d, N={n})")
    print(f"{'='*70}")
    print(f"  {'Method':<35} {'Coverage':>9}  {'Band Width':>10}")
    print(f"  {'-'*60}")

    methods = [
        ("KDE v3 selective prior (proposed)", "kde_covers",  "kde_bw"),
        ("Plain KDE (all history)",           "plain_cov",   "plain_bw"),
        ("Rolling KDE (250 days)",            "roll_cov",    "roll_bw"),
        ("Filtered Historical Simulation",    "fhs_cov",     "fhs_bw"),
    ]

    for name, cov_col, bw_col in methods:
        cov = valid[cov_col].dropna().mean() * 100
        bw  = valid[bw_col].dropna().mean()
        print(f"  {name:<35} {cov:>8.1f}%  {bw:>9.2f}%")

    print(f"{'='*70}\n")

    # ── LaTeX table ───────────────────────────────────────────────────────
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Comparison of density-based forecasting methods,")
    latex.append(r"$h = +15$ days, $N = " + str(n) + r"$ out-of-sample events.")
    latex.append(r"All methods produce full predictive densities evaluated")
    latex.append(r"using the same proper scoring rules. Plain KDE uses the")
    latex.append(r"full unconditional history; Rolling KDE uses the most")
    latex.append(r"recent 250 trading days; FHS applies KDE to")
    latex.append(r"GARCH(1,1)-standardised residuals.}")
    latex.append(r"\label{tab:kde_baselines}")
    latex.append(r"\begin{tabular}{lcc}")
    latex.append(r"\toprule")
    latex.append(r"Method & 90\% Coverage & Band Width \\")
    latex.append(r"\midrule")

    for name, cov_col, bw_col in methods:
        cov = valid[cov_col].dropna().mean() * 100
        bw  = valid[bw_col].dropna().mean()
        bold = "selective prior" in name
        if bold:
            latex.append(f"  \\textbf{{{name}}} & \\textbf{{{cov:.1f}\\%}} & "
                         f"\\textbf{{{bw:.2f}\\%}} \\\\")
        else:
            latex.append(f"  {name} & {cov:.1f}\\% & {bw:.2f}\\% \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)
    print(latex_str)
    (out_dir / "kde_baselines_table.tex").write_text(latex_str)
    log.info(f"[save] {out_dir}/kde_baselines_table.tex")
    log.info("Done.\n")


if __name__ == "__main__":
    main()