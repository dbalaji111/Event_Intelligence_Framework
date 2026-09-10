"""
compute_proper_scores.py
========================
Computes distribution-level proper scoring rules for IJF Paper 1:

  1. CRPS (Continuous Ranked Probability Score)
     - For KDE v4: computed from stored x_grid and density via numerical integration
     - For GARCH family: computed analytically from Gaussian CDF formula
     CRPS(F, y) = E_F|X - y| - 0.5 * E_F|X - X'|

  2. Log Predictive Score (Log Score)
     - For KDE v4: log(f(y)) evaluated at realised return using stored density
     - For GARCH family: log of Gaussian PDF at realised return

  Also runs Diebold-Mariano tests on CRPS differentials.

Usage (from project root):
    python scripts/compute_proper_scores.py \
        --kde_dir   kde_output_v4/per_event \
        --results   figures/kde_v4_results_garch_family.csv \
        --output    figures

Output:
    figures/proper_scores_summary.txt  — table ready for paper
    figures/proper_scores.csv          — full event-level results
"""

import argparse, json, sys, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm
from scipy.stats import wilcoxon
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CRPS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def crps_gaussian(mu: float, sigma: float, y: float) -> float:
    """
    Analytical CRPS for N(mu, sigma^2) at observation y.
    CRPS(N(mu,sigma), y) = sigma * (z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi))
    where z = (y - mu) / sigma.
    """
    if sigma <= 0 or not np.isfinite(sigma):
        return np.nan
    z = (y - mu) / sigma
    crps = sigma * (z * (2 * scipy_norm.cdf(z) - 1) +
                    2 * scipy_norm.pdf(z) -
                    1 / np.sqrt(np.pi))
    return float(crps)


def crps_kde(x_grid: np.ndarray, density: np.ndarray, y: float) -> float:
    """
    Numerical CRPS for KDE mixture posterior via integration.
    CRPS(F, y) = integral (F(x) - 1(x >= y))^2 dx
    Uses trapezoidal rule on stored x_grid and density.
    """
    if not np.isfinite(y):
        return np.nan
    dx  = np.diff(x_grid)
    # Build CDF numerically
    cdf = np.zeros(len(x_grid))
    for i in range(1, len(x_grid)):
        cdf[i] = cdf[i-1] + 0.5 * (density[i-1] + density[i]) * dx[i-1]
    # Normalise
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    # Heaviside
    H = (x_grid >= y).astype(float)
    integrand = (cdf - H) ** 2
    crps = np.trapz(integrand, x_grid)
    return float(crps)


def log_score_gaussian(mu: float, sigma: float, y: float) -> float:
    """Log predictive score = log p(y) for N(mu, sigma^2)."""
    if sigma <= 0 or not np.isfinite(sigma):
        return np.nan
    return float(scipy_norm.logpdf(y, mu, sigma))


def log_score_kde(x_grid: np.ndarray, density: np.ndarray, y: float) -> float:
    """Log predictive score for KDE via linear interpolation of stored density."""
    if not np.isfinite(y):
        return np.nan
    f_y = float(np.interp(y, x_grid, density))
    if f_y <= 0:
        return -np.inf
    return float(np.log(f_y))


# ─────────────────────────────────────────────────────────────────────────────
# DIEBOLD-MARIANO TEST
# ─────────────────────────────────────────────────────────────────────────────

def dm_test(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple:
    """
    Diebold-Mariano test: H0: E[d] = 0 where d = loss_a - loss_b.
    Negative DM stat means loss_a < loss_b (model A is better).
    Uses Newey-West variance with lag=1.
    Returns (DM statistic, p-value).
    """
    d    = loss_a - loss_b
    n    = len(d)
    d_bar = np.mean(d)
    # Newey-West with lag 1
    gamma0 = np.mean(d**2) - d_bar**2
    gamma1 = np.mean(d[1:] * d[:-1]) - d_bar**2
    nw_var = (gamma0 + 2 * gamma1) / n
    if nw_var <= 0:
        return np.nan, np.nan
    dm_stat = d_bar / np.sqrt(nw_var)
    from scipy.stats import norm as scipy_norm2
    p_val = 2 * (1 - scipy_norm2.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_val)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kde_dir",  required=True,
                    help="kde_output_v4/per_event/")
    ap.add_argument("--results",  required=True,
                    help="figures/kde_v4_results_garch_family.csv")
    ap.add_argument("--output",   default="figures")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load results CSV (has GARCH family sigma columns)
    df = pd.read_csv(args.results)
    df["base_date"] = pd.to_datetime(df["base_date"])
    log.info(f"Loaded {len(df)} events")

    kde_dir = Path(args.kde_dir)

    # ── Compute scores for each event ────────────────────────────────────
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="Scoring", unit=" ev"):
        event_id = row["event_id"]
        realised = row["realised_r"]
        if not np.isfinite(realised):
            continue

        # Load KDE posterior
        json_path = kde_dir / f"{event_id}_kde_v4.json"
        if not json_path.exists():
            continue
        with json_path.open() as f:
            data = json.load(f)

        x_grid  = np.array(data.get("x_grid",  []))
        density = np.array(data.get("density", []))

        if len(x_grid) == 0 or len(density) == 0:
            continue

        # KDE scores
        crps_kde_val = crps_kde(x_grid, density, realised)
        logs_kde_val = log_score_kde(x_grid, density, realised)

        # GARCH(1,1) scores
        g_sig  = row.get("garch_sigma")
        crps_g = crps_gaussian(0, g_sig, realised) \
                 if pd.notna(g_sig) else np.nan
        logs_g = log_score_gaussian(0, g_sig, realised) \
                 if pd.notna(g_sig) else np.nan

        # EGARCH scores
        e_sig  = row.get("egarch_sigma")
        crps_e = crps_gaussian(0, e_sig, realised) \
                 if pd.notna(e_sig) else np.nan
        logs_e = log_score_gaussian(0, e_sig, realised) \
                 if pd.notna(e_sig) else np.nan

        # GJR-GARCH scores
        j_sig  = row.get("gjr_sigma")
        crps_j = crps_gaussian(0, j_sig, realised) \
                 if pd.notna(j_sig) else np.nan
        logs_j = log_score_gaussian(0, j_sig, realised) \
                 if pd.notna(j_sig) else np.nan

        rows.append({
            "event_id":    event_id,
            "realised_r":  realised,
            "crps_kde":    crps_kde_val,
            "crps_garch":  crps_g,
            "crps_egarch": crps_e,
            "crps_gjr":    crps_j,
            "logs_kde":    logs_kde_val,
            "logs_garch":  logs_g,
            "logs_egarch": logs_e,
            "logs_gjr":    logs_j,
        })

    scores = pd.DataFrame(rows)
    scores.to_csv(out_dir / "proper_scores.csv", index=False)
    log.info(f"[save] {out_dir}/proper_scores.csv  ({len(scores)} events)")

    # ── Summary statistics ────────────────────────────────────────────────
    valid = scores.dropna(subset=["crps_kde","crps_garch","crps_gjr",
                                   "logs_kde","logs_garch"])
    n = len(valid)

    methods = [
        ("KDE v4",      "crps_kde",    "logs_kde"),
        ("GARCH(1,1)",  "crps_garch",  "logs_garch"),
        ("EGARCH(1,1)", "crps_egarch", "logs_egarch"),
        ("GJR-GARCH",   "crps_gjr",    "logs_gjr"),
    ]

    # DM tests on CRPS
    dm_garch_crps,  p_garch_crps  = dm_test(
        valid["crps_kde"].values, valid["crps_garch"].values)
    dm_egarch_crps, p_egarch_crps = dm_test(
        valid["crps_kde"].values, valid["crps_egarch"].values)
    dm_gjr_crps,    p_gjr_crps    = dm_test(
        valid["crps_kde"].values, valid["crps_gjr"].values)

    # ── Print and save table ──────────────────────────────────────────────
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"  PROPER SCORING RULES  (h=+15d, N={n})")
    lines.append("=" * 70)
    lines.append(f"  {'Method':<18} {'Mean CRPS':>10}  {'Mean Log Score':>14}")
    lines.append("  " + "-" * 50)

    for name, crps_col, logs_col in methods:
        c = valid[crps_col].dropna().mean()
        l = valid[logs_col].dropna().mean()
        marker = " ◄" if name == "KDE v4" else ""
        lines.append(f"  {name:<18} {c:>10.4f}  {l:>14.4f}{marker}")

    lines.append("  " + "-" * 50)
    lines.append(f"\n  DM tests on CRPS (KDE v4 vs benchmark):")
    lines.append(f"  vs GARCH(1,1):  DM={dm_garch_crps:+.3f}  p={p_garch_crps:.3f}")
    lines.append(f"  vs EGARCH(1,1): DM={dm_egarch_crps:+.3f}  p={p_egarch_crps:.3f}")
    lines.append(f"  vs GJR-GARCH:   DM={dm_gjr_crps:+.3f}  p={p_gjr_crps:.3f}")
    lines.append("  (Negative DM = KDE v4 has lower CRPS = better)")
    lines.append("=" * 70)

    # LaTeX table
    lines.append("\n% ── LaTeX table (paste into paper) ──────────────────────────────")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Distribution-level proper scoring rules at $h = +15$ days,")
    lines.append(r"$N = " + str(n) + r"$ out-of-sample events.")
    lines.append(r"CRPS = Continuous Ranked Probability Score (lower = better).")
    lines.append(r"Log Score = mean log predictive density (higher = better).")
    lines.append(r"DM = Diebold--Mariano test statistic (KDE v4 vs benchmark);")
    lines.append(r"negative values indicate KDE v4 is more accurate.")
    lines.append(r"$^{**}p<0.01$, $^*p<0.05$.}")
    lines.append(r"\label{tab:proper_scores}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Mean CRPS & Mean Log Score & DM vs KDE v4 \\")
    lines.append(r"\midrule")

    for name, crps_col, logs_col in methods:
        c = valid[crps_col].dropna().mean()
        l = valid[logs_col].dropna().mean()
        if name == "KDE v4":
            dm_str = "---"
            lines.append(
                f"  \\textbf{{{name}}} & \\textbf{{{c:.4f}}} & "
                f"\\textbf{{{l:.4f}}} & {dm_str} \\\\")
        else:
            dm_map = {
                "GARCH(1,1)":  (dm_garch_crps,  p_garch_crps),
                "EGARCH(1,1)": (dm_egarch_crps, p_egarch_crps),
                "GJR-GARCH":   (dm_gjr_crps,    p_gjr_crps),
            }
            dm_v, p_v = dm_map[name]
            sig = "^{**}" if p_v < 0.01 else ("^{*}" if p_v < 0.05 else "")
            dm_str = f"${dm_v:+.3f}{sig}$"
            lines.append(
                f"  {name} & {c:.4f} & {l:.4f} & {dm_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    table_str = "\n".join(lines)
    print(table_str)

    txt_path = out_dir / "proper_scores_summary.txt"
    txt_path.write_text(table_str)
    log.info(f"[save] {txt_path}")
    log.info("Done.\n")


if __name__ == "__main__":
    main()