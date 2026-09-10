"""
frl_firth_standalone.py
==========================
Self-contained Firth (1993) bias-reduced logistic regression,
implemented directly with numpy/scipy -- no sklearn dependency, to
avoid the firthlogist/scikit-learn version conflict.

Numerically stable even in separation-affected cells that fail to
converge under standard MLE.

Reference: Firth, D. (1993). Bias reduction of maximum likelihood
estimates. Biometrika, 80(1), 27-38.

Usage:
    python frl_firth_standalone.py \
        --kde_results results_run/kde_output_v4/kde_v4_results_h15.csv \
        --extreme_threshold 10.0
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats


def firth_logistic_regression(X, y, max_iter=50, tol=1e-6):
    n, p = X.shape
    beta = np.zeros(p)
    info_inv = np.eye(p)

    for iteration in range(max_iter):
        eta = X @ beta
        pi = 1 / (1 + np.exp(-eta))
        W = pi * (1 - pi)
        W = np.clip(W, 1e-10, None)

        Xw = X * np.sqrt(W)[:, None]
        try:
            XtWX_inv = np.linalg.inv(Xw.T @ Xw)
        except np.linalg.LinAlgError:
            return beta, np.full(p, np.nan), False

        H = Xw @ XtWX_inv @ Xw.T
        h = np.diag(H)

        score = X.T @ (y - pi + h * (0.5 - pi))

        info = X.T @ (X * W[:, None])
        try:
            info_inv = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            return beta, np.full(p, np.nan), False

        delta = info_inv @ score
        beta_new = beta + delta

        if np.max(np.abs(delta)) < tol:
            se = np.sqrt(np.diag(info_inv))
            return beta_new, se, True
        beta = beta_new

    se = np.sqrt(np.diag(info_inv))
    return beta, se, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kde_results", required=True)
    ap.add_argument("--extreme_threshold", type=float, default=10.0)
    args = ap.parse_args()

    df = pd.read_csv(args.kde_results).dropna(
        subset=["kde_is_bimodal", "kde_std", "realised_r"])
    df["extreme"] = (df["realised_r"].abs() > args.extreme_threshold).astype(int)

    X = np.column_stack([
        np.ones(len(df)),
        df["kde_std"].values,
        df["kde_is_bimodal"].values,
    ])
    y = df["extreme"].values.astype(float)

    beta, se, converged = firth_logistic_regression(X, y)
    names = ["const", "kde_std", "bimodal"]

    print(f"N={len(df)}, extreme={int(y.sum())}, threshold={args.extreme_threshold}%")
    print(f"Converged: {converged}\n")
    print(f"{'Term':<12}{'Coef':>10}{'SE':>10}{'z':>10}{'p':>10}{'OR':>10}")
    for i, name in enumerate(names):
        z = beta[i] / se[i] if se[i] > 0 else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        or_val = np.exp(beta[i])
        print(f"{name:<12}{beta[i]:>10.4f}{se[i]:>10.4f}{z:>10.3f}{p:>10.4f}{or_val:>10.4f}")

    bimodal_p = 2 * (1 - stats.norm.cdf(abs(beta[2] / se[2]))) if se[2] > 0 else np.nan
    print(f"\nCompare bimodal coefficient to standard MLE (-1.7807, p=0.0058):")
    print(f"Firth: {beta[2]:.4f}, p={bimodal_p:.4f}")
    print("\nIf close, standard MLE was already reliable at this threshold/horizon.")
    print("This method's real value is in the separation-affected cells")
    print("(h7@15%, h7@20%, h15@20%) -- rerun with --extreme_threshold set to")
    print("those values to get finite, interpretable estimates there instead")
    print("of the excluded/undefined standard MLE results.")


if __name__ == "__main__":
    main()
