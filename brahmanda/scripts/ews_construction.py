"""
ews_construction.py
=====================
Builds and validates an early-warning signal for extreme crude oil
price moves, using the predictors already shown to work in the FRL
regression (kde_std) plus a natural refinement (tail_weight), rather
than the bimodal flag (shown NOT to be a tail-risk signal).

Two disciplines applied that were missing from a naive "just check
the coefficient" approach:
  1. Evaluated via ROC/AUC and precision-recall, not just a p-value.
  2. Threshold selected via k-fold cross-validation, so reported
     performance isn't inflated by tuning on the same data it's
     evaluated on.

Usage:
    python ews_construction.py \
        --kde_results results_run/kde_output_v4/kde_v4_results_h15.csv \
        --extreme_threshold 10.0 \
        --n_folds 5
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve


def evaluate_signal(y_true, score, label):
    """ROC-AUC plus the Youden's-J-optimal threshold, precision/recall there."""
    auc = roc_auc_score(y_true, score)
    fpr, tpr, thresholds = roc_curve(y_true, score)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]

    pred = (score >= best_threshold).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    print(f"{label}: AUC={auc:.3f}, threshold={best_threshold:.3f} (Youden's J), "
          f"precision={precision:.3f}, recall={recall:.3f}, "
          f"n_flagged={pred.sum()}")

    # Full precision-recall tradeoff, not just one auto-selected point --
    # a real deployment decision needs to see the actual tradeoff curve.
    print(f"  {'Percentile':>12}{'Threshold':>12}{'Precision':>12}{'Recall':>12}{'N flagged':>12}")
    for pctile in [95, 90, 80, 70, 60, 50]:
        thresh = np.percentile(score, pctile)
        pred_p = (score >= thresh).astype(int)
        tp_p = ((pred_p == 1) & (y_true == 1)).sum()
        fp_p = ((pred_p == 1) & (y_true == 0)).sum()
        fn_p = ((pred_p == 0) & (y_true == 1)).sum()
        prec_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else np.nan
        rec_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else np.nan
        print(f"  {pctile:>11}%{thresh:>12.3f}{prec_p:>12.3f}{rec_p:>12.3f}{pred_p.sum():>12}")

    return {"label": label, "auc": auc, "threshold": best_threshold,
            "precision": precision, "recall": recall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kde_results", required=True)
    ap.add_argument("--extreme_threshold", type=float, default=10.0)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.kde_results)
    required = ["kde_std", "realised_r"]
    tail_col = next((c for c in ["kde_tail_weight", "tail_weight", "Pi_tail", "pi_tail"]
                      if c in df.columns), None)
    if tail_col:
        required.append(tail_col)
    df = df.dropna(subset=required)
    df["extreme"] = (df["realised_r"].abs() > args.extreme_threshold).astype(int)
    print(f"N={len(df)}, extreme={df['extreme'].sum()} "
          f"({df['extreme'].mean()*100:.1f}%)\n")

    print("=" * 70)
    print("IN-SAMPLE SIGNAL EVALUATION (baseline, before cross-validation)")
    print("=" * 70)
    evaluate_signal(df["extreme"].values, df["kde_std"].values, "kde_std alone")
    if tail_col:
        evaluate_signal(df["extreme"].values, df[tail_col].values, f"{tail_col} alone")
        combined = df["kde_std"].values * df[tail_col].values
        evaluate_signal(df["extreme"].values, combined, "kde_std x tail_weight")
    if "base_ovx" in df.columns:
        ovx_sub = df.dropna(subset=["base_ovx"])
        evaluate_signal(ovx_sub["extreme"].values, ovx_sub["base_ovx"].values, "OVX alone")

    print("\n" + "=" * 70)
    print(f"CROSS-VALIDATED EVALUATION ({args.n_folds}-fold, threshold selected")
    print("on train folds, evaluated on held-out fold -- avoids overfitting)")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    X = df["kde_std"].values
    y = df["extreme"].values

    cv_results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fpr, tpr, thresholds = roc_curve(y_train, X_train)
        best_idx = np.argmax(tpr - fpr)
        threshold = thresholds[best_idx]

        pred_test = (X_test >= threshold).astype(int)
        tp = ((pred_test == 1) & (y_test == 1)).sum()
        fp = ((pred_test == 1) & (y_test == 0)).sum()
        fn = ((pred_test == 0) & (y_test == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        auc_test = roc_auc_score(y_test, X_test) if len(set(y_test)) > 1 else np.nan

        print(f"Fold {fold+1}: threshold(train)={threshold:.3f}, "
              f"AUC(test)={auc_test:.3f}, precision(test)={precision:.3f}, "
              f"recall(test)={recall:.3f}")
        cv_results.append({"fold": fold+1, "threshold": threshold,
                            "auc": auc_test, "precision": precision, "recall": recall})

    cv_df = pd.DataFrame(cv_results)
    print(f"\nMean out-of-fold AUC: {cv_df['auc'].mean():.3f} (+/- {cv_df['auc'].std():.3f})")
    print(f"Mean out-of-fold precision: {cv_df['precision'].mean():.3f}")
    print(f"Mean out-of-fold recall: {cv_df['recall'].mean():.3f}")
    print(f"\nThis is the honest, non-overfit estimate of how this signal would")
    print(f"perform on genuinely new events -- use these numbers, not the")
    print(f"in-sample numbers above, for any claim about real-world performance.")

    cv_df.to_csv("ews_cv_results.csv", index=False)
    print(f"\n[saved] ews_cv_results.csv")


if __name__ == "__main__":
    main()