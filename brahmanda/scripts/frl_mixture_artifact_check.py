"""
frl_mixture_artifact_check.py
================================
Addresses reviewer point 4: is the negative bimodal-extreme
relationship a mechanical artefact of two-component mixture
construction, or does it exceed what mixture mechanics alone would
predict?

Logic: simulate two-component Gaussian mixtures with TOTAL VARIANCE
HELD FIXED (matched to the real corpus's kde_std distribution), but
varying mode separation. Compute P(|draw| > threshold) for each,
compare bimodal (separated) vs unimodal (zero separation) cases
under pure mechanics alone -- no economic story, just arithmetic.

If the mechanical effect alone is small relative to the empirical
finding (OR=0.169 at h15/10%), that is evidence the empirical result
is not merely a construction artefact. If the mechanical effect
alone is comparably large, the paper's interpretation needs
revising.

Usage:
    python frl_mixture_artifact_check.py \
        --kde_results results_run/kde_output_v4/kde_v4_results_h15.csv \
        --threshold 10.0 \
        --n_sims 20000
"""
import argparse
import numpy as np
import pandas as pd


def simulate_mixture_tail_prob(total_std, separation, threshold, n_draws=20000, rng=None):
    """
    Two-component equal-weight Gaussian mixture, components at
    +/- separation/2 from zero, each with within-component std
    chosen so the MIXTURE's total variance equals total_std^2
    (matching a real event's kde_std). separation=0 recovers a
    single Gaussian with std=total_std (the 'unimodal' case).

    Mixture variance = within_var + (separation/2)^2
    (standard decomposition of variance for an equal-weight 2-mode
    mixture with modes at +/- separation/2).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    within_var = max(total_std**2 - (separation / 2) ** 2, 1e-6)
    within_std = np.sqrt(within_var)

    component = rng.integers(0, 2, size=n_draws)
    means = np.where(component == 0, -separation / 2, separation / 2)
    draws = rng.normal(loc=means, scale=within_std)

    return np.mean(np.abs(draws) > threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kde_results", required=True)
    ap.add_argument("--threshold", type=float, default=10.0)
    ap.add_argument("--n_sims", type=int, default=20000)
    args = ap.parse_args()

    df = pd.read_csv(args.kde_results).dropna(subset=["kde_std", "kde_is_bimodal"])
    mean_std = df["kde_std"].mean()
    print(f"Mean kde_std across corpus: {mean_std:.3f}")
    print(f"Extreme threshold: {args.threshold}%\n")

    rng = np.random.default_rng(42)

    print("=" * 70)
    print("MECHANICAL EFFECT: mixture separation vs P(extreme), variance FIXED")
    print("=" * 70)
    print(f"{'Separation':>12} {'P(extreme)':>12} {'vs unimodal':>15}")

    separations = [0, 5, 10, 15, 20, 25, 30]
    p_unimodal = None
    results = []
    for sep in separations:
        p = simulate_mixture_tail_prob(mean_std, sep, args.threshold, args.n_sims, rng)
        if sep == 0:
            p_unimodal = p
        ratio = p / p_unimodal if p_unimodal else np.nan
        print(f"{sep:>12.0f} {p*100:>11.2f}% {ratio:>15.3f}")
        results.append({"separation": sep, "p_extreme": p, "odds_ratio_vs_unimodal": ratio})

    print("\n" + "=" * 70)
    print("COMPARISON TO EMPIRICAL FINDING")
    print("=" * 70)
    print("Empirical odds ratio (bimodal vs unimodal, h15, threshold=10%): 0.169")
    print("(i.e. bimodal events have ~17% of the odds of extreme moves)")
    print()
    print("If mechanical-only odds ratios at realistic separations (typically")
    print("observed bimodal mode separation is roughly 15-25 percentage points")
    print("in this corpus -- check against real data for exact figure) are")
    print("substantially CLOSER TO 1 than 0.169, this supports the paper's")
    print("interpretation that the empirical effect exceeds pure mixture")
    print("mechanics and reflects genuine information content, not just an")
    print("artefact of two-mode construction.")
    print()
    print("If mechanical-only odds ratios are ALREADY close to 0.169 at")
    print("realistic separations, the empirical finding may be substantially")
    print("explained by construction alone, and the paper's interpretation")
    print("needs revising accordingly.")

    pd.DataFrame(results).to_csv("mixture_artifact_check.csv", index=False)
    print("\n[saved] mixture_artifact_check.csv")


if __name__ == "__main__":
    main()
