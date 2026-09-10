"""
run_full_evaluation.py
=======================
Single entry point that regenerates every table in the paper's Results
section, in the correct dependency order, using the CORRECTED
kde_engine_v4.py (base_quantile bug fixed) as the foundation everything
else builds on.

What this DOES cover (confirmed against your actual project scripts'
real CLI signatures before being wired in here):
  1. kde_engine_v4.py    -- run at h=7, h=15, h=30  (Tables 1, 2, 4, 8;
                             feeds every downstream script)
  2. garch_family_benchmark.py  -- EGARCH/GJR-GARCH comparison (Table 6-adjacent,
                             Figures 4-5)
  3. kde_baselines.py    -- plain KDE / rolling KDE baselines (Table 2)
  4. compute_proper_scores.py   -- CRPS, log score, DM tests (Table 6)
  5. compute_fei_by_event_type.py -- divergence-based FEI (Table 13)
  6. validate_framework.py -- OVX validation (Table 7), pinball/DM,
                             rolling historical baseline
  7. ablation_study_combined.py -- three-rank weighting ablation
                             (Tables 11, 12) -- built earlier this session

What this DOES NOT cover, and why:
  - Table 9 (tau sensitivity sweep): no script generating this table
    could be located anywhere in the current project files. Rather than
    write new statistical logic that risks silently diverging from
    whatever originally produced Table 9's numbers, this is left as an
    explicit gap -- see the printed warning at the end of this script.
    If you have this script saved elsewhere, point this orchestrator at
    it once located, or flag it and it can be written properly with the
    real methodology confirmed first.
  - Table 10 (regime breakdown by M1/M2 spread): NOT a separate script --
    this is a simple groupby on columns already present in the h=15
    kde_v3_results CSV ("base_spread_M1M2"), so it's generated inline
    here at the end rather than needing its own script.

Usage:
    python run_full_evaluation.py \
        --analogues retrieval_output_v4/per_event \
        --prices "data/oil_prices_full.xlsx" \
        --ovx "crude_oil_data/oil_price_data/CBOE Crude Oil Volatility Historical Data.csv" \
        --metadata memory_2024_2026/df_major_metadata_2024_2026.csv \
        --output results_run

All outputs land under --output, one subfolder per stage, so nothing
overwrites your existing results until you're ready to compare and
swap them in.
"""
import argparse
import subprocess
import sys
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("orchestrator")

SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(name: str, cmd: list):
    log.info(f"\n{'='*70}")
    log.info(f"STEP: {name}")
    log.info(f"CMD:  {' '.join(str(c) for c in cmd)}")
    log.info(f"{'='*70}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        log.error(f"STEP FAILED: {name} (exit code {result.returncode})")
        return False
    log.info(f"STEP COMPLETE: {name}")
    return True


def generate_table10_regime_breakdown(kde_csv_h15: Path, output_dir: Path):
    """
    Table 10 is a simple groupby on the already-computed h=15 results --
    no separate script needed. Backwardation: spread > 0.10,
    contango: spread < -0.10, everything between is "flat", matching
    the paper's stated thresholds.
    """
    if not kde_csv_h15.exists():
        log.warning(f"Cannot generate Table 10 -- {kde_csv_h15} not found")
        return

    df = pd.read_csv(kde_csv_h15)
    if "base_spread_M1M2" not in df.columns or "covers_90" not in df.columns:
        log.warning("Table 10: required columns missing from h=15 results, skipping")
        return

    df = df.dropna(subset=["base_spread_M1M2", "covers_90"])

    def regime(spread):
        if spread > 0.10:  return "Backwardation"
        if spread < -0.10: return "Contango"
        return "Flat"

    df["regime"] = df["base_spread_M1M2"].apply(regime)

    rows = []
    for reg in ["Backwardation", "Flat", "Contango"]:
        sub = df[df["regime"] == reg]
        if len(sub) == 0:
            continue
        rows.append({
            "Regime": reg,
            "N": len(sub),
            "Selective Prior Coverage": round(sub["covers_90"].mean() * 100, 1),
            "GARCH Coverage": round(sub["garch_covers_90"].mean() * 100, 1)
                               if "garch_covers_90" in sub.columns else None,
            "Mean Spread": round(sub["base_spread_M1M2"].mean(), 2),
            "Bimodal %": round(sub["kde_is_bimodal"].mean() * 100, 1)
                          if "kde_is_bimodal" in sub.columns else None,
        })

    out_df = pd.DataFrame(rows)
    out_path = output_dir / "table10_regime_breakdown.csv"
    out_df.to_csv(out_path, index=False)
    log.info(f"[Table 10] saved to {out_path}")
    print(out_df.to_string(index=False))



def _find_width_column(df: pd.DataFrame, candidates):
    """Return the first matching column from a list of candidate names."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def generate_width_efficiency(kde_dir: Path, output_dir: Path):
    """
    Calculate the BAND-WIDTH RATIO separately from the divergence-based FEI.

    Width Efficiency Ratio (WER):
        WER = GARCH band width / selective-prior band width

    This is deliberately NOT called FEI because the manuscript's Table 13
    FEI is divergence-based. The function tries several common column names
    so it can work with the existing KDE-v4 result schema without silently
    substituting another metric.
    """
    rows = []

    for h in [7, 15, 30]:
        csv_path = kde_dir / f"kde_v4_results_h{h}.csv"
        if not csv_path.exists():
            log.warning(f"[Width ratio] {csv_path} not found -- skipping h={h}")
            continue

        df = pd.read_csv(csv_path)

        kde_col = _find_width_column(df, [
            "kde_band_width", "selective_band_width",
            "mean_band_width", "band_width", "kde_width"
        ])
        garch_col = _find_width_column(df, [
            "garch_band_width", "garch_mean_band_width",
            "garch_width", "garch_bandwidth"
        ])

        if kde_col is None or garch_col is None:
            log.warning(
                f"[Width ratio] h={h}: could not identify both width columns. "
                f"Available columns: {list(df.columns)}"
            )
            continue

        kde_width = pd.to_numeric(df[kde_col], errors="coerce")
        garch_width = pd.to_numeric(df[garch_col], errors="coerce")
        valid = pd.concat(
            [kde_width.rename("kde"), garch_width.rename("garch")], axis=1
        ).dropna()
        valid = valid[valid["kde"] > 0]

        if valid.empty:
            log.warning(f"[Width ratio] h={h}: no valid width observations.")
            continue

        # Event-level ratio, consistent with the paper's event-level reporting
        # convention, plus ratio of mean widths for transparent checking.
        event_ratio = valid["garch"] / valid["kde"]
        mean_kde = valid["kde"].mean()
        mean_garch = valid["garch"].mean()
        ratio_of_means = mean_garch / mean_kde

        rows.append({
            "Horizon": h,
            "N": len(valid),
            "Selective Mean Band Width": round(mean_kde, 4),
            "GARCH Mean Band Width": round(mean_garch, 4),
            "Width Ratio (GARCH/KDE)": round(ratio_of_means, 4),
            "GARCH Wider (%)": round((ratio_of_means - 1) * 100, 2),
            "Mean Event-Level Width Ratio": round(event_ratio.mean(), 4),
        })

    if rows:
        out_df = pd.DataFrame(rows)
        out_path = output_dir / "band_width_efficiency.csv"
        out_df.to_csv(out_path, index=False)
        log.info(f"[Width ratio] saved to {out_path}")
        print("\nBand-width efficiency -- DISTINCT FROM DIVERGENCE FEI")
        print(out_df.to_string(index=False))
    else:
        log.warning(
            "[Width ratio] No width-ratio table generated. "
            "Inspect the KDE-v4 result columns above and add the correct "
            "width column names to _find_width_column()."
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analogues", required=True,
        help="retrieval_output_v3/per_event")
    ap.add_argument("--prices", required=True,
        help="data/oil_prices_full.xlsx")
    ap.add_argument("--ovx", required=True,
        help="CBOE Crude Oil Volatility Historical Data.csv")
    ap.add_argument("--metadata", required=True,
        help="memory_2024_2026/df_major_metadata_2024_2026.csv "
             "(for compute_fei_by_event_type.py's event-type labels)")
    ap.add_argument("--output", default="results_run")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    kde_dir = out_dir / "kde_output_v4"

    log.info("="*70)
    log.info("FULL EVALUATION PIPELINE -- using CORRECTED kde_engine_v4.py")
    log.info("(base_quantile bug fixed: real price_percentile() now used)")
    log.info("="*70)

    failures = []

    # ── Step 1: core KDE engine, all three horizons ─────────────────────
    for h in [7, 15, 30]:
        ok = run_step(
            f"kde_engine_v4.py (h={h})",
            [sys.executable, "kde_engine_v4.py",
             "--analogues", args.analogues,
             "--prices", args.prices,
             "--ovx", args.ovx,
             "--output", str(kde_dir),
             "--horizon", str(h)],
        )
        if not ok:
            failures.append(f"kde_engine_v3.py h={h}")

    kde_csv_h15 = kde_dir / "kde_v4_results_h15.csv"
    if not kde_csv_h15.exists():
        log.error("h=15 results missing -- cannot proceed with downstream "
                   "scripts that depend on it. Stopping here.")
        sys.exit(1)

    # ── Step 2: GARCH family comparison ─────────────────────────────────
    ok = run_step(
        "garch_family_benchmark.py",
        [sys.executable, "garch_family_benchmark.py",
         "--results", str(kde_csv_h15),
         "--prices", args.prices,
         "--horizon", "15",
         "--output", str(out_dir / "garch_family")],
    )
    if not ok: failures.append("garch_family_benchmark.py")

    # ── Step 3: KDE baselines (plain / rolling) ─────────────────────────
    ok = run_step(
        "kde_baselines.py",
        [sys.executable, "kde_baselines.py",
         "--results", str(kde_csv_h15),
         "--prices", args.prices,
         "--horizon", "15",
         "--output", str(out_dir / "kde_baselines")],
    )
    if not ok: failures.append("kde_baselines.py")

    # ── Step 4: proper scoring rules (CRPS, log score, DM tests) ────────
    ok = run_step(
        "compute_proper_scores.py",
        [sys.executable, "compute_proper_scores.py",
         "--kde_dir", str(kde_dir),
         "--results", str(kde_csv_h15),
         "--output", str(out_dir / "proper_scores")],
    )
    if not ok: failures.append("compute_proper_scores.py")

    # ── Step 5: FEI by event type (Table 13) ────────────────────────────
    ok = run_step(
        "compute_fei_by_event_type.py",
        [sys.executable, "compute_fei_by_event_type.py",
         "--kde_dir", str(kde_dir),
         "--results", str(kde_csv_h15),
         "--metadata", args.metadata,
         "--output", str(out_dir / "fei_by_event_type")],
    )
    if not ok: failures.append("compute_fei_by_event_type.py")

    # ── Step 6: OVX validation + robustness + pinball/DM (Table 7) ──────
    ok = run_step(
        "validate_framework.py",
        [sys.executable, "validate_framework.py",
         "--kde_csv", str(kde_csv_h15),
         "--prices", args.prices,
         "--ovx", args.ovx,
         "--horizon", "15",
         "--output", str(out_dir / "validation")],
    )
    if not ok: failures.append("validate_framework.py")

    # ── Step 7: ablation study, all 3 horizons combined (Tables 11, 12) ─
    ok = run_step(
        "ablation_study_combined.py",
        [sys.executable, "ablation_study_combined.py",
         "--analogues", args.analogues,
         "--prices", args.prices,
         "--ovx", args.ovx,
         "--output", str(out_dir / "ablation")],
    )
    if not ok: failures.append("ablation_study_combined.py")

    # ── Step 8: Table 10, generated inline (no separate script needed) ──
    generate_table10_regime_breakdown(kde_csv_h15, out_dir)

    # ── Step 9: BAND-WIDTH RATIO, kept separate from divergence FEI ─────
    generate_width_efficiency(kde_dir, out_dir)

    # ── Final summary ────────────────────────────────────────────────────
    log.info("\n" + "="*70)
    log.info("PIPELINE COMPLETE")
    log.info("="*70)
    if failures:
        log.warning(f"{len(failures)} step(s) failed -- check output above:")
        for f in failures:
            log.warning(f"  - {f}")
    else:
        log.info("All steps completed successfully.")

    log.info("\nNOT COVERED BY THIS SCRIPT (see module docstring for why):")
    log.info("  - Table 9 (tau sensitivity sweep) -- no source script located")
    log.info("    in project files. Confirm methodology before regenerating.")
    log.info(f"\nAll outputs saved under: {out_dir.resolve()}")
    log.info("Next step: use the v4 outputs for the final Tables 2/4/6/7/10/11/12/13. "
              "Use band_width_efficiency.csv for width ratios; keep the "
              "divergence-based FEI from compute_fei_by_event_type.py separate.")


if __name__ == "__main__":
    main()