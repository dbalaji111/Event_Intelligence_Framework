"""
09b_batch_retrieve.py
======================
Batch orchestrator on top of script 09 (retrieve_analogues).

Use case: "for every cluster in year 2023, retrieve top-K analogues from
2000-2022". Loops over all base clusters that match the query year + window
filter, calls retrieve_analogues with max_candidate_year = query_year - 1
(strict no-leakage), saves one JSON per base cluster.

Reads
-----
  --cluster_index   clusters_full_history/cluster_memory.parquet (from script 08)

Writes
------
  <output_dir>/<query_year>_w<window>/analogues_<cluster_id>.json
  <output_dir>/<query_year>_w<window>/_batch_summary.csv
  <output_dir>/<query_year>_w<window>/_batch_summary.json

Run
---
    python 09b_batch_retrieve.py \\
        --cluster_index clusters_full_history/cluster_memory.parquet \\
        --query_year 2023 \\
        --window 15 \\
        --top_k 20

CLI flags
---------
  --query_year       year of the BASE clusters to retrieve for (e.g. 2023)
  --window           which window-size to query (3, 5, 10, 15, 20, 30)
  --top_k            top-K analogues per base (default 20)
  --require_role     only run for base clusters with this role
                     (e.g. REGIME-MARKER) — drops NOVEL/REPEAT noise
  --min_size         only run for base clusters with size >= this
  --limit            optional smoke: only first N base clusters
  --beta             three weights sem,prop,qtl (default 0.5,0.3,0.2)
  --gamma, --tau_s, --tau_eta, --tau_q, --dominance_penalty
                     same as in script 09; passed through unchanged
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import pandas as pd

# Locate the sibling script 09 so we can import its function
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Import the retrieval function and helpers from script 09
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_retrieve09", HERE / "09_retrieve_analogues.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_index        = _mod.load_index
retrieve_analogues = _mod.retrieve_analogues
_parse_stop_list   = _mod._parse_stop_list
DEFAULT_BETA      = _mod.DEFAULT_BETA
DEFAULT_GAMMA     = _mod.DEFAULT_GAMMA
DEFAULT_DOMINANCE_PENALTY = _mod.DEFAULT_DOMINANCE_PENALTY
DEFAULT_TAU_S     = _mod.DEFAULT_TAU_S
DEFAULT_TAU_ETA   = _mod.DEFAULT_TAU_ETA
DEFAULT_TAU_Q     = _mod.DEFAULT_TAU_Q
DEFAULT_TOP_K     = _mod.DEFAULT_TOP_K


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster_index", required=True)
    ap.add_argument("--query_year",    required=True, type=int,
                    help="year of the base clusters (analogues forced to "
                         "first_seen.year <= query_year - 1)")
    ap.add_argument("--window",        required=True, type=int,
                    help="window_days of base clusters AND of analogues "
                         "(only same-window candidates are considered)")
    ap.add_argument("--output_dir", default=None,
                    help="defaults to <cluster_index dir>/retrievals/<year>_w<W>/")
    ap.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--require_role", default=None,
                    help="restrict BASE clusters to this role "
                         "(e.g. REGIME-MARKER). Analogues are NOT role-restricted.")
    ap.add_argument("--min_size", type=int, default=0,
                    help="only run for base clusters with size >= this")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: only first N base clusters")
    # Pass-through to retrieve_analogues
    ap.add_argument("--beta", default=",".join(str(b) for b in DEFAULT_BETA))
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--dominance_penalty", type=float,
                    default=DEFAULT_DOMINANCE_PENALTY)
    ap.add_argument("--tau_s",   type=float, default=DEFAULT_TAU_S)
    ap.add_argument("--tau_eta", type=float, default=DEFAULT_TAU_ETA)
    ap.add_argument("--tau_q",   type=float, default=DEFAULT_TAU_Q)
    ap.add_argument("--allow_overlap", action="store_true",
                    help="permit candidate analogues whose date span overlaps base")
    ap.add_argument("--recency_tau", type=float, default=None,
                    help="exponential decay half-life in YEARS for recency weighting "
                         "(applied at Stage 1 entry to break BERT-blob bias)")
    ap.add_argument("--max_per_era", type=int, default=None,
                    help="cap analogues per era_size_years window (diversity)")
    ap.add_argument("--era_size_years", type=int, default=5)
    ap.add_argument("--entity_stop_list", default=None,
                    help="entities to strip from BOTH bags before Jaccard. "
                         "Use 'DEFAULT' for the built-in generic-actor list.")
    args = ap.parse_args()

    idx_path = Path(args.cluster_index).resolve()
    out_dir  = (Path(args.output_dir).resolve() if args.output_dir
                else idx_path.parent / "retrievals"
                     / f"{args.query_year}_w{args.window}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {idx_path.name}")
    t0 = time.time()
    idx = load_index(idx_path)
    print(f"[load]   {len(idx)} clusters in {time.time()-t0:.1f}s")

    # Select BASE clusters: year == query_year, window == args.window, optional role/size
    idx["_yr"] = pd.to_datetime(idx["first_seen"]).dt.year
    base_pool = idx[(idx["_yr"] == args.query_year) &
                    (idx["window_days"] == args.window)]
    if args.require_role is not None:
        base_pool = base_pool[base_pool["role"] == args.require_role]
    if args.min_size > 0:
        base_pool = base_pool[base_pool["size"] >= args.min_size]
    base_pool = base_pool.reset_index(drop=True)
    print(f"[base] {len(base_pool)} base clusters in year={args.query_year}, "
          f"window={args.window}d"
          + (f", role={args.require_role}" if args.require_role else "")
          + (f", min_size={args.min_size}" if args.min_size > 0 else ""))

    if args.limit:
        base_pool = base_pool.head(args.limit).copy()
        print(f"[limit] capped to first {args.limit}")

    if base_pool.empty:
        sys.exit("No base clusters match — check --query_year / --window / --require_role")

    beta = tuple(float(x) for x in args.beta.split(","))
    if len(beta) != 3:
        sys.exit("ERROR: --beta must have exactly 3 numbers")

    max_cand_yr = args.query_year - 1
    print(f"[batch] retrieving with max_candidate_year={max_cand_yr} "
          f"(strict no-leakage from year {args.query_year})")
    print(f"[batch] writing to {out_dir}")

    summary_rows = []
    t_start = time.time()
    for i, row in base_pool.iterrows():
        cid = row["cluster_id"]
        try:
            result = retrieve_analogues(
                cid, idx,
                top_k=args.top_k,
                beta=beta, gamma=args.gamma,
                dominance_penalty=args.dominance_penalty,
                tau_s=args.tau_s, tau_eta=args.tau_eta, tau_q=args.tau_q,
                same_window_only=True,
                exclude_overlapping=not args.allow_overlap,
                max_candidate_year=max_cand_yr,
                recency_tau=args.recency_tau,
                max_per_era=args.max_per_era,
                era_size_years=args.era_size_years,
                entity_stop_list=_parse_stop_list(args.entity_stop_list),
            )
        except Exception as e:
            print(f"  [{i+1}/{len(base_pool)}] {cid}  FAILED: {e}", flush=True)
            summary_rows.append({
                "base_cluster_id": cid, "n_analogues": 0,
                "error": str(e), "first_seen": row["first_seen"],
                "size": int(row["size"]), "role": row["role"],
                "label": row["label"],
            })
            continue

        out_json = out_dir / f"analogues_{cid}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        n_an = len(result.get("analogues", []))
        funnel = result.get("stage_funnel", {})
        summary_rows.append({
            "base_cluster_id": cid,
            "first_seen":      row["first_seen"],
            "size":            int(row["size"]),
            "role":            row["role"],
            "label":           row["label"],
            "n_analogues":     n_an,
            "stage1_semantic": funnel.get("stage1_semantic", 0),
            "stage2_entity":   funnel.get("stage2_entity", 0),
            "stage3_quantile": funnel.get("stage3_quantile", 0),
            "stage4_propagation": funnel.get("stage4_propagation", 0),
            "top_analogue_year": (
                result["analogues"][0]["first_seen"][:4] if n_an else ""
            ),
            "top_composite_w": (
                result["analogues"][0]["scores"]["composite_weight"] if n_an else 0.0
            ),
            "out_json":        out_json.name,
        })

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(base_pool) - i - 1) / rate / 60
            print(f"  [{i+1:>4}/{len(base_pool)}] {cid:<14}  "
                  f"n_analogues={n_an:>2}  "
                  f"({rate:5.1f} q/s · ETA {eta:5.1f} min)", flush=True)

    # Persist summary
    sdf = pd.DataFrame(summary_rows)
    s_csv = out_dir / "_batch_summary.csv"
    s_jsn = out_dir / "_batch_summary.json"
    sdf.to_csv(s_csv, index=False)
    with s_jsn.open("w", encoding="utf-8") as f:
        json.dump({
            "query_year":         args.query_year,
            "window":             args.window,
            "max_candidate_year": max_cand_yr,
            "n_base_clusters":    len(base_pool),
            "n_succeeded":        int((sdf["n_analogues"] > 0).sum()),
            "n_zero_results":     int((sdf["n_analogues"] == 0).sum()),
            "config": {
                "top_k": args.top_k, "beta": list(beta), "gamma": args.gamma,
                "dominance_penalty": args.dominance_penalty,
                "tau_s": args.tau_s, "tau_eta": args.tau_eta, "tau_q": args.tau_q,
                "require_role": args.require_role, "min_size": args.min_size,
            },
        }, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"\n[done] {len(base_pool)} base clusters processed in {elapsed/60:.1f} min")
    n_succeeded = int((sdf['n_analogues'] > 0).sum()) if len(sdf) else 0
    n_zero      = int((sdf['n_analogues'] == 0).sum()) if len(sdf) else 0
    print(f"[done] {n_succeeded} succeeded, {n_zero} returned zero")
    print(f"[done] outputs in {out_dir}")
    print(f"[done] summary: {s_csv}")


if __name__ == "__main__":
    main()
