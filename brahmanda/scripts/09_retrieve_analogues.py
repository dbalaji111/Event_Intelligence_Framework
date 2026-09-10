"""
09_retrieve_analogues.py
=========================
Multi-stage retrieval cascade for finding similar historical event-arc clusters.
This is the core retrieval engine that the API will wrap.

Stages (per the EventCast PDF, Section 2 + Stage 2 of the retrieval funnel):

  Stage 1  Semantic similarity   - cosine on BERT centroids
                                   keep candidates with s_sem >= tau_s
  Stage 2  Strict named-entity    - entity Jaccard >= tau_eta AND optional
                                   polarity / event-type match
  Stage 3  Quantile / regime      - d_qtl <= tau_q (Hamming-like over the
                                   {annual, monthly, cum_7d/15d/30d} bands)
  Stage 4  Propagation profile    - DTW distance between rho_base(t) and rho_i(t)
                                   keep top-K by smallest DTW

  Stage 2.5 Dominance / confounder - count concurrent clusters during candidate's
                                   span; compute dominance share and C(t) flag

  Final ranking: composite weight per the PDF formula
        w*_i = beta_1 * s_sem + beta_2 * s_prop + beta_3 * (1 - d_qtl)
               - gamma * C(t) - (1 - dominance) * dominance_penalty
        defaults: beta = (0.5, 0.3, 0.2), gamma = 0.5, dominance_penalty = 0.3

Usage as CLI
------------
    python 09_retrieve_analogues.py \\
        --cluster_index clusters_full_history/cluster_memory.parquet \\
        --base_cluster_id c15d_00042 \\
        --top_k 20 \\
        --output_dir clusters_full_history/retrievals

Usage as a library (for the API)
--------------------------------
    from retrieve_analogues import load_index, retrieve_analogues
    idx = load_index("clusters_full_history/cluster_memory.parquet")
    result = retrieve_analogues("c15d_00042", idx, top_k=20)
    # result is a dict ready to json.dump
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DEFAULT THRESHOLDS / WEIGHTS  (from the EventCast PDF, Section 2)
# ---------------------------------------------------------------------------
DEFAULT_BETA              = (0.5, 0.3, 0.2)   # (sem, prop, qtl)
DEFAULT_GAMMA             = 0.5               # confounder penalty
DEFAULT_DOMINANCE_PENALTY = 0.3               # how much low-dominance hurts
DEFAULT_TAU_S             = 0.6               # min semantic cosine sim
DEFAULT_TAU_ETA           = 0.2               # min entity Jaccard
DEFAULT_TAU_Q             = 0.65              # max quantile distance (1 = totally different)
DEFAULT_TOP_K             = 20

# Stage funnel pool sizes
POOL_AFTER_SEMANTIC = 200
POOL_AFTER_ENTITY   = 60
POOL_AFTER_QUANTILE = 30


# ---------------------------------------------------------------------------
# INDEX LOADING
# ---------------------------------------------------------------------------
def load_index(path: Path | str) -> pd.DataFrame:
    """Load cluster_memory.parquet (or .pkl fallback). Decodes the JSON-encoded
    bundle columns back into Python objects so they're usable downstream."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_pickle(path)
    # Decode JSON columns into native dicts/lists
    json_cols = ["entity_bag", "polarity_dist", "evtype_dist",
                 "quantile_sig", "propagation_prof", "realised_path", "members"]
    for c in json_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda s: json.loads(s) if isinstance(s, str) else s)
    return df


# ---------------------------------------------------------------------------
# DISTANCES / SIMILARITIES
# ---------------------------------------------------------------------------
def cosine_sim_batch(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Cosine similarity of query (D,) against matrix of N candidates (N,D)."""
    q = query_vec.astype(np.float32)
    qn = q / (np.linalg.norm(q) + 1e-12)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return (mn @ qn).astype(np.float32)


# Generic actors that appear in 60%+ of articles and carry no discriminative power.
# Override via --entity_stop_list "comma,separated,list".
DEFAULT_ENTITY_STOP_LIST = frozenset({
    "opec", "gulf oil producers", "usa", "european union", "europe",
    "united states", "u.s.", "us", "saudi arabia", "russia",
    "international energy agency", "iea",
})


def entity_jaccard(bag_a: Dict[str, int], bag_b: Dict[str, int],
                   stop_list: Optional[frozenset] = None) -> float:
    """Jaccard on entity bags, optionally with a stop-list to strip generic
    high-frequency actors that pollute every cluster's bag."""
    sa, sb = set(bag_a or {}), set(bag_b or {})
    if stop_list:
        sa = sa - stop_list
        sb = sb - stop_list
    if not sa and not sb:
        return 0.0   # both bags are generic-only -> no discriminative match
    inter = len(sa & sb)
    uni = len(sa | sb)
    return inter / uni if uni else 0.0


def quantile_distance(sig_a: Dict[str, str], sig_b: Dict[str, str]) -> float:
    """Hamming-like distance over the quantile signature dict.
    Returns value in [0, 1]; 0 = perfect match, 1 = all bands differ."""
    keys = ("annual", "monthly", "cum_7d", "cum_15d", "cum_30d")
    diffs, n = 0, 0
    for k in keys:
        a = (sig_a or {}).get(k, "UNK")
        b = (sig_b or {}).get(k, "UNK")
        if a == "UNK" and b == "UNK":
            continue
        n += 1
        if a != b:
            diffs += 1
    return (diffs / n) if n else 1.0


def dtw_distance(a: List[float], b: List[float]) -> float:
    """O(n*m) DTW with squared error cost. Returns sqrt of accumulated cost."""
    if not a or not b:
        return float("inf")
    a_arr, b_arr = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    n, m = len(a_arr), len(b_arr)
    D = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (a_arr[i - 1] - b_arr[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


# ---------------------------------------------------------------------------
# Stage 2.5 — DOMINANCE / CONFOUNDER analysis
# ---------------------------------------------------------------------------
def compute_dominance(candidate_row: pd.Series,
                      cluster_index: pd.DataFrame) -> Tuple[float, int, int]:
    """Returns (dominance_share, concurrent_count, larger_concurrent_count).
    Same window_days as the candidate; same ROLE doesn't matter (we count ALL)."""
    first = pd.to_datetime(candidate_row["first_seen"])
    last  = pd.to_datetime(candidate_row["last_seen"])
    same_w = cluster_index[
        (cluster_index["window_days"] == candidate_row["window_days"]) &
        (cluster_index["cluster_id"] != candidate_row["cluster_id"])
    ]
    fs = pd.to_datetime(same_w["first_seen"])
    ls = pd.to_datetime(same_w["last_seen"])
    overlap = (fs <= last) & (ls >= first)
    concurrent = same_w[overlap]
    n_concurrent = len(concurrent)
    if n_concurrent == 0:
        return 1.0, 0, 0
    larger = (concurrent["size"] > candidate_row["size"]).sum()
    total_size = candidate_row["size"] + concurrent["size"].sum()
    dominance = float(candidate_row["size"]) / float(total_size) if total_size else 1.0
    return dominance, int(n_concurrent), int(larger)


# ---------------------------------------------------------------------------
# THE RETRIEVAL CASCADE
# ---------------------------------------------------------------------------
def retrieve_analogues(
    base_cluster_id: str,
    cluster_index: pd.DataFrame,
    *,
    top_k: int = DEFAULT_TOP_K,
    beta: Tuple[float, float, float] = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    dominance_penalty: float = DEFAULT_DOMINANCE_PENALTY,
    tau_s: float = DEFAULT_TAU_S,
    tau_eta: float = DEFAULT_TAU_ETA,
    tau_q: float = DEFAULT_TAU_Q,
    same_window_only: bool = True,
    exclude_overlapping: bool = True,
    require_role: Optional[str] = None,
    min_size: int = 0,
    max_candidate_year: Optional[int] = None,
    min_candidate_year: Optional[int] = None,
    recency_tau: Optional[float] = None,
    max_per_era: Optional[int] = None,
    era_size_years: int = 5,
    entity_stop_list: Optional[frozenset] = None,
) -> Dict[str, Any]:
    """Run the four-stage retrieval cascade and return a JSON-ready dict.

    Parameters mirror the PDF defaults; tweak for sensitivity studies.
    `same_window_only`: only consider candidates with the same window_days
        (compares apples to apples).
    `exclude_overlapping`: drop candidates whose date span overlaps the base
        (avoids self-similarity / look-ahead).
    """
    # Locate base
    base_mask = cluster_index["cluster_id"] == base_cluster_id
    if not base_mask.any():
        raise KeyError(f"base_cluster_id {base_cluster_id} not found in index")
    base = cluster_index[base_mask].iloc[0]

    # Candidate pool
    cand = cluster_index[cluster_index["cluster_id"] != base_cluster_id].copy()
    if same_window_only:
        cand = cand[cand["window_days"] == base["window_days"]]
    if exclude_overlapping:
        b_first = pd.to_datetime(base["first_seen"])
        b_last  = pd.to_datetime(base["last_seen"])
        cf = pd.to_datetime(cand["first_seen"])
        cl = pd.to_datetime(cand["last_seen"])
        cand = cand[~((cf <= b_last) & (cl >= b_first))]
    if require_role is not None:
        cand = cand[cand["role"] == require_role]
    if min_size > 0:
        cand = cand[cand["size"] >= min_size]
    if max_candidate_year is not None or min_candidate_year is not None:
        cand_yr = pd.to_datetime(cand["first_seen"]).dt.year
        if max_candidate_year is not None:
            cand = cand[cand_yr <= max_candidate_year]
            cand_yr = pd.to_datetime(cand["first_seen"]).dt.year
        if min_candidate_year is not None:
            cand = cand[cand_yr >= min_candidate_year]
    cand = cand.reset_index(drop=True)

    if cand.empty:
        return _empty_result(base, "no candidates after pre-filtering")

    # ---- Stage 1: SEMANTIC ----
    base_vec = np.asarray(base["centroid"], dtype=np.float32)
    cand_mat = np.stack([np.asarray(v, dtype=np.float32)
                         for v in cand["centroid"].tolist()])
    s_sem = cosine_sim_batch(base_vec, cand_mat)
    cand["s_sem"] = s_sem
    # If recency_tau is set, weight s_sem at Stage 1 entry so the candidate
    # pool spans multiple eras instead of collapsing into one BERT-space blob
    if recency_tau is not None and recency_tau > 0:
        base_year = pd.to_datetime(base["first_seen"]).year
        cand_yr_s1 = pd.to_datetime(cand["first_seen"]).dt.year
        age_yrs_s1 = (base_year - cand_yr_s1).astype(float).clip(lower=0)
        cand["s_sem_recency"] = cand["s_sem"] * np.exp(-age_yrs_s1 / float(recency_tau))
        after1 = cand[cand["s_sem"] >= tau_s].nlargest(POOL_AFTER_SEMANTIC, "s_sem_recency")
    else:
        after1 = cand[cand["s_sem"] >= tau_s].nlargest(POOL_AFTER_SEMANTIC, "s_sem")
    n_after1 = len(after1)
    if after1.empty:
        return _empty_result(base, f"no candidates passed semantic filter (tau_s={tau_s})",
                             stages={"stage1_semantic": 0})

    # ---- Stage 2: STRICT ENTITY (+ optional polarity dominance) ----
    base_bag = base["entity_bag"]
    sl = entity_stop_list if entity_stop_list is not None else None
    after1["jaccard"] = after1["entity_bag"].apply(
        lambda b: entity_jaccard(base_bag, b, stop_list=sl)
    )
    after2 = after1[after1["jaccard"] >= tau_eta].nlargest(POOL_AFTER_ENTITY, "jaccard")
    n_after2 = len(after2)
    if after2.empty:
        return _empty_result(base, f"no candidates passed entity filter (tau_eta={tau_eta})",
                             stages={"stage1_semantic": n_after1, "stage2_entity": 0})

    # ---- Stage 3: QUANTILE / REGIME ----
    base_qsig = base["quantile_sig"]
    after2["d_qtl"] = after2["quantile_sig"].apply(
        lambda q: quantile_distance(base_qsig, q)
    )
    after3 = after2[after2["d_qtl"] <= tau_q].nsmallest(POOL_AFTER_QUANTILE, "d_qtl")
    n_after3 = len(after3)
    if after3.empty:
        return _empty_result(base, f"no candidates passed quantile filter (tau_q={tau_q})",
                             stages={"stage1_semantic": n_after1,
                                     "stage2_entity": n_after2,
                                     "stage3_quantile": 0})

    # ---- Stage 4: PROPAGATION DTW ----
    base_prof = base["propagation_prof"]
    after3["dtw_prop"] = after3["propagation_prof"].apply(
        lambda p: dtw_distance(base_prof, p)
    )
    # Convert DTW to a similarity in [0,1] for s_prop term
    max_dtw = after3["dtw_prop"].replace([np.inf], np.nan).max()
    if pd.notna(max_dtw) and max_dtw > 0:
        after3["s_prop"] = 1.0 - (after3["dtw_prop"] / max_dtw)
    else:
        after3["s_prop"] = 0.0
    after4 = after3.nlargest(top_k * 2, "s_prop")
    n_after4 = len(after4)

    # ---- Stage 2.5: DOMINANCE ----
    dom_vals, n_conc, n_larger = [], [], []
    for _, row in after4.iterrows():
        d, nc, nl = compute_dominance(row, cluster_index)
        dom_vals.append(d); n_conc.append(nc); n_larger.append(nl)
    after4["dominance"]            = dom_vals
    after4["concurrent_count"]     = n_conc
    after4["larger_concurrent"]    = n_larger
    after4["C_t"] = (after4["larger_concurrent"] > 0).astype(int)

    # ---- Composite ranking per PDF formula ----
    b1, b2, b3 = beta
    after4["composite_raw"] = (
        b1 * after4["s_sem"]
        + b2 * after4["s_prop"]
        + b3 * (1.0 - after4["d_qtl"])
        - gamma * after4["C_t"]
        - (1.0 - after4["dominance"]) * dominance_penalty
    )

    # ---- Recency penalty: exp(-age_years / recency_tau) ----
    base_year = pd.to_datetime(base["first_seen"]).year
    cand_yr = pd.to_datetime(after4["first_seen"]).dt.year
    age_yrs = (base_year - cand_yr).astype(float).clip(lower=0)
    if recency_tau is not None and recency_tau > 0:
        after4["recency_factor"] = np.exp(-age_yrs / float(recency_tau))
    else:
        after4["recency_factor"] = 1.0
    after4["composite_weight"] = after4["composite_raw"] * after4["recency_factor"]
    after4["age_years"] = age_yrs

    # ---- Era-stratified top-K (optional diversity constraint) ----
    if max_per_era is not None and max_per_era > 0:
        after4["era"] = (cand_yr // era_size_years) * era_size_years
        # Keep top max_per_era per era, then take top_k of those
        ranked = (after4.sort_values("composite_weight", ascending=False)
                        .groupby("era", sort=False).head(max_per_era)
                        .nlargest(top_k, "composite_weight")
                        .reset_index(drop=True))
    else:
        ranked = after4.nlargest(top_k, "composite_weight").reset_index(drop=True)

    # ---- Build JSON-ready result ----
    analogues = []
    for rank, (_, r) in enumerate(ranked.iterrows(), start=1):
        analogues.append({
            "rank": rank,
            "cluster_id": r["cluster_id"],
            "label": r["label"],
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
            "size": int(r["size"]),
            "spread_days": int(r.get("spread_days", 0)),
            "role": r["role"],
            "scores": {
                "s_sem":            round(float(r["s_sem"]), 4),
                "entity_jaccard":   round(float(r["jaccard"]), 4),
                "d_qtl":            round(float(r["d_qtl"]), 4),
                "dtw_prop":         round(float(r["dtw_prop"]), 4),
                "s_prop":           round(float(r["s_prop"]), 4),
                "dominance":        round(float(r["dominance"]), 4),
                "concurrent_count": int(r["concurrent_count"]),
                "C_t_confounder":   int(r["C_t"]),
                "age_years":        round(float(r.get("age_years", 0.0)), 2),
                "recency_factor":   round(float(r.get("recency_factor", 1.0)), 4),
                "composite_raw":    round(float(r["composite_raw"]), 4),
                "composite_weight": round(float(r["composite_weight"]), 4),
            },
            "factors": {
                "entity_bag":       r["entity_bag"],
                "polarity_dist":    r["polarity_dist"],
                "evtype_dist":      r["evtype_dist"],
                "quantile_sig":     r["quantile_sig"],
                "realised_path":    r["realised_path"],
                "realised_path_len": int(r["realised_path_len"]),
                "members":          r["members"],
            },
        })

    return {
        "query": _base_summary(base),
        "config": {
            "top_k": top_k,
            "beta": list(beta),
            "gamma": gamma,
            "dominance_penalty": dominance_penalty,
            "thresholds": {"tau_s": tau_s, "tau_eta": tau_eta, "tau_q": tau_q},
            "same_window_only": same_window_only,
            "exclude_overlapping": exclude_overlapping,
        },
        "stage_funnel": {
            "stage1_semantic":    n_after1,
            "stage2_entity":      n_after2,
            "stage3_quantile":    n_after3,
            "stage4_propagation": n_after4,
            "final_ranked":       len(ranked),
        },
        "analogues": analogues,
    }


def _base_summary(base: pd.Series) -> Dict[str, Any]:
    return {
        "cluster_id":      base["cluster_id"],
        "label":           base["label"],
        "window_days":     int(base["window_days"]),
        "first_seen":      base["first_seen"],
        "last_seen":       base["last_seen"],
        "size":            int(base["size"]),
        "spread_days":     int(base.get("spread_days", 0)),
        "role":            base["role"],
        "entity_bag":      base["entity_bag"],
        "polarity_dist":   base["polarity_dist"],
        "evtype_dist":     base["evtype_dist"],
        "quantile_sig":    base["quantile_sig"],
        "members":         base["members"],
    }


def _parse_stop_list(arg_value: Optional[str]) -> Optional[frozenset]:
    """Translate --entity_stop_list CLI value into the right object."""
    if arg_value is None:
        return None
    s = arg_value.strip()
    if s == "":
        return frozenset()                  # explicitly disable stop-list
    if s.upper() == "DEFAULT":
        return DEFAULT_ENTITY_STOP_LIST
    custom = {tok.strip().lower() for tok in s.split(",") if tok.strip()}
    return frozenset(custom)


def _empty_result(base: pd.Series, reason: str,
                  stages: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    return {
        "query": _base_summary(base),
        "stage_funnel": stages or {},
        "analogues": [],
        "reason_no_results": reason,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster_index", required=True,
                    help="cluster_memory.parquet from script 08")
    ap.add_argument("--base_cluster_id", required=True,
                    help="cluster_id of the query (e.g. c15d_00042)")
    ap.add_argument("--output_dir", default=None,
                    help="defaults to <cluster_index dir>/retrievals/")
    ap.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--beta", default=",".join(str(b) for b in DEFAULT_BETA),
                    help="three weights: sem,prop,qtl  (default 0.5,0.3,0.2)")
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--dominance_penalty", type=float,
                    default=DEFAULT_DOMINANCE_PENALTY)
    ap.add_argument("--tau_s",   type=float, default=DEFAULT_TAU_S)
    ap.add_argument("--tau_eta", type=float, default=DEFAULT_TAU_ETA)
    ap.add_argument("--tau_q",   type=float, default=DEFAULT_TAU_Q)
    ap.add_argument("--allow_cross_window", action="store_true",
                    help="if set, allow candidates with different window_days")
    ap.add_argument("--allow_overlap", action="store_true",
                    help="if set, allow candidates that overlap base in time")
    ap.add_argument("--require_role", default=None,
                    help="restrict candidates to one role (e.g. REGIME-MARKER)")
    ap.add_argument("--max_candidate_year", type=int, default=None,
                    help="only consider analogues with first_seen.year <= this "
                         "(e.g. 2022 to forbid 2023 leakage when querying 2023)")
    ap.add_argument("--min_candidate_year", type=int, default=None,
                    help="only consider analogues with first_seen.year >= this")
    ap.add_argument("--recency_tau", type=float, default=None,
                    help="exponential decay half-life in YEARS for recency "
                         "weighting; e.g. 8 means a 16-year-old analogue gets "
                         "weight ~0.13 of a contemporary one. None = no decay.")
    ap.add_argument("--max_per_era", type=int, default=None,
                    help="cap analogues per era_size_years window in the top-K "
                         "(diversity constraint). Default None = no cap.")
    ap.add_argument("--era_size_years", type=int, default=5,
                    help="size in years of an 'era' for --max_per_era (default 5)")
    ap.add_argument("--entity_stop_list", default=None,
                    help="comma-separated entities to strip from BOTH bags before "
                         "computing entity Jaccard. Use 'DEFAULT' for the built-in "
                         "list of generic actors (opec, gulf oil producers, usa, "
                         "european union, etc.). Use empty string '' to disable.")
    args = ap.parse_args()

    idx_path = Path(args.cluster_index).resolve()
    out_dir = (Path(args.output_dir).resolve() if args.output_dir
               else idx_path.parent / "retrievals")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {idx_path.name}")
    t0 = time.time()
    idx = load_index(idx_path)
    print(f"[load]   {len(idx)} clusters in {time.time()-t0:.1f}s")

    beta = tuple(float(x) for x in args.beta.split(","))
    if len(beta) != 3:
        sys.exit("ERROR: --beta must have exactly 3 numbers")

    print(f"[query] base_cluster_id = {args.base_cluster_id}")
    t0 = time.time()
    result = retrieve_analogues(
        args.base_cluster_id, idx,
        top_k=args.top_k,
        beta=beta, gamma=args.gamma,
        dominance_penalty=args.dominance_penalty,
        tau_s=args.tau_s, tau_eta=args.tau_eta, tau_q=args.tau_q,
        same_window_only=not args.allow_cross_window,
        exclude_overlapping=not args.allow_overlap,
        require_role=args.require_role,
        max_candidate_year=args.max_candidate_year,
        min_candidate_year=args.min_candidate_year,
        recency_tau=args.recency_tau,
        max_per_era=args.max_per_era,
        era_size_years=args.era_size_years,
        entity_stop_list=_parse_stop_list(args.entity_stop_list),
    )
    print(f"[query] retrieval finished in {time.time()-t0:.1f}s")

    out_json = out_dir / f"analogues_{args.base_cluster_id}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"[save] {out_json}  ({out_json.stat().st_size/1e3:.1f} kB)")

    # Brief stdout summary
    print(f"\n--- Funnel ---")
    for k, v in result.get("stage_funnel", {}).items():
        print(f"  {k:<22} {v}")
    print(f"\n--- Top-{min(args.top_k, len(result.get('analogues', [])))} analogues ---")
    for a in result.get("analogues", [])[:args.top_k]:
        s = a["scores"]
        print(f"  #{a['rank']:>2}  {a['cluster_id']:<14}  "
              f"w={s['composite_weight']:+.3f}  age={s.get('age_years',0):.0f}y  "
              f"sem={s['s_sem']:.2f}  jac={s['entity_jaccard']:.2f}  "
              f"qtl={s['d_qtl']:.2f}  dom={s['dominance']:.2f}  "
              f"({a['first_seen'][:10]} -> {a['last_seen'][:10]})  -> {a['label'][:60]}")
    if not result.get("analogues"):
        print(f"  (none) reason: {result.get('reason_no_results', '?')}")


if __name__ == "__main__":
    main()
