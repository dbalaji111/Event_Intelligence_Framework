"""
diagnose_weight_toggle.py
==========================
Isolates the exact mechanism behind the Full == -DTW duplicate.
Picks one real event and one real analogue, computes the three-rank
weight twice (W_DTW=0.30 vs W_DTW=0.00), and prints every intermediate
number -- sim_score, quantile_score, dtw_score, and the final weight --
so we can see directly whether toggling the global changes anything,
rather than continuing to reason about it from the code alone.

Usage:
    python scripts/diagnose_weight_toggle.py \
        --analogues retrieval_output_v3/per_event \
        --prices    "data/oil_prices_full.xlsx"
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import Kde_engine_v3 as eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analogues", required=True)
    ap.add_argument("--prices", required=True)
    args = ap.parse_args()

    prices = eng.PriceData(Path(args.prices), None)

    json_files = sorted(Path(args.analogues).glob("*_analogues.json"))
    jf = json_files[0]
    data = json.load(jf.open())
    base_date = pd.to_datetime(data["base_date"])
    analogues = data["analogues"]
    print(f"Event: {jf.stem}, base_date={base_date.date()}, "
          f"{len(analogues)} analogues in file\n")

    # replicate run_event_v3's own filtering so we use a REAL valid_pool
    valid_pool = []
    for a in analogues:
        a_date = pd.to_datetime(a.get("date", ""), errors="coerce")
        if pd.isna(a_date) or a_date >= base_date:
            continue
        if eng.COVID_START <= a_date <= eng.COVID_END:
            continue
        valid_pool.append(a)
    print(f"valid_pool size: {len(valid_pool)}\n")

    def get_path(rec):
        path = []
        for node in eng.PRICE_NODES:
            if node == 0:
                path.append(0.0)
            else:
                pp = rec.get("price_path", {})
                path.append(float(pp.get(str(node), pp.get(node, 0)) or 0) if pp else 0.0)
        return np.array(path, dtype=float)

    base_path_raw = np.zeros(len(eng.PRICE_NODES))
    for node_idx, node in enumerate(eng.PRICE_NODES):
        r = prices.forward_return(base_date, abs(node) if node != 0 else 1)
        if r is not None:
            base_path_raw[node_idx] = r if node > 0 else -r if node < 0 else 0.0
    print(f"base_path_raw: {base_path_raw}\n")

    all_dtw_vals = [eng.dtw_increments(base_path_raw, get_path(a)) for a in valid_pool]
    print(f"all_dtw_vals (first 8): {[round(v,3) for v in all_dtw_vals[:8]]}")
    print(f"median dtw: {np.median(all_dtw_vals):.4f}\n")

    # pick analogue index 5 (arbitrary, avoid edge index 0)
    a = valid_pool[5]
    a_path = get_path(a)
    base_quantile = "40%-60%"
    a_quantile = str(a.get("quantile_cat", a.get("quantile", ""))).strip()
    sim = float(a.get("similarity", 0.5))

    print(f"Testing analogue: {a.get('event_id')}, "
          f"similarity={sim}, quantile_cat={a_quantile!r}")
    print(f"a_path: {a_path}\n")

    for w_dtw_test in [0.30, 0.00]:
        eng.W_DTW = w_dtw_test
        eng.W_SIM = 0.50
        eng.W_QUANTILE = 0.20

        print(f"--- eng.W_DTW = {eng.W_DTW}, eng.W_SIM = {eng.W_SIM}, "
              f"eng.W_QUANTILE = {eng.W_QUANTILE} ---")

        w = eng.compute_three_rank_weight(
            similarity=sim,
            base_quantile=base_quantile,
            analogue_quantile=a_quantile,
            base_path=base_path_raw,
            analogue_path=a_path,
            all_dtw=all_dtw_vals,
        )
        print(f"  compute_three_rank_weight() returned: {w:.6f}\n")


if __name__ == "__main__":
    main()
