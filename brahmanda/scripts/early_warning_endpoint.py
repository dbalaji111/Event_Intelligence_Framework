"""
early_warning_endpoint.py
============================
Drop-in addition to redflag_api_v5.py. Adds a real, cross-validated
early-warning classification based on kde_std (dispersion),
replacing the earlier bimodal-flag-based BUY STRADDLE trigger --
bimodality was shown (FRL analysis) to NOT function as a tail-risk
alarm; kde_std does (AUC=0.826, cross-validated, h=15d).

THRESHOLDS BELOW ARE PROVISIONAL -- calibrated only on h=15d.
Confirm against h=7d and h=30d reruns before treating as final;
update THRESHOLDS dict below once confirmed.

Integration: import classify_early_warning() and call it wherever
the current /analyze or /kde/{event_id} response is assembled,
adding the result to the response payload.
"""

# Calibrated on h=15d, N=406, cross-validated 5-fold (AUC=0.826).
# TODO: confirm/update after h7 and h30 reruns.
THRESHOLDS = {
    "h15": {
        "high_confidence": 12.221,  # 95th percentile: precision 76.2%, recall 24.2%
        "watchlist": 8.435,          # 60th percentile: precision 30.7%, recall 75.8%
    },
    # "h7": {...},   # fill in after rerun
    # "h30": {...},  # fill in after rerun
}


def classify_early_warning(kde_std: float, horizon: str = "h15") -> dict:
    """
    Returns a tiered early-warning classification based on posterior
    dispersion (kde_std), the cross-validated signal -- NOT the
    bimodal flag, which was shown not to predict extreme moves.
    """
    thresholds = THRESHOLDS.get(horizon)
    if thresholds is None:
        return {
            "tier": "UNCALIBRATED",
            "note": f"No validated threshold for horizon '{horizon}' yet. "
                    f"Only h15 is currently calibrated."
        }

    if kde_std >= thresholds["high_confidence"]:
        tier = "HIGH_CONFIDENCE_ALERT"
        note = "Top 5% dispersion. ~76% precision, ~24% recall on extreme moves (h15, cross-validated)."
    elif kde_std >= thresholds["watchlist"]:
        tier = "WATCHLIST"
        note = "Elevated dispersion. ~31% precision, ~76% recall on extreme moves (h15, cross-validated)."
    else:
        tier = "NORMAL"
        note = "Dispersion below watchlist threshold."

    return {
        "tier": tier,
        "kde_std": round(kde_std, 3),
        "threshold_high_confidence": thresholds["high_confidence"],
        "threshold_watchlist": thresholds["watchlist"],
        "note": note,
        "signal_source": "kde_std (posterior dispersion) -- cross-validated AUC=0.826, h15",
        "not_used": "kde_is_bimodal -- confirmed NOT a tail-risk signal (FRL analysis); "
                    "bimodal events show LOWER extreme-move odds, controlling for dispersion.",
    }


# Example integration point in redflag_api_v5.py's /analyze or /kde/{event_id}
# handler, after the posterior stats are computed:
#
#   from early_warning_endpoint import classify_early_warning
#   ...
#   ews = classify_early_warning(stats["kde_std"], horizon="h15")
#   response["early_warning"] = ews
