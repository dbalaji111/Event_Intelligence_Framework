<!--
  TIMING REMINDER — READ BEFORE PUSHING THIS PUBLIC
  IJF submission is under double-blind review. Publishing this file to a
  personally-identified GitHub account before a decision is reached risks
  de-anonymizing the submission to a reviewer who searches for the method.
  Safe to push publicly: after the IJF decision, OR if a reviewer requests
  code during review, provide it via an anonymized link / the journal's own
  supplementary-material system instead of this repo. FRL has no such
  restriction once its own review process confirms non-blind handling.
  Delete this comment once it's safe to publish.
-->

# Brahmanda — Event-Conditioned Bayesian Forecasting for Commodity Markets

Retrieval-augmented, event-conditioned Bayesian forecasting for crude oil and gold, combining BERT-based semantic retrieval, LLM-driven structured event extraction, and kernel density estimation to produce full posterior forecast distributions rather than point estimates.

This repository accompanies two papers:

| Paper | Status | Core finding |
|---|---|---|
| *Event-Conditioned Bayesian Forecasting via Selective Analogue Retrieval* | Under review, International Journal of Forecasting | Beats the GARCH family at every horizon tested (7/15/30 days), with narrower, not wider, prediction intervals |
| *Does Bimodality Predict Extreme Moves? Evidence from Event-Conditioned Crude Oil Forecasts* | Prepared for submission, Finance Research Letters | Bimodal posteriors predict *lower*, not higher, extreme-move odds — evidence for scenario differentiation, not tail risk |

---

## Table of Contents

- [Data Availability and Licensing](#data-availability-and-licensing)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Reproducing the IJF Paper](#reproducing-the-ijf-paper)
- [Reproducing the FRL Paper](#reproducing-the-frl-paper)
- [The Dashboard](#the-dashboard)
- [Known Limitations](#known-limitations)
- [Citation](#citation)
- [License](#license)

---

## Data Availability and Licensing

**What is included:** BERT embeddings, LLM-extracted structured event attributes (entity, causal links, polarity, modality, event type), model outputs, evaluation code, and results CSVs for every table in both papers.

**What is NOT included:** the underlying news article text. Source articles are licensed from Refinitiv/Factiva and WSJ archives; redistribution of the raw text is prohibited under those licenses. All derived artifacts (embeddings, extracted attributes, labels) are original work and are shared here.

If you need to reconstruct the full corpus including source text, you will need your own Refinitiv/Factiva access; this repository provides the extraction and modeling pipeline to apply to it.

---

## Repository Structure

```
.
├── scripts/                        # All analysis and pipeline code
│   ├── kde_engine_v4.py            # Core forecasting engine (IJF)
│   ├── garch_family_benchmark.py   # GARCH/EGARCH/GJR-GARCH benchmarks (IJF)
│   ├── compute_fei_by_event_type.py
│   ├── compute_pinball_dm_all_benchmarks.py
│   ├── generate_figures.py
│   ├── frl_master_analysis.py      # FRL: core logistic regression + Table 1
│   ├── frl_robustness_v3.py        # FRL: 12-cell horizon x threshold sweep
│   ├── frl_firth_standalone.py     # FRL: separation-robust estimation
│   ├── frl_text_validation_v2.py   # FRL: modality/text-based check
│   ├── frl_specification_checks.py # FRL: clustering, probit, event-type controls
│   ├── frl_mixture_artifact_check.py
│   ├── frl_real_separation_check.py
│   ├── calibration_table.py        # FRL: reliability/calibration check
│   ├── ews_construction.py         # Early-warning signal validation
│   ├── retrieval_engine.py         # Analogue retrieval, temporal-exclusion enforced
│   ├── Retrieval_engine_basic.py
│   └── redflag_api_v5.py           # Dashboard backend (FastAPI)
├── results_run/kde_output_v4/      # Forecasting engine outputs, per horizon
├── memory_2001_2023/                # Training corpus (derived attributes only)
├── memory_2024_2026/                # Test corpus (derived attributes only)
├── frl_v4/                          # FRL analysis outputs
├── redflag_dashboard_v6.html        # Dashboard frontend
├── REPRODUCIBILITY.md               # Full table-by-table script/argument reference
└── README.md                        # This file
```

---

## Installation

```bash
git clone <repo-url>
cd brahmanda
pip install -r requirements.txt --break-system-packages
```

Core dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `faiss-cpu`, `tqdm`.

---

## Reproducing the IJF Paper

The forecasting engine is run once per horizon:

```bash
python scripts/kde_engine_v4.py \
  --analogues retrieval_output_v3/per_event \
  --prices "data/oil_prices_full.xlsx" \
  --ovx "crude_oil_data/oil_price_data/CBOE Crude Oil Volatility Historical Data.csv" \
  --output results_run/kde_output_v4 --horizon 15
```
Repeat with `--horizon 7` and `--horizon 30`. This produces the results CSVs that every downstream table and figure reads from.

For the full table-by-table breakdown (which script, which arguments, which table), see [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md).

---

## Reproducing the FRL Paper

All FRL analyses read directly from the `kde_engine_v4.py` output above. The core pipeline:

```bash
# Table 1 and the main logistic regression
python scripts/frl_master_analysis.py \
  --kde_results results_run/kde_output_v4/kde_v4_results_h15.csv \
  --metadata memory_2024_2026/df_major_metadata_2024_2026.csv \
  --output frl_v4

# Table 3: full 12-cell horizon x threshold robustness sweep
python scripts/frl_robustness_v3.py \
  --kde_h7  results_run/kde_output_v4/kde_v4_results_h7.csv \
  --kde_h15 results_run/kde_output_v4/kde_v4_results_h15.csv \
  --kde_h30 results_run/kde_output_v4/kde_v4_results_h30.csv \
  --output frl_v4

# Specification checks: clustering, probit, non-linear dispersion, event-type control
python scripts/frl_specification_checks.py \
  --kde_results results_run/kde_output_v4/kde_v4_results_h15.csv \
  --metadata memory_2024_2026/df_major_metadata_2024_2026.csv \
  --extreme_threshold 10.0
```

Full argument reference for every table, including the Firth's-correction and mechanical-artifact-check scripts, is in [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md).

---

## The Dashboard

A live terminal UI (`redflag_dashboard_v6.html`) backed by a FastAPI server (`redflag_api_v5.py`):

```bash
BASE_PATH="." \
OVX_PATH="crude_oil_data/oil_price_data/CBOE Crude Oil Volatility Historical Data.csv" \
CORPUS_PATH=scorer_corpus.jsonl \
PYTHONPATH=scripts \
uvicorn scripts.redflag_api_v5:app --port 8000 --workers 1
```

Then open `redflag_dashboard_v6.html` in a browser (it is hardcoded to `http://localhost:8000`).

> **Note:** the walk-forward backtest panel has a known path-configuration bug — see Known Limitations below.

---

## Known Limitations

Documented here in the interest of honest reproducibility, not buried:

- **IJF, Table 9 (tau sensitivity):** the source script for this table was not recovered; the table as it stands has an internal inconsistency with Table 4 and should not be cited without regenerating it.
- **IJF, `kde_baselines.py`:** fails with all-NaN output; root cause not yet resolved.
- **IJF, `compute_proper_scores.py` (CRPS):** provisional; treat CRPS figures in the paper as such.
- **FRL, causal-chain-length heuristic:** attempted and abandoned — the `Causal Links` field's actual text format doesn't match the delimiter-based parsing the heuristic assumed.
- **FRL, bimodal tail-weight calibration:** inconclusive. Only 2 of 80 bimodal events had a direction-matched extreme outcome, too sparse to distinguish a real pattern from noise.
- **Dashboard backtest:** `redflag_backtest_layer.py` looks for `kde_output_v3*`-named files; all current data uses `v4` naming. This is the most likely cause of live "Backtest error" messages. Not yet patched.

---

## Citation

If you use this code or these results, please cite both papers (BibTeX to be added upon acceptance/DOI assignment):

```bibtex
@article{author_ijf_2026,
  title={Event-Conditioned Bayesian Forecasting via Selective Analogue Retrieval},
  journal={International Journal of Forecasting},
  year={2026},
  note={Under review}
}

@article{author_frl_2026,
  title={Does Bimodality Predict Extreme Moves? Evidence from Event-Conditioned Crude Oil Forecasts},
  journal={Finance Research Letters},
  year={2026},
  note={Under review}
}
```

---

## License

Code: MIT License (see `LICENSE`). Data: derived artifacts only, see [Data Availability and Licensing](#data-availability-and-licensing) above. Raw source article text is not included and is not covered by this repository's license.
