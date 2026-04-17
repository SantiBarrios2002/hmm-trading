# GitHub Issues

This file defines the actual work items for the repository. It complements `IMPLEMENTATION_PLAN.md`.

- `IMPLEMENTATION_PLAN.md` = acceptance gates / review checks
- `GITHUB_ISSUES.md` = implementation tasks / branch plan

Each issue below is intentionally PR-sized.

---

## Issue 01 — Repository scaffold and quality tooling
**Branch:** `feat/01-repo-scaffold`
**Gate:** A

### Goal
Create the initial academic repo skeleton and enforce baseline quality rules.

### Tasks
- create `src/hft_hmm/` package layout
- create `tests/`, `notebooks/`, `scripts/`, `docs/`
- add `pyproject.toml`, `requirements.txt`, `.python-version`
- add formatting and lint configuration
- add placeholder `__init__.py` files

### Deliverables
- importable package
- working lint/test commands
- minimal CI-ready structure

### Acceptance notes
See Gate A in `IMPLEMENTATION_PLAN.md`.

---

## Issue 02 — Data IO and dataset contract
**Branch:** `feat/02-data-io`
**Gate:** B

### Goal
Define a reproducible interface for loading market data.

### Tasks
- implement CSV loader
- optionally add Yahoo Finance loader
- define canonical columns: timestamp, price, volume if available
- validate schema and dtypes
- add sample fixture data for tests

### Deliverables
- `data/io.py`
- fixture dataset for tests
- schema validation tests

### Acceptance notes
See Gate B.

---

## Issue 03 — Return preprocessing and sampling frequency utilities
**Branch:** `feat/03-return-preprocessing`
**Gate:** B

### Goal
Produce clean log returns and frequency-aware preprocessing utilities.

### Tasks
- implement log-return computation
- implement resampling helper for daily / 5-minute / 1-minute data
- implement NaN and duplicate timestamp handling
- add train/test split helper preserving time order

### Deliverables
- `data/preprocessing.py`
- synthetic-data tests for returns and resampling

### Acceptance notes
See Gate B.

---

## Issue 04 — State grid and paper reference helpers
**Branch:** `feat/04-state-grid-utils`
**Gate:** B

### Goal
Support paper-aligned terminology and explicit references in code.

### Tasks
- create utilities for state labels and metadata
- create helper for storing paper reference strings in docstrings or constants
- define `paper-faithful`, `engineering approximation`, `evaluation layer` tags

### Deliverables
- `core/references.py`
- `core/state_metadata.py`

### Acceptance notes
See Gate B.

---

## Issue 05 — Piecewise linear regression baseline
**Branch:** `feat/05-plr-baseline`
**Gate:** C

### Goal
Implement the paper's naive baseline idea using segmented trends.

### Tasks
- implement simple piecewise linear regression baseline
- estimate segment slopes and residual variances
- optionally compute Durbin-Watson diagnostic
- expose outputs as baseline state summaries

### Deliverables
- `models/plr_baseline.py`
- synthetic trend segmentation tests

### Acceptance notes
See Gate C.

---

## Issue 06 — Baseline Gaussian HMM wrapper
**Branch:** `feat/06-gaussian-hmm-wrapper`
**Gate:** C

### Goal
Wrap `hmmlearn` cleanly so the model is easy to inspect and test.

### Tasks
- implement fit / predict / predict_proba interface
- expose learned means, variances, transition matrix, initial distribution
- add deterministic random state support
- document which parts are delegated to `hmmlearn`

### Deliverables
- `models/gaussian_hmm.py`
- synthetic regime-switching tests

### Acceptance notes
See Gate C.

---

## Issue 07 — Model selection over number of hidden states
**Branch:** `feat/07-model-selection-k`
**Gate:** D

### Goal
Compare candidate hidden-state counts and select `K` reproducibly.

### Tasks
- implement loop over candidate `K`
- compute log-likelihood, AIC, BIC
- return ranked summary table
- add plotting helper for selection curves

### Deliverables
- `experiments/model_selection.py`
- tests for AIC/BIC logic

### Acceptance notes
See Gate D.

---

## Issue 08 — Forward inference / filtering distribution
**Branch:** `feat/08-forward-inference`
**Gate:** D

### Goal
Implement the forward recursion used for filtering and expected return prediction.

### Tasks
- implement normalized forward recursion
- use log-safe computations where needed
- compute filtering probabilities over states
- expose expected return from probabilities and state means

### Deliverables
- `inference/forward_filter.py`
- toy-system tests with normalization checks

### Acceptance notes
See Gate D.

---

## Issue 09 — Signal generation and simple trading policy
**Branch:** `feat/09-signal-policy`
**Gate:** E

### Goal
Turn expected returns into a basic long/short trading signal.

### Tasks
- implement sign-based signal
- optionally implement thresholded signal
- align predictions with next-step return realization correctly
- document no-cost and cost-aware evaluation modes

### Deliverables
- `strategy/signals.py`
- alignment and leakage tests

### Acceptance notes
See Gate E.

---

## Issue 10 — Evaluation metrics and backtest summaries
**Branch:** `feat/10-backtest-metrics`
**Gate:** E

### Goal
Create reusable evaluation functions and report-ready summaries.

### Tasks
- implement cumulative return
- implement Sharpe ratio
- implement drawdown and hit rate
- generate compact summary DataFrame

### Deliverables
- `evaluation/metrics.py`
- metric edge-case tests

### Acceptance notes
See Gate E.

---

## Issue 11 — Rolling-window training experiment
**Branch:** `feat/11-rolling-window`
**Gate:** F

### Goal
Reflect the paper's rolling retraining setup in a reproducible experiment.

### Tasks
- implement walk-forward training loop
- support configurable window length and retrain frequency
- log chosen parameters and outputs per window

### Deliverables
- `experiments/walk_forward.py`
- small integration test with fixture data

### Acceptance notes
See Gate F.

---

## Issue 12 — Experiment configuration and reproducibility log
**Branch:** `feat/12-experiment-config`
**Gate:** F

### Goal
Make experiment settings explicit and saved.

### Tasks
- create typed configuration object or YAML-based config
- log dataset, frequency, K, seed, windows, feature params
- save experiment summaries to disk

### Deliverables
- `config/experiment_config.py`
- config validation tests

### Acceptance notes
See Gate F.

---

## Issue 13 — Volatility ratio feature
**Branch:** `feat/13-volatility-ratio`
**Gate:** G

### Goal
Implement the first side-information predictor.

### Tasks
- implement EWMA volatility estimate
- implement fast/slow volatility ratio
- parameterize decay and window settings
- add visualization helper

### Deliverables
- `features/volatility_ratio.py`
- feature tests

### Acceptance notes
See Gate G.

---

## Issue 14 — Intraday seasonality feature
**Branch:** `feat/14-seasonality-feature`
**Gate:** G

### Goal
Implement the second side-information predictor.

### Tasks
- map timestamps to time-of-day index
- compute intraday seasonal buckets or normalized index
- expose feature series for spline fitting

### Deliverables
- `features/seasonality.py`
- feature construction tests

### Acceptance notes
See Gate G.

---

## Issue 15 — Spline-based predictor fitting
**Branch:** `feat/15-spline-predictor`
**Gate:** G

### Goal
Fit spline relationships between side-information predictors and returns.

### Tasks
- fit spline to predictor vs normalized return
- allow configurable number of knots
- optionally force approximate zero-mean over support
- create evaluation plot

### Deliverables
- `features/splines.py`
- fit/evaluate tests

### Acceptance notes
See Gate G.

---

## Issue 16 — IOHMM-style transition conditioning approximation
**Branch:** `feat/16-iohmm-approx`
**Gate:** H

### Goal
Approximate the paper's idea that transitions depend on side information.

### Tasks
- define transition model `P(m_t | m_{t-1}, x_t)` using logistic or softmax regression
- train transition conditioner on inferred or decoded states
- expose time-varying transition probabilities
- document deviation from full IOHMM learning

### Deliverables
- `models/iohmm_approx.py`
- normalization and deterministic tests

### Acceptance notes
See Gate H.

---

## Issue 17 — Integrated side-information experiment
**Branch:** `feat/17-side-info-experiment`
**Gate:** H

### Goal
Compare baseline HMM against side-information-enhanced versions.

### Tasks
- run baseline experiment
- run volatility-ratio enhancement
- run seasonality enhancement
- summarize differences in metrics and plots

### Deliverables
- `scripts/run_side_info_comparison.py`
- saved figures and summary outputs

### Acceptance notes
See Gate H.

---

## Issue 18 — Visualization package for report-ready figures
**Branch:** `feat/18-visualizations`
**Gate:** I

### Goal
Standardize figures for report and presentation use.

### Tasks
- hidden-state timeline plot
- return vs signal plot
- model selection plot
- side-information spline plot
- cumulative PnL plot

### Deliverables
- `visualization/plots.py`
- smoke tests for plotting functions

### Acceptance notes
See Gate I.

---

## Issue 19 — README academic usage example
**Branch:** `docs/19-readme-usage`
**Gate:** A

### Goal
Add a minimal end-to-end usage example suitable for a reviewer.

### Tasks
- document environment setup
- document one baseline run
- document one side-info run
- mention which library routines are used and why

### Deliverables
- updated `README.md`

### Acceptance notes
See Gate A.

---

## Issue 20 — Presentation support material
**Branch:** `docs/20-presentation-support`
**Gate:** I

### Goal
Prepare concise material for preview and final presentation.

### Tasks
- create `docs/paper_notes.md`
- create `docs/experiment_log.md`
- summarize faithful vs approximate components
- draft a short technical slide narrative

### Deliverables
- presentation support docs under `docs/`

### Acceptance notes
See Gate I.
