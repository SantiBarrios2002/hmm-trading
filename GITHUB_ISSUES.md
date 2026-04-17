# GitHub Issues Plan

This file can be copied into GitHub issues almost verbatim. Each issue is intentionally PR-sized.

---

## Issue 01 — Repository scaffold and quality tooling
**Branch:** `feat/01-repo-scaffold`

### Goal
Create the initial academic repo skeleton and enforce baseline quality rules.

### Tasks
- create `src/hft_hmm/` package layout
- create `tests/`, `notebooks/`, `scripts/`, `docs/`
- add `pyproject.toml`, `requirements.txt`, `.python-version`
- add formatting and lint configuration
- add placeholder `__init__.py` files

### Acceptance criteria
- package imports correctly
- `pytest`, `ruff`, and `black --check` run successfully
- README explains repo purpose and workflow

### References
- project organization only
- course brief on project realization and presentation

---

## Issue 02 — Data IO and dataset contract
**Branch:** `feat/02-data-io`

### Goal
Define a reproducible interface for loading market data.

### Tasks
- implement CSV loader and optional Yahoo Finance loader
- define canonical columns: timestamp, price, volume if available
- create typed data contract for downstream modules
- add sample fixture data for tests

### Acceptance criteria
- raw data loads into a pandas DataFrame with validated schema
- tests cover missing columns and bad timestamps

### References
- paper uses financial time series and return preprocessing
- project-only utility

---

## Issue 03 — Return preprocessing and sampling frequency utilities
**Branch:** `feat/03-return-preprocessing`

### Goal
Produce log returns and consistent sampling utilities.

### Tasks
- implement log-return computation
- implement resampling helper for daily / 5-minute / 1-minute data
- implement NaN and duplicate timestamp handling
- add train/test split helper preserving time order

### Acceptance criteria
- correct log-return values on synthetic data
- edge-case tests for repeated timestamps and insufficient rows

### References
- paper Section 2.2 defines returns as log price differences

---

## Issue 04 — State grid and paper reference helpers
**Branch:** `feat/04-state-grid-utils`

### Goal
Support paper-aligned terminology and explicit references in code.

### Tasks
- create utilities for state labels and metadata
- create helper for storing paper reference strings in docstrings or constants
- define `paper-faithful` vs `engineering approximation` tags

### Acceptance criteria
- helper utilities imported by other modules
- at least one test validates expected metadata structure

### References
- paper Section 2.2 and 2.3

---

## Issue 05 — Piecewise linear regression baseline
**Branch:** `feat/05-plr-baseline`

### Goal
Implement the paper's naive initialization / baseline idea using segmented trends.

### Tasks
- implement simple piecewise linear regression baseline
- estimate segment slopes and residual variances
- optionally compute Durbin-Watson diagnostic
- expose outputs as baseline state summaries

### Acceptance criteria
- synthetic piecewise-trend signal is segmented sensibly
- regression outputs are test-covered

### References
- paper Section 3.1 (PLR baseline)
- status should be marked approximation unless segmentation is closely matched

---

## Issue 06 — Baseline Gaussian HMM wrapper
**Branch:** `feat/06-gaussian-hmm-wrapper`

### Goal
Wrap `hmmlearn` cleanly so the model is easy to inspect and test.

### Tasks
- implement fit / predict / predict_proba interface
- expose learned means, variances, transition matrix, initial distribution
- add deterministic random state support
- document which parts are delegated to `hmmlearn`

### Acceptance criteria
- model trains on synthetic Gaussian regime-switching data
- learned parameter shapes are correct
- unit tests cover fit and inference API

### References
- paper Section 2.3 and Section 3.2
- Baum-Welch / EM based learning

---

## Issue 07 — Model selection over number of hidden states
**Branch:** `feat/07-model-selection-k`

### Goal
Compare candidate hidden-state counts and select K reproducibly.

### Tasks
- implement loop over candidate K values
- compute log-likelihood, AIC, BIC
- return ranked summary table
- add plotting helper for selection curves

### Acceptance criteria
- synthetic experiments favor the correct or near-correct K
- tests validate AIC/BIC calculation logic

### References
- paper Section 2.3.4 and Section 3.2

---

## Issue 08 — Forward inference / filtering distribution
**Branch:** `feat/08-forward-inference`

### Goal
Implement the forward recursion used for real-time filtering and expected return prediction.

### Tasks
- implement normalized forward recursion in log-safe form where possible
- compute filtering probabilities over states
- expose expected return from state probabilities and state means
- compare wrapper output with library posterior probabilities where appropriate

### Acceptance criteria
- probabilities are normalized at every step
- tests cover simple two-state toy system with known behavior

### References
- paper Section 6 and Algorithm 4 style prediction logic
- paper-faithful target

---

## Issue 09 — Signal generation and simple trading policy
**Branch:** `feat/09-signal-policy`

### Goal
Turn expected returns into a basic long/short trading signal for evaluation.

### Tasks
- implement sign-based signal
- optional thresholded signal
- align predictions with next-step return realization correctly
- document no-cost and cost-aware evaluation modes

### Acceptance criteria
- no look-ahead leakage
- tests verify alignment between signal and realized return indices

### References
- paper inference-to-decision logic
- engineering evaluation layer

---

## Issue 10 — Evaluation metrics and backtest summaries
**Branch:** `feat/10-backtest-metrics`

### Goal
Create reusable evaluation functions and presentation-ready summaries.

### Tasks
- implement cumulative return
- implement Sharpe ratio
- implement drawdown and hit rate
- generate compact summary DataFrame

### Acceptance criteria
- metrics validated on synthetic known examples
- tests include zero-variance and empty-series edge cases

### References
- paper performance reporting sections
- engineering evaluation layer

---

## Issue 11 — Rolling-window training experiment
**Branch:** `feat/11-rolling-window`

### Goal
Reflect the paper's rolling retraining spirit in a reproducible experiment.

### Tasks
- implement walk-forward training loop
- support configurable window length and retrain frequency
- log chosen parameters and outputs per window

### Acceptance criteria
- no future data leakage
- integration test on small fixture dataset

### References
- paper Section 2.3 discusses rolling monthly estimation windows

---

## Issue 12 — Volatility ratio feature
**Branch:** `feat/12-volatility-ratio`

### Goal
Implement the first side-information predictor.

### Tasks
- implement EWMA volatility estimate
- implement fast/slow volatility ratio
- parameterize decay and window settings
- add visualization helper

### Acceptance criteria
- synthetic volatility changes behave as expected
- tests validate monotonic intuition and output shape

### References
- paper Section 4.2

---

## Issue 13 — Intraday seasonality feature
**Branch:** `feat/13-seasonality-feature`

### Goal
Implement the second side-information predictor.

### Tasks
- map timestamps to time-of-day index
- compute intraday seasonal buckets or normalized index
- expose feature series for spline fitting

### Acceptance criteria
- feature construction works across different frequencies
- tests validate index behavior on known timestamps

### References
- paper Section 4.3

---

## Issue 14 — Spline-based predictor fitting
**Branch:** `feat/14-spline-predictor`

### Goal
Fit spline relationships between side-information predictors and returns.

### Tasks
- fit spline to predictor vs normalized return
- allow configurable number of knots
- optionally force approximate zero-mean over support
- create evaluation plot

### Acceptance criteria
- spline fit returns deterministic outputs for fixed config
- tests cover fit/evaluate interface and shape

### References
- paper Section 4.1 and Algorithm 2
- likely engineering approximation in Python

---

## Issue 15 — IOHMM-style transition conditioning approximation
**Branch:** `feat/15-iohmm-approx`

### Goal
Approximate the paper's idea that transitions depend on side information.

### Tasks
- define transition model `P(m_t | m_{t-1}, x_t)` using logistic or softmax regression
- train transition conditioner on inferred or decoded states
- expose time-varying transition probabilities
- document deviation from full IOHMM learning

### Acceptance criteria
- conditioned transitions vary sensibly with features
- tests validate shapes, normalization, and deterministic behavior

### References
- paper Section 5.1 and 5.2
- explicitly mark as engineering approximation unless full IOHMM EM is implemented

---

## Issue 16 — Integrated side-information experiment
**Branch:** `feat/16-side-info-experiment`

### Goal
Compare baseline HMM against side-information-enhanced version.

### Tasks
- run baseline experiment
- run volatility-ratio enhancement
- run seasonality enhancement
- summarize differences in metrics and plots

### Acceptance criteria
- one reproducible experiment script creates all summary artifacts
- output saved to `docs/figures/` or similar

### References
- paper Sections 4–7

---

## Issue 17 — Visualization package for report-ready figures
**Branch:** `feat/17-visualizations`

### Goal
Standardize figures for the report and presentation.

### Tasks
- hidden-state timeline plot
- return vs signal plot
- model selection plot
- side-information spline plot
- cumulative PnL plot

### Acceptance criteria
- figures render from scripts without notebook-only code
- smoke tests cover plotting functions

### References
- figures throughout the paper
- project presentation support

---

## Issue 18 — Experiment configuration and reproducibility log
**Branch:** `feat/18-experiment-config`

### Goal
Make experiment settings explicit and reportable.

### Tasks
- create typed configuration object or YAML-based config
- log dataset, frequency, K, seed, windows, feature params
- save experiment summaries to disk

### Acceptance criteria
- one run produces a saved machine-readable config and result summary
- tests cover config parsing or validation

### References
- project reproducibility requirement

---

## Issue 19 — README academic usage example
**Branch:** `docs/19-readme-usage`

### Goal
Add a minimal end-to-end usage example suitable for a reviewer.

### Tasks
- document environment setup
- document one baseline run
- document one side-info run
- mention which library routines are used and why

### Acceptance criteria
- another student could run the baseline from the README

### References
- project presentation and evaluation constraints

---

## Issue 20 — Presentation support material
**Branch:** `docs/20-presentation-support`

### Goal
Prepare concise material for the preview and final presentation.

### Tasks
- create `docs/paper_notes.md`
- create `docs/experiment_log.md`
- summarize faithful vs approximate components
- draft 5-slide technical narrative

### Acceptance criteria
- enough material exists to build the oral presentation quickly

### References
- ASPTA project preview and presentation requirements
