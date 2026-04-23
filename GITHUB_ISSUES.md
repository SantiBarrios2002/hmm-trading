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
- `src/hft_hmm/data.py`
- fixture dataset for tests
- schema validation tests

### Acceptance notes
See Gate B.

---

## Issue 02b — Databento parquet loader
**Branch:** `feat/02b-databento-loader`
**Gate:** B

### Goal
Load local databento 1-minute OHLCV parquet files into the repository's
canonical market-data contract so side-information experiments can run on the
paper's exact contract (ES) at 1-minute frequency.

### Tasks
- add `pyarrow` to runtime dependencies in `pyproject.toml` and `requirements.txt`
- implement `load_databento_parquet(path, *, symbol=None, spec=None)` in `src/hft_hmm/data.py` that:
  - reads the parquet file via pandas/pyarrow
  - promotes the `ts_event` UTC DatetimeIndex to a `timestamp` column
  - maps `close` → `price` and preserves `volume`
  - drops databento metadata columns (`rtype`, `publisher_id`, `instrument_id`, `symbol`)
  - optionally filters by `symbol` value before dropping it (files may contain multiple continuous-contract rolls)
  - routes through `validate_market_data` so the existing contract tests apply uniformly
- export the new loader from `hft_hmm/__init__.py`
- add a small parquet fixture (one or two days, single symbol) under `tests/fixtures/`, keeping it under ~200 KB
- add unit tests covering: happy path, missing `close` / `ts_event`, symbol filter, and malformed metadata

### Deliverables
- `load_databento_parquet` in `src/hft_hmm/data.py`
- `pyarrow` added to runtime deps
- parquet fixture under `tests/fixtures/`
- additions to `tests/test_data_io.py`

### Acceptance notes
See Gate B. The loader must produce a DataFrame indistinguishable from the
CSV loader's output once run through `validate_market_data`, so downstream
code does not need a source-specific branch. Large parquet files live under
`data/databento/` and are not tracked in git.

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
- `src/hft_hmm/preprocessing.py`
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
- support an `init_from_plr=True` option that seeds means, variances, and state sequence from Issue 05's PLR output, matching the paper's initialization strategy
- document which parts are delegated to `hmmlearn` and which are engineering wrappers around it

### Deliverables
- `models/gaussian_hmm.py`
- synthetic regime-switching tests
- a test that `init_from_plr=True` produces reproducible fits on a fixed fixture

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
- `src/hft_hmm/selection/model_selection.py`
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
- implement the forward recursion in log-space using log-sum-exp stabilization
- return filtering probabilities `p(m_t | Δy_{1:t})` normalized at every step
- expose expected return `E[Δy_{t+1} | Δy_{1:t}]` from filtering probabilities and state means
- add a no-underflow regression test on a synthetic sequence of at least 5,000 steps

### Deliverables
- `inference/forward_filter.py`
- toy-system tests with normalization checks
- long-sequence stability test

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
- `src/hft_hmm/strategy/signals.py`
- alignment and leakage tests

### Acceptance notes
See Gate E.

---

## Issue 10 — Evaluation metrics and backtest summaries
**Branch:** `feat/10-backtest-metrics`
**Gate:** E

### Goal
Create reusable evaluation functions and report-ready summaries, reported in
both pre-cost and post-cost form so comparison to the paper's pre-cost
Sharpe is honest.

### Tasks
- implement cumulative return
- implement Sharpe ratio in both pre-cost and post-cost modes with an explicit cost model (basis points per turnover)
- implement drawdown and hit rate
- generate compact summary DataFrame that labels each row pre-cost or post-cost
- document the cost model and default assumptions in the module docstring

### Deliverables
- `src/hft_hmm/evaluation/metrics.py`
- metric edge-case tests (zero-variance, constant signal, single-period)
- a test that post-cost Sharpe equals pre-cost Sharpe when cost = 0

### Acceptance notes
See Gate E.

---

## Issue 11 — Rolling-window training experiment
**Branch:** `feat/11-rolling-window`
**Gate:** F

### Goal
Reflect the paper's retraining setup in a reproducible experiment. The
paper's scheme is a fixed-length training window followed by a forecast
period, so the default here matches that: train on the most recent `H` days,
forecast one step ahead over the subsequent `T` days, then advance the
window and retrain.

### Tasks
- implement a walk-forward loop with the default "train H days → forecast T days → advance" scheme
- parameterize window length `H`, forecast horizon `T`, and retrain frequency (default: retrain once per forecast period)
- assert at the boundary of every window that no timestamp in the training slice is ≥ the first forecast timestamp
- log per-window: window index, train/test time ranges, chosen `K`, log-likelihood, and metrics summary

### Deliverables
- `src/hft_hmm/experiments/walk_forward.py`
- small integration test with fixture data covering at least two windows
- an explicit no-leakage test that fails if the loop accidentally trains on future data

### Acceptance notes
See Gate F.

---

## Issue 12 — Experiment configuration and reproducibility log
**Branch:** `feat/12-experiment-config`
**Gate:** F

### Goal
Make experiment settings explicit and saved. The reproducibility contract is
that any run can be re-executed from its saved config alone.

### Tasks
- create a typed configuration object with YAML (de)serialization
- cover dataset, frequency, `K`, seed, walk-forward windows, feature params, and cost model
- define the run-artifact layout under `runs/<run_id>/`:
  - `config.yaml` (resolved, deterministic serialization)
  - `metrics.json` (summary metrics, pre- and post-cost)
  - `figures/` (any plots produced)
  - `log.jsonl` (per-window log entries, one JSON object per line)
- `run_id = sha256(resolved_config_yaml)[:12]` so identical configs map to identical run ids
- add `scripts/repro.py <config.yaml>` that loads a saved config and re-executes the run end to end

### Deliverables
- `src/hft_hmm/config/experiment_config.py`
- `scripts/repro.py`
- config validation tests
- a round-trip test: run on fixture → re-run via `repro.py` → metrics match bit-for-bit

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
Implement the second side-information predictor. The paper's seasonality
spline is defined over Chicago local time (CME exchange TZ), so the feature
must convert away from the loader's canonical UTC before bucketing.

### Tasks
- convert timestamps from UTC to an exchange-local timezone (default `America/Chicago` for ES; configurable for other venues)
- map converted timestamps to a time-of-day index
- compute intraday seasonal buckets or a normalized index suitable for spline input
- expose the feature as a pd.Series aligned with the price series
- add a test that a known 09:30 Chicago timestamp maps to the same bucket across daylight-saving transitions

### Deliverables
- `features/seasonality.py`
- feature construction tests
- explicit TZ-conversion test (includes a DST boundary)

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
Two routes are offered: the paper's **spline-bucketed transition matrix** is
the primary deliverable; a **softmax-conditioned** variant is an optional
stretch for comparison.

### Tasks (primary — bucketed-A route)
- discretize each spline predictor into `R` buckets using the spline's roots as boundaries (paper uses `R = 5` for the two splines in Fig. 5)
- align each side-information value `x_t` with the corresponding return and assign it to a bucket
- train a separate transition matrix `A_r` per bucket via Baum-Welch on the concatenated per-bucket data
- expose a time-varying transition probability lookup `A(x_t)` that selects the matrix for the current bucket
- document the concatenation shortcut as an approximation to the paper's formulation

### Tasks (optional stretch — softmax route)
- implement `P(m_t | m_{t-1}, x_t)` as a softmax-conditioned model fitted to decoded or inferred states
- compare against the bucketed-A route on a small fixture

### Deliverables
- `models/iohmm_approx.py` containing the bucketed-A implementation
- normalization tests: every `A_r` row sums to 1 within tolerance
- a deterministic fit test on a synthetic fixture
- a short note in the module docstring describing deviations from the paper (concatenation, finite bucket count, optional softmax variant)

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
- `src/hft_hmm/visualization/plots.py`
- smoke tests for plotting functions

### Acceptance notes
See Gate I.

---

## Issue 19 — README academic usage example
**Branch:** `docs/19-readme-usage`
**Gate:** A

### Goal
Add a minimal end-to-end usage example suitable for a reviewer without
depending on later side-information milestones.

### Tasks
- document environment setup
- document one baseline run
- point readers to the later side-information workflow once Gates G/H land,
  but do not require it for Gate A
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

---

## Issue 21 — Variance floor and EM monotonicity for the Gaussian HMM
**Branch:** `fix/21-hmm-em-stability`
**Gate:** C

### Goal
Close two paper-fidelity gaps in the Gaussian HMM wrapper that currently
cause `Model is not converging` warnings on the example walk-forward run:
(1) no explicit minimum-variance floor tied to the instrument tick size,
and (2) Baum-Welch log-likelihood is allowed to decrease between EM
iterations under the default `hmmlearn` settings.

### Tasks
- add a `min_variance` argument to the HMM wrapper with a documented default
  derived from the instrument tick size (ES tick = 0.25 → minute log-return
  floor), and expose it through `ExperimentConfig`
- clamp or reject `hmmlearn`'s `covars_` post-fit if any diagonal entry
  falls below `min_variance`
- tighten `hmmlearn` EM settings (`tol`, `n_iter`, `init_params`, seed
  control) so all tracked fixtures fit with monotone non-decreasing
  log-likelihood; document the chosen settings in the wrapper docstring
- if monotonicity cannot be achieved with `hmmlearn`, replace the EM loop
  with a custom log-space forward-backward + M-step and label it an
  engineering approximation
- propagate chosen settings through `ExperimentConfig` so runs stay
  deterministic and reproducible

### Deliverables
- updated `src/hft_hmm/models/gaussian_hmm.py`
- new tests in `tests/test_gaussian_hmm_wrapper.py`:
  - monotone EM log-likelihood across iterations on a deterministic fixture
  - variance-floor enforcement (clamped or raises, per the documented rule)
- `scripts/repro.py configs/example_es_csv.yaml` runs with no convergence
  warnings; update the tracked run-hash if affected

### Acceptance notes
See Gate C. This is the prerequisite for Gate J: results comparisons are
only meaningful once EM is numerically well-behaved on every window.

---

## Issue 22 — Standalone predictor backtests
**Branch:** `feat/22-standalone-predictors`
**Gate:** G

### Goal
Before wiring side-information predictors into the IOHMM (Issue 16),
validate each predictor's own signal on the walk-forward rig, mirroring the
paper's §4 structure where each spline-based predictor is evaluated in
isolation. This gives a diagnostic before the Gate H comparison and a
per-predictor baseline for the Gate J results table.

### Tasks
- add an experiment runner that takes a fitted spline predictor (volatility
  ratio or seasonality) and produces a sign-based trading signal directly
  from the spline evaluation at `x_t`, without going through an HMM
- reuse the existing walk-forward loop, signal module, and metrics module so
  the artifact shape under `runs/<run_id>/` is identical to the baseline
  HMM run
- add two configs under `configs/`:
  - `example_es_vol_ratio_standalone.yaml`
  - `example_es_seasonality_standalone.yaml`
- regenerate both via `scripts/repro.py` and document the outcome (pre-cost
  Sharpe, hit rate) in the PR notes; once Gate I lands, mirror the same
  summary into `docs/experiment_log.md`. If a predictor has no traction,
  record that and flag it as an input to the Gate H discussion

### Deliverables
- `src/hft_hmm/experiments/standalone_predictor.py`
- two new config YAMLs under `configs/`
- two tracked `runs/<run_id>/` artifacts (or documented commands to
  regenerate)
- integration test on the tracked month-long ES fixture covering both
  predictors
- test asserting the standalone signal path does not instantiate a
  Gaussian HMM

### Acceptance notes
See Gate G. The separation from the HMM is what makes the Gate H IOHMM
comparison interpretable: if a standalone predictor already carries the
signal, the IOHMM gain should be measured against that baseline, not
against the vanilla HMM alone.

---

## Issue 23 — Paper spec and ambiguity table
**Branch:** `docs/23-paper-spec`
**Gate:** I

### Goal
Replace the informal "faithful-vs-approximate" prose required by Gate I
with a single structured document that a thesis committee can read
top-to-bottom. This is the artifact that defends interpretive choices
during the oral.

### Tasks
- create `docs/paper_spec.md` with a table whose columns are:
  - **component** (e.g., "emission distribution", "transition conditioning")
  - **paper says** (with §/figure reference)
  - **repo interpretation** (what we implemented)
  - **deviation type**: paper-faithful / engineering approximation /
    evaluation-layer / excluded by §2.5
  - **acceptance risk** (one short phrase)
- cover at least: observation construction, emission distribution,
  transition model (baseline and IOHMM), EM vs MCMC, `K` selection,
  variance floor, side-information predictors, spline fitting,
  walk-forward scheme, cost model, and everything currently excluded by
  `IMPLEMENTATION_PLAN.md §2.5`
- cross-link each row to the relevant module docstring or issue number
- link to the spec from `README.md` and `IMPLEMENTATION_PLAN.md`

### Deliverables
- `docs/paper_spec.md`
- links from `README.md` and `IMPLEMENTATION_PLAN.md`

### Acceptance notes
See Gate I. Any later change in scope (e.g., re-including MCMC or an
ML/DL inference engine outside the paper) must be reflected here as an
**extension** row, not buried in module docstrings.

---

## Issue 24 — Paper-comparison results document
**Branch:** `feat/24-results-vs-paper`
**Gate:** J

### Goal
Produce the single document a reader opens to see whether the replication
reproduces anything the paper claims. Not required to match the paper's
numbers exactly; required to map our numbers to theirs honestly and call
out where and why they diverge.

### Tasks
- create `docs/results_vs_paper.md` with a comparison table spanning:
  - baseline HMM (Gate F)
  - volatility-ratio IOHMM (Gate H)
  - seasonality IOHMM (Gate H)
  - long-only benchmark
  and columns: pre-cost Sharpe, post-cost Sharpe, hit rate, cumulative
  return, chosen `K`, sample window, paper reference (§/table), repo
  `run_id`
- include a short narrative categorizing each gap as: data scope, model
  scope (§2.5 exclusions), implementation approximation, or numerical /
  stochastic variation
- state plainly which directional claim from the paper the repo reproduces
  (e.g., "vol-conditioned transitions beat baseline on pre-cost Sharpe by
  Δ on fixture Y, vs paper's reported Δ′ in §4.4")
- produce one comparison figure using Gate I's plotting code and store
  under `docs/figures/`

### Deliverables
- `docs/results_vs_paper.md`
- `docs/figures/cumulative_return_vs_paper.png` (or equivalent)
- all cited `runs/<run_id>/` artifacts reproducible via `scripts/repro.py`

### Acceptance notes
See Gate J. A PR opening this issue must cite live `run_id`s; the table
cannot be filled with placeholders. This is the gate that answers "did
the project work."
