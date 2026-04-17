# Implementation Plan — Acceptance Checks

This document is the repo's implementation contract. It is **not** a task list and should not duplicate GitHub issues. Each section defines a gate that must be passed before the next milestone is considered complete.

Use this file during PR review.
Use `GITHUB_ISSUES.md` to create and track the actual work items.

---

## 1. Academic objective

The repository should implement a clean, reviewable, academically defensible Python reproduction of the paper's HMM-based momentum framework, with explicit separation between:

- **paper-faithful components**
- **engineering approximations**
- **evaluation-only utilities**

The repo must support the course goal of simulating some results of the paper and presenting the methods clearly, not necessarily reproducing every original experiment exactly.

---

## 2. Global rules that every PR must satisfy

### 2.1 Code quality
- Every public function must have:
  - type hints
  - a docstring
  - a short "References" note pointing to the paper section or saying `Engineering utility`
- No notebook-only logic in core modules.
- No hidden global state.
- Random seeds must be explicit in experiments.

### 2.2 Testing
- Every function added in a PR must be covered by at least one test.
- Each bug fix must include a regression test.
- Numerical functions must have at least one deterministic synthetic-data test.
- Time-series functions must have at least one no-leakage or alignment test when relevant.

### 2.3 Documentation
- Every module must state whether it is:
  - `paper-faithful`
  - `engineering approximation`
  - `evaluation layer`
- Every library used must be mentioned in the README or module docs.
- Every experiment script must save its configuration.

### 2.4 Reviewability
- One PR = one coherent idea.
- Each PR must reference one primary issue.
- Branch names must match the pattern:
  - `feat/NN-short-name`
  - `docs/NN-short-name`
  - `fix/NN-short-name`

### 2.5 Scope exclusions
These parts of the original paper are intentionally out of scope for the
coursework replication. They are listed here so the grader does not mistake
their absence for oversight.

- **MCMC parameter estimation.** The paper fits Θ by both Baum-Welch and Metropolis-Hastings. This repo uses only Baum-Welch.
- **MCMC bridge sampling for model selection.** The paper picks `K` using cross-validation, AIC/BIC, and marginal likelihood via MCMC bridge sampling. This repo uses only cross-validation plus AIC/BIC.
- **Asynchronous IOHMM.** The paper sketches an asynchronous variant for mixed-frequency inputs. This repo implements only the synchronous IOHMM approximation.
- **Multi-security / portfolio backtest.** Evaluation is single-security (ES or an equivalent proxy). No cross-asset construction.
- **Production execution concerns.** No latency modeling, slippage beyond a flat cost-per-turnover, venue microstructure, or order-book effects.

---

## 3. Acceptance gates

Each gate below is a **check set**, not an implementation checklist.
A gate passes only when all listed conditions are satisfied.

---

## Gate A — Repository and environment
**Covers:** Issues 01, 19

### Must pass
- The repo installs on Python **3.11**.
- `pyproject.toml`, `requirements.txt`, and `.python-version` are present and consistent.
- The package imports without path hacks.
- `pytest` runs successfully.
- `ruff` and `black --check` run successfully.
- The README explains:
  - project purpose
  - environment setup
  - baseline run command
  - where paper-faithful vs approximate parts live

### Evidence expected in PR review
- CI or local command output pasted in PR
- a minimal import test
- README setup section verified by a fresh environment

---

## Gate B — Data contract and preprocessing
**Covers:** Issues 02, 02b, 03, 04

### Must pass
- Data loading accepts a documented schema.
- Timestamp parsing is validated.
- At least one CSV / yfinance loader and one databento parquet loader exist and both route through the same validation path.
- The databento parquet loader maps `ts_event` → `timestamp`, `close` → `price`, and filters by `symbol` when requested.
- The replication dataset decision (yfinance daily for development, databento 1-minute ES for paper-replication runs) is documented in the repo.
- Log returns are computed correctly.
- Resampling preserves time order and documents frequency assumptions.
- Duplicate timestamps and missing values are handled explicitly.
- Train/test split utilities do not leak future data.
- State metadata and paper reference helpers exist and are used.

### Evidence expected in PR review
- unit tests on synthetic price data
- tests for malformed input
- parquet fixture + loader test exercising the databento path
- one example showing daily and intraday preprocessing

---

## Gate C — Baseline modeling scaffold
**Covers:** Issues 05, 06

### Must pass
- A piecewise linear regression baseline exists and returns interpretable trend summaries.
- A Gaussian HMM wrapper exists with a stable, documented API.
- The wrapper exposes at least:
  - fit
  - predict
  - predict_proba
  - means
  - variances/covariances
  - transition matrix
  - initial distribution
- Library usage is documented explicitly.
- Synthetic-data tests show the model can recover regime-like structure at least qualitatively.

### Evidence expected in PR review
- test fixtures for segmented trends
- synthetic regime-switching example
- docstrings referencing the corresponding paper sections

---

## Gate D — Model selection and filtering inference
**Covers:** Issues 07, 08

### Must pass
- Candidate state counts `K` can be compared reproducibly.
- AIC and BIC calculations are tested.
- Forward filtering returns normalized state probabilities at every time step.
- Expected return from filtering probabilities and state means is exposed as an API.
- The forward recursion runs in log-space with log-sum-exp stabilization and does not underflow on long sequences.

### Evidence expected in PR review
- tests verifying normalization
- tests for AIC/BIC formulas
- a no-underflow test on a synthetic sequence of at least 5,000 steps
- one toy two-state example with known behavior

---

## Gate E — Trading signal and evaluation layer
**Covers:** Issues 09, 10

### Must pass
- A sign-based trading signal can be generated from expected returns.
- Signal alignment with future realized returns is correct.
- No look-ahead leakage exists in the signal path.
- Evaluation functions exist for at least:
  - cumulative return
  - Sharpe ratio (reported in both pre-cost and post-cost form)
  - drawdown
  - hit rate
- The cost model (basis points per turnover) is documented and post-cost equals pre-cost when cost = 0.
- Metric edge cases are tested.

### Evidence expected in PR review
- alignment tests
- zero-variance metric test
- pre-cost vs post-cost parity test at cost = 0
- example summary table from a small experiment, labeled pre- and post-cost

---

## Gate F — Walk-forward experiment
**Covers:** Issues 11, 12

### Must pass
- A walk-forward training loop exists with the default scheme: train on the most recent `H` days → forecast one step ahead over the subsequent `T` days → advance the window and retrain.
- `H`, `T`, and retrain frequency are configurable; the default retrains once per forecast period.
- Future data is never used during fitting, verified by an explicit boundary assertion inside the loop.
- Each run produces a `runs/<run_id>/` directory containing:
  - `config.yaml` (resolved, deterministic serialization)
  - `metrics.json` (pre- and post-cost summary)
  - `figures/` (any plots)
  - `log.jsonl` (one JSON entry per window)
- `run_id = sha256(resolved_config_yaml)[:12]` so identical configs map to identical artifact directories.
- `scripts/repro.py <config.yaml>` re-executes a run end to end and the resulting metrics match bit-for-bit.

### Evidence expected in PR review
- integration test on a fixture covering at least two windows
- saved config + run artifact example
- round-trip reproducibility test via `scripts/repro.py`
- explicit no-leakage review notes

---

## Gate G — Side-information predictors
**Covers:** Issues 13, 14, 15

### Must pass
- Volatility ratio feature is implemented and tested.
- Intraday seasonality feature is implemented and tested.
- Spline fitting exists with a documented Python approximation.
- The spline interface is deterministic for fixed inputs and configuration.
- The code labels clearly which parts are paper-faithful and which are approximate.

### Evidence expected in PR review
- feature construction tests
- spline fit/evaluate tests
- at least one visualization produced from script code

---

## Gate H — IOHMM-style transition conditioning
**Covers:** Issues 16, 17

### Must pass
- The repo contains a clearly labeled **approximate** transition-conditioning implementation following the paper's spline-bucketed approach (discretize each spline into buckets using its roots as boundaries; train a separate transition matrix per bucket).
- A softmax-conditioned variant may exist as an optional stretch for comparison; if present it is labeled as an engineering approximation rather than a paper-faithful route.
- Transition probabilities vary with side information and remain normalized per row.
- The experiment compares:
  - baseline HMM
  - volatility-enhanced version
  - seasonality-enhanced version
- Deviations from the paper — concatenation shortcut, finite bucket count, any softmax variant — are explicitly documented in the module docstring.

### Evidence expected in PR review
- tests for shape and per-row normalization of every per-bucket transition matrix
- one experiment script producing comparison outputs
- written note naming the approximation route(s) implemented and listing deviations from the paper

---

## Gate I — Figures and presentation support
**Covers:** Issues 18, 20

### Must pass
- Plotting functions run from scripts, not only notebooks.
- The repo can generate figures suitable for:
  - state timeline
  - model selection
  - spline predictor
  - cumulative returns
- A concise paper notes document exists.
- A concise experiment log exists.
- A short faithful-vs-approximate summary exists for presentation use.

### Evidence expected in PR review
- generated figures stored under `docs/figures/` or equivalent
- smoke tests for plotting functions
- reviewed `docs/paper_notes.md` and `docs/experiment_log.md`

---

## 4. Definition of done for the project

The project is ready for academic submission when:

- Gates A through F are fully passed
- at least part of Gate G is passed
- Gate H is passed in approximate form at minimum
- Gate I is passed well enough to support the oral presentation

A strong project should pass all gates except perhaps the "full IOHMM" variant, which is optional.

---

## 5. Recommended delivery order

Recommended order for merging PRs:

1. Gate A
2. Gate B
3. Gate C
4. Gate D
5. Gate E
6. Gate F
7. Gate G
8. Gate H
9. Gate I

This keeps the repo academically coherent and easy to review.
