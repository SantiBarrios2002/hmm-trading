# Implementation Plan — Acceptance Checks

This document is the repo's implementation contract. It is **not** a task list and it should not duplicate GitHub issues. Each section defines a gate that must be passed before the next milestone is considered complete.

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
**Covers:** Issues 02, 03, 04

### Must pass
- Data loading accepts a documented schema.
- Timestamp parsing is validated.
- Log returns are computed correctly.
- Resampling preserves time order and documents frequency assumptions.
- Duplicate timestamps and missing values are handled explicitly.
- Train/test split utilities do not leak future data.
- State metadata and paper reference helpers exist and are used.

### Evidence expected in PR review
- unit tests on synthetic price data
- tests for malformed input
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
- Numerical stability is addressed and documented.

### Evidence expected in PR review
- tests verifying normalization
- tests for AIC/BIC formulas
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
  - Sharpe ratio
  - drawdown
  - hit rate
- Metric edge cases are tested.

### Evidence expected in PR review
- alignment tests
- zero-variance metric test
- example summary table from a small experiment

---

## Gate F — Walk-forward experiment
**Covers:** Issues 11, 12

### Must pass
- A rolling or walk-forward training loop exists.
- Window size and retraining frequency are configurable.
- Future data is never used during fitting.
- Every experiment saves:
  - dataset info
  - frequency
  - seed
  - K
  - feature settings
  - metrics summary
- Results can be reproduced from a saved configuration.

### Evidence expected in PR review
- one integration test on a small fixture
- saved config artifact example
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
- The repo contains either:
  1. a clearly labeled **approximate** transition-conditioning implementation, or
  2. a clearly labeled **full** IOHMM implementation
- Transition probabilities vary with side information and remain normalized.
- The experiment compares:
  - baseline HMM
  - volatility-enhanced version
  - seasonality-enhanced version
- Deviations from the paper are explicitly documented.

### Evidence expected in PR review
- tests for shape and normalization
- one experiment script producing comparison outputs
- written note stating whether the implementation is full IOHMM or approximation

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
