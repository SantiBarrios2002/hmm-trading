# Hidden Markov Models Applied To Intraday Momentum Trading — Implementation Plan

## Academic objective
This repository is designed as an **academic replication and interpretation project** for ASPTA. The goal is **not** to build a production trading system and **not** to claim exact reproduction of the paper’s reported trading metrics. The goal is to:

1. implement the paper’s probabilistic pipeline in Python,
2. understand which parts are core methodology versus optional extensions,
3. reproduce a defensible subset of the paper’s results,
4. keep the repository clean enough for stepwise PR review and oral presentation.

This fits the course brief, which asks for simulating some results of a selected paper and presenting the problem, methods, and implementation choices.

---

## Source papers and mandatory referencing rule
Primary paper:
- Christensen, Turner, Godsill, **"Hidden Markov Models Applied To Intraday Momentum Trading With Side Information"** (arXiv:2006.08307)

Course brief:
- ASPTA course introduction / project instructions PDF

### Mandatory code reference rule
Every public function must contain, in its docstring or comments:
- the **paper section** it implements,
- the **equation / algorithm / figure / page reference** if applicable,
- whether the function is:
  - `paper-faithful`
  - `engineering approximation`
  - `project-only utility`

### Example
```python

def compute_filtering_distribution(...):
    """
    Compute ω_{t|t,k} for the HMM forward recursion.

    Reference:
        Paper Section 6, Eq. for filtering distribution on p.16,
        Algorithm 4 (HMM Prediction).
    Status:
        paper-faithful
    """
```

---

## Recommended Python version
Use **Python 3.11**.

Why:
- broad support across scientific Python stack,
- stable wheels for NumPy / SciPy / pandas / scikit-learn / matplotlib,
- good compatibility with pytest, mypy, ruff,
- avoids edge-case packaging friction that sometimes appears with 3.12+ for less maintained packages.

---

## Recommended libraries
These are allowed and should be named explicitly in the report.

### Core scientific stack
- `numpy` — arrays, linear algebra, vectorization
- `scipy` — distributions, interpolation, optimization utilities
- `pandas` — time-indexed data handling
- `matplotlib` — figures for report / presentation

### Modeling
- `hmmlearn` — baseline Gaussian HMM with EM / Baum-Welch
- `scikit-learn` — model selection helpers, logistic regression, preprocessing, metrics
- `statsmodels` — optional for Durbin-Watson statistic and basic time-series diagnostics

### Testing / quality
- `pytest` — unit and integration tests
- `pytest-cov` — coverage
- `ruff` — linting
- `black` — formatting
- `mypy` — optional but recommended static typing

### Why libraries are allowed here
The paper itself is about applying known estimation and inference tools to a finance problem; it does **not** introduce a brand-new optimization method. For an academic replication project, using well-known libraries is appropriate so long as:
- you understand what each library routine is doing,
- you document where a library replaces handwritten math,
- you isolate paper-specific logic in your own code.

---

## Dependencies file
Use `requirements.txt` for simplicity.

Suggested contents:
- numpy
- scipy
- pandas
- matplotlib
- seaborn  # optional; only if you want quick exploratory plots
- scikit-learn
- hmmlearn
- statsmodels
- yfinance  # optional if market data is downloaded directly
- pytest
- pytest-cov
- black
- ruff
- mypy

Note: `seaborn` is optional and not required by the project architecture.

---

## Repository structure
```text
repo/
├─ src/
│  └─ hft_hmm/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ data/
│     │  ├─ io.py
│     │  ├─ preprocess.py
│     │  └─ sampling.py
│     ├─ diagnostics/
│     │  ├─ residuals.py
│     │  └─ model_selection.py
│     ├─ features/
│     │  ├─ volatility_ratio.py
│     │  ├─ seasonality.py
│     │  └─ spline_signal.py
│     ├─ models/
│     │  ├─ baseline_plr.py
│     │  ├─ gaussian_hmm.py
│     │  ├─ iohmm_transition.py
│     │  └─ inference.py
│     ├─ backtest/
│     │  ├─ signal.py
│     │  ├─ pnl.py
│     │  └─ metrics.py
│     ├─ plots/
│     │  └─ figures.py
│     └─ utils/
│        ├─ math.py
│        ├─ grids.py
│        └─ references.py
├─ tests/
│  ├─ test_preprocess.py
│  ├─ test_plr.py
│  ├─ test_gaussian_hmm.py
│  ├─ test_inference.py
│  ├─ test_features.py
│  ├─ test_iohmm_transition.py
│  └─ test_backtest.py
├─ notebooks/
│  ├─ 01_data_overview.ipynb
│  ├─ 02_baseline_hmm.ipynb
│  ├─ 03_side_information.ipynb
│  └─ 04_results.ipynb
├─ scripts/
│  ├─ run_baseline.py
│  ├─ run_model_selection.py
│  ├─ run_side_info.py
│  └─ run_backtest.py
├─ docs/
│  ├─ paper_notes.md
│  ├─ experiment_log.md
│  └─ figures/
├─ requirements.txt
├─ pyproject.toml  # optional later
├─ .python-version
└─ README.md
```

---

## Branching strategy
Every issue gets its own short-lived branch.

### Branch naming template
```text
feat/<issue-number>-<short-name>
fix/<issue-number>-<short-name>
docs/<issue-number>-<short-name>
exp/<issue-number>-<short-name>
```

### Examples
- `feat/01-repo-scaffold`
- `feat/03-return-preprocessing`
- `feat/07-gaussian-hmm-baseline`
- `feat/11-forward-inference`
- `feat/14-volatility-ratio-feature`
- `feat/18-iohmm-approx-transition`
- `exp/21-es-vs-spy-comparison`

### Pull request rule
Each PR must include:
- what paper section it targets,
- what was implemented exactly,
- what is faithful versus approximated,
- test evidence,
- one figure or one printed artifact if relevant.

---

## Development rules

### Rule 1 — Every function gets tested
At minimum, each public function must have:
- one correctness test,
- one shape / type test,
- one edge-case test where relevant.

### Rule 2 — Every module names paper references
Each file begins with a short module note:
- target paper section,
- equations / algorithms implemented,
- known deviations.

### Rule 3 — Separate math from experiments
Do not mix:
- model math,
- plotting,
- data download,
- backtest execution.

### Rule 4 — No hidden notebook logic
A notebook may explore, but final logic must live in `src/` and be imported by notebooks.

### Rule 5 — Reproducibility first
Set seeds where randomness appears.
Record:
- dataset choice,
- sampling frequency,
- date range,
- train/test split,
- chosen `K`,
- key hyperparameters.

### Rule 6 — Explicit approximation labels
Whenever you do **not** reproduce the paper literally, say so in code and docs.
Examples:
- replacing custom discretized Gaussian emissions with `hmmlearn` Gaussian emissions,
- replacing full IOHMM learning with a conditional transition approximation,
- using lower-frequency data instead of 1-minute ES futures.

---

## Academic scope: what to implement fully vs partially

### High-priority, course-aligned components
These are the parts most aligned with ASPTA topics and should be implemented first:
1. Gaussian HMM formulation
2. EM / Baum-Welch learning
3. Model selection over number of states
4. Forward inference / filtering distribution
5. Prediction from state probabilities
6. At least one side-information feature
7. Clean experimental evaluation

### Medium-priority components
1. Piecewise linear regression baseline
2. Rolling-window retraining
3. Simple transaction-cost-aware PnL
4. A practical approximation to IOHMM transitions

### Low-priority components
1. Full MCMC learning
2. Bridge sampling for marginal likelihood
3. Exact paper-level trading replication on 1-minute ES futures

These are valuable to discuss academically, but not essential for a strong project repository.

---

# Issue-by-issue implementation plan

## Issue 01 — Repository scaffold
**Branch:** `feat/01-repo-scaffold`

### Goal
Create the clean project skeleton and tooling.

### Deliverables
- `src/` package structure
- `tests/` layout
- `README.md`
- `requirements.txt`
- `.python-version`
- basic CI-ready commands in README

### Acceptance criteria
- package imports cleanly,
- `pytest` runs,
- `ruff` and `black --check` run,
- top-level README explains project scope.

### Tests
- smoke test importing `hft_hmm`

### Paper relevance
Project support issue, no direct paper section.

---

## Issue 02 — Paper notes and replication scope
**Branch:** `docs/02-paper-notes`

### Goal
Write a concise internal document mapping paper sections to implementation tasks.

### Deliverables
- `docs/paper_notes.md`
- list of equations / algorithms to reproduce
- list of explicit simplifications

### Acceptance criteria
- each major paper section has a corresponding repo task
- faithful vs approximate parts are clearly marked

### Paper relevance
Paper structure and method overview.

---

## Issue 03 — Data ingestion and preprocessing
**Branch:** `feat/03-data-ingestion`

### Goal
Load price series and transform to returns.

### What the paper does
The paper works on price series `y_t` and log returns `Δy_t = log(y_t / y_{t-1})`.

### Deliverables
- `data/io.py`
- `data/preprocess.py`
- configurable loader for CSV / parquet / yfinance
- log-return computation
- missing-value handling
- date filtering

### Acceptance criteria
- raw prices load into time-indexed DataFrame
- returns computed correctly
- first NaN is handled deterministically

### Tests
- log-return formula test
- missing timestamp handling test
- monotonic index validation test

### Paper relevance
Observation definition and return preprocessing.

---

## Issue 04 — Sampling and experiment datasets
**Branch:** `feat/04-sampling-config`

### Goal
Support daily / 5-minute / 1-minute experiment configurations.

### Deliverables
- `data/sampling.py`
- config-driven resampling
- metadata object storing sampling frequency and experiment name

### Acceptance criteria
- one command can produce a clean experiment dataset for each supported frequency
- experiment metadata is saved

### Tests
- resampling output shape test
- frequency label propagation test

### Paper relevance
Supports the paper’s intraday setting while allowing academic simplification.

---

## Issue 05 — Return diagnostics and sanity checks
**Branch:** `feat/05-diagnostics`

### Goal
Check whether the data resembles the assumptions enough to proceed.

### Deliverables
- histogram plot
- rolling mean / variance diagnostics
- optional normality summary
- stationarity notes in docs

### Acceptance criteria
- one diagnostics notebook or script produces figures reproducibly

### Tests
- diagnostic helper functions return expected objects / shapes

### Paper relevance
Supports emission modeling assumptions.

---

## Issue 06 — Piecewise linear regression baseline
**Branch:** `feat/06-plr-baseline`

### Goal
Implement the paper’s baseline learning idea using segmented linear trends.

### What the paper claims
PLR is used as a baseline to estimate change points, latent state sequence, means, and variances; cross-validation selects `K = 2` in the default case.

### Deliverables
- `models/baseline_plr.py`
- segmented regression utility
- residual variance estimate per segment
- optional Durbin-Watson residual check

### Acceptance criteria
- synthetic piecewise-linear series recovers approximate segment slopes
- segment variances are finite and interpretable

### Tests
- known-breakpoint synthetic test
- slope recovery tolerance test
- residual variance positivity test

### Paper relevance
Section 3.1 baseline.

---

## Issue 07 — Gaussian HMM wrapper
**Branch:** `feat/07-gaussian-hmm`

### Goal
Build a clean wrapper around `hmmlearn` for the baseline HMM.

### What the paper claims
The model uses hidden states and Gaussian emissions with state-specific means and variances.

### Deliverables
- `models/gaussian_hmm.py`
- model config dataclass
- fit / predict state / predict probabilities / score methods
- extraction of transition matrix, means, covariances

### Acceptance criteria
- model fits on synthetic Gaussian-switching data
- recovered means roughly match generating means
- transition matrix rows sum to one

### Tests
- synthetic fit test
- probability normalization test
- parameter extraction test

### Paper relevance
Core HMM formulation and Baum-Welch learning.

---

## Issue 08 — Model selection for number of states
**Branch:** `feat/08-model-selection`

### Goal
Compare candidate values of `K`.

### What the paper claims
Different approaches suggest `K = 2` or `K = 3`; Baum-Welch with penalized likelihood criteria gives `K = 3`.

### Deliverables
- `diagnostics/model_selection.py`
- AIC/BIC utilities
- sweep script over candidate `K`
- plot of criterion versus `K`

### Acceptance criteria
- reproducible sweep over chosen `K` range
- plot generated and saved
- chosen `K` recorded in experiment log

### Tests
- BIC/AIC computation sanity test
- model ranking deterministic for fixed seed on synthetic data

### Paper relevance
Section on unknown number of hidden states and Section 3.2.

---

## Issue 09 — Experiment log and reproducibility metadata
**Branch:** `feat/09-experiment-log`

### Goal
Track every run like a small academic study.

### Deliverables
- `docs/experiment_log.md`
- JSON or YAML artifact per run with:
  - dataset
  - date range
  - frequency
  - train/test split
  - K
  - random seed
  - metrics

### Acceptance criteria
- every scripted experiment emits one metadata file

### Tests
- metadata serialization test

### Paper relevance
Project rigor support.

---

## Issue 10 — HMM forward inference implementation
**Branch:** `feat/10-forward-inference`

### Goal
Implement the paper’s filtering and prediction logic explicitly, even if training uses a library.

### What the paper claims
The inference phase computes the predictive distribution using the forward algorithm; it initializes from the prior, predicts one step ahead through the transition matrix, updates with observation likelihoods, and obtains `Δŷ_t` as an expectation under the predictive distribution.

### Deliverables
- `models/inference.py`
- likelihood vector calculation
- filtering recursion
- prediction recursion
- expected return estimate from state probabilities and means

### Acceptance criteria
- recursive probabilities remain normalized
- inferred filtering distribution is sensible on synthetic data
- one-step prediction behaves consistently with known state means

### Tests
- normalization test at each step
- synthetic known-parameter recursion test
- expected-return computation test

### Paper relevance
Inference phase, Section 6, Algorithm 4.

---

## Issue 11 — Signal extraction and trading rule
**Branch:** `feat/11-trading-signal`

### Goal
Convert predicted return into a simple academic trading signal.

### Deliverables
- `backtest/signal.py`
- sign-based rule: long / short / flat
- optional thresholding to reduce noise

### Acceptance criteria
- signal generation is deterministic and documented

### Tests
- sign rule tests
- threshold behavior tests

### Paper relevance
Prediction-to-trading interpretation.

---

## Issue 12 — PnL and evaluation metrics
**Branch:** `feat/12-backtest-metrics`

### Goal
Evaluate academic trading performance without overselling claims.

### Deliverables
- `backtest/pnl.py`
- `backtest/metrics.py`
- cumulative return
- Sharpe ratio
- max drawdown
- turnover estimate
- optional simple transaction-cost model

### Acceptance criteria
- metrics reproduce known values on synthetic return streams

### Tests
- Sharpe ratio test
- drawdown test
- transaction cost deduction test

### Paper relevance
Simulation and results section.

---

## Issue 13 — Rolling-window training
**Branch:** `feat/13-rolling-window`

### Goal
Match the paper’s idea of re-estimating parameters on rolling historical windows.

### What the paper claims
Parameter estimation is done offline on recent historical data, with mention of a rolling window of 23 trading days for stable estimation of state means.

### Deliverables
- rolling fit utility
- walk-forward evaluation script
- saved per-window parameters

### Acceptance criteria
- no look-ahead leakage
- each prediction uses only past data

### Tests
- leakage-prevention test
- window boundary test

### Paper relevance
Learning-phase implementation details.

---

## Issue 14 — Volatility ratio predictor
**Branch:** `feat/14-volatility-ratio`

### Goal
Implement the first side-information feature.

### What the paper claims
The volatility ratio uses two realized-volatility estimates based on exponentially weighted variance with different windows, then takes their ratio.

### Deliverables
- `features/volatility_ratio.py`
- EWMA volatility estimator
- fast/slow ratio feature
- configuration for lambda and windows

### Acceptance criteria
- volatility estimate is numerically stable
- ratio feature is finite after warm-up period

### Tests
- EWMA recurrence test
- positive volatility test
- ratio finite-value test

### Paper relevance
Section 4.2 predictor I.

---

## Issue 15 — Seasonality predictor
**Branch:** `feat/15-seasonality`

### Goal
Implement the second side-information feature.

### What the paper claims
Intraday seasonality is modeled by indexing time within the day and estimating a cyclic relationship to returns.

### Deliverables
- `features/seasonality.py`
- intraday position index feature
- bucket construction for within-day time

### Acceptance criteria
- feature repeats cleanly by trading day
- indices are reproducible and interpretable

### Tests
- within-day index mapping test
- reset-per-day test

### Paper relevance
Section 4.3 predictor II.

---

## Issue 16 — Spline fitting for predictive side signals
**Branch:** `feat/16-spline-signals`

### Goal
Replicate the paper’s use of splines to convert predictors into smooth predictive signals.

### What the paper claims
A B-spline is used to estimate the nonlinear relationship between predictor and normalized return; the spline is constrained to have zero mean over its support.

### Deliverables
- `features/spline_signal.py`
- fit spline to `(x_t, normalized return)`
- evaluate spline on new `x_t`
- optional zero-mean adjustment utility

### Acceptance criteria
- spline fit returns a callable or model object
- evaluation works on train and test domains
- zero-mean adjustment is documented

### Tests
- fit/evaluate smoke test
- zero-mean adjustment numerical test
- monotonic input validation test

### Paper relevance
Section 4.1 and Algorithm 2.

---

## Issue 17 — Standalone side-signal evaluation
**Branch:** `feat/17-side-signal-eval`

### Goal
Evaluate the spline-derived predictors before integrating them into the HMM.

### What the paper claims
The two predictors are first developed and tested as stand-alone signals before being used in the IOHMM learning stage.

### Deliverables
- experiment script for each side signal
- cumulative return and Sharpe plots
- correlation table versus baseline strategy if desired

### Acceptance criteria
- one report figure per predictor
- clean separation between feature validation and HMM integration

### Tests
- side-signal pipeline smoke test

### Paper relevance
Section 4.4 simulation and results.

---

## Issue 18 — IOHMM-style conditional transition approximation
**Branch:** `feat/18-iohmm-approx`

### Goal
Capture the paper’s IOHMM idea without committing to a full bespoke IOHMM learning engine.

### What the paper claims conceptually
In the IOHMM, transitions become conditional on side information: `p(m_t | m_{t-1}, x_t)` instead of only `p(m_t | m_{t-1})`.

### Practical project choice
Approximate conditional transitions with a supervised transition model, e.g. multinomial logistic regression or per-previous-state logistic models.

### Deliverables
- `models/iohmm_transition.py`
- transition feature builder
- conditional transition probability model
- interface returning `A_t` for each time step

### Acceptance criteria
- time-varying transition matrix is valid
- each row sums to one
- feature-driven changes in transition probabilities are visible

### Tests
- row-normalization test
- time-varying matrix shape test
- synthetic conditional-transition sanity test

### Paper relevance
Section 5 conceptually, but labeled as **engineering approximation** unless a fully faithful implementation is built.

---

## Issue 19 — IOHMM-style forward inference
**Branch:** `feat/19-iohmm-inference`

### Goal
Reuse forward inference with time-varying transitions `A_t`.

### What the paper claims
Inference in the IOHMM case is similar to the HMM case, except the parameters are conditioned on `x_t`.

### Deliverables
- extension of `models/inference.py` to accept `A_t`
- prediction with conditional transitions

### Acceptance criteria
- filtering remains normalized across time
- predictions differ from fixed-transition baseline when features vary

### Tests
- normalization test with dynamic transitions
- regression test versus fixed-transition case when features are constant

### Paper relevance
Section 6, IOHMM inference logic.

---

## Issue 20 — Main experiment: baseline HMM
**Branch:** `exp/20-baseline-hmm-results`

### Goal
Produce the first presentation-ready result set.

### Deliverables
- state sequence plot
- state probability plot
- transition matrix heatmap
- predicted return plot
- cumulative PnL plot
- concise markdown summary

### Acceptance criteria
- one fully reproducible baseline run
- all figures saved under `docs/figures/`

### Tests
- end-to-end smoke test for script

### Paper relevance
Sections 3, 6, 7.

---

## Issue 21 — Main experiment: side information comparison
**Branch:** `exp/21-side-info-comparison`

### Goal
Compare baseline HMM against side-information-enhanced model.

### Deliverables
- side-by-side metrics table
- figure comparing state probabilities or predicted returns
- short discussion of whether side information helps

### Acceptance criteria
- identical train/test split across models
- comparison produced from one script entry point

### Tests
- end-to-end smoke test for comparison script

### Paper relevance
Sections 4, 5, 6, 7.

---

## Issue 22 — Optional issue: MCMC reading note only
**Branch:** `docs/22-mcmc-note`

### Goal
Document the Bayesian part without implementing it unless there is extra time.

### Deliverables
- `docs/mcmc_note.md`
- explanation of posterior sampling, bridge sampling, and why omitted from core repo

### Acceptance criteria
- clearly distinguishes paper content from implemented content

### Paper relevance
Section 3.3.

---

## Issue 23 — Presentation artifact generation
**Branch:** `docs/23-presentation-material`

### Goal
Create final presentation-ready assets.

### Deliverables
- 5-minute preview outline
- final slide figure checklist
- key claims with cautious wording

### Acceptance criteria
- all figures and tables traced back to code and experiments

### Paper relevance
Project presentation support.

---

# Suggested implementation order

## Phase A — foundation
1. Issue 01
2. Issue 02
3. Issue 03
4. Issue 04
5. Issue 05

## Phase B — core paper replication
6. Issue 07
7. Issue 08
8. Issue 10
9. Issue 11
10. Issue 12
13. Issue 20

## Phase C — stronger academic replication
11. Issue 06
12. Issue 13
13. Issue 14
14. Issue 15
15. Issue 16
16. Issue 17
17. Issue 21

## Phase D — paper-inspired extension
18. Issue 18
19. Issue 19

## Phase E — optional documentation
20. Issue 22
21. Issue 23

---

# Minimal viable academic project
If time gets tight, the minimum strong submission is:
- data preprocessing,
- Gaussian HMM with EM,
- model selection over `K`,
- forward inference,
- simple prediction and evaluation,
- one clear discussion section about why IOHMM and MCMC were not fully reproduced.

This is already well aligned with course topics on estimation, EM, Bayesian inference, and tracking.

---

# Definition of done for the whole repository
The project is done when:
- the baseline HMM experiment runs end-to-end from raw data to figures,
- each public function is tested,
- each core module names its paper references,
- all simplifications are explicit,
- one side-information experiment exists,
- the final README can truthfully state what was replicated and what was approximated.

---

# README wording suggestion
Suggested one-sentence project description:

> Academic Python replication of a Hidden Markov Model / IOHMM-inspired intraday momentum paper, focused on clean implementation, interpretable inference, and presentation-ready experimental results.

