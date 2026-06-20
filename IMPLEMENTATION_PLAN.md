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

### 2.5 Scope: faithful replication target + planned extensions

The project's scope was widened on 2026-05-17 from a "targeted replication
plus extensions" framing to a **100% paper-faithful replication plus extensions**
framing. The original exclusions of MCMC on Θ and bridge sampling for K were
based on the argument that "Baum-Welch and MCMC converge to the same answer";
the paper itself (p.18) contradicts that — MCMC HMM is reported as
*worse* than Baum-Welch on the same instrument and timeframe, and that result
is precisely what a faithful replication should reproduce or refute on the
local Databento ES sample.

#### Paper-faithful trading-model table (target: 6/6)

The paper's headline trading-model comparison (Figure 8) lists six models. Of
those:

- ✅ **Baum-Welch HMM**, **Volatility-Ratio IOHMM**, **Seasonality IOHMM**, **Long-only** — implemented and reported in `docs/results_vs_paper.md`.
- ✅ **Default HMM** (PLR-derived emission means/variances + uniform transition matrix A) — Gate N, registered as `default_hmm` in the side-information comparison runner.
- ❌ **MCMC HMM** (Metropolis-Hastings on Θ) — *missing*, planned as Gate O.

#### Paper-faithful K-selection trio (target: 3/3)

Paper §4 picks K via three independent routes:

- ✅ **AIC / BIC** — implemented; documented BIC overfit on `h_days=60` (selects K=4 in every window, Sharpe drops).
- ❌ **Cross-validation** for K — *missing*, planned as Gate M. Expected to resolve the BIC overfit empirically.
- ❌ **MCMC bridge sampling** for K — *missing*, planned as Gate P. Reuses MCMC HMM (Gate O) posterior samples; not a standalone MCMC build.

#### Still excluded by paper scope
These genuinely remain out of scope for the coursework replication. Listed
here so the grader does not mistake their absence for oversight.

- **Asynchronous IOHMM.** The paper sketches an asynchronous variant for mixed-frequency inputs. This repo implements only the synchronous IOHMM approximation; the project's single-frequency 1-minute ES data does not exercise the asynchronous path.
- **Multi-security / portfolio backtest.** Evaluation is single-security (ES or an equivalent proxy). No cross-asset construction.
- **Production execution concerns.** No latency modeling, slippage beyond a flat cost-per-turnover, venue microstructure, or order-book effects.

#### Planned extensions beyond paper scope
These are not in the paper but extend the replication along directions the
syllabus invites (state-space ladder: HMM → Kalman / PF → DMM; Bayesian
inference for IOHMM transition functions).

- **HMC on continuous-parametric IOHMM transitions (Gate K).** ✅ Shipped via PR #48. Replaces the bucketed approximation in §8 with `A(x_t) = softmax(W x_t + b)` fit via NUTS in NumPyro. Distinct from Gate O's "MCMC on Θ": HMC here samples the *transition function*, not the Gaussian HMM parameters. Headline result is a negative Sharpe finding; see `docs/results_vs_paper.md`.
- **Deep Markov Model benchmark (Gate L).** Pyro-based DMM (Krishnan et al. 2017, paper PDF in `docs/`) as the nonlinear-state-space generalization of the HMM. Trained with variational inference (its native algorithm), compared against the HMM / IOHMM / HMC IOHMM variants on the same walk-forward rig.

**Gates K and L are extensions; Gates M, N, O, P are faithful-replication completions.** All six (M, N, O, P, K, L) are tracked under §5 "Recommended delivery order" below.

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
**Covers:** Issues 05, 06, 21

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
- The wrapper exposes a minimum-variance floor with a documented default tied to the instrument tick size. The paper flags that Gaussian emission variance can collapse below a meaningful threshold on tick-grid data, so this floor must be a first-class configuration knob rather than an implicit backend default.
- Baum-Welch EM produces a monotone non-decreasing log-likelihood across iterations on every tracked fixture. The chosen backend settings (`tol`, `n_iter`, `min_covar`, init scheme) are documented in the wrapper module docstring. If monotonicity cannot be achieved with `hmmlearn`, the wrapper falls back to a custom log-space forward-backward + M-step, labeled as an engineering approximation.

### Evidence expected in PR review
- test fixtures for segmented trends
- synthetic regime-switching example
- docstrings referencing the corresponding paper sections
- a deterministic test asserting monotone EM log-likelihood across iterations
- a variance-floor test (either clamps to the floor or raises with a documented rule)
- a clean `scripts/repro.py configs/example_es_csv.yaml` run with no `Model is not converging` warnings

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
**Covers:** Issues 13, 14, 15, 22

### Must pass
- Volatility ratio feature is implemented and tested.
- Intraday seasonality feature is implemented and tested.
- Spline fitting exists with a documented Python approximation.
- The spline interface is deterministic for fixed inputs and configuration.
- The code labels clearly which parts are paper-faithful and which are approximate.
- Each side-information predictor (volatility ratio, intraday seasonality) has a standalone walk-forward backtest running on the same experiment rig as the baseline HMM, producing its own `runs/<run_id>/` artifact. This mirrors the paper's §4 structure, where each spline-based predictor is evaluated in isolation before being folded into the IOHMM, and it makes the Gate H comparison interpretable.
- The standalone-predictor signal path contains no HMM state object; the sign of the predicted return comes from evaluating the fitted spline at `x_t`.
- If a predictor has no standalone traction on the evaluation window, the result is recorded in the PR notes and then copied into `docs/experiment_log.md` during Gate I so the Gate H outcome can cite it later.

### Evidence expected in PR review
- feature construction tests
- spline fit/evaluate tests
- at least one visualization produced from script code
- two tracked `runs/<run_id>/` artifacts (one per predictor) reproducible via `scripts/repro.py`
- an integration test that the standalone-predictor signal path does not instantiate a Gaussian HMM

---

## Gate H — IOHMM-style transition conditioning
**Covers:** Issues 16, 17, 42

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
**Covers:** Issues 18, 20, 23

### Must pass
- Plotting functions run from scripts, not only notebooks.
- The repo can generate figures suitable for:
  - state timeline
  - model selection
  - spline predictor
  - cumulative returns
- A concise paper notes document exists.
- A concise experiment log exists.
- `docs/paper_spec.md` exists as a structured table with columns: **component**, **paper says** (with §/figure reference), **repo interpretation**, **deviation type** (paper-faithful / engineering approximation / evaluation-layer / excluded by §2.5), **acceptance risk**. This replaces any informal faithful-vs-approximate prose and is the single document a thesis committee reads to understand interpretive choices.
  The current paper-fidelity map lives at [`docs/paper_spec.md`](docs/paper_spec.md).

### Evidence expected in PR review
- generated figures stored under `docs/figures/` or equivalent
- smoke tests for plotting functions
- reviewed `docs/paper_notes.md`, `docs/experiment_log.md`, and `docs/paper_spec.md`

---

## Gate J — Paper-comparison results
**Covers:** Issue 24

The goal of this gate is not to reproduce the paper's numbers exactly, but to show honestly where the repo lands relative to them and which directional claims it reproduces.

### Must pass
- `docs/results_vs_paper.md` exists and contains a comparison table with rows for baseline HMM, volatility-ratio IOHMM, seasonality IOHMM, and long-only benchmark, and columns for pre-cost Sharpe, post-cost Sharpe at the documented `cost_bps`, hit rate, cumulative return, chosen `K`, sample window, paper reference (§/table), and repo `run_id`.
- Every repo number in the table is backed by a live `runs/<run_id>/` artifact reproducible via `scripts/repro.py`. No placeholders.
- Every gap between a repo number and its paper counterpart is categorized as one of: **data scope** (different window / vendor), **model scope** (§2.5 exclusions), **implementation approximation** (bucketed-A, continuous Gaussian emissions, etc.), or **numerical / stochastic variation**.
- At least one directional claim from the paper is reproduced or explicitly discussed:
  - model selection points to `K ∈ {2, 3}`
  - volatility-conditioned IOHMM outperforms baseline HMM on pre-cost Sharpe
  - seasonality IOHMM outperforms baseline HMM on pre-cost Sharpe
- Pre-cost Sharpe of the best side-information variant is reported alongside the paper's reference figure (≈2 per §4.4) with a short honest gap analysis.

### Evidence expected in PR review
- `docs/results_vs_paper.md` rendered with the full table and gap narrative
- all `runs/<run_id>/` artifacts cited in the table present under `runs/`
- one comparison figure under `docs/figures/` produced by Gate I's plotting code (e.g., cumulative-return overlay or Sharpe bar chart)

---

## Gate K — HMC on continuous-parametric IOHMM transitions (extension)
**Covers:** §2.5 planned extension. Issue and branch to be opened when work starts (suggested: `feat/NN-hmc-iohmm`).

This gate is the MCMC half of the planned extensions. It is *not* required
for Gates A–J to pass; it is the contribution that converts the bucketed
§8 engineering approximation into the paper's continuous-parametric form
and supplies the Tema 2 MCMC piece of the syllabus.

### Must pass
- A NumPyro or PyMC model fits the transition function `A(x_t) = softmax(W x_t + b)` with discrete HMM states marginalized in the forward filter.
- NUTS (or HMC) is used; chain count, warmup, and adaptation settings are recorded in the run artifact.
- The HMC IOHMM is registered as an additional variant alongside the existing bucketed-grid and bucketed-quantile variants, producing a three-way ablation (grid / quantile / HMC continuous) that isolates bucket-placement effects from continuous-conditioning effects.
- Posterior diagnostics (R-hat for each parameter, ESS per chain) are stored per window in the run artifact and clearly flagged when convergence is poor.
- A short note in the module docstring distinguishes "HMC on transition logits" (this gate) from "MCMC on Θ" (explicitly excluded by §2.5).

### Evidence expected in PR review
- Convergence-diagnostic summary across walk-forward windows
- Three-way ablation table (grid / quantile / HMC) on the same fixture
- A run artifact under `runs/<run_id>/` reproducible via `scripts/repro.py`

---

## Gate L — Deep Markov Model benchmark (extension)
**Covers:** §2.5 planned extension. Issue and branch to be opened when work starts (suggested: `feat/NN-dmm-benchmark`).

This gate is the deep-learning half of the planned extensions, positioned
as the nonlinear-state-space generalization of the HMM. It is *not*
required for Gates A–J to pass.

### Must pass
- A Pyro-based DMM (Krishnan et al. 2017 — paper PDF in `docs/`) wraps emission MLP, transition MLP gated on side information, and a structured inference network.
- The DMM produces a per-window `expected_next_returns` series (`E[Δy_{t+1} | Δy_{1:t}]`) consumed by the signal layer (`strategy/signals.py`, `build_signal`) — the repo's actual forecast contract; there is no `predict_next_return` method. It is registered in `EXPECTED_VARIANTS` and dispatched through the existing `_checkpointed_stage` per-variant runner machinery (alongside `_run_side_info_variant` / `_run_default_hmm_variant`), with no parallel code path.
- Walk-forward integration uses the same data loaders and evaluation layer as the HMM variants; any sampling-frequency change (e.g., 5-min training to keep compute tractable) is documented and recorded in the config.
- Training settings (epochs, batch size, optimizer, KL annealing schedule) are recorded in the run artifact for reproducibility.
- A negative-result discussion is acceptable and expected — if DMM does not beat the HMM on Sharpe, the result is reported alongside the conviction-weighted negative-result precedent.

### Evidence expected in PR review
- Reproducibility check: rerun via `scripts/repro.py` yields the same metrics within documented stochastic tolerance
- Comparison table including HMM / IOHMM (grid + quantile) / HMC IOHMM (if Gate K landed) / DMM
- One figure showing latent-state trajectories or filtered-state probabilities under the DMM

---

## Gate M — Cross-validation for K selection (faithful replication)
**Covers:** §2.5 paper-faithful K-selection trio. Issue and branch to be opened when work starts (suggested: `feat/NN-cv-k-selection`).

Paper §4 picks K via three independent routes: cross-validation, AIC/BIC,
and MCMC bridge sampling. The repo currently implements only AIC/BIC, and
the BIC route exhibits a documented overfit on `h_days=60` (selects K=4 in
every walk-forward window, Sharpe drops on every variant). This gate adds
the missing CV route and tests whether CV resolves the BIC overfit
empirically.

### Must pass
- A walk-forward cross-validation routine selects `K` by held-out predictive performance (per-bar log-likelihood or expected-return-aligned predictive score), wired into the existing `WalkForwardConfig.k_values` sweep so a config can specify `k_selection: cv` alongside the existing `best_by_bic` path.
- CV is exercised on **both** the canonical `h_days=23` regime and the BIC-overfit `h_days=60` regime so the chosen-K distribution can be compared head-to-head against BIC on the same windows.
- The selection procedure (fold count, split policy, scoring metric) is deterministic for a given seed and documented in the config and module docstring.
- The chosen-K-per-window distribution is persisted in the run artifact alongside the existing BIC chosen-K record.
- Tests verify the CV selector on a synthetic sequence with a known optimal K (e.g., generated from K=2 then evaluated over K ∈ {2, 3, 4}).

### Evidence expected in PR review
- Side-by-side BIC vs CV chosen-K histograms on `h_days=23` and `h_days=60`
- Sharpe-table comparison under CV vs BIC on both training-window lengths
- One paragraph in `docs/results_vs_paper.md` interpreting the BIC-vs-CV result (does CV pick K=2 on `h_days=60`, restoring the Sharpe?)

---

## Gate N — Default HMM trading variant (faithful replication)
**Covers:** §2.5 paper-faithful trading-model table. Issue and branch to be opened when work starts (suggested: `feat/NN-default-hmm`).

Paper Figure 8 includes a "Default HMM" variant — emission means derived
from PLR segments, transition matrix `A` held at uniform `(1/K) · ones((K, K))`
(no transition learning). The repo already implements PLR
(`plr_baseline.py`) but does not wire it as a runnable trading variant.

### Must pass
- A `default_hmm` variant is added to the side-information comparison runner. Emission means/variances come from PLR segment statistics on the training window; `π` and `A` are uniform.
- The variant uses the existing forward filter, signal builder, and walk-forward rig — no parallel code path.
- The PLR seeding is deterministic for a given seed.
- The variant is included in the headline comparison table alongside Baum-Welch HMM, vol-ratio IOHMM, seasonality IOHMM, and long-only, so the table now lists 5 of the paper's 6 models (the 6th, MCMC HMM, lands in Gate O).
- Tests verify uniform-`A` behavior and that PLR-seeded means propagate through the variant unchanged.

### Evidence expected in PR review
- A row for `default_hmm` in `docs/results_vs_paper.md` headline table
- One figure or short comparison showing that the Default HMM's Sharpe is the worst of the group (paper's claim, p.17), and an honest commentary if it isn't on the local sample

---

## Gate O — MCMC on Θ for the Gaussian HMM (faithful replication) ⏸️ DEFERRED — documented future work
**Covers:** §2.5 paper-faithful trading-model table. Issue and branch to be opened when work starts (suggested: `feat/NN-mcmc-hmm`).

> **Deferred (2026-06-19).** Not on the active path. The MCMC contribution is
> already delivered by Gate K (HMC on the IOHMM transition logits), so this
> gate's only added value is table completeness (6/6 paper models) for an
> outcome the paper and this project both predict to be a null — MH on Θ for a
> well-identified diagonal-Gaussian HMM lands within numerical noise of the
> Baum-Welch fit. The spec below is retained should the gate be revisited; the
> deferral rationale is `dmm_mcmc_roadmap.md` §7.1 and §4 of this document.

Paper §3 fits Θ = {π, A, μ, σ²} by Metropolis-Hastings as an alternative to
Baum-Welch. Paper Figure 8 reports the MCMC HMM as **worse** than Baum-Welch
(MCMC fails to beat long-only while BW does); the paper attributes this to
prior selection and proposal-density issues. This gate replicates that
finding (or refutes it) on the local Databento ES sample.

### Must pass
- A NumPyro Metropolis-Hastings sampler (or PyMC; same toolchain as Gate K is preferred) samples Θ = {π, A, μ, σ²} for K ∈ {2, 3} on the training window.
- The paper's priors are followed where specified: uniform on `A` (Dirichlet with concentration 1), weakly-informative on `μ` and `σ²`. Any deviation is documented in the module docstring with the paper-page citation it replaces.
- The point estimate retained for trading is the highest-posterior-probability sample (paper convention, p.17).
- The variant is registered as `mcmc_hmm` in the side-information comparison runner, so the headline table reaches 6/6 paper models.
- Convergence diagnostics (R-hat, ESS per parameter, acceptance rate) are stored per walk-forward window in the run artifact and flagged when poor.
- The module docstring distinguishes "MCMC on Θ" (this gate) from "HMC on transition logits" (Gate K) — the two are different problems, not the same one.

### Evidence expected in PR review
- Convergence-diagnostic summary across walk-forward windows
- A row for `mcmc_hmm` in `docs/results_vs_paper.md` headline table
- A direct comparison of Baum-Welch Θ vs MCMC Θ on at least one window, with discussion of whether the paper's "MCMC underperforms BW" finding reproduces on the local data
- A reproducibility check via `scripts/repro.py`

---

## Gate P — Bridge sampling for K selection (faithful replication) ⏸️ DEFERRED — documented future work
**Covers:** §2.5 paper-faithful K-selection trio. Issue and branch to be opened when work starts (suggested: `feat/NN-bridge-sampling`). **Depends on Gate O** — reuses MCMC HMM posterior samples.

> **Deferred (2026-06-19).** Depends on Gate O (also deferred) for posterior
> samples, and its standalone value is marginal: the BIC-overfit-on-`h_days=60`
> story is already resolved by Gate M's cross-validation route. Bridge sampling
> would add a confirming *third* K-selection witness, not a new result. Retained
> as documented future work; rationale in `dmm_mcmc_roadmap.md` §7.5 and §4 below.

Paper §4's third K-selection route is Bayesian marginal likelihood via
bridge sampling. With Gate O's MCMC posterior samples in hand, bridge
sampling is a wrapper that computes the marginal likelihood `p(Y | M_k)`
from the same chains; this gate completes the §4 K-selection trio
(CV + AIC/BIC + bridge sampling).

### Must pass
- A bridge-sampling estimator wraps the Gate O posterior samples and produces a marginal-likelihood estimate per candidate K on each walk-forward window.
- The estimator is tested against an analytically tractable toy model (e.g., conjugate Gaussian with a known marginal likelihood) before being applied to the HMM posterior.
- Bridge sampling is exercised on **both** `h_days=23` and `h_days=60` so the K-selection trio (BIC vs CV vs bridge sampling) can be compared head-to-head, mirroring Gate M's evaluation regime.
- The chosen-K distribution from bridge sampling is persisted in the run artifact alongside the existing BIC and Gate M CV records.

### Evidence expected in PR review
- Three-way K-selection comparison table (BIC vs CV vs bridge sampling) on both `h_days=23` and `h_days=60`
- A short paragraph in `docs/results_vs_paper.md` interpreting whether all three routes agree on this data
- A reproducibility check via `scripts/repro.py`

---

## Gate Q — Combined volatility-ratio + seasonality IOHMM (faithful replication) ✅ SHIPPED — negative result
**Covers:** §4 combined predictor variant. Landed via PR #53 (branch `feat/52-combined-iohmm`), headline rerun `runs/22bbd2c8d0a4` (2026-06-02).

Paper §4 evaluates IOHMM-Predictor-I (vol-ratio), IOHMM-Predictor-II
(seasonality), **and a combined IOHMM** that conditions transitions on
both predictors jointly. The repo had both single-predictor variants in
bucketed and HMC-continuous form; this gate added the combined variant as a
natural extension of the Gate K continuous-parametric form to vector-valued
side information (`x_t ∈ ℝ²`).

**Outcome — negative result.** The combined variant scored **0.5453** pre-cost
Sharpe on the 6-year Databento ES rerun — *below* both single predictors
(vol-ratio HMC 0.6515, seasonality HMC 0.6281) and barely above baseline
(0.5298); the best configuration found remains vol-ratio **bucketed** (0.7583).
Joint conditioning diluted the vol-ratio edge rather than stacking the signals;
all 92 windows converged cleanly (rhat ≈ 1.00–1.003), so it is a real finding,
not a sampler artifact. This **falsifies** the prior "single most likely Sharpe
lift" prediction and makes Gate Q the project's third consistent negative
result. See [`docs/results_vs_paper.md`](docs/results_vs_paper.md).

### Must pass
- A new variant `vol_ratio_seasonality_hmc_continuous` is registered in `EXPECTED_VARIANTS`, fit via NumPyro NUTS with `A_ij(x_t) = softmax_j(W_i · x_t + b_i)` where `x_t ∈ ℝ²` carries vol-ratio and seasonality jointly.
- The `iohmm_continuous` module is generalized from scalar `x_t` to vector `x_t` so the same forward filter, posterior-mean transition function, and convergence diagnostics are reused — no parallel code path. Existing D=1 callers (vol-ratio HMC, seasonality HMC) remain bit-for-bit identical.
- Per-feature standardization on the training slice (each component of `x_t` has its own training mean/std). Leakage-free.
- Convergence is gated by the existing `ContinuousIOHMMConfig` thresholds; per-window divergent fits flagged as in Gate K.
- Tests verify (a) D=1 collapses to the existing Gate K result on the single-feature variants, (b) combined transitions are row-stochastic and finite, (c) standardization fits only on the training slice, (d) the variant uses the existing forward filter and walk-forward rig with no parallel code path.

### Evidence expected in PR review
- A new row for `vol_ratio_seasonality_hmc_continuous` in the headline table or in a dedicated combined-IOHMM subsection of `docs/results_vs_paper.md`
- Side-by-side comparison: combined vs vol-ratio HMC vs seasonality HMC pre-cost Sharpe, with an interpretation of whether joint conditioning captures information neither single predictor catches alone
- Per-window convergence summary (R-hat, ESS) for the combined variant, matching the Gate K reporting convention
- A reproducibility check via `scripts/repro.py`

### Why this gate was high-leverage (and how it resolved)
- **Paper-faithful coverage (achieved):** paper §4 has the combined variant; the repo now matches it. Closed a real coverage gap, not a refinement.
- **Hypothesized Sharpe lift (falsified):** vol-ratio and seasonality are individually useful (0.76 and 0.63 pre-cost Sharpe vs 0.53 baseline), and the conviction-weighted, K-sweep, and h=60 ablations had all failed without testing joint conditioning — so cross-effects ("high vol-ratio behaves differently at the open than at midday") were the unexplored axis. The rerun tested it: joint conditioning **did not** capture additive or cross-predictor structure; it diluted the vol-ratio edge (0.5453, below either predictor alone). The coverage value held; the Sharpe hypothesis did not.
- **Methodologically distinct from Gate K:** new transition structure, not a new inference method. Preserves the §2.5 exclusion (NUTS still samples only `(W, b)`).

### Cost (actual)
Implementation + tests + docs, plus one headline rerun (`runs/22bbd2c8d0a4`, ~30h elapsed on local CPU including HMC across all variants).

---

## 4. Definition of done for the project

Two tiers, reflecting the scope widening of 2026-05-17:

### Minimum submittable

- Gates A through F are fully passed
- at least part of Gate G is passed
- Gate H is passed in approximate form at minimum
- Gate I is passed well enough to support the oral presentation
- Gate J produces at least one reproduced directional claim from the paper, with an honest gap analysis

This level corresponds to the "targeted replication" framing the project
originally operated under. It is enough to clear the coursework brief.

### Paper-faithful core + extensions (current target, revised 2026-06-19)

In addition to the minimum above:

- **Trading-model table reaches 5/6 paper variants:** Default HMM (Gate N ✅) lands alongside the existing four (Baum-Welch HMM, vol-ratio IOHMM, seasonality IOHMM, long-only). The 6th, MCMC HMM (Gate O), is **deliberately deferred** — see scope note below.
- **K-selection reaches 2/3 paper routes:** cross-validation (Gate M) lands alongside AIC/BIC. The third route, bridge sampling (Gate P), is **deliberately deferred** — see scope note below.
- **Combined IOHMM (Gate Q) ✅ landed** (PR #53) so the §4 trading-variant inventory matches the paper's combined-predictor experiment in addition to the single-predictor ones — headline outcome is a negative result on Sharpe.
- **Extensions land in order:** Gate K (HMC IOHMM, ✅ shipped), then Gate L (DMM benchmark) — the next active extension once Gate M lands.

A strong project at this level passes Gates A–N and Q, lands the Gate L
extension, and documents Gates O and P as deferred future work with the
rationale below.

#### Scope decision (2026-06-19): defer Gate O and Gate P

Gates O (MCMC on Θ) and P (bridge sampling) are removed from the active path
and demoted to documented future work. Rationale:

- **The MCMC contribution is already delivered by Gate K.** Gate K puts HMC on
  the IOHMM transition logits `(W, b)` — a problem where MCMC *changes* the
  answer rather than restating it. MH on Θ for a well-identified
  diagonal-Gaussian HMM converges to essentially the Baum-Welch estimate, so
  Gate O's expected outcome is a null result the paper (Fig 8) already reports.
  It buys table completeness (6/6) at ~1–2 weeks of compute for no new finding.
- **The BIC-overfit story is already carried by Gate M.** Bridge sampling
  (Gate P) would add a confirming *third* K-selection route, but cross-validation
  (Gate M) resolving the overfit is sufficient for the clean positive narrative;
  Gate P also depends on Gate O, which is deferred.
- **The deep-model benchmark (Gate L) is the higher-novelty, longer-pole work.**
  Starting it sooner is the better use of remaining runway than shipping two
  expected-null results for inventory completeness.

This reverts the 2026-05-17 scope-widening's re-inclusion of Gate O/P and
re-aligns the plan with `dmm_mcmc_roadmap.md` §7.1 and §7.5, which already
argued for the same exclusion. The defense framing — "MCMC was placed where it
changes the answer (Gate K), not where it restates the EM fit (Gate O)" — is
the same one written there.

The BIC overfit on `h_days=60` documented in `docs/results_vs_paper.md` is
revisited once Gate M lands: if cross-validation picks K=2 on `h_days=60`, the
negative result becomes "BIC fails on long windows; the paper's
cross-validation route correctly resolves it" — a clean defense story that does
not require the deferred bridge-sampling route.

---

## 5. Recommended delivery order

Merge order, organized by phase. Phases 1–2 are already complete; phase 3
is the current path-B work.

### Phase 1 — Replication scaffold (complete)

1. Gate A
2. Gate B
3. Gate C
4. Gate D
5. Gate E
6. Gate F
7. Gate G
8. Gate H
9. Gate I
10. Gate J

### Phase 2 — Gate K (complete)

11. Gate K — HMC continuous-parametric IOHMM (✅ merged in #48; negative-result outcome documented in `docs/results_vs_paper.md`).

### Phase 3 — Path B: paper-faithful completion + DMM (current)

In dependency order, with each step landing as its own PR. Cross-cutting
constraint: Gates M and P should both be exercised on `h_days=23` *and*
`h_days=60` so the BIC overfit can be compared against CV and bridge
sampling head-to-head on the same windows.

12. **Gate N — Default HMM variant** ✅ shipped (PR #51). Closed the trading-table gap; smallest piece of work in path B.
13. **Gate Q — Combined vol-ratio + seasonality IOHMM** ✅ shipped (PR #53, run `22bbd2c8d0a4`). Closed the §4 combined-predictor coverage gap; the hypothesized Sharpe lift was **falsified** (0.5453, below either single predictor).
14. **Gate M — Cross-validation for K selection** (~1 week, no dependencies). Resolves the BIC overfit empirically; standalone, no MCMC machinery needed. ← **in progress** (branch `feat/50-cv-k-selection`).
15. **Gate L — Deep Markov Model benchmark** (~3–4 weeks, depends on nothing in path-B; the largest single piece of work). Closes the modern-alternative story. **Next active extension after Gate M.**

**Deferred (2026-06-19), documented future work — see §4 scope decision:**

- **Gate O — MCMC on Θ** (~1–2 weeks, would reuse NumPyro infra from Gate K). Expected-null result (MH on Θ ≈ Baum-Welch); the MCMC contribution is already delivered by Gate K.
- **Gate P — Bridge sampling for K** (~0.5 week, depends on Gate O). Confirming third K-selection route; the BIC-overfit story is already carried by Gate M.

### Phase 3 calendar estimate

With Gates O and P deferred (see §4), the active remaining path is Gate M
(~1 week, in progress) followed by Gate L (~3–4 weeks). Gates N and Q are
already shipped. The deferred Gate O (~1–2 weeks) and Gate P (~0.5 week) are
not counted in the remaining estimate.

This keeps the repo academically coherent and easy to review, with each
phase a clean "what's faithfully replicated, what extends the paper" line
to defend.
