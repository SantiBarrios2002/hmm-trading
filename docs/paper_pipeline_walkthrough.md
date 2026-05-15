# Paper Pipeline Walkthrough — What We Have vs What We Excluded

This document walks through the Christensen / Turner / Godsill paper as a
step-by-step pipeline and notes, at each step, what the repo implements
(`✅` paper-faithful, `🟡` engineering approximation, `❌` excluded) and the
reason for each exclusion. It is the narrative companion to the structured
deviation table in [`docs/paper_spec.md`](paper_spec.md) and to the results
in [`docs/results_vs_paper.md`](results_vs_paper.md).

---

### 1. Data and preprocessing
- **Paper (§7):** 1-min ES futures, log returns `Δy_t`.
- **Repo:** ✅ Same. Databento parquet ES 1-min, `compute_log_returns` does `ln(p_t/p_{t-1})`. Window is 2019–2024 instead of the paper's exact sample but same instrument and frequency.

### 2. Hidden state model (baseline HMM, §3)
- **Paper:** Diagonal Gaussian HMM on returns; states ordered by mean return; one fixed transition matrix.
- **Repo:** ✅ Same. `GaussianHMMWrapper` wraps `hmmlearn`, exposes means/variances/transition/initial. States are ordered by mean. Variance floor (`min_variance`, §21) is an engineering knob the paper flags as needed on tick-grid data.

### 3. State initialization (§3.1)
- **Paper:** PLR (piecewise linear regression) used to seed the HMM.
- **Repo:** 🟡 PLR baseline exists (`plr_baseline.py`) and produces interpretable trend summaries, but it's not wired as the HMM init — `hmmlearn`'s default kmeans init is used. Could be wired in if you want full §3.1 fidelity; small change.

### 4. Parameter estimation Θ
- **Paper:** Two parallel routes —
  - **Baum-Welch EM**
  - **Metropolis-Hastings MCMC** on Θ
- **Repo:**
  - ✅ Baum-Welch via `hmmlearn`, with stabilized priors and monotone-LL test.
  - ❌ **MCMC on Θ excluded by `IMPLEMENTATION_PLAN.md §2.5`.** Reason: for a well-behaved Gaussian HMM with K=2–4 on minute returns, MH and BW converge to essentially the same Θ. MCMC adds days of compute and ~500–1000 LOC for a near-identical answer. The exclusion is a deliberate scope call, not an oversight.

### 5. Model selection (choosing K)
- **Paper (§4):** Three parallel routes —
  - **Cross-validation**
  - **AIC/BIC**
  - **Bridge sampling for marginal likelihood** (MCMC-based)
- **Repo:**
  - ✅ AIC/BIC fully implemented (`selection/model_selection.py`).
  - ❌ **CV not implemented.** Not formally excluded; just a gap. This is the cheapest of the three to add and is the most relevant fix to the BIC=4 overfit our recent ablation exposed.
  - ❌ **Bridge sampling excluded by §2.5.** Reason: heavy MCMC infrastructure for a comparison the AIC/BIC route already gives a defensible answer to. Same reason as MCMC on Θ.

### 6. Forward filtering and one-step prediction (§6)
- **Paper:** Standard normalized log-space forward recursion; `E[Δy_{t+1}] = P(m_{t+1}|Δy_{1:t}) · μ`.
- **Repo:** ✅ Same. `forward_filter.py` is paper-faithful, log-space, stabilized.

### 7. Side-information predictors (§4.2)
- **Paper:** Two predictors used independently and then combined in the IOHMM —
  - Predictor I: **volatility ratio** (short/long RiskMetrics EWMA).
  - Predictor II: **intraday seasonality** (exchange-local time of day).
- **Repo:**
  - ✅ Volatility ratio paper-faithful (`λ=0.79`, `ψ_fast=50`, `ψ_slow=100`).
  - 🟡 Seasonality: UTC→Chicago conversion is paper-faithful; the scalar bucket encoding fed into the spline is an engineering approximation.
  - 🟡 Spline fitting (§4.1) — deterministic least-squares cubic splines with quantile knots; the paper isn't fully specific on the procedure.

### 8. IOHMM transition conditioning (§4)
- **Paper:** Continuous parametric conditioning — `A` is a function of `x_t` through a side-information model.
- **Repo:** Both forms are now implemented and run side-by-side for ablation:
  - ✅ **Continuous-parametric HMC** (`models/iohmm_continuous.py`, Gate K, branch `feat/47-hmc-iohmm` / PR #48). `A_ij(x_t) = softmax_j(W_i · x_t + b_i)` fit with NumPyro NUTS per walk-forward window. Paper-faithful transition parameterization. Emissions remain at the Baum-Welch fit (the §2.5 exclusion on MCMC-of-Θ still stands) — NUTS samples only `(W, b)`.
  - 🟡 **Bucketed-transition approximation** (`models/iohmm_approx.py`). We discretize `x_t` into 3 buckets and fit one transition matrix per bucket with smoothing toward the pooled baseline. Retained as the engineering-approximation baseline for the grid-vs-continuous comparison.

### 9. Variants tested (§4)
- **Paper:** Baseline HMM, IOHMM-Predictor-I, IOHMM-Predictor-II, **and a combined IOHMM** that uses both predictors jointly.
- **Repo:**
  - ✅ Baseline HMM.
  - ✅ IOHMM with vol-ratio (bucketed).
  - ✅ IOHMM with seasonality (bucketed).
  - ✅ IOHMM with vol-ratio (HMC continuous-parametric, Gate K).
  - ✅ IOHMM with seasonality (HMC continuous-parametric, Gate K).
  - ❌ **Combined vol+seasonality IOHMM not implemented.** Not formally excluded — just deferred for plumbing reasons (would change `EXPECTED_VARIANTS` and break existing comparison_id hashes). The single most likely change to lift Sharpe and the recommended next variant.

The two HMC variants and their bucketed counterparts are run in one comparison (`configs/example_es_databento_side_info_comparison_hmc.yaml`) so the grid-vs-continuous ablation has a fair within-config baseline. As of writing, the full 6-year ES walk-forward run is in flight on CPU; results will land in `results_vs_paper.md` once it completes.

### 10. Asynchronous IOHMM
- **Paper:** Sketches an asynchronous variant for mixed-frequency inputs (e.g., daily macro features mixed with minute returns).
- **Repo:** ❌ **Excluded by §2.5.** Reason: synchronous IOHMM is already nontrivial; the asynchronous variant doubles the implementation complexity for a feature that doesn't apply at the single-frequency 1-min ES setup we're evaluating.

### 11. Walk-forward / retraining (§2.3)
- **Paper:** Train on previous H days (≈1 month), forecast forward, retrain.
- **Repo:** ✅ `walk_forward.py` defaults to `h_days=23, t_days=1, retrain_every_days=t_days`. Per-window leakage assertion is enforced. Configurable retrain cadence is a small extension on top of the paper's defaults.

### 12. Trading signal (§8)
- **Paper:** `Signal_t = TF(yhat_t)` — but `TF` is **underspecified**; the paper uses sign without saying so explicitly.
- **Repo:**
  - ✅ Sign policy (paper-faithful default).
  - 🟡 `thresholded_hold` (turnover-aware diagnostic, our addition).
  - 🟡 `conviction_weighted` (continuous positions, our recent addition — empirically negative on this data).

### 13. Strategy evaluation (§4.4, §7)
- **Paper:** Sum intraday returns by day, compute daily Sharpe, annualize by `sqrt(258)`.
- **Repo:** ✅ Same. `daily_annualized_sharpe_ratio` uses the paper's exact convention.

### 14. Cost model
- **Paper:** Vague — "post-cost Sharpe falls by about 15%" with no specification of bps, per-side/round-trip, spread, slippage, or fill model.
- **Repo:** 🟡 Linear `cost_bps_per_turnover` convention with default 1.0 bp. Conservative at 1-min cadence. Isolated as a diagnostic, not a paper-matching target.

### 15. Multi-asset / portfolio
- **Paper:** Sits in a broader intraday momentum literature; the paper itself focuses on single-asset.
- **Repo:** ❌ **Excluded by §2.5.** Single-asset by design. We do have parquet files for 6A, CL, GC, NQ, etc. but no portfolio aggregation.

### 16. Production execution
- **Paper:** Not addressed.
- **Repo:** ❌ **Excluded by §2.5.** No latency, slippage beyond flat cost, venue, or order-book modeling.

---

### What this means for your "extra"

Status as of now:

- ✅ **HMC continuous-parametric IOHMM (Gate K) is done** — closes the §8 approximation gap *and* delivers the Tema 2 MCMC contribution in one PR. Full benchmark is mid-run; results to be folded into `results_vs_paper.md`.
- 🟡 Gate L (DMM benchmark) is the planned next contribution (modern alternative on the state-space ladder, see scope justification §6).

Remaining gap-closing extras the doc still tracks, ranked by paper-fidelity vs cost:

| Extra | Closes which §2.5 / paper gap? | Implementation cost | Probable Sharpe impact |
|---|---|---|---|
| **Combined vol+seasonality IOHMM** | §4 combined predictor variant (the paper does this, we don't) | ~1 week | Moderate-to-high lift expected |
| **CV-based K selection** | §4 model-selection trio (we have 1/3) | ~1 week | Likely picks K=2 even with longer h, would invalidate the BIC=4 overfit |
| **PLR seeding of HMM init** | §3.1 init scheme (we have PLR but don't wire it in) | ~2 days | Small, but full §3.1 fidelity |
| **Quantile bucket boundaries** (Issue 42) | Tightens the bucketed IOHMM as the *fair baseline* for the Gate K HMC comparison; without quantile boundaries, part of any HMC win is attributable to bucket imbalance rather than continuous-parametric conditioning | ~3 days | Small on its own, but cleans up the grid-vs-continuous ablation story |
| **MCMC bridge sampling for K only** | §2.5 exclusion (now partially neutralized — Gate K already delivers MCMC credibility) | 2–3 weeks | Probably picks K=2, like CV would; redundant given Gate K is in |
| **Full MCMC on Θ** | §2.5 exclusion (the "I implemented MCMC on Θ" credential) | 3+ weeks | Negligible — Θ converges to BW result on this data |

If the goal is **better Sharpe**: combined IOHMM.
If the goal is **defensible "we addressed the paper's selection trio" narrative**: CV K selection.
The Tema 2 / MCMC ask is **already satisfied by Gate K** — bridge sampling for K is no longer the headline MCMC story.

---

## Brainstorm — original-contribution directions

The replication itself fulfills the course brief, but the project also needs
something *original* — not just a faithful re-implementation. Non-paper
directions that build naturally on the existing pipeline:

- **Cross-asset IOHMM with shared regimes.** The repo has 25+ futures
  parquets (6A, CL, GC, NQ, ZB, …). Fit one HMM with regimes inferred from a
  basket and trade each constituent on the shared state. Genuine extension,
  not in the paper.
- **Calibrated cost model from public ES microstructure.** Replace the
  `1.0` bp/turnover convention with a spread + impact model estimated from
  the data itself. Closes the §14 gap honestly, and gives the post-cost
  diagnostic real teeth.
- **Posterior-uncertainty-aware sizing.** Instead of `E[Δy]` (which we showed
  is a bad conviction signal), size positions by the *entropy* of the
  filtered state distribution, or by the predicted variance of the next
  return. This directly responds to the negative finding from the conviction
  ablation: maybe the right "conviction" measure isn't magnitude, but
  posterior concentration.
- **Regime-stability diagnostic.** Quantify how stable the fitted state means
  and transition probabilities are across walk-forward windows. The paper
  assumes stability; measuring it on 6 years of ES would be a small but
  original empirical contribution and a natural figure for the defense.

---

## Brainstorm — MCMC / Monte Carlo extensions

The paper's §2.5 exclusion of MCMC on Θ is defensible (BW converges to the
same answer), but there are MCMC-shaped contributions that *aren't* redundant
with Baum-Welch and that close real gaps in the current pipeline:

- **HMC on a continuous-parametric IOHMM transition.** ✅ **Implemented as
  Gate K** (`models/iohmm_continuous.py`, PR #48). `A_ij(x_t) = softmax_j(W_i · x_t + b_i)`
  fit with NumPyro NUTS per walk-forward window, NUTS samples only the
  transition logits `(W, b)` while emissions stay at the Baum-Welch fit. This
  closes the §8 approximation gap and delivers the MCMC contribution in one
  PR. Full benchmark is mid-run.
- **Bayesian model averaging across K.** Instead of picking one K, compute
  posterior weights over K ∈ {2, 3, 4} (bridge sampling or marginal
  likelihoods) and average forecasts. Closes §5 differently from CV —
  CV picks one K, BMA hedges. Original framing relative to the paper,
  which commits to a single K.
- **Posterior predictive checks via simulation.** Sample forward paths from
  the fitted HMM, build the model-implied Sharpe distribution, and locate
  the realized Sharpe in that distribution. A clean Monte-Carlo-flavored
  model-criticism figure ("realized Sharpe is at the 32nd percentile of
  the model-implied distribution") that's easy to defend.
- **Particle filter for non-Gaussian emissions.** Drop the Gaussian
  assumption and use SMC for online filtering with a heavy-tailed
  (Student-t) emission. The variance floor in §2 exists because tick-grid
  returns have fat tails — a particle filter fixes the root cause rather
  than clipping the symptom.

## Brainstorm — ML / DL extensions

An ML/DL block is feasible and natural here. Each of these ties back to a
specific gap or underspecification in the paper:

- **Neural IOHMM** (ML mirror of the HMC idea above). Parameterize `A(x_t)`
  with a small MLP, train end-to-end with the HMM likelihood via PyTorch
  and a custom forward-backward. Same §8 gap as the HMC variant, neural
  rather than Bayesian — lets the thesis say "we tested both Bayesian and
  neural fixes to the bucketed approximation."
- **LSTM / small Transformer baseline.** Apples-to-apples sequence-model
  competitor to the HMM on the same walk-forward windows. Answers the
  question every committee asks: "does an HMM with explicit regimes beat
  a generic sequence model on this signal?" Single ML addition that
  strengthens the replication story the most.
- **HMM-features → gradient-boosted classifier.** Feed filtered state
  probabilities, `E[Δy_{t+1}]`, and side info into LightGBM predicting the
  sign of the next return. Cheap (~3 days), hybrid model, natural ablation
  against pure HMM and against pure GBM.
- **RL for the trading signal `TF`.** The paper leaves §12 underspecified
  ("`Signal_t = TF(yhat_t)`" without saying what `TF` is). Train PPO or
  DQN on filtered posteriors with a cost-aware reward. High prestige,
  high overfitting risk on this much data — flag as an exploration, not
  a headline deliverable.
- **Deep Markov Model (Krishnan et al. 2017).** Neural emission and
  transition functions in a state-space model — the "full DL replacement"
  for the HMM. Most ambitious; least likely to win on Sharpe; strongest
  "we engaged seriously with modern alternatives" defense answer.

### Opinionated pairing

The "one MCMC + one ML item" pairing for defense-day strength:

1. **HMC IOHMM** — ✅ done as Gate K; closes §8 and satisfies the MCMC ask in
   one PR. Full benchmark mid-run.
2. **DMM benchmark (Gate L)** — chosen modern alternative; see scope
   justification §6. The "state-space ladder" framing makes the comparison
   tighter than an LSTM baseline.

---

## Scope and methodology justification

This section consolidates the reasoning behind the major inclusion and
exclusion decisions in the replication, intended as a single defensible
narrative for thesis review.

### 1. Replication is targeted, not exhaustive

Christensen/Turner/Godsill propose a multi-pronged methodology
(HMM + IOHMM, EM + MCMC, AIC/BIC + CV + bridge sampling,
synchronous + asynchronous variants). A faithful end-to-end re-
implementation is roughly 12–18 months of work and produces a Sharpe
number already in the paper. The ASPTA brief asks instead for a
defensible simulation of the central results plus an original
contribution. The repo reflects that: the *core forecasting pipeline*
(HMM, IOHMM, Baum-Welch, forward filter, walk-forward, Sharpe
evaluation) is paper-faithful; the *methodological side branches* (MCMC
on Θ, bridge sampling for K, asynchronous IOHMM) are excluded with
reason.

### 2. MCMC on Θ — excluded because Baum-Welch already wins this argument

The paper presents Baum-Welch and Metropolis-Hastings MCMC on Θ as
alternative routes to the same posterior. For a diagonal-Gaussian HMM
with K ∈ {2, 3, 4} on minute returns, the likelihood surface is well-
behaved — EM finds the same optimum on every restart we have tested,
and the implied posterior is sharply concentrated. MCMC and BW converge
to essentially identical Θ in this regime. Implementing a generic MH
sampler costs ~500–1000 LOC and days of compute for an answer within
numerical noise of what `hmmlearn` already produces. The Tema 2 / MCMC
contribution is reallocated to a place where MCMC actually changes the
answer: the IOHMM transition function (see `dmm_mcmc_roadmap.md`).

### 3. Bridge sampling for K — excluded for the same reason

The paper's third model-selection route is Bayesian marginal likelihood
via bridge sampling. AIC and BIC are already implemented and, on this
dataset, the disagreement between them (BIC overfits to K=4 with
h_days=60) is more cleanly resolved by cross-validation than by bridge
sampling — and CV is roughly one tenth the implementation cost. Bridge
sampling would close the §4 selection-trio formally but is unlikely to
yield a different K than CV.

### 4. Asynchronous IOHMM — excluded by single-frequency data

The asynchronous variant exists to handle mixed-frequency inputs (daily
macro features alongside minute returns). Our data is all 1-min ES plus
features derived from that same series, so synchronous IOHMM is
sufficient. Asynchronous machinery would roughly double the IOHMM code
surface for a feature with nothing to operate on in this setup.

### 5. Bucketed IOHMM transitions — the approximation that motivated Gate K

The paper specifies a continuous-parametric `A(x_t)`. The repo initially
discretized `x_t` into three buckets and fit one transition matrix per
bucket — an engineering approximation labeled as such in `paper_spec.md`.
**Gate K (`models/iohmm_continuous.py`, PR #48) now implements the
paper-faithful continuous-parametric form**, fit with NumPyro NUTS per
walk-forward window. This closes the §8 approximation gap and absorbs
the Tema 2 MCMC requirement in a single contribution. The bucketed
variant is retained as the engineering-approximation baseline so the
grid-vs-continuous ablation has a fair within-config comparison.

### 6. DMM as the chosen modern-alternative benchmark

LSTM, Transformer, and deep state-space models could all serve as the
"modern baseline." DMM is chosen because it sits one rung above HMM on
the state-space ladder — generalizing the discrete latent
(→ continuous) and the linear dynamics (→ neural) — while sharing the
same forward-filtering / latent-state-inference structure. Both models
forecast `E[Δy_{t+1}]` from a latent state estimate, so the comparison
is interpretable. An LSTM comparison would pit a latent-variable model
against a non-latent sequence model and would not produce the
state-space-ladder arc the course syllabus invites.

### 7. Multi-asset and production execution — out of scope by design

The paper itself is single-asset. The repo has parquet files for 25+
futures contracts but no portfolio layer. Adding portfolio aggregation
changes the evaluation surface (covariance, sizing, rebalancing) and
the question being asked. Production execution (latency, slippage,
venue, order book) is a separate research program unrelated to the
regime-detection focus of this paper. Both are listed in
`paper_spec.md §2.5` as scope-bounded exclusions, not gaps.

