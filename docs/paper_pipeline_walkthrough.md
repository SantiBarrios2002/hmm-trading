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
  - ✅ **Continuous-parametric HMC** (`models/iohmm_continuous.py`, Gate K, merged in #48). `A_ij(x_t) = softmax_j(W_i · x_t + b_i)` fit with NumPyro NUTS per walk-forward window. Paper-faithful transition parameterization. Emissions remain at the Baum-Welch fit (the §2.5 exclusion on MCMC-of-Θ still stands) — NUTS samples only `(W, b)`. Full 6-year ES benchmark complete (run `d8b6e7eef6c2`); 183/184 fits converged cleanly, one isolated divergence on vol-ratio window 19. **Outcome is a negative result on Sharpe** — see `results_vs_paper.md`.
  - 🟡 **Bucketed-transition approximation** (`models/iohmm_approx.py`). We discretize `x_t` into 3 buckets and fit one transition matrix per bucket with smoothing toward the pooled baseline. Retained as the engineering-approximation baseline for the grid-vs-continuous comparison.

### 9. Variants tested (§4)
- **Paper:** Baseline HMM, IOHMM-Predictor-I, IOHMM-Predictor-II, **and a combined IOHMM** that uses both predictors jointly. Figure 8 also reports a **Default HMM** (PLR-derived emissions, uniform `A`) as the deliberately-weak floor of the trading-model comparison.
- **Repo:**
  - ✅ Baseline HMM.
  - ✅ IOHMM with vol-ratio (bucketed).
  - ✅ IOHMM with seasonality (bucketed).
  - ✅ IOHMM with vol-ratio (HMC continuous-parametric, Gate K).
  - ✅ IOHMM with seasonality (HMC continuous-parametric, Gate K).
  - ✅ Combined vol+seasonality IOHMM (HMC continuous-parametric, Gate Q) — natural extension of the Gate K continuous form to vector `x_t ∈ ℝ²`, with volatility-ratio and intraday-seasonality conditioning transitions jointly. NUTS still samples only `(W, b)` and emissions remain at the Baum-Welch fit.
  - ✅ Default HMM (`models/default_hmm.py`, Gate N) — PLR segment statistics seed the emission means/variances, `A` and `π` are held at `1/K`. Uses the shared forward filter and walk-forward rig with no parallel code path.

The single-predictor HMC variants and their bucketed counterparts are run in one comparison (`configs/example_es_databento_side_info_comparison_hmc.yaml`) so the grid-vs-continuous ablation has a fair within-config baseline. Gate Q adds `configs/example_es_databento_side_info_comparison_combined.yaml` for the combined-predictor capability; the full headline Sharpe rerun is tracked separately from the capability PR. The full Gate K 6-year ES walk-forward run completed (artifacts at `runs/d8b6e7eef6c2`); see `results_vs_paper.md` for the headline numbers and the negative-result discussion.

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

- ✅ **HMC continuous-parametric IOHMM (Gate K) is shipped** — closes the §8 approximation gap *and* delivers the Tema 2 MCMC contribution in one PR (merged in #48). Full 6-year ES benchmark complete (run `d8b6e7eef6c2`); methodology delivered but **the outcome is a negative result on Sharpe** — bucketed beats HMC on vol-ratio (0.76 vs 0.65), tied within noise on seasonality. See `results_vs_paper.md` for the table and the three plausible explanations.
- 🟡 Gate L (DMM benchmark) is the planned next contribution (modern alternative on the state-space ladder, see scope justification §6).

Remaining gap-closing extras the doc still tracks, ranked by paper-fidelity vs cost:

| Extra | Closes which §2.5 / paper gap? | Implementation cost | Probable Sharpe impact |
|---|---|---|---|
| **Combined vol+seasonality IOHMM** | §4 combined predictor variant; scoped as Gate Q in `IMPLEMENTATION_PLAN.md` | ~1 week | Moderate-to-high lift expected — highest documented Sharpe-flip candidate |
| **CV-based K selection** | §4 model-selection trio (we have 1/3) | ~1 week | Likely picks K=2 even with longer h, would invalidate the BIC=4 overfit |
| **PLR seeding of HMM init** | §3.1 init scheme (we have PLR but don't wire it in) | ~2 days | Small, but full §3.1 fidelity |
| **Quantile-boundary rerun of the Gate K comparison** | Issue 42 *capability* shipped via PR #46 (`boundary_mode: "quantile"`); the Gate K run `d8b6e7eef6c2` still used `boundary_mode: grid`. A rerun with quantile mode would clean up the grid-vs-continuous attribution and isolate "is the bucketed advantage real, or grid-placement luck?" | ~half a day (rerun only) | Small on its own, but cleans up the negative-result story |
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

- **HMC on a continuous-parametric IOHMM transition.** ✅ **Shipped as
  Gate K** (`models/iohmm_continuous.py`, merged in #48). `A_ij(x_t) = softmax_j(W_i · x_t + b_i)`
  fit with NumPyro NUTS per walk-forward window, NUTS samples only the
  transition logits `(W, b)` while emissions stay at the Baum-Welch fit. This
  closes the §8 approximation gap and delivers the MCMC contribution in one
  PR. Full 6-year ES benchmark complete; negative-result outcome on Sharpe
  documented in `results_vs_paper.md`.
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

1. **HMC IOHMM** — ✅ shipped as Gate K (merged in #48); closes §8 and
   satisfies the MCMC ask in one PR. Full 6-year benchmark complete;
   negative-result outcome on Sharpe documented in `results_vs_paper.md`.
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
**Gate K (`models/iohmm_continuous.py`, merged in #48) implements the
paper-faithful continuous-parametric form**, fit with NumPyro NUTS per
walk-forward window. This closes the §8 approximation gap and absorbs
the Tema 2 MCMC requirement in a single contribution — but the full
6-year ES benchmark is a *negative result on Sharpe*: the bucketed
approximation still wins on vol-ratio, the two are tied within noise on
seasonality (see `results_vs_paper.md`). The bucketed variant is
retained as the engineering-approximation baseline so the
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

---

## In-depth defense reference: anticipated tutor questions

This section expands six likely questions that the higher-level pipeline
walkthrough above only summarizes. Use it as a defense crib sheet.

### 1. Why isn't PLR initialization used?

**What PLR is.** Piecewise Linear Regression — paper §3.1 segments the
return series into trend / no-trend pieces, then uses each segment's
statistics (mean, variance, length) as a prior on what the HMM states
should look like. Concretely: if PLR finds three trend segments, you seed
K=3 states with the segment means/variances and let Baum-Welch refine.

**What we do instead.** `hmmlearn`'s default k-means initialization —
clusters the *observations* into K groups and uses cluster centroids as
starting means. Standard practice in the HMM literature.

**Why this is fine.**
- For K∈{2,3,4} on minute returns, the likelihood surface is well-behaved.
  EM converges to essentially the same Θ across initialization schemes;
  variance across multiple restarts is numerical noise.
- PLR's value-add is when the likelihood has multiple modes that EM can
  get stuck in. That is not the regime here.
- The repo *does* implement PLR (`plr_baseline.py`) — used as an
  interpretable baseline trend summary, just not wired into the HMM init.
  Wiring it in is ~2 days of work and would close the §3.1 fidelity gap
  formally, but on this data it would not change the answer.

### 2. MCMC for parameter estimation — and where does HMC come in?

This is the question where the distinction matters most.

The HMM has parameters

```
Θ = {π, A, μ₁..μ_K, σ²₁..σ²_K}
```

where π is the initial-state distribution, A is the transition matrix,
and μ_k / σ²_k are the state-conditional Gaussian emission parameters.

**The paper proposes two estimation routes for Θ:**
- Baum-Welch / EM — iterative MLE.
- Metropolis-Hastings MCMC — random-walk sampling of the posterior over Θ.

**We use only EM. MCMC-on-Θ stays excluded** because:
- For a diagonal-Gaussian HMM with K=2-4 on minute returns the likelihood
  is sharply concentrated and well-behaved.
- EM and MH converge to the same Θ in this regime; the posterior is so
  tight that the point estimate and posterior mean differ only in
  numerical noise.
- A generic MH sampler is ~500–1000 LOC and days of compute per fit for
  an answer that matches `hmmlearn` to ~4 decimals.

**Where HMC enters (Gate K).** The paper's IOHMM transition function
`A(x_t)` is a different problem from Θ. Gate K models

```
A_ij(x_t) = softmax_j(W_i · x_t + b_i)
```

and uses NumPyro NUTS (a Hamiltonian Monte Carlo variant) to sample the
posterior over **(W, b)** — the transition-function parameters. The
emission parameters {μ_k, σ²_k} stay fixed at the Baum-Welch fit.

So the precise summary:

- MCMC on Θ → still excluded (BW dominates it on this data).
- MCMC on (W, b) → implemented as Gate K (the paper's continuous-parametric
  form, which EM cannot fit naturally without per-bar transition
  observations).

This satisfies the Tema 2 MCMC requirement and closes the §8 IOHMM
approximation gap in one contribution.

### 3. AIC/BIC — what they are, why only these

Paper §4 specifies three model-selection routes for choosing K:
cross-validation, AIC/BIC, and MCMC bridge sampling. The repo implements
AIC/BIC.

**Definitions.**

```
AIC = 2p − 2·log L        (Akaike Information Criterion)
BIC = p·log(n) − 2·log L  (Bayesian Information Criterion)
```

where `p` = number of fitted parameters, `L` = maximized likelihood,
`n` = number of observations. Lower is better — pick the K that minimizes
the criterion.

**Intuition.**
- Both balance fit (high `log L`) against complexity (high `p`).
- AIC penalizes each parameter by a constant `2`.
- BIC penalizes by `log(n)`, which grows with sample size. For `n > 7`
  we have `log(n) > 2`, so BIC penalizes complexity more heavily than AIC
  on any non-tiny dataset.
- AIC asymptotically minimizes prediction error (under iid).
- BIC asymptotically picks the "true" model (under iid + true model in
  candidate set).

**Parameter counts** for a K-state diagonal-Gaussian HMM are `K² + K − 1`:

| K | p |
|---|---|
| 2 | 5  |
| 3 | 11 |
| 4 | 19 |

**Why only AIC/BIC.**
- **Bridge sampling** is excluded by §2.5 for the same reason as MCMC-on-Θ:
  it is MCMC machinery to confirm what AIC/BIC already say. With Gate K
  in, the "we did MCMC" credential no longer depends on bridge sampling.
- **CV** is just a gap — the cheapest of the three to add, and is listed
  as the most defensible model-selection extra. **Flag this to the tutor
  as a known follow-up.**

**Honest empirical caveat.** In the long-window ablation (`h_days=60`)
BIC selects K=4 in every walk-forward window for every variant. Forecast
Sharpe drops. Reason: BIC's `p·log(n)` penalty grows slowly in `n`, so a
long training window over-justifies higher-state models. The extra states
fit training-window noise rather than predictive structure. CV would
likely overrule BIC here because CV scores on held-out prediction —
the metric we actually care about.

### 4. Seasonality and spline fitting in depth

#### Seasonality (§4.3)

Intraday returns have time-of-day structure — opening volatility burst,
midday lull, closing flurry. Encoding "time of day" as side information
lets the model condition transitions on which intraday regime is active.

**Implementation.**
1. Take the UTC timestamp of each bar.
2. Convert to **Chicago local time** (`America/Chicago`) — ES futures
   clear at CME in Chicago, so exchange-local time is what matters;
   UTC blurs daylight-savings transitions.
3. Bucket each minute into a time-of-day bin. Config: `bucket_minutes=1`
   means each minute of the day is its own bin.
4. Optionally normalize: `bucket_index / total_buckets ∈ [0, 1]` so the
   downstream spline receives a clean scalar input.

**Where this is an approximation.**
- Time-zone conversion: paper-faithful (Chicago time for ES).
- Bucket encoding: the paper isn't fully specific — one-hot, scalar, or
  multi-dim time-of-day bases are all defensible. The repo uses the
  scalar `bucket_index / total_buckets`.

#### Spline fitting (§4.1)

Goal: learn `f(x_t) ≈ E[r_{t+1} | x_t]` — a smooth, nonlinear function of
the side-information signal.

**Why splines.** Linear regression is too restrictive (the relationship is
plausibly nonlinear). Kernel methods are flexible but harder to make
deterministic. Splines hit the sweet spot: piecewise polynomial, smooth at
the joins ("knots"), fit by closed-form least squares.

**The repo's choices.**

```yaml
spline:
  degree: 3       # cubic polynomial pieces
  n_knots: 5      # 5 knot points → 4 piecewise segments
  demean: false   # do not subtract the mean over the support
  min_obs: 20     # need ≥20 points before fitting
```

- **Cubic (degree 3):** smooth first and second derivatives at knots;
  standard default.
- **5 knots at quantiles:** knot placement at the 0%, 25%, 50%, 75%,
  100% quantiles of training `x_t`. Quantile knots = balanced data per
  segment, which avoids fitting a segment to a few outliers.
- **Demean off** in the headline config: the paper specifies
  `∫ f(x) dx = 0` (zero-mean predictor) but the bucketed IOHMM downstream
  doesn't require it; demeaning is left as a config knob.
- **Deterministic least-squares:** B-spline basis matrix `B`,
  solve `min ‖y − Bβ‖²` for coefficients `β`. No regularization —
  overfitting is controlled by `n_knots`.

**Where this is an approximation.** The paper says "splines fit the
conditional mean" but does not specify number of knots, knot placement
(uniform vs quantile vs adaptive), basis (B-spline vs natural vs other),
or regularization. All defensible engineering choices.

**Where the spline plugs into the pipeline.**
- Standalone-predictor mode: `f(x_t)` is the prediction directly — used in
  `standalone_predictor` runs.
- IOHMM mode: the spline's output (or `x_t` itself) feeds the transition
  conditioning. The bucketed IOHMM buckets the spline's domain; Gate K
  HMC takes (preprocessed) `x_t` directly into the softmax.

### 5. IOHMM transition — bucketed vs continuous-parametric HMC

The heart of Gate K.

**Baseline HMM (no side info):** single fixed `K×K` matrix,
`A_{ij} = P(m_t = j | m_{t-1} = i)`, same for all `t`.

**IOHMM (paper §4):** the transition depends on side info `x_t`,

```
A_{ij}(x_t) = P(m_t = j | m_{t-1} = i, x_t)
```

The paper specifies this as a continuous parametric function of `x_t`.
They don't fully nail down the parametric form; one natural choice — and
what the repo implements in Gate K — is the row-wise multinomial-logit
(softmax):

```
A_{ij}(x_t) = exp(W_{ij} · x_t + b_{ij}) / Σ_k exp(W_{ik} · x_t + b_{ik})
```

Each row stays a valid probability distribution; the dependence on `x_t`
is smooth (small changes in `x_t` → small changes in `A(x_t)`).

#### Bucketed approximation (Gate H, `iohmm_approx.py`)

The engineering shortcut:

1. Discretize `x_t` into **B buckets** (B=3 in the headline config: low /
   mid / high).
2. Fit one transition matrix `A⁽ʳ⁾` per bucket `r` by counting transitions
   on the training data restricted to bars where `x_t ∈ bucket r`.
3. Smooth toward the pooled baseline to handle low-count buckets.
4. At prediction time: when `x_t` lands in bucket `r`, use `A⁽ʳ⁾` for the
   forward step.

Why this is an *approximation*:

- A real continuous function changes smoothly as `x_t` crosses any
  boundary. The bucketed version has **step discontinuities** at bucket
  edges — every `x_t` in the same bucket gets the *same* `A`, then it
  jumps when you cross the boundary. Unphysical.
- Within a bucket all variation in `x_t` is thrown away. A "barely-low-vol"
  and a "very-low-vol" bar get treated identically.
- The bucket boundaries are a hyperparameter — different choices give
  different answers. Quantile-derived boundaries are available
  (`boundary_mode: "quantile"`, shipped via PR #46 / Issue 42); the
  Gate K headline run used the default `boundary_mode: grid`, so a
  follow-up rerun with quantile boundaries would isolate how much of
  the bucketed advantage is real vs grid-placement luck.

#### Continuous-parametric HMC (Gate K, `iohmm_continuous.py`)

**Paper-faithful** because the repo implements the continuous functional
form the paper specifies, fit per walk-forward window with NumPyro NUTS.

**Why HMC instead of MLE for (W, b).**
- The logit IOHMM has no closed-form M-step. MLE-by-gradient-descent is
  one option, but HMC gives the full posterior over (W, b), not just a
  point estimate.
- HMC / NUTS exploits the gradient of the log-posterior efficiently — far
  better mixing than random-walk MH on parametric models with continuous
  parameters.
- The parameter space is small (≈ K²·dim(x) ≈ 10 params), well-suited to
  HMC.
- Bayesian framing = direct tie to the Tema 2 MCMC syllabus block.

**Side-by-side comparison.**

| Property | Bucketed (Gate H) | HMC continuous (Gate K) |
|---|---|---|
| Functional form | step (piecewise constant in `x_t`) | smooth softmax in `x_t` |
| Continuity in `x_t` | discontinuous at boundaries | continuous everywhere |
| Information use | discards within-bucket variation | uses full `x_t` precision |
| Bucket boundaries as a hyperparameter | yes — sensitive to placement | none |
| Matches paper's spec | engineering approximation | yes — continuous parametric |

The bucketed variant is retained as the engineering-approximation
**baseline** for the grid-vs-continuous ablation. Without the bucketed run
alongside Gate K you cannot isolate "how much of the Sharpe difference is
from going continuous?" from other shifts.

### 6. Trading signal, pre/post-cost Sharpe, and the cost model

#### The trading signal

The HMM / IOHMM produces

```
ŷ_t = E[r_{t+1} | r_{1:t}] = Σ_j ω_{t+1|t, j} · μ_j
```

This is a continuous-valued expected next return. To trade, convert it
into a position.

The paper writes `Signal_t = TF(ŷ_t)` but doesn't define `TF`. This is the
underspecification the repo works around with three policies:

**a) `sign` (paper-faithful default).** `s_t = sign(ŷ_t) ∈ {−1, +1}`.
Always fully invested in one direction. Long if expected return positive,
short if negative. Maximally aggressive (no flat state) and maximally
turnover-prone.

**b) `thresholded_hold`.**

```
if |ŷ_t| < θ:  s_t = s_{t-1}   # hold previous position
else:          s_t = sign(ŷ_t)
```

Skips trades when the predicted edge is small. Empirical result on ES
1-min: turnover cut by ~78%, but pre-cost Sharpe drops by half. Most of
the directional signal lives in *small-magnitude* predictions that fall
inside the dead zone.

**c) `conviction_weighted`.** `s_t = clip(ŷ_t / σ_train, −1, +1)` —
continuous position size scaling with prediction magnitude. Idea: trust
large predictions more. Empirical result: **negative on this data** —
reduces Sharpe across every variant. The HMM's prediction magnitude does
not correlate with directional accuracy on ES 1-min: the strategy wins
because many small bets compound, not because large-magnitude predictions
are more accurate.

#### Strategy return

```
R^strategy_{t+1} = s_t · r_{t+1}
```

The `t → t+1` lag is critical: signal formed at `t` can only trade the
return at `t+1`. The repo asserts this in `align_signal_with_future_return`
to prevent off-by-one look-ahead bugs.

#### Sharpe ratio

```
Daily Sharpe = sqrt(258) · mean(R_daily) / std(R_daily)
```

`258` ≈ trading days per year (paper convention). The repo aggregates
intraday returns to daily *first* (sum by UTC date), then computes Sharpe
on the daily series. **Doing it the other way — annualizing minute-level
Sharpe by `sqrt(252·390)` — would be wrong**, because minute returns are
not iid daily so the variance scaling breaks.

#### Pre-cost vs post-cost

Pre-cost: `R^pre_{t+1} = s_t · r_{t+1}`.

Post-cost: `R^post_{t+1} = s_t · r_{t+1} − cost(|s_t − s_{t-1}|)`.

Pre-cost is the **academic comparison target** — it answers "does the
model contain directional information?". Post-cost is a stress test that
adds "and can you trade it after a transparent cost assumption?".

#### The cost model

```
cost(|Δs|) = cost_bps_per_turnover × |Δs| / 10000
```

with `cost_bps_per_turnover = 1.0` as default.

**Turnover convention.** `|Δs|` is the absolute change in position size.
Going from `0 → +1` is `|Δs| = 1` → 1 bp cost; going from `+1 → −1` (full
flip) is `|Δs| = 2` → 2 bp cost. So an *entry* costs 1 bp, a *flip* costs
2 bp (round trip).

**Numerical impact on the headline run** (baseline HMM, sign policy):
- 82,044 position changes over the sample
- × 2 turnover units per flip × 1 bp = ~164,000 bp of cumulative cost
- Pre-cost cumulative return `0.94` → post-cost `−1.00` (full wipe-out)
- Pre-cost Sharpe `0.53` → post-cost `−8.9`

#### Why pre-cost is honest and post-cost is diagnostic

**Paper-comparison side (pre-cost).** The paper reports pre-cost Sharpe
as the academic comparison number. It does not specify its cost model in
any reproducible detail ("post-cost falls by ~15%" without bps, per-side
convention, slippage, fees, or fill model). The repo uses pre-cost when
comparing to the paper. Best pre-cost Sharpe in the headline run is
**0.7577** (vol-ratio, sign); paper reports `≈ 2.0`. Large absolute gap,
but the directional claim (side-info beats baseline) holds.

**Diagnostic side (post-cost at 1 bp).** The 1 bp/turnover convention is
**conservative for ES futures** — real ES round-trip cost for a retail
trader using market orders is around 0.5–1.0 bp, and lower for
institutional flow. At 1-minute cadence with the sign policy flipping
~25 bars on average, the cost drag dominates. The strategy is
unprofitable under this assumption.

**Why we don't tune the cost number to match the paper's "15% drop":**
that would be reverse-engineering an undocumented number, and every reader
would correctly suspect we chose the value that produced a 15% gap.
Better to publish both sides transparently and let the reader decide.

#### Likely tutor follow-ups

1. *Why softmax for the IOHMM transition and not something else?* — rows
   sum to 1 automatically; gradients are smooth; the multinomial-logit
   form is the natural Bayesian generalization of a transition matrix to
   continuous covariates.
2. *What's the prior on (W, b)?* — `Normal(0, 1)` per element, weakly
   informative. Row-centering applied post-hoc to handle the
   rank-deficiency of softmax (rows are identifiable only up to an
   additive constant).
3. *Why not use cross-validation now if BIC is overfitting?* — it's the
   scoped next extra; the negative-result narrative around BIC is
   pedagogically useful for the defense (concrete demonstration of when
   each criterion fails).
4. *Are there windows where HMC fails to converge?* — yes, occasional
   divergences are expected at the default `num_warmup=500`,
   `target_accept_prob=0.8`. The runner reports rhat and ess_bulk per
   window so non-mixing chains are flagged rather than absorbed silently.
   Mitigation options if a window diverges: bump `num_warmup` or
   `target_accept_prob`, or tighten the spline prior on that data slice.
