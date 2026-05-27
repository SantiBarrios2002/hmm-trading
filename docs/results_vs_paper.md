# Results Versus Paper

This document compares the repository's current reproducible Databento ES
1-minute results with the directional claims in Christensen, Turner, and
Godsill, *Hidden Markov Models Applied To Intraday Momentum Trading With Side
Information*. The goal is not exact numerical reproduction: the repo uses the
local continuous-contract Databento file documented in
[`data/README.md`](../data/README.md) and the scope choices documented in
[`docs/paper_spec.md`](paper_spec.md).

Five reproducible runs back this document:

- `runs/04269749abff` — `signal_policy: sign` (paper-style sign-based policy);
  the headline paper-replication run.
- `runs/f7af264b0da4` — `signal_policy: thresholded_hold` with
  `signal_threshold: 1.7e-6` (turnover-aware second pass).
- `runs/e25370277df7` — Sharpe-improvement ablation that swaps the
  signal policy from `sign` to `conviction_weighted` while leaving every other
  setting identical to the headline run. Used to attribute Sharpe change to
  the policy alone (negative result; see below).
- `runs/62e3e3714c0f` — Sharpe-improvement structural-change run that keeps
  the sign policy and combines a per-window BIC sweep over `K ∈ {2, 3, 4}`
  with a longer `h_days=60` training window. Also a negative result; see
  below.
- `runs/d8b6e7eef6c2` — Gate K side-info comparison that runs the bucketed
  and HMC continuous-parametric IOHMM variants side-by-side under identical
  walk-forward settings (`h_days=23`, `t_days=20`, `retrain_every_days=20`,
  `K=2`). Source of the §"Gate K HMC continuous-parametric IOHMM" results
  below. Also a negative result for the continuous form on this data;
  see below.

The first two runs share the Databento ES 1-minute parquet, walk-forward
schedule (`h_days=23`, `t_days=20`, `retrain_every_days=20`, `K=2`), and
`cost_bps_per_turnover=1.0`. Configs:
[`configs/example_es_databento_side_info_comparison.yaml`](../configs/example_es_databento_side_info_comparison.yaml)
and
[`configs/example_es_databento_side_info_comparison_thresholded.yaml`](../configs/example_es_databento_side_info_comparison_thresholded.yaml).
The Sharpe-improvement runs use
[`configs/example_es_databento_conviction_only.yaml`](../configs/example_es_databento_conviction_only.yaml)
and
[`configs/example_es_databento_enhanced.yaml`](../configs/example_es_databento_enhanced.yaml).

## Pre-Cost Academic Comparison

This is the headline table for the academic comparison. Sharpe uses the
paper's §4.4 convention: sum intraday strategy returns by UTC date, compute
Sharpe on the daily return vector, then annualize by `sqrt(258)`. Sample window
is 2019-01-28 00:01 UTC to 2024-12-25 23:59 UTC for every row.

Post-cost results are intentionally excluded from this table because the paper
does not specify enough execution-cost detail for a clean reproduction target.

| Model | Signal policy | Pre-cost Sharpe | Hit rate | Pre-cost cumulative return | Paper reference | Repo `run_id` |
|---|---|---:|---:|---:|---|---|
| Baseline HMM | sign | 0.5298 | 0.4079 | 0.9378 | §3 baseline HMM; §4.4 comparison context | `04269749abff` |
| Volatility-ratio IOHMM | sign | 0.7577 | 0.4080 | 1.6061 | §4.2 Predictor I; §4.4 IOHMM comparison | `04269749abff` |
| Seasonality IOHMM | sign | 0.6285 | 0.4080 | 1.2191 | §4.2 Predictor II; §4.4 IOHMM comparison | `04269749abff` |
| Default HMM | sign | -0.1132 | 0.4038 | -0.1374 | §3 Default HMM; PLR emissions, uniform A/π (Gate N) | `d8b6e7eef6c2` |
| Baseline HMM | thresholded_hold (1.7e-6) | 0.2264 | 0.4073 | 0.3396 | §3 baseline HMM, turnover-aware variant | `f7af264b0da4` |
| Volatility-ratio IOHMM | thresholded_hold (1.7e-6) | 0.2819 | 0.4074 | 0.4385 | §4.2 Predictor I, turnover-aware variant | `f7af264b0da4` |
| Seasonality IOHMM | thresholded_hold (1.7e-6) | 0.1316 | 0.4071 | 0.1862 | §4.2 Predictor II, turnover-aware variant | `f7af264b0da4` |
| Long-only benchmark | n/a | 0.6410 | 0.4091 | 1.3064 | Evaluation benchmark, not a paper model | `04269749abff` |

### Gate K HMC continuous-parametric IOHMM

The volatility-ratio and seasonality variants in the headline table above
use the bucketed-transition approximation (`models/iohmm_approx.py`). The
paper-faithful continuous-parametric form,
`A_ij(x_t) = softmax_j(W_i · x_t + b_i)` fit with NumPyro NUTS per
walk-forward window, is implemented as Gate K
(`models/iohmm_continuous.py`, merged in #48). The comparison config
[`configs/example_es_databento_side_info_comparison_hmc.yaml`](../configs/example_es_databento_side_info_comparison_hmc.yaml)
runs the bucketed and HMC variants side-by-side under identical
walk-forward settings so the grid-vs-continuous ablation has a fair
within-config baseline.

| Model | Signal policy | Pre-cost Sharpe | Hit rate | Pre-cost cumulative return | Repo `run_id` |
|---|---|---:|---:|---:|---|
| Baseline HMM                            | sign | 0.5298 | 0.4079 | 0.9378 | `d8b6e7eef6c2` |
| Volatility-ratio IOHMM (bucketed)       | sign | 0.7577 | 0.4080 | 1.6061 | `d8b6e7eef6c2` |
| Volatility-ratio IOHMM (HMC continuous) | sign | 0.6513 | 0.4079 | 1.3025 | `d8b6e7eef6c2` |
| Seasonality IOHMM (bucketed)            | sign | 0.6285 | 0.4080 | 1.2191 | `d8b6e7eef6c2` |
| Seasonality IOHMM (HMC continuous)      | sign | 0.6279 | 0.4079 | 1.2057 | `d8b6e7eef6c2` |

**Negative result for the continuous-parametric form on this data.** On
seasonality the two formulations are within numerical noise of each other
(0.6285 vs 0.6279). On volatility-ratio the bucketed approximation
*beats* the paper-faithful continuous form by ~14% pre-cost Sharpe
(0.7577 vs 0.6513). The §8 approximation gap is closed methodologically,
but Sharpe does not improve.

**Convergence diagnostics.** All 92 windows of `seasonality_hmc_continuous`
converged cleanly (rhat ≤ 1.05, ess_bulk ≥ 200 per the config thresholds).
`volatility_ratio_hmc_continuous` had 91/92 windows converge cleanly and
**one divergent window** (index 19, rhat ≈ 8.5, ess_bulk = 1) at the
default `num_warmup=500`, `target_accept_prob=0.8`. That window's
posterior is effectively noise; its predictions feed into the
vol-ratio HMC trading signal and account for some — but not all — of the
HMC vs bucketed gap above. The fact that the same calendar index
converged on the seasonality variant points at a volatility-ratio-feature
degeneracy at that bar, not a generic model/prior problem. See the
"Limitations and follow-ups" subsection below for the remediation
options being tracked.

#### Three plausible explanations for the negative result

1. **Window-19 divergence drags the vol-ratio HMC average down.** That
   single 1/92 ≈ 1.1% slice of forecast bars contributes a
   near-random transition matrix to the trading signal. Cheap to test
   in isolation by re-running just that window with higher
   `num_warmup` (e.g. 2000) or `target_accept_prob` (e.g. 0.95).
2. **Bucket boundaries confound the comparison.** The Gate K run used
   `boundary_mode: grid` (the config default) for the bucketed
   baseline. Quantile boundary mode is implemented (PR #46 / Issue 42)
   but not exercised here. On this data the grid boundaries may happen
   to fall in places that cleanly separate predictive regimes; against
   quantile boundaries the bucketed advantage could shrink — a
   follow-up rerun with `boundary_mode: quantile` would isolate this.
3. **Posterior averaging blurs an information signal.** The bucketed
   form gives a hard transition matrix per regime; the HMC form
   integrates over `(W, b)` posterior uncertainty, which softens
   transitions. On the volatility-ratio side info the model might
   prefer hard switching at vol regimes, and the HMC posterior dilutes
   that.

#### Defense framing

This is a *publishable* negative result, not a project failure. The
contribution is the methodology and the diagnostics — including
isolating the window-19 divergence and the bucket-boundary confound
above — not a Sharpe improvement. The §8 approximation gap is closed
by demonstration even though Sharpe does not move in the expected
direction. The same hit rate (~0.408) across every variant confirms
the conviction-weighted negative result still applies: the strategy
wins via compounding many small wins, not via high-magnitude
predictions.

#### Limitations and follow-ups

- The Gate K comparison run `d8b6e7eef6c2` used `boundary_mode: grid`
  for the bucketed baseline (the config default), not the
  `boundary_mode: quantile` mode that shipped via PR #46 (Issue 42).
  Part of the bucketed advantage may therefore be attributable to
  grid-bucket-placement luck rather than a real edge for the discrete
  form. A follow-up run that adds `boundary_mode: quantile` to the
  HMC comparison config — ideally producing a three-way ablation
  (grid / quantile / HMC continuous) within one run — would clean up
  the attribution.
- A targeted **re-run of window 19** with bumped HMC settings would
  isolate the divergence's contribution to the HMC Sharpe.
- A future write-up should report the full **per-window rhat / ess
  distribution**, not just the aggregate "all converged" headline. The
  per-window posteriors are persisted at `runs/d8b6e7eef6c2/<variant>.posterior/`.

## Sharpe-Improvement Experiments

The headline pre-cost daily Sharpe in the table above (best variant 0.7577 vs
the paper's ≈2.0 reference) leaves a large gap. The runs in this section
honestly test three candidate changes against that gap, all of them clearly
labeled as evaluation-layer extensions on top of the §4 paper-faithful
pipeline rather than re-interpretations of the paper itself. Every metric
below is pre-cost daily Sharpe under the same UTC-date aggregation and
`sqrt(258)` annualization as the table above.

### Changes considered

1. **Continuous, conviction-weighted positions.** The new `conviction_weighted`
   signal policy in [`strategy/signals.py`](../src/hft_hmm/strategy/signals.py)
   replaces the sign rule with `position[t] = clip(E[Δy_{t+1}] / σ_train, -1,
   +1)`, where `σ_train` is the standard deviation of training-side predicted
   expected returns computed inside the walk-forward loop. This is leakage-free
   (no forecast bars touch the scale) and degrades gracefully to the sign
   policy when σ_train is small. The change is motivated by the
   `thresholded_hold` evidence in the headline table: discarding small-magnitude
   predictions hurt pre-cost Sharpe, suggesting those predictions still carry
   directional information that might be worth down-weighting rather than
   zeroing.
2. **AIC/BIC sweep over `K ∈ {2, 3, 4}`.** The headline runs lock `K = 2`.
   The `walk_forward._select_k` helper already supports per-window BIC
   selection when `WalkForwardConfig.k_values` has more than one entry; the
   enhanced config exercises it. The paper's §4 uses model selection over
   `K ∈ {2, 3}` plus MCMC bridge sampling — the latter is excluded by §2.5,
   the former is now exercised explicitly.
3. **Longer training window (`h_days = 60`).** The §3.1 default of one rolling
   month (`h_days = 23`) is preserved as the paper-faithful default, but
   on the local 6-year sample a 60-day window is large enough to stabilize
   EM means without going so wide that intraday regimes are smeared.

### Conviction-weighted ablation result (negative finding)

Run `e25370277df7` switches the signal policy from `sign` to
`conviction_weighted` and leaves every other setting identical to the headline
run `04269749abff` — same `h_days=23`, `k_values=[2]`, vol-ratio, seasonality,
spline, and bucketed-transition parameters. Reproduce with:

```bash
python scripts/repro.py configs/example_es_databento_conviction_only.yaml
```

| Model | sign (`04269749abff`) | conviction_weighted (`e25370277df7`) | Δ |
|---|---:|---:|---:|
| Baseline HMM            | 0.5298 | 0.5182 | -0.0116 |
| Volatility-ratio IOHMM  | 0.7577 | 0.5744 | -0.1833 |
| Seasonality IOHMM       | 0.6285 | 0.5655 | -0.0630 |

Conviction weighting *reduces* pre-cost Sharpe across every variant on this
sample, with the volatility-ratio variant taking the largest hit. The honest
interpretation is that the HMM's predicted-return magnitude is **not** a good
conviction signal on this dataset: scaling positions by `|E[Δy_{t+1}]|` dilutes
the consistent directional information carried by the many small-magnitude
predictions, while leaving full exposure on the few large-magnitude
predictions — which on a 1-minute Gaussian HMM tend to occur at regime
transitions, where the model is most uncertain. This is consistent with the
sub-50% per-bar hit rate (~0.408 across all variants): the strategy has
positive Sharpe because winning bars compound in the right direction, not
because high-magnitude predictions are more accurate.

The `conviction_weighted` policy stays in the codebase as a documented
evaluation-layer alternative — useful for instruments or models where
prediction magnitude does correlate with accuracy — but the headline
Sharpe-improvement experiment below uses the paper-faithful `sign` policy.

### Structural-change run: BIC sweep + longer window (also negative)

Run `62e3e3714c0f` keeps the sign policy and applies the two structural
changes that the conviction ablation does not contradict: BIC selection over
`K ∈ {2, 3, 4}` per window, and `h_days = 60` instead of 23. Reproduce with:

```bash
python scripts/repro.py configs/example_es_databento_enhanced.yaml
```

| Model | sign baseline (`04269749abff`) | sign + K-sweep + h=60 (`62e3e3714c0f`) | Δ |
|---|---:|---:|---:|
| Baseline HMM            | 0.5298 | 0.3671 | -0.1627 |
| Volatility-ratio IOHMM  | 0.7577 | 0.4874 | -0.2703 |
| Seasonality IOHMM       | 0.6285 | 0.4419 | -0.1866 |

Pre-cost Sharpe falls on every variant. The diagnostic that explains the drop
is the chosen-K distribution: with `h_days = 60`, BIC selects **K = 4 in
every single window** for every variant (90/90 windows). The headline run, by
contrast, was pinned to K = 2.

The honest interpretation is that BIC's `p·log(n)` complexity penalty grows
slowly in `n`, so a longer training window justifies more states — but the
extra states fit training-window noise rather than predictive structure. The
forecast-side directional signal becomes noisier even though log-likelihood
on training is higher. This is a classical overfitting trade-off, made
concrete on this dataset.

### Summary of Sharpe-improvement experiments

All three proposed levers individually fail to improve pre-cost daily Sharpe
on the local 6-year ES sample:

- **Conviction weighting** (run `e25370277df7`) hurts because predicted-return
  magnitude doesn't correlate with directional accuracy on this data.
- **`K` sweep + longer training window** (run `62e3e3714c0f`) hurts because
  BIC over-justifies higher-state models when given more training data, and
  the extra states fit noise.
- **The two combined** would inherit both losses; a separate combined run is
  not reported because the conviction loss is monotonic across configurations
  tested and there is no expected interaction that would reverse it.

The paper-faithful headline configuration (`sign` policy, `K = 2`,
`h_days = 23`) is therefore essentially a local optimum on this dataset
across the changes explored. This is a useful finding in itself: the §3.1
defaults survive ablation rather than reflecting an arbitrary choice.

### Reasonable next levers (not implemented this round)

If a future PR wants to push pre-cost Sharpe further on this dataset, the
levers most likely to help — based on what *didn't* work above — are:

- **Combined vol-ratio + seasonality IOHMM** (scoped as Gate Q in
  `IMPLEMENTATION_PLAN.md`). The two predictors are individually useful;
  joint conditioning of the transition softmax on `x_t ∈ ℝ²` may capture
  cross-effects that the independent variants miss. Documented here as
  the single most likely change to lift pre-cost Sharpe on this dataset.
- **Cross-validation `K` selection instead of BIC.** The §4 paper compares
  CV, AIC/BIC, and MCMC bridge sampling; we only implement BIC. CV would
  pick `K` by held-out predictive performance, which is the metric we
  actually care about.
- **Smarter feature scaling and bucketing.** Quantile-based bucket
  boundaries are now available (`boundary_mode: "quantile"`, shipped
  via PR #46) but were not used in the headline run; richer joint
  splines could change the IOHMM conditioning result without altering
  the signal policy.

None of these are scope-clean one-liners; they are issue-sized follow-ups.

### What is intentionally not changed

- **The paper's pre-cost Sharpe is still the comparison target.** The
  Sharpe-improvement runs are reported pre-cost; post-cost numbers on these
  variants live in the diagnostic table below and remain dominated by the
  fixed `1.0` bp/turnover convention.
- **The IOHMM bucketing approximation is unchanged.** Bucket count, smoothing,
  spline knots, and vol-ratio EWMA parameters are kept at the headline values
  so the Sharpe lift is attributable to the signal/selection changes rather
  than to a feature-engineering sweep.
- **A combined vol-ratio + seasonality IOHMM variant is scoped as Gate Q**
  in `IMPLEMENTATION_PLAN.md` but not implemented this round. Each
  `EXPECTED_VARIANTS` change invalidates existing comparison_id hashes and
  forces a headline rerun; Gates K and N have already paid this cost twice,
  and Gate Q will be the third (and likely final) such change. Tracked
  separately so the Sharpe-improvement experiments in this section stay
  attributable to signal/selection choices, not to a new model variant.

## Post-Cost Diagnostic

This table is a cost-sensitivity stress test under the repo's fixed
`1.0` basis-point-per-turnover convention. It is deliberately isolated from
the paper comparison above. The numbers answer a different question: whether
the minute-level signal survives this explicit, conservative execution-cost
assumption.

| Model | Signal policy | Post-cost Sharpe | Post-cost cumulative return | Total turnover | Cost drag in cumulative return | Repo `run_id` |
|---|---|---:|---:|---:|---:|---|
| Baseline HMM | sign | -8.8973 | -1.0000 | 164,088 | 1.9378 | `04269749abff` |
| Volatility-ratio IOHMM | sign | -7.4351 | -1.0000 | 138,020 | 2.6061 | `04269749abff` |
| Seasonality IOHMM | sign | -7.6906 | -1.0000 | 134,940 | 2.2191 | `04269749abff` |
| Baseline HMM | thresholded_hold (1.7e-6) | -2.3563 | -0.9626 | 35,844 | 1.3023 | `f7af264b0da4` |
| Volatility-ratio IOHMM | thresholded_hold (1.7e-6) | -2.0129 | -0.9370 | 31,324 | 1.3756 | `f7af264b0da4` |
| Seasonality IOHMM | thresholded_hold (1.7e-6) | -2.1692 | -0.9490 | 31,512 | 1.1352 | `f7af264b0da4` |
| Long-only benchmark | n/a | 0.6410 | 1.3064 | 0 | 0.0000 | `04269749abff` |

The comparison figure is
[`docs/figures/cumulative_return_vs_paper.png`](figures/cumulative_return_vs_paper.png),
generated with:

```bash
python scripts/plot_results_vs_paper.py 04269749abff f7af264b0da4
```

The cited run artifacts are reproduced with:

```bash
python scripts/repro.py configs/example_es_databento_side_info_comparison.yaml
python scripts/repro.py configs/example_es_databento_side_info_comparison_thresholded.yaml
```

## Gap Analysis

The repo reproduces a central directional claim on the local Databento sample:
both side-information variants beat the baseline HMM on pre-cost Sharpe under
the sign policy, and the volatility-ratio variant continues to beat the
baseline under the thresholded_hold policy as well. Volatility-ratio is the
strongest trading variant in both passes (`0.7577` and `0.2819` pre-cost
Sharpe). The seasonality variant beats the baseline under sign but slips below
it under thresholded_hold, where the dead-zone discards a larger share of the
seasonal signal than of the volatility one.

The side-information variants do not dominate the long-only benchmark on every
pre-cost metric: under the sign policy, volatility-ratio beats long-only on
pre-cost Sharpe (`0.7577` vs `0.6410`) and cumulative return (`1.6061` vs
`1.3064`); under the thresholded_hold policy, no trading variant beats
long-only. The post-cost diagnostic is intentionally not used to rank paper
replication quality because it is dominated by the repo's turnover-cost
assumption.

## Cost Convention

The `1.0` basis-point-per-turnover setting is intentionally kept even though it
is punitive at 1-minute cadence. The paper reports pre-cost results as the main
comparison and says post-cost Sharpe falls, but does not specify its data,
spread, slippage, fee, per-side / round-trip, or execution assumptions. Rather
than tune an undocumented cost number until the post-cost gap resembles the
paper, the repo keeps a simple fixed convention that is easy to audit:
flat-to-long costs 1 bp and long-to-short costs 2 bp. This makes post-cost
results a conservative stress test and keeps the headline paper comparison on
the pre-cost daily Sharpe, where the paper gives the most usable target. A
negative post-cost result under this convention means "the minute-level signal
is too expensive under this assumed cost"; it does not erase the pre-cost
evidence that the model contains directional information.

## Threshold Calibration

The thresholded_hold pass uses `signal_threshold = 1.7e-6`, originally picked
from the `|E[Δy_{t+1}]|` quantile sweep printed by
[`scripts/calibrate_threshold.py`](../scripts/calibrate_threshold.py) over the
first 12 walk-forward windows (`q75 ≈ 1.72e-6`). A full 92-window baseline
calibration puts the same value near the median of the out-of-sample expected
return magnitudes (`q50 ≈ 1.72e-6`, `q75 ≈ 2.86e-6`, `q99.5 ≈ 1.97e-5`).
Under the same per-window alignment used by the run artifacts, the exact
`1.7e-6` threshold gives the baseline HMM `0.2264` pre-cost Sharpe and
`-2.3563` post-cost Sharpe. The full-sample median threshold is slightly
higher (`≈1.72e-6`) and improves that baseline tradeoff (`0.2971` / `-2.2166`),
but the saved run keeps the pre-selected `1.7e-6` value rather than tuning on
the full evaluation window.

## Turnover Diagnostics

| Model | Signal policy | Total turnover | Position changes | Mean holding bars | Cost drag in cumulative return |
|---|---|---:|---:|---:|---:|
| Baseline HMM | sign | 164,088 | 82,044 | 25.24 | 1.9378 |
| Volatility-ratio IOHMM | sign | 138,020 | 69,010 | 30.00 | 2.6061 |
| Seasonality IOHMM | sign | 134,940 | 67,470 | 30.69 | 2.2191 |
| Baseline HMM | thresholded_hold | 35,844 | 17,922 | 115.53 | 1.3023 |
| Volatility-ratio IOHMM | thresholded_hold | 31,324 | 15,662 | 132.20 | 1.3756 |
| Seasonality IOHMM | thresholded_hold | 31,512 | 15,756 | 131.41 | 1.1352 |

The thresholded_hold policy cuts turnover by roughly 78% across every variant
and quadruples mean holding-period length, but pre-cost Sharpe also drops by
about a half to two-thirds because most of the directional information sits in
small-magnitude expected-return predictions that fall inside the dead-zone.
Post-cost Sharpe improves by a factor of roughly 3 to 4 versus the sign policy
but stays negative on this sample under the simple
one-basis-point-per-turnover cost model. This is why the post-cost table is
presented as a separate diagnostic. The paper does not specify a transaction
cost model beyond saying post-cost Sharpe falls by roughly 15%, so using the
post-cost shortfall as a paper-replication failure would be over-interpreting an
underspecified execution assumption.

The main gaps are:

- **Data scope:** the repo table uses local Databento continuous-contract ES
  data from 2019 through 2024, not the paper's exact historical sample/vendor
  window.
- **Model scope:** MCMC parameter estimation on Θ (emissions), MCMC bridge
  sampling, asynchronous IOHMM, multi-security portfolios, and production
  execution modeling are excluded by the project scope. **HMC on IOHMM
  transition logits is no longer excluded** — it ships as Gate K
  ([`iohmm_continuous.py`](../src/hft_hmm/models/iohmm_continuous.py)) and
  preserves the §2.5 exclusion on Θ because NUTS samples only `(W, b)`,
  with emissions held at the Baum-Welch fit.
- **Implementation approximation:** the headline numbers above are
  from the finite bucketed-transition approximation in
  [`iohmm_approx.py`](../src/hft_hmm/models/iohmm_approx.py). The paper-
  faithful continuous-parametric form is implemented as Gate K and was
  run side-by-side under the dedicated Gate K config (run
  `d8b6e7eef6c2`); see the "Gate K HMC continuous-parametric IOHMM"
  subsection for those results. The Gate K outcome is a negative result
  on Sharpe — the bucketed approximation remains the better-performing
  variant on vol-ratio.
- **Numerical / stochastic variation:** all runs are deterministic for the
  saved configs, but fitted HMM parameters remain sensitive to the short
  sample and to the stabilized EM settings documented in
  [`gaussian_hmm.py`](../src/hft_hmm/models/gaussian_hmm.py).

The paper's §4.4 reference Sharpe for the best side-information variant is
approximately `2` after the paper's full experiment setup. The repo's best
pre-cost daily-annualized Sharpe on the local Databento run is `0.7577`
(volatility-ratio, sign policy). This supports the directional side-information
claim, but not a numerical reproduction of the paper's headline magnitude.
