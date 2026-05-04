# Results Versus Paper

This document compares the repository's current reproducible Databento ES
1-minute results with the directional claims in Christensen, Turner, and
Godsill, *Hidden Markov Models Applied To Intraday Momentum Trading With Side
Information*. The goal is not exact numerical reproduction: the repo uses the
local continuous-contract Databento file documented in
[`data/README.md`](../data/README.md) and the scope choices documented in
[`docs/paper_spec.md`](paper_spec.md).

Two reproducible runs back this document:

- `runs/04269749abff` — `signal_policy: sign` (paper-style sign-based policy).
- `runs/f7af264b0da4` — `signal_policy: thresholded_hold` with
  `signal_threshold: 1.7e-6` (turnover-aware second pass).

Both share the Databento ES 1-minute parquet, walk-forward schedule
(`h_days=23`, `t_days=20`, `retrain_every_days=20`, `K=2`), and
`cost_bps_per_turnover=1.0`. Configs:
[`configs/example_es_databento_side_info_comparison.yaml`](../configs/example_es_databento_side_info_comparison.yaml)
and
[`configs/example_es_databento_side_info_comparison_thresholded.yaml`](../configs/example_es_databento_side_info_comparison_thresholded.yaml).

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
| Baseline HMM | thresholded_hold (1.7e-6) | 0.2264 | 0.4073 | 0.3396 | §3 baseline HMM, turnover-aware variant | `f7af264b0da4` |
| Volatility-ratio IOHMM | thresholded_hold (1.7e-6) | 0.2819 | 0.4074 | 0.4385 | §4.2 Predictor I, turnover-aware variant | `f7af264b0da4` |
| Seasonality IOHMM | thresholded_hold (1.7e-6) | 0.1316 | 0.4071 | 0.1862 | §4.2 Predictor II, turnover-aware variant | `f7af264b0da4` |
| Long-only benchmark | n/a | 0.6410 | 0.4091 | 1.3064 | Evaluation benchmark, not a paper model | `04269749abff` |

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
- **Model scope:** MCMC parameter estimation, MCMC bridge sampling,
  asynchronous IOHMM, multi-security portfolios, and production execution
  modeling are excluded by the project scope.
- **Implementation approximation:** side-information conditioning uses the
  finite bucketed-transition approximation in
  [`iohmm_approx.py`](../src/hft_hmm/models/iohmm_approx.py), not the exact
  continuous IOHMM formulation.
- **Numerical / stochastic variation:** all runs are deterministic for the
  saved configs, but fitted HMM parameters remain sensitive to the short
  sample and to the stabilized EM settings documented in
  [`gaussian_hmm.py`](../src/hft_hmm/models/gaussian_hmm.py).

The paper's §4.4 reference Sharpe for the best side-information variant is
approximately `2` after the paper's full experiment setup. The repo's best
pre-cost daily-annualized Sharpe on the local Databento run is `0.7577`
(volatility-ratio, sign policy). This supports the directional side-information
claim, but not a numerical reproduction of the paper's headline magnitude.
