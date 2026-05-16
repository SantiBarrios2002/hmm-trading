# Supervisor Notes: HMM Intraday Momentum Project

> **Status note (2026-05-16):** this document is the original April-27 pitch
> to the supervisor. It is preserved here as a historical record of the
> proposed scope. Two material scope changes have happened since:
>
> 1. **MCMC is no longer excluded.** Gate K
>    ([`models/iohmm_continuous.py`](../src/hft_hmm/models/iohmm_continuous.py),
>    merged in #48) implements the paper-faithful continuous-parametric IOHMM
>    transition `A_ij(x_t) = softmax_j(W_i · x_t + b_i)` and fits `(W, b)`
>    with NumPyro NUTS per walk-forward window. MCMC on the HMM
>    emission parameters Θ remains excluded by `IMPLEMENTATION_PLAN.md §2.5`.
>    Full 6-year ES benchmark complete (run `d8b6e7eef6c2`). **Outcome is a
>    negative result on Sharpe**: the paper-faithful continuous form does
>    not beat the bucketed approximation on this data (vol-ratio 0.65 vs
>    0.76; seasonality tied within noise). The contribution is the
>    methodology and convergence diagnostics, not a Sharpe improvement.
>    See [`results_vs_paper.md`](results_vs_paper.md) for the table.
> 2. **A Deep Markov Model benchmark (Gate L) is planned as a second
>    extension**, climbing the state-space ladder beyond the HMM/IOHMM
>    pair. See [`dmm_mcmc_roadmap.md`](dmm_mcmc_roadmap.md).
>
> Current authoritative documents:
> - Pipeline status: [`paper_pipeline_walkthrough.md`](paper_pipeline_walkthrough.md)
> - Deviation table: [`paper_spec.md`](paper_spec.md)
> - Results: [`results_vs_paper.md`](results_vs_paper.md)
> - Extension plan: [`dmm_mcmc_roadmap.md`](dmm_mcmc_roadmap.md)

## Purpose

I would like to validate whether the paper *Hidden Markov Models Applied To Intraday Momentum Trading With Side Information* by Christensen, Turner, and Godsill is suitable for my ASPTA assignment project.

The ASPTA project asks for selecting a paper connected to the course, validating it with a teacher, and simulating some of its results for the final presentation. This paper is connected to several course topics: maximum likelihood estimation, the EM algorithm, Bayesian filtering/inference, hidden state models, and time-series signal processing.

## Paper Summary

The paper models intraday financial returns with a Hidden Markov Model. The hidden state represents the latent market momentum or trend regime, while the observed variable is the one-step log return. The central motivation is that classical momentum filters, such as moving-average based indicators, can lag when the market changes direction. The HMM instead estimates a latent state distribution and updates it recursively as new returns arrive.

The paper has two main parts:

1. A baseline HMM for intraday momentum:
   - observations are log returns
   - hidden states represent upward, downward, or neutral momentum regimes
   - emissions are univariate Gaussian distributions
   - parameters are estimated using piecewise linear regression, Baum-Welch/EM, and MCMC
   - model selection suggests two or three hidden states

2. A side-information extension:
   - two external predictors are considered: volatility ratio and intraday seasonality
   - splines are used to learn nonlinear relationships between these predictors and future returns
   - an IOHMM-style approximation conditions the transition matrix on the external predictor
   - the paper reports that side-information versions improve the baseline HMM in pre-cost Sharpe ratio

The original experiment uses one-minute e-mini S&P 500 futures data. The paper reports that the baseline and side-information models can produce positive pre-cost performance, with side-information variants improving over the plain Baum-Welch HMM.

## Main Formulas To Replicate

The formulas below are the ones I would use directly in the implementation. The notation is adapted slightly so it maps cleanly to code.

### Paper Reference Map

| Project formula | Paper location |
| --- | --- |
| Hidden-state index `m_t in {1, ..., K}` | Eq. (1), Section 2.2 |
| Noisy trend / return observation model | Eq. (2), Section 2.2 |
| Gaussian emission distribution `phi_k` | Eq. (3), Section 2.3.2 |
| EWMA volatility forecast | Eq. (4), Section 4.2 |
| Gaussian likelihood used in filtering | Eq. (5), Section 6 |
| HMM joint distribution | Section 2.2, immediately after Eq. (2), unnumbered |
| Baum-Welch forward/backward and parameter updates | Algorithm 1, Section 3.2 |
| Spline learning and evaluation | Algorithm 2, Section 4.1 |
| IOHMM bucketed transition learning | Algorithm 3, Section 5.2 |
| HMM/IOHMM prediction recursion | Algorithms 4 and 5, Section 6 |
| Sharpe ratio | Section 4.4, unnumbered |
| Transaction-cost-adjusted strategy return | Project evaluation formula, based on the paper's transaction-cost discussion |

### Return Observation (Paper Section 2.2, Unnumbered)

The observed signal is the log return:

```text
r_t = log(y_t / y_{t-1})
```

where `y_t` is the sampled price and `r_t` is the one-step return. The paper uses one-minute ES futures returns after synchronizing tick data onto a regular grid.

### Baseline HMM (Paper Eq. (1), Eq. (2), Eq. (3))

The hidden state is the latent momentum regime:

```text
m_t in {1, ..., K}
```

The Gaussian emission model is:

```text
r_t | m_t = k ~ Normal(mu_k, sigma_k^2)
```

This is the code-level version of the paper's noisy trend model in Eq. (2):

```text
Delta y_t = mu_{m_t} + epsilon_t
epsilon_t ~ Normal(0, sigma_{m_t}^2)
```

The transition probabilities are:

```text
A_{ij} = P(m_t = j | m_{t-1} = i)
sum_j A_{ij} = 1
```

The paper's emission matrix form in Eq. (3) is the discretized/normalized Gaussian likelihood for each state:

```text
phi_k(r_t)
  proportional to Normal(r_t; mu_k, sigma_k^2)
```

The joint HMM likelihood factorizes as:

```text
p(r_{1:T}, m_{1:T})
  = p(m_1)
    prod_{t=2:T} p(m_t | m_{t-1})
    prod_{t=1:T} p(r_t | m_t)
```

This is the core generative model I would reproduce first.

### Baum-Welch / EM Learning (Paper Algorithm 1)

The learning objective is maximum likelihood estimation of the HMM parameters:

```text
Theta_hat = argmax_Theta p(r_{1:T} | Theta)
```

where:

```text
Theta = {pi, A, mu_1:K, sigma_1:K^2}
```

The EM algorithm uses posterior state probabilities:

```text
gamma_t(k) = P(m_t = k | r_{1:T}, Theta)
xi_t(i,j) = P(m_t = i, m_{t+1} = j | r_{1:T}, Theta)
```

The paper gives the Baum-Welch forward/backward recursion and parameter estimation in Algorithm 1. In code, I would use the standard normalized M-step form:

```text
pi_k = gamma_1(k)

A_{ij}
  = sum_{t=1:T-1} xi_t(i,j)
    / sum_{t=1:T-1} gamma_t(i)

mu_k
  = sum_{t=1:T} gamma_t(k) r_t
    / sum_{t=1:T} gamma_t(k)

sigma_k^2
  = sum_{t=1:T} gamma_t(k) (r_t - mu_k)^2
    / sum_{t=1:T} gamma_t(k)
```

In the project implementation I would enforce a small variance floor, because the paper notes that Gaussian variance can collapse below a meaningful tick-grid scale.

### Model Selection (Paper Section 2.3.4 and Figure 2)

For candidate state counts `K`, I would compare fitted HMMs with AIC and BIC:

```text
AIC = 2p - 2 log L
BIC = p log(n) - 2 log L
```

where `p` is the number of fitted parameters, `n` is the number of observations, and `L` is the maximized likelihood. This replaces the paper's heavier MCMC bridge-sampling route while preserving a formal model-order selection step.

### Forward Filtering And Prediction (Paper Eq. (5), Algorithm 4)

The likelihood vector in Eq. (5) is:

```text
p(r_t | m_t = k, Theta)
  proportional to Normal(r_t; mu_k, sigma_k^2)
```

Let:

```text
omega_{t|t,k} = P(m_t = k | r_{1:t})
```

Prediction step:

```text
omega_{t+1|t,j}
  = sum_i omega_{t|t,i} A_{ij}
```

Update step after observing `r_{t+1}`:

```text
omega_{t+1|t+1,j}
  = omega_{t+1|t,j} Normal(r_{t+1}; mu_j, sigma_j^2)
    / sum_l omega_{t+1|t,l} Normal(r_{t+1}; mu_l, sigma_l^2)
```

The one-step expected return is:

```text
E[r_{t+1} | r_{1:t}]
  = sum_j omega_{t+1|t,j} mu_j
```

This is the inference formula used to turn the fitted HMM into a predictive signal.

### Trading Signal And Strategy Return (Paper Algorithm 4 and Section 7)

The basic trading rule is sign-based:

```text
s_t = sign(E[r_{t+1} | r_{1:t}])
```

The next-period strategy return is:

```text
R_{t+1}^{strategy} = s_t r_{t+1}
```

This one-period lag is important: the signal formed at time `t` is evaluated on the return at `t+1`, avoiding look-ahead bias.

With transaction costs:

```text
R_{t+1}^{post-cost}
  = R_{t+1}^{strategy}
    - (cost_bps / 10000) * |s_t - s_{t-1}|
```

The cost equation is a project-side implementation detail. The paper discusses transaction costs and slippage in the simulations, but does not give this exact compact formula.

### Volatility-Ratio Predictor (Paper Eq. (4), Section 4.2)

The paper's volatility forecast uses a truncated EWMA:

```text
sigma_{t+1|t}(psi)
  = sqrt((1 - lambda) sum_{tau=0:psi} lambda^tau r_{t-tau}^2)
```

The volatility-ratio side-information feature is:

```text
X_t^{vol}
  = sigma_{t+1|t}(psi_fast) / sigma_{t+1|t}(psi_slow)
```

The paper uses one-minute defaults close to:

```text
lambda = 0.79
psi_fast = 50
psi_slow = 100
```

### Intraday Seasonality Predictor (Paper Section 4.3, Unnumbered)

The seasonality predictor maps each timestamp to its location within the exchange-local trading day:

```text
X_t^{season} = bucket(local_time_t)
```

For implementation this means converting UTC timestamps to Chicago local time for ES futures, then mapping each minute to a fixed time-of-day index. A normalized version can be used for spline fitting:

```text
X_t^{season, normalized} = bucket_t / number_of_buckets
```

### Spline Predictor (Paper Algorithm 2, Section 4.1)

The side-information step fits a nonlinear function between a predictor and future returns:

```text
f(x_t) ~= E[r_{t+1} | x_t]
```

The paper uses splines and forces the fitted function to have zero mean over its support:

```text
integral f(x) dx = 0
```

In the project, this would be implemented as a deterministic spline fit from aligned pairs `(x_t, r_{t+1})`, with careful handling of missing values and no look-ahead leakage.

### IOHMM-Style Transition Conditioning (Paper Algorithm 3 and Algorithm 5)

The side-information HMM changes the transition model from:

```text
P(m_t | m_{t-1})
```

to:

```text
P(m_t | m_{t-1}, x_t)
```

The planned approximation is to discretize the side-information feature into buckets and estimate one transition matrix per bucket:

```text
A(x_t) = A_r if x_t is in bucket r
P(m_t = j | m_{t-1} = i, x_t) = A_r[i,j]
```

The IOHMM prediction step then becomes:

```text
omega_{t+1|t,j}
  = sum_i omega_{t|t,i} A(x_t)_{ij}
```

This preserves the key idea from the paper: external information changes the transition dynamics between hidden momentum states.

### Evaluation Metrics (Paper Section 4.4, Unnumbered)

The main comparison metric is the annualized Sharpe ratio:

```text
Sharpe = sqrt(N) * mean(R) / std(R)
```

For daily aggregated strategy returns, the paper uses approximately:

```text
N = 258 trading days
```

I would also report:

```text
cumulative_return = prod_t (1 + R_t) - 1
hit_rate = mean(1{R_t > 0})
max_drawdown = max peak-to-trough loss of cumulative returns
```

These metrics make the replication easier to assess than Sharpe alone.

## Why I Think It Fits ASPTA

The project is a good fit because it is not just a finance exercise; it is fundamentally a signal processing and statistical inference problem:

- hidden state estimation through an HMM
- parameter learning through EM/Baum-Welch
- model order selection for the number of hidden states
- recursive filtering for prediction
- side-information fusion through a transition-conditioned model
- careful simulation and evaluation of a signal-generation system

This maps naturally to the course content on estimation theory, EM, Bayesian inference/filtering, and time-series signal processing.

## Proposed Implementation Scope

I propose to implement a clear, reproducible subset rather than the full paper.

Core implementation:

- load and preprocess one-minute or sampled price data
- compute log returns
- implement a Gaussian HMM momentum model
- estimate parameters using Baum-Welch/EM
- compare candidate state counts using AIC/BIC
- run forward filtering to estimate the next-return expectation
- convert predicted returns into long/short trading signals
- evaluate cumulative return, Sharpe ratio, drawdown, hit rate, and transaction-cost sensitivity

Side-information extension, if time allows:

- implement volatility-ratio and intraday-seasonality predictors
- fit spline predictors for future returns
- test each predictor standalone
- implement an approximate IOHMM transition-conditioning method
- compare baseline HMM against volatility- and seasonality-enhanced variants

## Deliberate Simplifications

To keep the project feasible and reviewable, I would exclude or approximate some parts of the original paper:

- No full MCMC parameter estimation unless required; the main implementation would use Baum-Welch/EM.
- No MCMC bridge sampling for model selection; I would use AIC/BIC instead.
- No asynchronous tick-level model; I would use synchronous sampled returns.
- Single-security experiment only, not a multi-asset portfolio.
- The IOHMM would be implemented as an explicit approximation using bucketed transition matrices conditioned on side-information values.

These simplifications still preserve the main ASPTA-relevant ideas: EM learning, hidden-state inference, recursive filtering, and signal evaluation.

## Intended Deliverables

- A tested Python package implementing the main model components.
- Reproducible experiment configuration files.
- A small set of plots:
  - hidden-state timeline
  - model-selection curve
  - cumulative return comparison
  - side-information spline curves
- A final comparison table showing baseline HMM, side-information variants, and a long-only benchmark.
- A short explanation of which parts are paper-faithful and which are engineering approximations.

## Questions For Supervisor

1. Is this paper appropriate for the ASPTA project given its use of HMMs, EM, Bayesian filtering, and signal prediction?
2. The course slide mentions simulating results in Matlab/C. Would a Python implementation be acceptable if it is reproducible, tested, and clearly documented?
3. Is it acceptable to focus on Baum-Welch/EM and AIC/BIC model selection, while excluding full MCMC and bridge sampling?
4. Is an approximate IOHMM transition-conditioning implementation sufficient for the side-information part, or should the project stay with the baseline HMM only?
5. For the final presentation, should the emphasis be on the statistical signal-processing method or on the financial trading simulation results?

## Proposed Short Pitch

My proposed project is to reproduce the main signal-processing pipeline from Christensen, Turner, and Godsill's paper on HMMs for intraday momentum trading. I would model returns as observations generated by a latent momentum regime, learn the HMM parameters using Baum-Welch/EM, select the number of states with AIC/BIC, and use forward filtering to generate one-step-ahead trading signals. If time allows, I would also implement the paper's side-information idea using volatility ratio and intraday seasonality predictors, approximating the IOHMM with bucketed transition matrices. The goal is not to reproduce every number exactly, but to simulate the core result and evaluate whether the side-information HMM improves over the baseline.
