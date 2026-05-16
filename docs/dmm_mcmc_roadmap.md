# DMM + MCMC Extension Roadmap

This roadmap plans two original contributions on top of the HMM replication:

- ✅ **MCMC extension (Gate K) — shipped.** Closes the §8 IOHMM gap (continuous
  parametric `A(x_t)` replaces the bucketed approximation) and satisfies
  Tema 2 of the syllabus. Implemented in
  [`models/iohmm_continuous.py`](../src/hft_hmm/models/iohmm_continuous.py),
  merged via #48. Full 6-year ES walk-forward benchmark complete
  (run `d8b6e7eef6c2`). **Outcome: negative result on Sharpe** — the
  paper-faithful continuous form does not beat the bucketed approximation
  on this data (vol-ratio 0.65 vs 0.76; seasonality tied within noise).
  See [`results_vs_paper.md`](results_vs_paper.md) for the full table and
  the three plausible explanations.
- 🟡 **Deep Markov Model extension (Gate L) — planned next.** Closes the
  "modern alternatives" gap, generalizes the HMM along the state-space
  ladder.

It is the planning companion to the brainstorm in
[`paper_pipeline_walkthrough.md`](paper_pipeline_walkthrough.md).
Reference paper for the DMM piece: Krishnan, Shalit, Sontag (2017),
*Structured Inference Networks for Nonlinear State Space Models* (AAAI),
PDF in `docs/`.

---

## 1. Framing: state-space ladder

The professor's hint — *"HMM (que es el modelo considerado en Kalman y PF)"* —
points at the same family the syllabus already covers. HMM, Kalman, particle
filter, and DMM differ only in the latent type and the (non)linearity of the
dynamics and emissions:

| Model | Latent | Dynamics | Emission | Inference |
|---|---|---|---|---|
| **HMM** (paper baseline, done) | discrete | linear (transition matrix) | Gaussian | Baum-Welch (EM, Tema 1) |
| **IOHMM** (paper §4, partial in repo) | discrete | conditional on `x_t` | Gaussian | EM + bucketing (engineering approx.) |
| **Kalman filter** | continuous | linear-Gaussian | Gaussian | closed form |
| **Particle filter** | continuous | nonlinear | any | SMC (Tema 2) |
| **DMM** (Krishnan 2017) | continuous | **neural nonlinear** | **neural** | variational (default), optional HMC |

The thesis arc becomes "replicate the paper's HMM, then climb the ladder by
relaxing linearity and discreteness." Every rung is in the syllabus.

---

## 2. How DMM fits the existing repo

- **Reuses:** `compute_log_returns`, walk-forward windowing,
  signal/evaluation layer, side-info featurizer (vol ratio + seasonality
  become exogenous inputs to the DMM transition net).
- **New code:** `src/hft_hmm/models/dmm.py` (PyTorch / Pyro) — emission MLP,
  transition MLP gated on `x_t`, structured inference network (combiner RNN).
  Pyro's `dmm` example is ~400 LOC and adapts cleanly.
- **Plugs into walk-forward** as another `Variant` alongside `baseline_hmm`
  and `iohmm_*` — no infrastructure change beyond extending
  `EXPECTED_VARIANTS`.
- **Closes which gap?** §8 of `paper_pipeline_walkthrough.md` (continuous-
  parametric transition conditioning). DMM's transition net is the deep
  version of what the paper specifies and what we currently bucket-approximate.

---

## 3. Where MCMC enters — Option A picked and shipped

### Option A — HMC on IOHMM transitions only ✅ DONE

Implemented and merged as Gate K (`models/iohmm_continuous.py`, merged in #48).
NumPyro NUTS samples `(W, b)` of `A_ij(x_t) = softmax_j(W_i · x_t + b_i)` per
walk-forward window. Full 6-year benchmark complete; converged cleanly on
183/184 fits. DMM (Gate L) will be trained with variational inference, as
originally planned — MCMC stays on the small, well-identified transition
problem where it converges cleanly.

The bucketed IOHMM remains in the comparison as the engineering-
approximation baseline, so the comparison is grid (Gate H) vs continuous
(Gate K) within a single config.

### Option B — HMC/NUTS on DMM latents or parameters — not chosen

Train DMM with VI as a baseline, then run HMC over a *subset* of parameters
(e.g., emission MLP last layer) or over latent trajectories on a held-out
window, and compare to the variational posterior. Methodologically novel but
substantially riskier — see §6 for the four reinforcing reasons HMC on a
neural state-space model rarely produces clean posteriors in this kind of
timeline. Option B remains a documented future direction if the DMM
benchmark (Gate L) produces a clean enough VI posterior to make the
HMC-vs-VI comparison interesting.

---

## 4. Build phases (Option A path)

| Phase | Work | Status |
|---|---|---|
| 0 — paper read + toolchain check | Read DMM §3–4 carefully; reproduce the polyphonic-music likelihood numbers in Pyro's `dmm` example to confirm the toolchain works on your hardware. | pending |
| 1a — Quantile bucket boundaries (Issue 42) | Add `quantile` boundary mode to `BucketedTransitionConfig` so the bucketed IOHMM has balanced bucket counts. Tracked at GitHub Issue 42. | ✅ shipped via PR #46 (`bucket_boundaries_from_quantiles`, `boundary_mode: "quantile"`). Note: the headline Gate K run `d8b6e7eef6c2` used `boundary_mode: grid`, so a follow-up rerun with `boundary_mode: quantile` is still needed for a clean grid-vs-continuous attribution. |
| 1b — HMC IOHMM (Gate K) | NumPyro NUTS samples `(W, b)` of `A_ij(x_t) = softmax_j(W_i · x_t + b_i)` per walk-forward window. Reuses existing IOHMM data flow. Compared against the grid-bucketed baseline within `configs/example_es_databento_side_info_comparison_hmc.yaml`. | ✅ shipped; full 6-year benchmark complete (run `d8b6e7eef6c2`); negative-result outcome (see `results_vs_paper.md`) |
| 2 — DMM walk-forward integration (Gate L) | Wrap Pyro DMM, expose `predict_next_return`, register as a variant in `EXPECTED_VARIANTS`, run side-by-side with HMM / IOHMM. | planned |
| 3 — write-up | State-space ladder figure, IOHMM ablation table (grid / quantile / HMC), HMC-posterior-vs-BW-point comparison plot, Sharpe table across all variants, negative-result discussion if DMM doesn't lift Sharpe. | pending Gate K results + Gate L |

---

## 5. Tutor mapping

- **Phase 1 (HMC IOHMM, Gate K)** → Ramon Morros (Tema 2 / MCMC). ✅ implemented.
- **Phase 2 (DMM, Gate L)** → either tutor. Closer to Antonio Pascual's territory if
  framed as "EM / variational generalization of the HMM"; closer to Ramon's
  if framed under Option B.
- **Recommendation:** lead with Ramon as primary tutor — the MCMC piece is
  the more syllabus-load-bearing contribution. Antonio secondary for the
  DMM / EM-lineage piece.

---

## 6. Risks

### Compute

DMM on six years of 1-min ES (~2M observations) is non-trivial. Plan:
subsample to 5-min for DMM training, evaluate on 1-min like the paper.

### Negative-result possibility

DMM may not beat the HMM on Sharpe. That is still a thesis-worthy finding
given the conviction-weighted negative result already in
`project_conviction_negative_result.md` — the project already has a track
record of taking negative results seriously.

### MCMC scaling — why HMC on DMM parameters is hard

This is the reason Option A keeps MCMC on the (small) IOHMM transition
parameters rather than on the DMM. Four reinforcing problems:

1. **Per-step cost scales with parameter count.** A small DMM has on the
   order of 10⁴–10⁵ parameters (two MLPs plus a combiner RNN). HMC's
   leapfrog integrator computes a gradient of the log-posterior w.r.t.
   every parameter at every step. The IOHMM transition logits, by contrast,
   are roughly `K² × n_buckets ≈ 20–50` parameters — three to four orders
   of magnitude cheaper per leapfrog step.

2. **Per-step cost also scales with sequence length.** Vanilla HMC is
   full-batch — every leapfrog step unrolls the inference network and the
   generative model through the entire training sequence. For T ≈ 2M
   minute bars, one gradient is already expensive; HMC needs hundreds of
   them per posterior sample. Stochastic-gradient HMC variants exist but
   bias the posterior and lose HMC's main selling point. The IOHMM forward
   filter is also O(T·K²), but K is tiny and there's no autograd through
   an RNN.

3. **Pathological posterior geometry.** Neural-network posteriors are
   multimodal with built-in symmetries (neuron permutations, scale-
   invariance through ReLU). The leapfrog integrator needs very small
   step sizes to stay stable, and chains mix slowly because they get
   stuck wandering equivalent reparameterizations. Effective sample size
   per second is often 100× worse than for a well-conditioned model. The
   IOHMM transition softmax is well-identified by comparison.

4. **Memory.** HMC stores gradients for the full model at every leapfrog
   step. With an RNN-based inference network unrolled through T steps,
   peak memory is O(T · hidden_dim) — same as training the DMM, but
   sustained across hundreds of leapfrog steps per sample. The IOHMM
   has no such hidden-state memory cost.

Combined, these mean HMC on DMM parameters typically needs days of
compute for a few hundred effective samples, with no guarantee the chains
are exploring the true posterior rather than a single mode. HMC on the
IOHMM transition logits, by contrast, converges in minutes. Option A
isolates MCMC where it actually works.

---

## 7. Scope and methodology justification

This section consolidates why the roadmap is structured as it is —
intended as a single defensible narrative for thesis review with both
professors. It complements the methodology justification at the end of
[`paper_pipeline_walkthrough.md`](paper_pipeline_walkthrough.md), which
covers the *replication* decisions; this section covers the *extension*
decisions.

### 7.1 Why we do not replicate the paper's full MCMC pipeline

The paper presents Baum-Welch and Metropolis-Hastings MCMC on Θ as
parallel routes to the HMM parameters. For a diagonal-Gaussian HMM with
K ∈ {2, 3, 4} on minute returns the likelihood is well-behaved and the
implied posterior is sharply concentrated — both routes converge to
essentially identical Θ. Implementing a generic MH sampler over Θ would
cost ~500–1000 LOC and days of compute per fit for an answer within
numerical noise of `hmmlearn`'s EM fit.

The MCMC contribution is reallocated to where the paper's methodology
is *not* already covered by EM: the IOHMM transition function, which
the repo currently bucket-approximates (§8 of the walkthrough) and
which the paper itself only sketches in continuous-parametric form
without committing to a specific estimator. This puts MCMC on a problem
where it changes the answer rather than restating it.

### 7.2 Why Option A (HMC on IOHMM) over Option B (HMC on DMM)

Option A places MCMC on a 20–50-parameter, well-identified,
well-conditioned model that converges in minutes. It is also the
targeted fix to the labeled §8 engineering approximation, so it closes
a paper-fidelity gap *and* delivers the Tema 2 contribution in a single
contribution.

Option B places MCMC on a 10⁴–10⁵-parameter, multi-modal,
RNN-unrolled posterior whose mixing is the open research problem
described in §6 of this roadmap — days of compute per posterior, no
guarantee of multi-modal coverage. Methodologically novel, but
Option A satisfies the syllabus requirement with substantially lower
risk and clearer defensibility.

We pick Option A also because it produces two cleanly-separated
contributions (HMC IOHMM, DMM benchmark), each mappable to one
professor's expertise. Option B is one combined contribution that can
only be descoped by abandoning it entirely.

### 7.3 Why DMM is the chosen DL benchmark

DMM sits one rung above HMM on the state-space ladder (continuous
latent, neural dynamics) while sharing the forward-filtering /
latent-state-inference structure. The comparison is interpretable —
both models forecast `E[Δy_{t+1}]` from a latent state estimate.

An LSTM or Transformer benchmark would compare a latent-variable model
against a non-latent sequence model, which conflates two separate
hypotheses (does *latency* help? does *non-linearity* help?). DMM
isolates the non-linearity question while keeping the latent-variable
structure constant. It also matches the syllabus's "tour of state-space
models" framing (HMM → Kalman → PF → neural SSM) more naturally than a
generic sequence model would.

### 7.4 How this maps to the course and to the two professors

- **Tema 1 (EM / Baum-Welch)** is covered by the existing HMM
  replication, already coordinated with Antonio Pascual.
- **Tema 2 (MCMC / Monte Carlo sampling)** is covered by the HMC IOHMM
  contribution, suitable for Ramon Morros.
- **DMM** straddles the two: variational-EM by default (Tema 1 lineage)
  but introducing continuous-latent state-space inference (Tema 2
  lineage via the Kalman / particle-filter tradition the syllabus
  already covers).

This is why the recommendation is Ramon as primary tutor, Antonio as
secondary — the MCMC piece is the more syllabus-load-bearing
contribution and Ramon's tutoring covers it most directly. The DMM
piece is co-tutored.

### 7.5 Tradeoffs we are explicitly accepting

- **DMM may not beat the HMM on Sharpe.** The project already has a
  documented negative-result discipline (the conviction-weighted policy
  ablation), so a negative DMM result is publishable as a finding,
  not a failure.
- **HMC IOHMM samples transition logits, not spline coefficients.**
  The spline layer of §4.1 remains a least-squares fit. Extending HMC
  to spline coefficients is a possible Phase 4, not part of the
  baseline plan.
- **Bridge sampling for K is not in the plan.** AIC/BIC and a
  cross-validation extension cover the model-selection trio at lower
  cost. Bridge sampling is excluded for the same reason as MCMC on Θ —
  it restates an answer rather than producing a new one.
- **No portfolio layer.** Single-asset by design, consistent with the
  paper. Adding portfolio aggregation would shift the question being
  asked and is out of scope for the ASPTA timeline.
