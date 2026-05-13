"""Continuous-parametric IOHMM transition conditioning fit with NumPyro NUTS.

This module implements the Gate K transition-function upgrade:

``A_ij(x_t) = softmax_j(W_i· x_t + b_i)``

for scalar side information ``x_t``.  It is a paper-faithful transition
parameterization, with the repo's deliberately scoped inference simplification:
emission parameters stay fixed at the per-window Baum-Welch fit and the decoded
training state path is used as the supervisory signal.  NUTS samples only the
transition logits ``(W, b)``; it does not sample the Gaussian HMM parameters
``Theta`` that remain excluded by ``IMPLEMENTATION_PLAN.md §2.5``.

Rows of softmax logits are identifiable only up to an additive constant.  The
NumPyro sample sites use the requested weakly informative ``Normal(0, 1)``
priors, and returned posterior samples are row-centered before diagnostics and
forecast use so ``W`` and ``b`` have a stable zero-sum-per-row convention.

Note: this row-centering is applied as a *transformation* of free Normal(0, 1)
draws, so HMC still explores the unidentified row-mean direction (one wasted
dimension per row, sampling its prior). R-hat and ESS are computed on the
centered samples, so the reported diagnostics reflect mixing in the
identifiable subspace. A cleaner future refactor would parameterize K-1 free
logits per row directly (e.g. fix the last column to zero) and drop the
post-hoc centering; the math is equivalent and the posterior space is
smaller.

References: §4 IOHMM side-information transitions / Gate K HMC extension
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import lax
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer import MCMC, NUTS

from hft_hmm.core import ENGINEERING_APPROXIMATION, PaperReference, reference

__category__: Final[str] = ENGINEERING_APPROXIMATION
CONTINUOUS_IOHMM_REFERENCE: Final[PaperReference] = reference(
    "§4", "continuous-parametric IOHMM transitions / Gate K HMC"
)

_PARAMETER_NAMES: Final[tuple[str, str]] = ("W", "b")
_MIN_STD: Final[float] = 1e-12


@dataclass(frozen=True)
class ContinuousIOHMMConfig:
    """NumPyro NUTS settings and convergence thresholds for the Gate K IOHMM.

    ``num_chains``, ``num_warmup``, ``num_samples``, and
    ``target_accept_prob`` are passed directly to NumPyro's NUTS/MCMC runner.
    ``seed`` controls JAX's PRNG key. ``rhat_threshold`` and
    ``ess_bulk_threshold`` define the artifact-level ``converged`` flag.

    References: §4 IOHMM side-information transitions / Gate K HMC
    """

    num_chains: int = 2
    num_warmup: int = 500
    num_samples: int = 1000
    target_accept_prob: float = 0.8
    seed: int = 0
    rhat_threshold: float = 1.05
    ess_bulk_threshold: int = 200

    def __post_init__(self) -> None:
        _validate_int(self.num_chains, "num_chains")
        if self.num_chains < 2:
            raise ValueError(f"num_chains must be at least 2, got {self.num_chains}.")
        _validate_int(self.num_warmup, "num_warmup")
        if self.num_warmup < 1:
            raise ValueError(f"num_warmup must be positive, got {self.num_warmup}.")
        _validate_int(self.num_samples, "num_samples")
        if self.num_samples < 4:
            raise ValueError(
                "num_samples must be at least 4 so split R-hat is defined, "
                f"got {self.num_samples}."
            )
        accept = float(self.target_accept_prob)
        if not np.isfinite(accept) or not 0.0 < accept < 1.0:
            raise ValueError(
                "target_accept_prob must be a finite float strictly between 0 and 1, "
                f"got {self.target_accept_prob!r}."
            )
        object.__setattr__(self, "target_accept_prob", accept)
        _validate_int(self.seed, "seed")
        rhat = float(self.rhat_threshold)
        if not np.isfinite(rhat) or rhat <= 1.0:
            raise ValueError(f"rhat_threshold must be a finite float greater than 1, got {rhat!r}.")
        object.__setattr__(self, "rhat_threshold", rhat)
        _validate_int(self.ess_bulk_threshold, "ess_bulk_threshold")
        if self.ess_bulk_threshold < 1:
            raise ValueError(
                "ess_bulk_threshold must be positive, " f"got {self.ess_bulk_threshold}."
            )


@dataclass(frozen=True)
class ContinuousIOHMMResult:
    """Posterior and diagnostics for one continuous-parametric IOHMM fit.

    ``posterior_samples`` contains row-centered samples for keys ``"W"`` and
    ``"b"`` with shape ``(chains, samples, K, K)``. ``posterior_mean_W`` and
    ``posterior_mean_b`` are coefficient means under the same row-centered
    convention. ``standardization`` stores the training-slice ``(mean, std)``
    used to transform side information before evaluating ``A(x_t)``.

    References: §4 IOHMM side-information transitions / Gate K HMC
    """

    config: ContinuousIOHMMConfig
    posterior_samples: dict[str, np.ndarray]
    rhat: dict[str, np.ndarray]
    ess_bulk: dict[str, np.ndarray]
    ess_tail: dict[str, np.ndarray]
    converged: bool
    posterior_mean_W: np.ndarray
    posterior_mean_b: np.ndarray
    standardization: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.config, ContinuousIOHMMConfig):
            raise TypeError(
                "config must be a ContinuousIOHMMConfig, " f"got {type(self.config).__name__}."
            )

        samples = _freeze_parameter_dict(
            self.posterior_samples,
            name="posterior_samples",
            ndim=4,
            allow_negative=True,
        )
        sample_shape = samples["W"].shape
        if samples["b"].shape != sample_shape:
            raise ValueError(
                "posterior_samples['W'] and posterior_samples['b'] must have the same "
                f"shape; got {sample_shape} and {samples['b'].shape}."
            )
        if sample_shape[0] != self.config.num_chains or sample_shape[1] != self.config.num_samples:
            raise ValueError(
                "posterior sample leading dimensions must match the config "
                f"({self.config.num_chains}, {self.config.num_samples}); "
                f"got {sample_shape[:2]}."
            )
        if sample_shape[2] != sample_shape[3] or sample_shape[2] < 2:
            raise ValueError(f"posterior samples must end with shape (K, K), got {sample_shape}.")
        k = sample_shape[2]

        rhat = _freeze_parameter_dict(self.rhat, name="rhat", ndim=2, allow_negative=False)
        ess_bulk = _freeze_parameter_dict(
            self.ess_bulk,
            name="ess_bulk",
            ndim=2,
            allow_negative=False,
        )
        ess_tail = _freeze_parameter_dict(
            self.ess_tail,
            name="ess_tail",
            ndim=2,
            allow_negative=False,
        )
        for metric_name, metric in (
            ("rhat", rhat),
            ("ess_bulk", ess_bulk),
            ("ess_tail", ess_tail),
        ):
            for param_name, values in metric.items():
                if values.shape != (k, k):
                    raise ValueError(
                        f"{metric_name}[{param_name!r}] must have shape ({k}, {k}), "
                        f"got {values.shape}."
                    )

        mean_w = np.asarray(self.posterior_mean_W, dtype=float).copy()
        mean_b = np.asarray(self.posterior_mean_b, dtype=float).copy()
        for attr_name, values in (
            ("posterior_mean_W", mean_w),
            ("posterior_mean_b", mean_b),
        ):
            if values.shape != (k, k):
                raise ValueError(f"{attr_name} must have shape ({k}, {k}), got {values.shape}.")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{attr_name} must contain only finite values.")
            values.setflags(write=False)

        mean, std = self.standardization
        mean = float(mean)
        std = float(std)
        if not np.isfinite(mean):
            raise ValueError(f"standardization mean must be finite, got {mean!r}.")
        if not np.isfinite(std) or std <= 0.0:
            raise ValueError(
                f"standardization std must be finite and strictly positive, got {std!r}."
            )

        object.__setattr__(self, "posterior_samples", samples)
        object.__setattr__(self, "rhat", rhat)
        object.__setattr__(self, "ess_bulk", ess_bulk)
        object.__setattr__(self, "ess_tail", ess_tail)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "posterior_mean_W", mean_w)
        object.__setattr__(self, "posterior_mean_b", mean_b)
        object.__setattr__(self, "standardization", (mean, std))


def fit_continuous_iohmm(
    *,
    state_sequence: np.ndarray,
    side_information: np.ndarray,
    n_states: int,
    config: ContinuousIOHMMConfig | None = None,
) -> ContinuousIOHMMResult:
    """Fit ``A(x_t) = softmax(W x_t + b)`` from decoded training states.

    The transition likelihood is supervised by the decoded state path:
    transition ``state_sequence[t] -> state_sequence[t + 1]`` is conditioned on
    ``side_information[t]``.  Side information is standardized using only this
    training slice and the same transformation is stored for forecast-time
    transition evaluation.

    References: §4 IOHMM side-information transitions / Gate K HMC
    """
    if config is None:
        config = ContinuousIOHMMConfig()
    elif not isinstance(config, ContinuousIOHMMConfig):
        raise TypeError(f"config must be a ContinuousIOHMMConfig, got {type(config).__name__}.")

    K = _validate_n_states(n_states)
    states = _coerce_state_sequence(state_sequence, n_states=K)
    side_info = _coerce_side_information(side_information, expected_len=len(states))
    mean = float(side_info.mean())
    std = float(side_info.std())
    if std <= _MIN_STD:
        std = 1.0
    side_info_std = (side_info - mean) / std

    kernel = NUTS(_numpyro_model, target_accept_prob=config.target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(config.seed),
        jnp.asarray(side_info_std, dtype=jnp.float32),
        jnp.asarray(states, dtype=jnp.int32),
        K,
    )

    raw_samples = {
        name: np.asarray(value, dtype=float)
        for name, value in mcmc.get_samples(group_by_chain=True).items()
        if name in _PARAMETER_NAMES
    }
    samples = {name: _center_rows(value) for name, value in raw_samples.items()}
    diagnostics = _diagnostics(samples)
    diagnostics = _mark_unidentified_rows(
        diagnostics,
        states=states,
        n_states=K,
        config=config,
    )

    converged = _all_converged(
        diagnostics["rhat"],
        diagnostics["ess_bulk"],
        rhat_threshold=config.rhat_threshold,
        ess_bulk_threshold=config.ess_bulk_threshold,
    )

    return ContinuousIOHMMResult(
        config=config,
        posterior_samples=samples,
        rhat=diagnostics["rhat"],
        ess_bulk=diagnostics["ess_bulk"],
        ess_tail=diagnostics["ess_tail"],
        converged=converged,
        posterior_mean_W=samples["W"].reshape(-1, K, K).mean(axis=0),
        posterior_mean_b=samples["b"].reshape(-1, K, K).mean(axis=0),
        standardization=(mean, std),
    )


def transition_probabilities_at(
    *,
    result: ContinuousIOHMMResult,
    x_values: np.ndarray,
) -> np.ndarray:
    """Evaluate posterior-mean transition matrices at side-information values.

    Returns an array with shape ``(len(x_values), K, K)``.  The training-window
    standardization stored on ``result`` is reapplied before evaluating the
    softmax logits, and every transition row sums to one.

    References: §4 IOHMM side-information transitions / Gate K HMC
    """
    if not isinstance(result, ContinuousIOHMMResult):
        raise TypeError("result must be a ContinuousIOHMMResult, " f"got {type(result).__name__}.")
    x = np.asarray(x_values, dtype=float)
    if x.ndim == 0:
        x = x.reshape(1)
    if x.ndim != 1:
        raise ValueError(f"x_values must be one-dimensional, got shape {x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x_values must contain only finite values.")

    mean, std = result.standardization
    standardized = (x - mean) / std
    logits = result.posterior_mean_W[None, :, :] * standardized[:, None, None]
    logits = logits + result.posterior_mean_b[None, :, :]
    return _softmax(logits, axis=2)


def _numpyro_model(side_info_std: jnp.ndarray, states: jnp.ndarray, n_states: int) -> None:
    """NumPyro transition-logit model with a scanned categorical likelihood."""
    W_raw = numpyro.sample("W", dist.Normal(0.0, 1.0).expand([n_states, n_states]).to_event(2))
    b_raw = numpyro.sample("b", dist.Normal(0.0, 1.0).expand([n_states, n_states]).to_event(2))
    W = W_raw - jnp.mean(W_raw, axis=1, keepdims=True)
    b = b_raw - jnp.mean(b_raw, axis=1, keepdims=True)

    transition_x = side_info_std[:-1]
    from_states = states[:-1]
    to_states = states[1:]

    def scan_step(total_log_lik: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
        x_t, from_state, to_state = inputs
        logits = W * x_t + b
        log_probs = jax.nn.log_softmax(logits, axis=1)
        total_log_lik = total_log_lik + log_probs[from_state, to_state]
        return total_log_lik, None

    log_lik, _ = lax.scan(
        scan_step,
        jnp.asarray(0.0, dtype=side_info_std.dtype),
        (transition_x, from_states, to_states),
    )
    numpyro.factor("transition_loglik", log_lik)


def _diagnostics(samples: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    rhat: dict[str, np.ndarray] = {}
    ess_bulk: dict[str, np.ndarray] = {}
    ess_tail: dict[str, np.ndarray] = {}
    for name, values in samples.items():
        rhat[name] = _sanitize_metric(split_gelman_rubin(values), fallback=10.0)
        ess_bulk[name] = _sanitize_metric(effective_sample_size(values), fallback=0.0)
        ess_tail[name] = _sanitize_metric(_tail_ess(values), fallback=0.0)
    return {"rhat": rhat, "ess_bulk": ess_bulk, "ess_tail": ess_tail}


def _tail_ess(values: np.ndarray) -> np.ndarray:
    flat = values.reshape((-1,) + values.shape[2:])
    lower = np.quantile(flat, 0.05, axis=0)
    upper = np.quantile(flat, 0.95, axis=0)
    lower_indicator = (values <= lower[None, None, :, :]).astype(float)
    upper_indicator = (values >= upper[None, None, :, :]).astype(float)
    return cast(
        np.ndarray,
        np.minimum(
            np.asarray(effective_sample_size(lower_indicator), dtype=float),
            np.asarray(effective_sample_size(upper_indicator), dtype=float),
        ),
    )


def _mark_unidentified_rows(
    diagnostics: dict[str, dict[str, np.ndarray]],
    *,
    states: np.ndarray,
    n_states: int,
    config: ContinuousIOHMMConfig,
) -> dict[str, dict[str, np.ndarray]]:
    outgoing_counts = np.bincount(states[:-1], minlength=n_states)
    unidentified = np.flatnonzero(outgoing_counts == 0)
    if unidentified.size == 0:
        return diagnostics

    marked = {
        metric: {name: values.copy() for name, values in by_param.items()}
        for metric, by_param in diagnostics.items()
    }
    high_rhat = max(config.rhat_threshold * 2.0, config.rhat_threshold + 1.0)
    for name in _PARAMETER_NAMES:
        marked["rhat"][name][unidentified, :] = high_rhat
        marked["ess_bulk"][name][unidentified, :] = 0.0
        marked["ess_tail"][name][unidentified, :] = 0.0
    return marked


def _all_converged(
    rhat: dict[str, np.ndarray],
    ess_bulk: dict[str, np.ndarray],
    *,
    rhat_threshold: float,
    ess_bulk_threshold: int,
) -> bool:
    all_rhat = np.concatenate([values.ravel() for values in rhat.values()])
    all_ess = np.concatenate([values.ravel() for values in ess_bulk.values()])
    return bool(
        np.all(np.isfinite(all_rhat))
        and np.all(all_rhat < rhat_threshold)
        and np.all(np.isfinite(all_ess))
        and np.all(all_ess > ess_bulk_threshold)
    )


def _coerce_state_sequence(state_sequence: np.ndarray, *, n_states: int) -> np.ndarray:
    states_raw = np.asarray(state_sequence)
    if states_raw.ndim != 1:
        raise ValueError(f"state_sequence must be one-dimensional, got shape {states_raw.shape}.")
    if states_raw.shape[0] < 2:
        raise ValueError("state_sequence must contain at least 2 observations.")
    if not np.all(np.isfinite(states_raw)):
        raise ValueError("state_sequence must contain only finite values.")
    if not np.allclose(states_raw, np.round(states_raw)):
        raise ValueError("state_sequence must be integer-valued.")
    states = states_raw.astype(int)
    if np.any(states < 0) or np.any(states >= n_states):
        raise ValueError(f"state_sequence contains a state index outside [0, {n_states - 1}].")
    return states


def _coerce_side_information(side_information: np.ndarray, *, expected_len: int) -> np.ndarray:
    side_info = np.asarray(side_information, dtype=float)
    if side_info.ndim != 1:
        raise ValueError(f"side_information must be one-dimensional, got shape {side_info.shape}.")
    if side_info.shape[0] != expected_len:
        raise ValueError(
            "side_information must have the same length as state_sequence; "
            f"got {side_info.shape[0]} vs {expected_len}."
        )
    if not np.all(np.isfinite(side_info)):
        raise ValueError("side_information must contain only finite values.")
    return side_info


def _validate_n_states(n_states: int) -> int:
    _validate_int(n_states, "n_states")
    if n_states < 2:
        raise ValueError(f"n_states must be at least 2, got {n_states}.")
    return int(n_states)


def _validate_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")


def _center_rows(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=float) - np.mean(values, axis=-1, keepdims=True)
    return cast(np.ndarray, centered)


def _softmax(values: np.ndarray, *, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return cast(np.ndarray, exp_values / exp_values.sum(axis=axis, keepdims=True))


def _sanitize_metric(values: np.ndarray, *, fallback: float) -> np.ndarray:
    metric = np.asarray(values, dtype=float).copy()
    metric[~np.isfinite(metric)] = fallback
    metric[metric < 0.0] = fallback
    return metric


def _freeze_parameter_dict(
    raw: dict[str, np.ndarray],
    *,
    name: str,
    ndim: int,
    allow_negative: bool,
) -> dict[str, np.ndarray]:
    if not isinstance(raw, dict):
        raise TypeError(f"{name} must be a dict, got {type(raw).__name__}.")
    missing = [param for param in _PARAMETER_NAMES if param not in raw]
    if missing:
        raise ValueError(f"{name} is missing keys {missing!r}.")
    frozen: dict[str, np.ndarray] = {}
    for param in _PARAMETER_NAMES:
        values = np.asarray(raw[param], dtype=float).copy()
        if values.ndim != ndim:
            raise ValueError(f"{name}[{param!r}] must be {ndim}-D, got shape {values.shape}.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name}[{param!r}] must contain only finite values.")
        if not allow_negative and np.any(values < 0.0):
            raise ValueError(f"{name}[{param!r}] must contain non-negative values.")
        values.setflags(write=False)
        frozen[param] = values
    return frozen
