"""Log-space forward filtering for one-dimensional Gaussian HMM return series.

The forward recursion here is the core filtering step behind the paper's
expected-return prediction logic: it updates the filtered state probabilities
``p(m_t | Δy_{1:t})`` and then projects them one step ahead through the
transition matrix to obtain ``E[Δy_{t+1} | Δy_{1:t}]``.

References: §6 forward filtering and prediction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hft_hmm.core import PAPER_FAITHFUL, StateGrid
from hft_hmm.models.gaussian_hmm import GaussianHMMResult

__category__: Final[str] = PAPER_FAITHFUL


@dataclass(frozen=True)
class ForwardFilterResult:
    """Immutable output of the Gaussian-HMM forward filter.

    ``filtering_probabilities[t, i]`` is the normalized filtering probability
    ``p(m_t = i | Δy_{1:t})``. ``predicted_next_state_probabilities[t, i]`` is
    ``p(m_{t+1} = i | Δy_{1:t})`` obtained by pushing the filtered
    distribution one step ahead through the transition matrix. The expected
    next return is the dot product of the predicted next-state probabilities
    with the state-mean vector.
    """

    state_grid: StateGrid
    filtering_probabilities: np.ndarray
    predicted_next_state_probabilities: np.ndarray
    expected_next_returns: np.ndarray
    log_likelihood: float

    def __post_init__(self) -> None:
        k = self.state_grid.k
        filtering = np.asarray(self.filtering_probabilities, dtype=float).copy()
        predicted = np.asarray(self.predicted_next_state_probabilities, dtype=float).copy()
        expected = np.asarray(self.expected_next_returns, dtype=float).copy()

        if filtering.ndim != 2:
            raise ValueError("filtering_probabilities must be a 2-D array.")
        if filtering.shape[1] != k:
            raise ValueError(
                f"filtering_probabilities must have shape (n_obs, {k}), got {filtering.shape}."
            )
        if predicted.shape != filtering.shape:
            raise ValueError(
                "predicted_next_state_probabilities must have the same shape as "
                "filtering_probabilities."
            )
        if expected.shape != (filtering.shape[0],):
            raise ValueError(
                "expected_next_returns must have shape (n_obs,), got "
                f"{expected.shape} for n_obs={filtering.shape[0]}."
            )
        if filtering.shape[0] < 1:
            raise ValueError("forward filter results must contain at least one observation.")
        if not np.all(np.isfinite(filtering)):
            raise ValueError("filtering_probabilities must contain only finite values.")
        if not np.all(np.isfinite(predicted)):
            raise ValueError("predicted_next_state_probabilities must contain only finite values.")
        if not np.all(np.isfinite(expected)):
            raise ValueError("expected_next_returns must contain only finite values.")
        if not np.allclose(filtering.sum(axis=1), 1.0):
            raise ValueError("filtering_probabilities rows must sum to 1.")
        if not np.allclose(predicted.sum(axis=1), 1.0):
            raise ValueError("predicted_next_state_probabilities rows must sum to 1.")
        if not np.isfinite(self.log_likelihood):
            raise ValueError("log_likelihood must be finite.")

        for array in (filtering, predicted, expected):
            array.setflags(write=False)

        object.__setattr__(self, "filtering_probabilities", filtering)
        object.__setattr__(self, "predicted_next_state_probabilities", predicted)
        object.__setattr__(self, "expected_next_returns", expected)


def forward_filter(
    returns: pd.Series | np.ndarray,
    model: GaussianHMMResult,
) -> ForwardFilterResult:
    """Run the normalized log-space forward recursion for a fitted Gaussian HMM.

    The returned filtering probabilities are normalized at each time step to
    prevent numerical underflow on long sequences. Expected next returns are
    computed as ``P(m_{t+1} | Δy_{1:t}) @ μ``, where ``μ`` is the canonical
    state-mean vector from ``model``.

    References: §6 forward filtering and one-step-ahead prediction
    """

    observations = _coerce_returns(returns)
    log_emissions = _gaussian_log_emissions(
        observations,
        means=model.means,
        variances=model.variances,
    )

    with np.errstate(divide="ignore"):
        log_start = np.log(model.initial_distribution)
        log_transition = np.log(model.transition_matrix)

    n_obs, k = log_emissions.shape
    log_filtering = np.empty((n_obs, k), dtype=float)
    log_normalizers = np.empty(n_obs, dtype=float)

    initial_log_alpha = log_start + log_emissions[0]
    initial_log_normalizer = float(logsumexp(initial_log_alpha))
    log_filtering[0] = initial_log_alpha - initial_log_normalizer
    log_normalizers[0] = initial_log_normalizer

    for t in range(1, n_obs):
        log_predicted = logsumexp(log_filtering[t - 1][:, None] + log_transition, axis=0)
        log_alpha = log_emissions[t] + log_predicted
        log_normalizer = float(logsumexp(log_alpha))
        log_filtering[t] = log_alpha - log_normalizer
        log_normalizers[t] = log_normalizer

    filtering = np.exp(log_filtering)
    predicted_next = filtering @ model.transition_matrix
    expected_next_returns = predicted_next @ model.means
    log_likelihood = float(log_normalizers.sum())

    return ForwardFilterResult(
        state_grid=model.state_grid,
        filtering_probabilities=filtering,
        predicted_next_state_probabilities=predicted_next,
        expected_next_returns=expected_next_returns,
        log_likelihood=log_likelihood,
    )


def _coerce_returns(returns: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"returns must be one-dimensional, got shape {values.shape}.")
    if values.size == 0:
        raise ValueError("returns must contain at least one observation.")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite values; drop NaN/inf first.")
    return values


def _gaussian_log_emissions(
    returns: np.ndarray,
    *,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    if returns.ndim != 1:
        raise ValueError(f"returns must be one-dimensional, got shape {returns.shape}.")
    if means.ndim != 1:
        raise ValueError(f"means must be one-dimensional, got shape {means.shape}.")
    if variances.shape != means.shape:
        raise ValueError(
            f"variances must have shape {means.shape}, got {variances.shape}."
        )

    centered = returns[:, None] - means[None, :]
    log_emissions = -0.5 * (
        np.log(2.0 * np.pi * variances[None, :]) + (centered * centered) / variances[None, :]
    )
    return np.asarray(log_emissions, dtype=float)
