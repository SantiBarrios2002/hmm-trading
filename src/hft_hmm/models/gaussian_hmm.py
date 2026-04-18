"""Gaussian HMM wrapper around ``hmmlearn`` for one-dimensional return series.

This module is an engineering wrapper: the EM/Baum-Welch fit, Viterbi decoding,
and forward-backward posterior computation are delegated to
``hmmlearn.hmm.GaussianHMM``. The wrapper adds:

- a frozen result dataclass exposing the learned parameters as numpy arrays,
- canonical state ordering (ascending mean return) so downstream filtering and
  signal code can assume a stable ``StateGrid`` convention,
- deterministic ``random_state`` plumbing,
- an ``init_from_plr`` path that seeds ``startprob_``, ``transmat_``,
  ``means_``, and ``covars_`` from an Issue 05 PLR baseline result, matching
  the paper's recommended initialization strategy.

See ``GAUSSIAN_HMM_REFERENCE`` for the paper pointer used in docstrings and
review notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from hft_hmm.core import (
    ENGINEERING_APPROXIMATION,
    PaperReference,
    StateGrid,
    default_labels,
    reference,
)
from hft_hmm.models.plr_baseline import PLRBaselineResult

__category__: Final[str] = ENGINEERING_APPROXIMATION
GAUSSIAN_HMM_REFERENCE: Final[PaperReference] = reference("§3", "Gaussian HMM baseline")

_MIN_VARIANCE: Final[float] = 1e-8
_TRANSMAT_SMOOTHING: Final[float] = 1.0


@dataclass(frozen=True)
class GaussianHMMResult:
    """Immutable snapshot of a fitted Gaussian HMM.

    Arrays are ordered by ascending mean return (the ``StateGrid`` convention),
    so ``means[i]``, ``variances[i]``, ``transition_matrix[i, :]``, and
    ``initial_distribution[i]`` all refer to the same state ``i``.
    """

    state_grid: StateGrid
    means: np.ndarray
    variances: np.ndarray
    transition_matrix: np.ndarray
    initial_distribution: np.ndarray
    log_likelihood: float
    n_observations: int
    converged: bool
    n_iter: int
    random_state: int | None

    def __post_init__(self) -> None:
        k = self.state_grid.k
        means = np.asarray(self.means, dtype=float).copy()
        variances = np.asarray(self.variances, dtype=float).copy()
        transmat = np.asarray(self.transition_matrix, dtype=float).copy()
        startprob = np.asarray(self.initial_distribution, dtype=float).copy()

        if means.shape != (k,):
            raise ValueError(f"means must have shape ({k},), got {means.shape}.")
        if variances.shape != (k,):
            raise ValueError(f"variances must have shape ({k},), got {variances.shape}.")
        if transmat.shape != (k, k):
            raise ValueError(f"transition_matrix must have shape ({k}, {k}), got {transmat.shape}.")
        if startprob.shape != (k,):
            raise ValueError(f"initial_distribution must have shape ({k},), got {startprob.shape}.")
        state_grid_means = np.asarray(self.state_grid.means, dtype=float)
        if state_grid_means.shape != (k,):
            raise ValueError(
                f"state_grid.means must have shape ({k},), got {state_grid_means.shape}."
            )
        if not np.all(np.isfinite(means)):
            raise ValueError("means must contain only finite values.")
        if not np.all(np.isfinite(variances)):
            raise ValueError("variances must contain only finite values.")
        if not np.all(variances > 0.0):
            raise ValueError("variances must be strictly positive.")
        if not np.all(np.isfinite(transmat)):
            raise ValueError("transition_matrix must contain only finite values.")
        if not np.all(transmat >= 0.0):
            raise ValueError("transition_matrix entries must be non-negative.")
        if not np.allclose(transmat.sum(axis=1), 1.0):
            raise ValueError("transition_matrix rows must sum to 1.")
        if not np.all(np.isfinite(startprob)):
            raise ValueError("initial_distribution must contain only finite values.")
        if not np.all(startprob >= 0.0):
            raise ValueError("initial_distribution entries must be non-negative.")
        if not np.isclose(startprob.sum(), 1.0):
            raise ValueError("initial_distribution must sum to 1.")
        if not np.all(np.isfinite(state_grid_means)):
            raise ValueError("state_grid.means must contain only finite values.")
        if not np.allclose(means, state_grid_means):
            raise ValueError("means must match state_grid.means.")
        if self.n_observations < 1:
            raise ValueError("n_observations must be positive.")

        for array in (means, variances, transmat, startprob):
            array.setflags(write=False)

        object.__setattr__(self, "means", means)
        object.__setattr__(self, "variances", variances)
        object.__setattr__(self, "transition_matrix", transmat)
        object.__setattr__(self, "initial_distribution", startprob)


class GaussianHMMWrapper:
    """Reviewable wrapper around ``hmmlearn.hmm.GaussianHMM`` for 1-D returns.

    The wrapper fits a diagonal-covariance Gaussian HMM on a univariate series,
    re-orders the learned states by ascending mean return, and exposes the
    result as a ``GaussianHMMResult``. ``predict`` and ``predict_proba`` return
    state indices consistent with that canonical ordering.

    References: §3 Gaussian HMM baseline (engineering wrapper around hmmlearn)
    """

    def __init__(
        self,
        n_states: int,
        *,
        random_state: int | None = None,
        n_iter: int = 100,
        tol: float = 1e-4,
    ) -> None:
        if n_states < 2:
            raise ValueError(f"n_states must be at least 2, got {n_states}.")
        if n_iter < 1:
            raise ValueError(f"n_iter must be positive, got {n_iter}.")
        if tol <= 0.0:
            raise ValueError(f"tol must be strictly positive, got {tol}.")

        self.n_states = n_states
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self._model: GaussianHMM | None = None
        self._permutation: np.ndarray | None = None

    def fit(
        self,
        returns: pd.Series | np.ndarray,
        *,
        init_from_plr: PLRBaselineResult | None = None,
    ) -> GaussianHMMResult:
        """Fit the Gaussian HMM and return a canonical result snapshot.

        Passing ``init_from_plr`` seeds EM from the supplied PLR baseline.
        Without it, ``hmmlearn`` draws its own initialization using
        ``random_state``.
        """
        observations = _coerce_returns(returns)
        reshaped = observations.reshape(-1, 1)

        if init_from_plr is not None and init_from_plr.state_grid.k != self.n_states:
            raise ValueError(
                "init_from_plr.state_grid.k does not match wrapper n_states; "
                f"got {init_from_plr.state_grid.k} vs {self.n_states}."
            )

        init_params = "" if init_from_plr is not None else "stmc"
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            init_params=init_params,
            params="stmc",
        )

        if init_from_plr is not None:
            _seed_from_plr(model, init_from_plr)

        model.fit(reshaped)
        log_likelihood = float(model.score(reshaped))

        means_raw = model.means_.reshape(-1)
        variances_raw = model.covars_.reshape(-1)
        permutation = np.argsort(means_raw, kind="mergesort")

        means_sorted = means_raw[permutation]
        variances_sorted = variances_raw[permutation]
        transmat_sorted = model.transmat_[permutation][:, permutation]
        startprob_sorted = model.startprob_[permutation]

        state_grid = StateGrid(
            k=self.n_states,
            means=means_sorted,
            labels=default_labels(self.n_states),
        )

        self._model = model
        self._permutation = permutation

        return GaussianHMMResult(
            state_grid=state_grid,
            means=means_sorted,
            variances=variances_sorted,
            transition_matrix=transmat_sorted,
            initial_distribution=startprob_sorted,
            log_likelihood=log_likelihood,
            n_observations=int(observations.shape[0]),
            converged=bool(model.monitor_.converged),
            n_iter=int(model.monitor_.iter),
            random_state=self.random_state,
        )

    def predict(self, returns: pd.Series | np.ndarray) -> np.ndarray:
        """Return the Viterbi-most-likely state sequence in canonical ordering."""
        model, permutation = self._require_fitted()
        reshaped = _coerce_returns(returns).reshape(-1, 1)
        raw_states = model.predict(reshaped)
        return _apply_permutation(raw_states, permutation)

    def predict_proba(self, returns: pd.Series | np.ndarray) -> np.ndarray:
        """Return forward-backward posteriors ordered by canonical state index."""
        model, permutation = self._require_fitted()
        reshaped = _coerce_returns(returns).reshape(-1, 1)
        raw_posteriors = np.asarray(model.predict_proba(reshaped), dtype=float)
        return raw_posteriors[:, permutation]

    def _require_fitted(self) -> tuple[GaussianHMM, np.ndarray]:
        if self._model is None or self._permutation is None:
            raise RuntimeError("Call fit() before predict() or predict_proba().")
        return self._model, self._permutation


def _coerce_returns(returns: pd.Series | np.ndarray) -> np.ndarray:
    """Validate and return a finite 1-D float array of return observations."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"returns must be one-dimensional, got shape {values.shape}.")
    if values.size == 0:
        raise ValueError("returns must contain at least one observation.")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite values; drop NaN/inf first.")
    return values


def _apply_permutation(raw_states: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """Map raw hmmlearn state indices into the ascending-mean canonical order."""
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size)
    return np.asarray(inverse[raw_states], dtype=permutation.dtype)


def _seed_from_plr(model: GaussianHMM, plr: PLRBaselineResult) -> None:
    """Set model parameters from a PLR baseline result before fitting.

    Transition probabilities are smoothed with an additive Laplace prior so
    states that the PLR sequence never left keep non-zero off-diagonal mass,
    which prevents EM from locking into degenerate solutions.
    """
    k = plr.state_grid.k
    means = plr.state_grid.means.reshape(k, 1).astype(float, copy=True)
    variances = np.maximum(plr.state_variances.astype(float, copy=True), _MIN_VARIANCE)
    covars = variances.reshape(k, 1)

    state_sequence = np.asarray(plr.state_sequence, dtype=int)
    transmat = np.full((k, k), _TRANSMAT_SMOOTHING, dtype=float)
    for previous, current in zip(state_sequence[:-1], state_sequence[1:], strict=True):
        transmat[previous, current] += 1.0
    transmat /= transmat.sum(axis=1, keepdims=True)

    startprob = np.full(k, _TRANSMAT_SMOOTHING, dtype=float)
    startprob[int(state_sequence[0])] += 1.0
    startprob /= startprob.sum()

    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars
