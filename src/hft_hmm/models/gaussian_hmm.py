"""Gaussian HMM wrapper around ``hmmlearn`` for one-dimensional return series.

This module is an engineering wrapper: the EM/Baum-Welch fit, Viterbi decoding,
and forward-backward posterior computation are delegated to
``hmmlearn.hmm.GaussianHMM``. The wrapper adds:

- a frozen result dataclass exposing the learned parameters as numpy arrays,
- canonical state ordering (ascending mean return) so downstream filtering and
  signal code can assume a stable ``StateGrid`` convention,
- deterministic ``random_state`` plumbing,
- an explicit minimum-variance floor ``min_variance`` with a tracked ES
  default of ``1e-8`` in log-return units, which is roughly the squared
  log return of a two-tick move (``2 * 0.25`` index points) near an ES
  price level of ``5,000``, enforced by a ``variance_floor_policy`` of
  ``"clamp"`` (default, emits a ``logging.warning`` with the affected
  state indices) or ``"raise"`` (raise ``ValueError`` naming the
  offending state indices — intended for tests and strict pipelines),
- stabilized EM backend settings chosen so the tracked ES fixture fits with a
  monotone non-decreasing log-likelihood under ``hmmlearn``'s log-space
  implementation: ``min_covar=min_variance``, ``startprob_prior=2.0``,
  ``transmat_prior=2.0``, ``covars_prior=min_variance``, and
  ``covars_weight=2.0``,
- an ``init_from_plr`` path that seeds ``startprob_``, ``transmat_``,
  ``means_``, and ``covars_`` from an Issue 05 PLR baseline result, matching
  the paper's recommended initialization strategy.

See ``GAUSSIAN_HMM_REFERENCE`` for the paper pointer used in docstrings and
review notes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final, Literal

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

_DEFAULT_MIN_VARIANCE: Final[float] = 1e-8
_TRANSMAT_SMOOTHING: Final[float] = 1.0
# Dirichlet / inverse-gamma pseudo-counts applied to the M-step. A value of 2.0
# (one prior observation per off-diagonal direction / state in addition to the
# uniform Laplace baseline of 1.0) is the smallest integer weight that keeps
# Baum-Welch monotone on tests/fixtures/es_1min_sample.csv across the tracked
# K=2 configuration without materially biasing the posterior away from the MLE.
_STARTPROB_PRIOR: Final[float] = 2.0
_TRANSMAT_PRIOR: Final[float] = 2.0
# Inverse-gamma weight on covars_prior. Setting weight > 1 regularizes the
# variance update so tiny-cluster states cannot collapse between EM iterations,
# which is what produced non-monotone log-likelihoods under the library default.
_COVARS_WEIGHT: Final[float] = 2.0
_MONOTONICITY_ATOL: Final[float] = float(np.finfo(float).eps ** 0.5)
_VARIANCE_FLOOR_POLICIES: Final[frozenset[str]] = frozenset({"clamp", "raise"})

_LOGGER = logging.getLogger(__name__)

VarianceFloorPolicy = Literal["clamp", "raise"]


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
    min_variance: float = _DEFAULT_MIN_VARIANCE
    em_log_likelihood_history: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))

    def __post_init__(self) -> None:
        k = self.state_grid.k
        means = np.asarray(self.means, dtype=float).copy()
        variances = np.asarray(self.variances, dtype=float).copy()
        transmat = np.asarray(self.transition_matrix, dtype=float).copy()
        startprob = np.asarray(self.initial_distribution, dtype=float).copy()
        history = np.asarray(self.em_log_likelihood_history, dtype=float).copy()

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
        if not np.isfinite(self.min_variance) or self.min_variance <= 0.0:
            raise ValueError(
                "min_variance must be a finite strictly positive float, "
                f"got {self.min_variance!r}."
            )

        if history.ndim != 1:
            raise ValueError(
                "em_log_likelihood_history must be one-dimensional, " f"got shape {history.shape}."
            )
        if history.size == 0:
            history = np.array([float(self.log_likelihood)], dtype=float)
        if not np.all(np.isfinite(history)):
            raise ValueError("em_log_likelihood_history must contain only finite values.")

        for array in (means, variances, transmat, startprob, history):
            array.setflags(write=False)

        object.__setattr__(self, "means", means)
        object.__setattr__(self, "variances", variances)
        object.__setattr__(self, "transition_matrix", transmat)
        object.__setattr__(self, "initial_distribution", startprob)
        object.__setattr__(self, "em_log_likelihood_history", history)

    @property
    def em_log_likelihood_is_monotone(self) -> bool:
        """Return whether the stored EM log-likelihood history is non-decreasing."""
        return _is_monotone_non_decreasing(self.em_log_likelihood_history)


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
        min_variance: float = _DEFAULT_MIN_VARIANCE,
        variance_floor_policy: VarianceFloorPolicy = "clamp",
    ) -> None:
        if n_states < 2:
            raise ValueError(f"n_states must be at least 2, got {n_states}.")
        if n_iter < 1:
            raise ValueError(f"n_iter must be positive, got {n_iter}.")
        if tol <= 0.0:
            raise ValueError(f"tol must be strictly positive, got {tol}.")
        if not np.isfinite(min_variance) or min_variance <= 0.0:
            raise ValueError(
                "min_variance must be a finite strictly positive float, " f"got {min_variance!r}."
            )
        if variance_floor_policy not in _VARIANCE_FLOOR_POLICIES:
            raise ValueError(
                "variance_floor_policy must be one of "
                f"{sorted(_VARIANCE_FLOOR_POLICIES)}, got {variance_floor_policy!r}."
            )

        self.n_states = n_states
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.min_variance = float(min_variance)
        self.variance_floor_policy: VarianceFloorPolicy = variance_floor_policy
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
            min_covar=self.min_variance,
            startprob_prior=_STARTPROB_PRIOR,
            transmat_prior=_TRANSMAT_PRIOR,
            covars_prior=self.min_variance,
            covars_weight=_COVARS_WEIGHT,
            random_state=self.random_state,
            init_params=init_params,
            params="stmc",
            implementation="log",
        )

        if init_from_plr is not None:
            _seed_from_plr(model, init_from_plr, min_variance=self.min_variance)

        model.fit(reshaped)
        _enforce_variance_floor(
            model,
            min_variance=self.min_variance,
            policy=self.variance_floor_policy,
        )
        log_likelihood = float(model.score(reshaped))
        em_log_likelihood_history = np.array(model.monitor_.history, dtype=float)

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
            min_variance=self.min_variance,
            em_log_likelihood_history=em_log_likelihood_history,
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


def _seed_from_plr(
    model: GaussianHMM,
    plr: PLRBaselineResult,
    *,
    min_variance: float,
) -> None:
    """Set model parameters from a PLR baseline result before fitting.

    Transition probabilities are smoothed with an additive Laplace prior so
    states that the PLR sequence never left keep non-zero off-diagonal mass,
    which prevents EM from locking into degenerate solutions.
    """
    k = plr.state_grid.k
    means = plr.state_grid.means.reshape(k, 1).astype(float, copy=True)
    variances = np.maximum(plr.state_variances.astype(float, copy=True), min_variance)
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


def _enforce_variance_floor(
    model: GaussianHMM,
    *,
    min_variance: float,
    policy: VarianceFloorPolicy,
) -> None:
    """Apply the configured variance floor to the fitted model's diagonal covars.

    Reads the compact ``(n_components, n_features)`` diagonal via the backing
    attribute because the public ``covars_`` getter returns a ``(n, f, f)``
    diagonal-expanded tensor under ``covariance_type="diag"``. Writes through
    the public ``covars_`` setter so hmmlearn re-runs its own shape/validity
    checks on the clamped value.

    State indices in the warning / error message follow the unsorted hmmlearn
    order present at the moment of clamping; the wrapper applies its
    ascending-mean permutation afterward.
    """
    diagonal = np.asarray(model._covars_, dtype=float).reshape(model.n_components, -1)
    below_floor = np.any(diagonal < min_variance, axis=1)
    if not below_floor.any():
        return

    affected = np.where(below_floor)[0].tolist()
    if policy == "raise":
        minima = diagonal[below_floor].min(axis=1).tolist()
        raise ValueError(
            f"fitted variance(s) fell below min_variance={min_variance!r} "
            f"in state index/indices {affected} (per-state minima={minima}); "
            "refit with a lower floor or variance_floor_policy='clamp'."
        )

    _LOGGER.warning(
        "Clamping fitted variances up to min_variance=%g in state index/indices %s",
        min_variance,
        affected,
    )
    clamped = np.maximum(diagonal, min_variance)
    model.covars_ = clamped


def _is_monotone_non_decreasing(history: np.ndarray) -> bool:
    """Return whether ``history`` is non-decreasing up to float noise."""
    if history.shape[0] < 2:
        return True
    return bool(np.all(np.diff(history) >= -_MONOTONICITY_ATOL))
