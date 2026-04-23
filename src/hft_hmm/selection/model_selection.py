"""Model selection over candidate hidden-state counts ``K``.

This module fits ``GaussianHMMWrapper`` for each requested ``K`` and scores the
fit with log-likelihood, Akaike Information Criterion, and Bayesian Information
Criterion. The parameter-count formula is specialized to a diagonal-covariance
Gaussian HMM on one-dimensional return series:

- ``K`` state means
- ``K`` diagonal variances
- ``K * (K - 1)`` free transition-matrix entries (rows sum to 1)
- ``K - 1`` free initial-distribution entries (sums to 1)

for a total of ``K^2 + 2K - 1`` free parameters.

The paper compares ``K`` via cross-validation, AIC/BIC, and marginal likelihood
under MCMC bridge sampling. This module implements only the AIC/BIC route;
cross-validation and MCMC are explicitly out of scope per
``IMPLEMENTATION_PLAN.md`` §2.5.

See ``MODEL_SELECTION_REFERENCE`` for the paper pointer used in docstrings and
review notes.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from hft_hmm.core import ENGINEERING_APPROXIMATION, PaperReference, reference
from hft_hmm.models.gaussian_hmm import GaussianHMMWrapper

__category__: Final[str] = ENGINEERING_APPROXIMATION
MODEL_SELECTION_REFERENCE: Final[PaperReference] = reference("§4", "model selection via AIC/BIC")


@dataclass(frozen=True)
class ModelSelectionRow:
    """Single-row summary of a Gaussian HMM fit at one candidate ``K``."""

    k: int
    log_likelihood: float
    n_parameters: int
    n_observations: int
    aic: float
    bic: float
    converged: bool
    random_state: int | None

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError(f"k must be at least 2, got {self.k}.")
        if self.n_parameters < 1:
            raise ValueError(f"n_parameters must be positive, got {self.n_parameters}.")
        if self.n_observations < 1:
            raise ValueError(f"n_observations must be positive, got {self.n_observations}.")


@dataclass(frozen=True)
class ModelSelectionResult:
    """Immutable summary of a model-selection sweep sorted by ``k`` ascending."""

    rows: tuple[ModelSelectionRow, ...]
    best_by_aic: int
    best_by_bic: int

    @property
    def any_non_converged(self) -> bool:
        """Return whether any fitted candidate failed to converge."""
        return any(not row.converged for row in self.rows)

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("rows must contain at least one entry.")
        if [row.k for row in self.rows] != sorted(row.k for row in self.rows):
            raise ValueError("rows must be sorted by k ascending.")
        if len({row.k for row in self.rows}) != len(self.rows):
            raise ValueError("rows must not contain duplicate k values.")
        k_values = {row.k for row in self.rows}
        if self.best_by_aic not in k_values:
            raise ValueError(f"best_by_aic={self.best_by_aic} missing from rows.")
        if self.best_by_bic not in k_values:
            raise ValueError(f"best_by_bic={self.best_by_bic} missing from rows.")


def count_gaussian_hmm_parameters(n_states: int) -> int:
    """Return the number of free parameters in a diagonal 1-D Gaussian HMM.

    The closed form ``K^2 + 2K - 1`` decomposes as ``K`` means, ``K`` diagonal
    variances, ``K(K-1)`` free transition-matrix entries, and ``K-1`` free
    initial-distribution entries.

    References: §4 model selection (engineering utility)
    """
    if n_states < 2:
        raise ValueError(f"n_states must be at least 2, got {n_states}.")
    return n_states * n_states + 2 * n_states - 1


def aic(log_likelihood: float, n_parameters: int) -> float:
    """Akaike Information Criterion: ``2p - 2*logL`` (smaller is better)."""
    if n_parameters < 1:
        raise ValueError(f"n_parameters must be positive, got {n_parameters}.")
    return 2.0 * n_parameters - 2.0 * log_likelihood


def bic(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    """Bayesian Information Criterion: ``p*ln(n) - 2*logL`` (smaller is better)."""
    if n_parameters < 1:
        raise ValueError(f"n_parameters must be positive, got {n_parameters}.")
    if n_observations < 1:
        raise ValueError(f"n_observations must be positive, got {n_observations}.")
    return n_parameters * math.log(n_observations) - 2.0 * log_likelihood


def compare_state_counts(
    returns: pd.Series | np.ndarray,
    k_values: Iterable[int],
    *,
    random_state: int | None = None,
    n_iter: int = 100,
    tol: float = 1e-4,
    min_variance: float = 1e-8,
) -> ModelSelectionResult:
    """Fit a Gaussian HMM at each ``K`` in ``k_values`` and rank by AIC/BIC.

    The same ``random_state`` is passed to every fit so successive sweeps with
    identical arguments produce bit-identical results. Duplicate ``K`` values
    are rejected up-front to keep the summary table well-formed. The sweep must
    contain at least two distinct candidates, otherwise selection is trivial.

    References: §4 model selection (engineering utility)
    """
    sorted_k = sorted(set(_coerce_k_values(k_values)))
    if len(sorted_k) < 2:
        raise ValueError("k_values must contain at least two distinct candidates.")

    observations = _coerce_returns_for_count(returns)
    n_observations = int(observations.shape[0])

    rows: list[ModelSelectionRow] = []
    for k in sorted_k:
        wrapper = GaussianHMMWrapper(
            n_states=k,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            min_variance=min_variance,
        )
        result = wrapper.fit(observations)
        if not result.converged:
            warnings.warn(
                (
                    f"GaussianHMMWrapper.fit did not converge for k={k} "
                    f"with random_state={random_state}."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        n_parameters = count_gaussian_hmm_parameters(k)
        rows.append(
            ModelSelectionRow(
                k=k,
                log_likelihood=result.log_likelihood,
                n_parameters=n_parameters,
                n_observations=n_observations,
                aic=aic(result.log_likelihood, n_parameters),
                bic=bic(result.log_likelihood, n_parameters, n_observations),
                converged=result.converged,
                random_state=random_state,
            )
        )

    aic_scores = np.array([row.aic for row in rows], dtype=float)
    bic_scores = np.array([row.bic for row in rows], dtype=float)
    best_by_aic = rows[int(np.argmin(aic_scores))].k
    best_by_bic = rows[int(np.argmin(bic_scores))].k

    return ModelSelectionResult(
        rows=tuple(rows),
        best_by_aic=best_by_aic,
        best_by_bic=best_by_bic,
    )


def _coerce_k_values(k_values: Iterable[int]) -> list[int]:
    """Validate a candidate-K iterable and return it as a concrete list."""
    materialized = list(k_values)
    if not materialized:
        raise ValueError("k_values must be non-empty.")
    for k in materialized:
        if isinstance(k, (bool, np.bool_)):
            raise TypeError(f"k_values entries must be int, got {type(k).__name__}.")
        if not isinstance(k, (int, np.integer)):
            raise TypeError(f"k_values entries must be int, got {type(k).__name__}.")
        if k < 2:
            raise ValueError(f"k_values entries must be >= 2, got {k}.")
    return [int(k) for k in materialized]


def _coerce_returns_for_count(returns: pd.Series | np.ndarray) -> np.ndarray:
    """Validate returns shape for downstream parameter counting and fitting."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"returns must be one-dimensional, got shape {values.shape}.")
    if values.size < 2:
        raise ValueError("returns must contain at least two observations.")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite values; drop NaN/inf first.")
    return values
