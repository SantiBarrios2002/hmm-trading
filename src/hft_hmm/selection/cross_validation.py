"""Forward-chaining cross-validation over candidate hidden-state counts ``K``.

This module implements the cross-validation route from the paper's §4 model
selection discussion as an engineering approximation. It deliberately does not
use random k-fold splits: the input sequence is divided into ``n_folds + 1``
contiguous blocks in temporal order, then fold ``i`` trains on the expanding
prefix ``[0, boundary[i + 1])`` and scores the immediately-following held-out
block ``[boundary[i + 1], boundary[i + 2])``. The fold count and split policy
are explicit and deterministic, so identical inputs plus ``random_state``
produce bit-identical choices.

Scoring is per-bar held-out predictive log-likelihood. For each fold and
candidate ``K``, a :class:`~hft_hmm.models.gaussian_hmm.GaussianHMMWrapper` is
fitted on the training prefix, then
``forward_filter(held_out, fitted).log_likelihood / len(held_out)`` is used as
the fold score. The held-out block is scored from the fitted model's initial
distribution rather than from the posterior at the training boundary; this is
documented as an engineering approximation because the paper does not fully
specify its CV protocol. An expected-return-aligned directional score was
considered, but predictive log-likelihood is the narrower §4 model-selection
metric and avoids entangling K selection with the trading signal policy.

See ``CROSS_VALIDATION_REFERENCE`` for the paper pointer used in docstrings and
review notes.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from hft_hmm.core import ENGINEERING_APPROXIMATION, PaperReference, reference
from hft_hmm.inference import forward_filter
from hft_hmm.models.gaussian_hmm import (
    _VARIANCE_FLOOR_POLICIES,
    GaussianHMMWrapper,
    VarianceFloorPolicy,
)

__category__: Final[str] = ENGINEERING_APPROXIMATION
CROSS_VALIDATION_REFERENCE: Final[PaperReference] = reference(
    "§4", "cross-validation K selection"
)
_SPLIT_POLICY: Final[str] = "expanding_forward_chaining"


@dataclass(frozen=True)
class CrossValidationFoldScore:
    """Single held-out fold score for one candidate ``K``."""

    k: int
    fold_index: int
    train_start_index: int
    train_end_index: int
    heldout_start_index: int
    heldout_end_index: int
    heldout_log_likelihood: float
    per_bar_log_likelihood: float
    converged: bool
    random_state: int | None

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError(f"k must be at least 2, got {self.k}.")
        if self.fold_index < 0:
            raise ValueError(f"fold_index must be non-negative, got {self.fold_index}.")
        if self.train_start_index < 0:
            raise ValueError(
                f"train_start_index must be non-negative, got {self.train_start_index}."
            )
        if self.train_end_index <= self.train_start_index:
            raise ValueError("train_end_index must be greater than train_start_index.")
        if self.heldout_start_index != self.train_end_index:
            raise ValueError(
                "heldout_start_index must equal train_end_index for forward chaining."
            )
        if self.heldout_end_index <= self.heldout_start_index:
            raise ValueError("heldout_end_index must be greater than heldout_start_index.")
        if not np.isfinite(self.heldout_log_likelihood):
            raise ValueError("heldout_log_likelihood must be finite.")
        if not np.isfinite(self.per_bar_log_likelihood):
            raise ValueError("per_bar_log_likelihood must be finite.")


@dataclass(frozen=True)
class CrossValidationRow:
    """Per-``K`` summary of forward-chaining held-out scores."""

    k: int
    fold_scores: tuple[CrossValidationFoldScore, ...]
    mean_heldout_log_likelihood: float

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError(f"k must be at least 2, got {self.k}.")
        if not self.fold_scores:
            raise ValueError("fold_scores must contain at least one entry.")
        fold_indices = [score.fold_index for score in self.fold_scores]
        if fold_indices != sorted(fold_indices):
            raise ValueError("fold_scores must be sorted by fold_index ascending.")
        if len(set(fold_indices)) != len(fold_indices):
            raise ValueError("fold_scores must not contain duplicate fold indices.")
        if any(score.k != self.k for score in self.fold_scores):
            raise ValueError("fold_scores entries must all match row k.")
        if not np.isfinite(self.mean_heldout_log_likelihood):
            raise ValueError("mean_heldout_log_likelihood must be finite.")
        expected = float(np.mean([score.per_bar_log_likelihood for score in self.fold_scores]))
        if not np.isclose(self.mean_heldout_log_likelihood, expected):
            raise ValueError(
                "mean_heldout_log_likelihood must equal the mean fold per-bar score."
            )


@dataclass(frozen=True)
class CrossValidationResult:
    """Immutable summary of a forward-chaining CV sweep sorted by ``k`` ascending."""

    rows: tuple[CrossValidationRow, ...]
    chosen_k: int
    n_folds: int
    split_policy: str

    @property
    def best_k(self) -> int:
        """Alias for ``chosen_k`` for callers using model-selection vocabulary."""
        return self.chosen_k

    @property
    def mean_scores_by_k(self) -> dict[int, float]:
        """Return mean per-bar held-out log-likelihood keyed by candidate ``K``."""
        return {row.k: row.mean_heldout_log_likelihood for row in self.rows}

    @property
    def any_non_converged(self) -> bool:
        """Return whether any fitted fold candidate failed to converge."""
        return any(not score.converged for row in self.rows for score in row.fold_scores)

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("rows must contain at least one entry.")
        if [row.k for row in self.rows] != sorted(row.k for row in self.rows):
            raise ValueError("rows must be sorted by k ascending.")
        if len({row.k for row in self.rows}) != len(self.rows):
            raise ValueError("rows must not contain duplicate k values.")
        k_values = {row.k for row in self.rows}
        if self.chosen_k not in k_values:
            raise ValueError(f"chosen_k={self.chosen_k} missing from rows.")
        if self.n_folds < 1:
            raise ValueError(f"n_folds must be positive, got {self.n_folds}.")
        if self.split_policy != _SPLIT_POLICY:
            raise ValueError(f"split_policy must be {_SPLIT_POLICY!r}, got {self.split_policy!r}.")
        for row in self.rows:
            if len(row.fold_scores) != self.n_folds:
                raise ValueError(
                    f"row k={row.k} has {len(row.fold_scores)} folds, expected {self.n_folds}."
                )


@dataclass(frozen=True)
class _ForwardChainingSplit:
    train_start_index: int
    train_end_index: int
    heldout_start_index: int
    heldout_end_index: int


def select_k_by_cv(
    returns: pd.Series | np.ndarray,
    k_values: Iterable[int],
    *,
    n_folds: int,
    random_state: int | None,
    n_iter: int,
    tol: float,
    min_variance: float,
    variance_floor_policy: VarianceFloorPolicy,
) -> CrossValidationResult:
    """Select ``K`` by forward-chaining held-out predictive log-likelihood.

    Candidate ``K`` values are fitted independently on each expanding training
    prefix and scored on the immediately-following contiguous held-out block.
    The selected ``K`` maximizes mean per-bar held-out log-likelihood; ties are
    resolved by the sorted candidate order, so the smallest tied ``K`` wins.

    References: §4 model selection via cross-validation (engineering utility)
    """
    sorted_k = sorted(set(_coerce_k_values(k_values)))
    if len(sorted_k) < 2:
        raise ValueError("k_values must contain at least two distinct candidates.")

    normalized_random_state = _coerce_random_state(random_state)
    _validate_fit_hyperparameters(
        n_folds=n_folds,
        n_iter=n_iter,
        tol=tol,
        min_variance=min_variance,
        variance_floor_policy=variance_floor_policy,
    )
    observations = _coerce_returns_for_cv(returns)
    splits = _forward_chaining_splits(
        n_observations=int(observations.shape[0]),
        n_folds=int(n_folds),
        min_train_size=max(sorted_k),
    )

    rows: list[CrossValidationRow] = []
    for k in sorted_k:
        fold_scores: list[CrossValidationFoldScore] = []
        for fold_index, split in enumerate(splits):
            train = observations[split.train_start_index : split.train_end_index]
            heldout = observations[split.heldout_start_index : split.heldout_end_index]
            wrapper = GaussianHMMWrapper(
                n_states=k,
                random_state=normalized_random_state,
                n_iter=n_iter,
                tol=tol,
                min_variance=min_variance,
                variance_floor_policy=variance_floor_policy,
            )
            fitted = wrapper.fit(train)
            if not fitted.converged:
                warnings.warn(
                    (
                        f"GaussianHMMWrapper.fit did not converge for k={k} "
                        f"fold={fold_index} with random_state={normalized_random_state}."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
            heldout_log_likelihood = forward_filter(heldout, fitted).log_likelihood
            per_bar = heldout_log_likelihood / float(heldout.shape[0])
            fold_scores.append(
                CrossValidationFoldScore(
                    k=k,
                    fold_index=fold_index,
                    train_start_index=split.train_start_index,
                    train_end_index=split.train_end_index,
                    heldout_start_index=split.heldout_start_index,
                    heldout_end_index=split.heldout_end_index,
                    heldout_log_likelihood=heldout_log_likelihood,
                    per_bar_log_likelihood=per_bar,
                    converged=fitted.converged,
                    random_state=normalized_random_state,
                )
            )
        rows.append(
            CrossValidationRow(
                k=k,
                fold_scores=tuple(fold_scores),
                mean_heldout_log_likelihood=float(
                    np.mean([score.per_bar_log_likelihood for score in fold_scores])
                ),
            )
        )

    mean_scores = np.array([row.mean_heldout_log_likelihood for row in rows], dtype=float)
    chosen_k = rows[int(np.argmax(mean_scores))].k

    return CrossValidationResult(
        rows=tuple(rows),
        chosen_k=chosen_k,
        n_folds=int(n_folds),
        split_policy=_SPLIT_POLICY,
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


def _coerce_random_state(random_state: int | None) -> int | None:
    if random_state is None:
        return None
    if isinstance(random_state, (bool, np.bool_)):
        raise TypeError(f"random_state must be int or None, got {type(random_state).__name__}.")
    if not isinstance(random_state, (int, np.integer)):
        raise TypeError(f"random_state must be int or None, got {type(random_state).__name__}.")
    return int(random_state)


def _validate_fit_hyperparameters(
    *,
    n_folds: int,
    n_iter: int,
    tol: float,
    min_variance: float,
    variance_floor_policy: VarianceFloorPolicy,
) -> None:
    if isinstance(n_folds, (bool, np.bool_)) or not isinstance(n_folds, (int, np.integer)):
        raise TypeError(f"n_folds must be int, got {type(n_folds).__name__}.")
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}.")
    if isinstance(n_iter, (bool, np.bool_)) or not isinstance(n_iter, (int, np.integer)):
        raise TypeError(f"n_iter must be int, got {type(n_iter).__name__}.")
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}.")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be a finite strictly positive float, got {tol!r}.")
    if not np.isfinite(min_variance) or min_variance <= 0.0:
        raise ValueError(
            "min_variance must be a finite strictly positive float, "
            f"got {min_variance!r}."
        )
    if variance_floor_policy not in _VARIANCE_FLOOR_POLICIES:
        raise ValueError(
            "variance_floor_policy must be one of "
            f"{sorted(_VARIANCE_FLOOR_POLICIES)}, got {variance_floor_policy!r}."
        )


def _coerce_returns_for_cv(returns: pd.Series | np.ndarray) -> np.ndarray:
    """Validate returns shape for temporal CV splitting and fitting."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"returns must be one-dimensional, got shape {values.shape}.")
    if values.size < 2:
        raise ValueError("returns must contain at least two observations.")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite values; drop NaN/inf first.")
    return values


def _forward_chaining_splits(
    *,
    n_observations: int,
    n_folds: int,
    min_train_size: int,
) -> tuple[_ForwardChainingSplit, ...]:
    if n_observations < n_folds + 1:
        raise ValueError(
            "returns must contain at least n_folds + 1 observations for non-empty "
            f"forward-chaining blocks; got {n_observations} observations and "
            f"n_folds={n_folds}."
        )

    n_blocks = n_folds + 1
    block_sizes = np.full(n_blocks, n_observations // n_blocks, dtype=int)
    block_sizes[: n_observations % n_blocks] += 1
    if int(block_sizes[0]) < min_train_size:
        raise ValueError(
            "first training fold must contain at least max(k_values) observations; "
            f"got {int(block_sizes[0])} observations and max(k_values)={min_train_size}."
        )

    boundaries = np.concatenate(([0], np.cumsum(block_sizes)))
    splits = []
    for fold_index in range(n_folds):
        train_start = 0
        train_end = int(boundaries[fold_index + 1])
        heldout_start = train_end
        heldout_end = int(boundaries[fold_index + 2])
        splits.append(
            _ForwardChainingSplit(
                train_start_index=train_start,
                train_end_index=train_end,
                heldout_start_index=heldout_start,
                heldout_end_index=heldout_end,
            )
        )
    return tuple(splits)
