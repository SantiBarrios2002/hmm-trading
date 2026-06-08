"""Tests for forward-chaining cross-validation K selection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hft_hmm.core import ENGINEERING_APPROXIMATION, module_category
from hft_hmm.selection import (
    CROSS_VALIDATION_REFERENCE,
    CrossValidationResult,
    select_k_by_cv,
)


def _two_regime_returns(*, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    for block in range(8):
        mean = -1.0 if block % 2 == 0 else 1.0
        chunks.append(rng.normal(loc=mean, scale=0.2, size=120))
    return np.concatenate(chunks)


def test_cross_validation_module_declares_engineering_category() -> None:
    import hft_hmm.selection.cross_validation as module

    assert module_category(module) == ENGINEERING_APPROXIMATION
    assert CROSS_VALIDATION_REFERENCE.section == "§4"


def test_select_k_by_cv_recovers_two_regimes() -> None:
    result = select_k_by_cv(
        _two_regime_returns(seed=3),
        k_values=[2, 3, 4],
        n_folds=4,
        random_state=42,
        n_iter=100,
        tol=1e-4,
        min_variance=1e-8,
        variance_floor_policy="clamp",
    )

    assert isinstance(result, CrossValidationResult)
    assert result.chosen_k == 2
    assert result.best_k == 2
    assert [row.k for row in result.rows] == [2, 3, 4]
    assert result.mean_scores_by_k[2] > result.mean_scores_by_k[3]
    assert result.mean_scores_by_k[2] > result.mean_scores_by_k[4]


def test_select_k_by_cv_is_deterministic_under_fixed_seed() -> None:
    returns = _two_regime_returns(seed=4)
    kwargs = dict(
        k_values=[2, 3, 4],
        n_folds=4,
        random_state=42,
        n_iter=100,
        tol=1e-4,
        min_variance=1e-8,
        variance_floor_policy="clamp",
    )

    first = select_k_by_cv(returns, **kwargs)
    second = select_k_by_cv(returns, **kwargs)

    assert first.chosen_k == second.chosen_k
    assert first.n_folds == second.n_folds
    assert first.split_policy == second.split_policy
    assert first.mean_scores_by_k == pytest.approx(second.mean_scores_by_k)
    for first_row, second_row in zip(first.rows, second.rows, strict=True):
        assert first_row.k == second_row.k
        for first_score, second_score in zip(
            first_row.fold_scores,
            second_row.fold_scores,
            strict=True,
        ):
            assert first_score.train_start_index == second_score.train_start_index
            assert first_score.train_end_index == second_score.train_end_index
            assert first_score.heldout_start_index == second_score.heldout_start_index
            assert first_score.heldout_end_index == second_score.heldout_end_index
            assert first_score.per_bar_log_likelihood == pytest.approx(
                second_score.per_bar_log_likelihood
            )


def test_select_k_by_cv_uses_expanding_forward_chaining_splits(monkeypatch) -> None:
    class _FakeWrapper:
        def __init__(
            self,
            n_states: int,
            *,
            random_state: int | None,
            n_iter: int,
            tol: float,
            min_variance: float,
            variance_floor_policy: str,
        ) -> None:
            self.n_states = n_states

        def fit(self, returns):
            return SimpleNamespace(k=self.n_states, converged=True)

    def fake_forward_filter(heldout, fitted):
        return SimpleNamespace(log_likelihood=float(fitted.k * len(heldout)))

    monkeypatch.setattr("hft_hmm.selection.cross_validation.GaussianHMMWrapper", _FakeWrapper)
    monkeypatch.setattr("hft_hmm.selection.cross_validation.forward_filter", fake_forward_filter)

    result = select_k_by_cv(
        np.arange(23, dtype=float),
        k_values=[2, 3],
        n_folds=4,
        random_state=0,
        n_iter=10,
        tol=1e-4,
        min_variance=1e-8,
        variance_floor_policy="clamp",
    )

    expected_boundaries = [(0, 5, 5, 10), (0, 10, 10, 15), (0, 15, 15, 19), (0, 19, 19, 23)]
    for row in result.rows:
        boundaries = [
            (
                score.train_start_index,
                score.train_end_index,
                score.heldout_start_index,
                score.heldout_end_index,
            )
            for score in row.fold_scores
        ]
        assert boundaries == expected_boundaries
        assert all(score.train_end_index == score.heldout_start_index for score in row.fold_scores)
        assert all(score.train_end_index <= score.heldout_start_index for score in row.fold_scores)


def test_select_k_by_cv_rejects_invalid_inputs() -> None:
    returns = _two_regime_returns(seed=5)
    base = dict(
        returns=returns,
        k_values=[2, 3],
        n_folds=4,
        random_state=42,
        n_iter=10,
        tol=1e-4,
        min_variance=1e-8,
        variance_floor_policy="clamp",
    )

    with pytest.raises(ValueError, match="non-empty"):
        select_k_by_cv(**{**base, "k_values": []})
    with pytest.raises(ValueError, match="at least two"):
        select_k_by_cv(**{**base, "k_values": [2]})
    with pytest.raises(ValueError, match=">= 2"):
        select_k_by_cv(**{**base, "k_values": [1, 2]})
    with pytest.raises(TypeError, match="int"):
        select_k_by_cv(**{**base, "k_values": [2.0, 3]})
    with pytest.raises(ValueError, match="one-dimensional"):
        select_k_by_cv(**{**base, "returns": np.zeros((10, 2))})
    with pytest.raises(ValueError, match="n_folds"):
        select_k_by_cv(**{**base, "n_folds": 0})
    with pytest.raises(TypeError, match="random_state"):
        select_k_by_cv(**{**base, "random_state": True})
    with pytest.raises(ValueError, match="first training fold"):
        select_k_by_cv(**{**base, "returns": np.arange(8, dtype=float), "n_folds": 4})
