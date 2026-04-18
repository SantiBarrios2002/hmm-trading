"""Tests for the model-selection sweep and information criteria."""

from __future__ import annotations

import math
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from hft_hmm.selection.model_selection import (
    ModelSelectionResult,
    ModelSelectionRow,
    aic,
    bic,
    compare_state_counts,
    count_gaussian_hmm_parameters,
)
from hft_hmm.selection.plots import plot_selection_curves


def _two_regime_returns(*, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    for block in range(4):
        mean = -1.0 if block % 2 == 0 else 1.0
        chunks.append(rng.normal(loc=mean, scale=0.3, size=400))
    return np.concatenate(chunks)


@pytest.mark.parametrize(
    "n_states,expected",
    [
        (2, 7),  # 4 + 4 - 1
        (3, 14),  # 9 + 6 - 1
        (5, 34),  # 25 + 10 - 1
    ],
)
def test_count_gaussian_hmm_parameters_formula(n_states, expected):
    assert count_gaussian_hmm_parameters(n_states) == expected


def test_count_gaussian_hmm_parameters_rejects_small_k():
    with pytest.raises(ValueError, match="at least 2"):
        count_gaussian_hmm_parameters(1)


def test_aic_formula():
    assert aic(log_likelihood=-100.0, n_parameters=7) == pytest.approx(2 * 7 - 2 * -100.0)
    assert aic(log_likelihood=0.0, n_parameters=3) == pytest.approx(6.0)


def test_bic_formula():
    expected = 7 * math.log(1000) - 2 * -100.0
    computed = bic(log_likelihood=-100.0, n_parameters=7, n_observations=1000)
    assert computed == pytest.approx(expected)


def test_aic_rejects_non_positive_parameters():
    with pytest.raises(ValueError, match="positive"):
        aic(log_likelihood=-1.0, n_parameters=0)


def test_bic_rejects_non_positive_observations():
    with pytest.raises(ValueError, match="positive"):
        bic(log_likelihood=-1.0, n_parameters=7, n_observations=0)


def test_compare_state_counts_recovers_two_regimes():
    returns = _two_regime_returns(seed=1)
    result = compare_state_counts(
        returns,
        k_values=[2, 3, 4, 5],
        random_state=42,
        n_iter=200,
    )

    assert isinstance(result, ModelSelectionResult)
    assert [row.k for row in result.rows] == [2, 3, 4, 5]
    assert result.best_by_bic == 2


def test_compare_state_counts_rows_are_model_selection_row():
    returns = _two_regime_returns(seed=2)
    result = compare_state_counts(returns, k_values=[2, 3], random_state=42)
    for row in result.rows:
        assert isinstance(row, ModelSelectionRow)
        assert row.n_parameters == count_gaussian_hmm_parameters(row.k)
        assert row.n_observations == returns.shape[0]
        assert row.aic == pytest.approx(aic(row.log_likelihood, row.n_parameters))
        assert row.bic == pytest.approx(
            bic(row.log_likelihood, row.n_parameters, row.n_observations)
        )


def test_compare_state_counts_deterministic_under_fixed_seed():
    returns = _two_regime_returns(seed=3)
    first = compare_state_counts(returns, k_values=[2, 3], random_state=7)
    second = compare_state_counts(returns, k_values=[2, 3], random_state=7)
    for row_a, row_b in zip(first.rows, second.rows, strict=True):
        assert row_a.log_likelihood == pytest.approx(row_b.log_likelihood)
        assert row_a.aic == pytest.approx(row_b.aic)
        assert row_a.bic == pytest.approx(row_b.bic)


def test_compare_state_counts_deduplicates_and_sorts_k_values():
    returns = _two_regime_returns(seed=4)
    result = compare_state_counts(returns, k_values=[3, 2, 3], random_state=42)
    assert [row.k for row in result.rows] == [2, 3]


def test_compare_state_counts_rejects_empty_k_values():
    with pytest.raises(ValueError, match="non-empty"):
        compare_state_counts(_two_regime_returns(seed=5), k_values=[])


def test_compare_state_counts_rejects_singleton_k_values():
    with pytest.raises(ValueError, match="at least two"):
        compare_state_counts(_two_regime_returns(seed=6), k_values=[2])


def test_compare_state_counts_rejects_k_below_two():
    with pytest.raises(ValueError, match=">= 2"):
        compare_state_counts(_two_regime_returns(seed=7), k_values=[1, 2])


def test_compare_state_counts_rejects_non_integer_k():
    with pytest.raises(TypeError, match="int"):
        compare_state_counts(_two_regime_returns(seed=8), k_values=[2.5, 3])


@pytest.mark.parametrize("bool_k", [True, False, np.bool_(True), np.bool_(False)])
def test_compare_state_counts_rejects_boolean_k(bool_k):
    with pytest.raises(TypeError, match="int"):
        compare_state_counts(_two_regime_returns(seed=8), k_values=[bool_k, 3])


def test_compare_state_counts_rejects_multidim_returns():
    returns = np.zeros((100, 2))
    with pytest.raises(ValueError, match="one-dimensional"):
        compare_state_counts(returns, k_values=[2, 3])


def test_compare_state_counts_rejects_nan_returns():
    returns = _two_regime_returns(seed=9)
    returns[5] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compare_state_counts(returns, k_values=[2, 3])


def test_model_selection_result_rejects_unsorted_rows():
    row_a = _make_row(k=3)
    row_b = _make_row(k=2)
    with pytest.raises(ValueError, match="sorted"):
        ModelSelectionResult(rows=(row_a, row_b), best_by_aic=2, best_by_bic=2)


def test_model_selection_result_rejects_duplicate_k():
    row_a = _make_row(k=2)
    row_b = _make_row(k=2)
    with pytest.raises(ValueError, match="duplicate"):
        ModelSelectionResult(rows=(row_a, row_b), best_by_aic=2, best_by_bic=2)


def test_model_selection_result_rejects_unknown_best_k():
    row_a = _make_row(k=2)
    with pytest.raises(ValueError, match="best_by_aic"):
        ModelSelectionResult(rows=(row_a,), best_by_aic=5, best_by_bic=2)


def test_model_selection_result_any_non_converged_property():
    converged_rows = (
        _make_row(k=2, converged=True),
        _make_row(k=3, converged=True),
    )
    all_converged = ModelSelectionResult(rows=converged_rows, best_by_aic=2, best_by_bic=2)
    assert all_converged.any_non_converged is False

    mixed_rows = (
        _make_row(k=2, converged=True),
        _make_row(k=3, converged=False),
    )
    mixed = ModelSelectionResult(rows=mixed_rows, best_by_aic=2, best_by_bic=2)
    assert mixed.any_non_converged is True


def test_compare_state_counts_warns_for_non_converged_fits(monkeypatch):
    class _FakeWrapper:
        def __init__(self, n_states: int, *, random_state: int | None, n_iter: int, tol: float) -> None:
            self.n_states = n_states
            self.random_state = random_state

        def fit(self, returns):
            return SimpleNamespace(
                log_likelihood=-100.0 - float(self.n_states),
                converged=self.n_states != 3,
            )

    monkeypatch.setattr("hft_hmm.selection.model_selection.GaussianHMMWrapper", _FakeWrapper)

    with pytest.warns(RuntimeWarning, match=r"did not converge for k=3 with random_state=42"):
        result = compare_state_counts(_two_regime_returns(seed=11), k_values=[2, 3], random_state=42)

    assert [row.converged for row in result.rows] == [True, False]
    assert result.any_non_converged is True


def test_plot_selection_curves_smoke():
    import matplotlib.pyplot as plt

    returns = _two_regime_returns(seed=10)
    result = compare_state_counts(returns, k_values=[2, 3], random_state=42)

    fig, ax = plt.subplots()
    try:
        returned_ax = plot_selection_curves(result, ax=ax)
        assert returned_ax is ax
        line_labels = [line.get_label() for line in ax.get_lines()]
        assert "AIC" in line_labels
        assert "BIC" in line_labels
    finally:
        plt.close(fig)


def _make_row(*, k: int, converged: bool = True) -> ModelSelectionRow:
    n_parameters = count_gaussian_hmm_parameters(k)
    return ModelSelectionRow(
        k=k,
        log_likelihood=-100.0,
        n_parameters=n_parameters,
        n_observations=1000,
        aic=aic(-100.0, n_parameters),
        bic=bic(-100.0, n_parameters, 1000),
        converged=converged,
        random_state=None,
    )
