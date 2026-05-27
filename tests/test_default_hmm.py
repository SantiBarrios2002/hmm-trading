"""Tests for the Default HMM model (Gate N)."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from hft_hmm.core import PAPER_FAITHFUL, module_category
from hft_hmm.models.default_hmm import fit_default_hmm
from hft_hmm.models.plr_baseline import fit_piecewise_linear_regression

default_hmm_module = importlib.import_module("hft_hmm.models.default_hmm")


def _make_returns(n: int = 200, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1e-3, size=n)


# ---------------------------------------------------------------------------
# Module taxonomy
# ---------------------------------------------------------------------------


def test_default_hmm_module_is_paper_faithful() -> None:
    assert module_category(default_hmm_module) == PAPER_FAITHFUL


# ---------------------------------------------------------------------------
# Uniform-A behavior
# ---------------------------------------------------------------------------


def test_transition_matrix_is_uniform_k2() -> None:
    result = fit_default_hmm(_make_returns(), n_states=2)
    expected = np.full((2, 2), 0.5)
    np.testing.assert_allclose(result.transition_matrix, expected)


def test_transition_matrix_is_uniform_k3() -> None:
    result = fit_default_hmm(_make_returns(), n_states=3)
    expected = np.full((3, 3), 1.0 / 3.0)
    np.testing.assert_allclose(result.transition_matrix, expected, atol=1e-12)


def test_initial_distribution_is_uniform_k2() -> None:
    result = fit_default_hmm(_make_returns(), n_states=2)
    expected = np.full(2, 0.5)
    np.testing.assert_allclose(result.initial_distribution, expected)


def test_initial_distribution_is_uniform_k3() -> None:
    result = fit_default_hmm(_make_returns(), n_states=3)
    expected = np.full(3, 1.0 / 3.0)
    np.testing.assert_allclose(result.initial_distribution, expected, atol=1e-12)


def test_transition_rows_sum_to_one() -> None:
    result = fit_default_hmm(_make_returns(), n_states=4)
    np.testing.assert_allclose(result.transition_matrix.sum(axis=1), np.ones(4))


# ---------------------------------------------------------------------------
# PLR-seeded mean propagation
# ---------------------------------------------------------------------------


def test_means_match_plr_state_means() -> None:
    returns = _make_returns()
    plr = fit_piecewise_linear_regression(returns, n_segments=2)
    result = fit_default_hmm(returns, n_states=2)
    np.testing.assert_allclose(result.means, plr.state_means)


def test_variances_match_plr_state_variances_when_above_floor() -> None:
    # Use a low floor so PLR variances pass through unclamped.
    returns = _make_returns()
    plr = fit_piecewise_linear_regression(returns, n_segments=2)
    result = fit_default_hmm(returns, n_states=2, min_variance=1e-30)
    np.testing.assert_allclose(result.variances, plr.state_variances)


def test_means_are_sorted_ascending() -> None:
    result = fit_default_hmm(_make_returns(), n_states=3)
    assert np.all(np.diff(result.means) >= 0.0)


def test_variances_at_least_min_variance() -> None:
    result = fit_default_hmm(_make_returns(), n_states=2, min_variance=1e-6)
    assert np.all(result.variances >= 1e-6)


def test_min_variance_floor_applied_to_zero_residuals() -> None:
    # A perfectly linear series has zero PLR residuals; floor must kick in.
    returns = np.linspace(0.0, 1.0, 100)
    result = fit_default_hmm(returns, n_states=2, min_variance=1e-5)
    assert np.all(result.variances >= 1e-5)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_fit_is_deterministic_for_same_input() -> None:
    returns = _make_returns()
    r1 = fit_default_hmm(returns, n_states=2)
    r2 = fit_default_hmm(returns, n_states=2)
    np.testing.assert_array_equal(r1.means, r2.means)
    np.testing.assert_array_equal(r1.transition_matrix, r2.transition_matrix)
    np.testing.assert_array_equal(r1.variances, r2.variances)


def test_fit_differs_for_different_inputs() -> None:
    r1 = fit_default_hmm(_make_returns(seed=1), n_states=2)
    r2 = fit_default_hmm(_make_returns(seed=2), n_states=2)
    assert not np.allclose(r1.means, r2.means)


# ---------------------------------------------------------------------------
# Result validity
# ---------------------------------------------------------------------------


def test_result_shape_k2() -> None:
    result = fit_default_hmm(_make_returns(), n_states=2)
    assert result.means.shape == (2,)
    assert result.variances.shape == (2,)
    assert result.transition_matrix.shape == (2, 2)
    assert result.initial_distribution.shape == (2,)


def test_log_likelihood_is_finite() -> None:
    result = fit_default_hmm(_make_returns(), n_states=2)
    assert np.isfinite(result.log_likelihood)


def test_accepts_pandas_series() -> None:
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=200, freq="1min", tz="UTC")
    returns = pd.Series(rng.normal(0.0, 1e-3, size=200), index=index)
    result = fit_default_hmm(returns, n_states=2)
    assert result.n_observations == 200


def test_n_observations_matches_input_length() -> None:
    returns = _make_returns(n=150)
    result = fit_default_hmm(returns, n_states=2)
    assert result.n_observations == 150


def test_converged_is_true() -> None:
    result = fit_default_hmm(_make_returns(), n_states=2)
    assert result.converged is True


def test_n_iter_is_zero() -> None:
    result = fit_default_hmm(_make_returns(), n_states=2)
    assert result.n_iter == 0
