"""Tests for the GaussianHMMWrapper and init_from_plr seeding path."""

from __future__ import annotations

import numpy as np
import pytest

from hft_hmm.models.gaussian_hmm import GaussianHMMResult, GaussianHMMWrapper
from hft_hmm.models.plr_baseline import fit_piecewise_linear_regression


def _sample_two_regime_returns(
    *,
    low_mean: float = -1.0,
    high_mean: float = 1.0,
    sigma: float = 0.3,
    per_regime: int = 400,
    n_blocks: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic two-regime returns with means separated at several sigma.

    The absolute magnitudes are intentionally large (unit-scale) so that
    hmmlearn's default KMeans-based initialization finds the two modes; the
    assertions only care about regime recovery, not realistic return scales.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for block in range(n_blocks):
        mean = low_mean if block % 2 == 0 else high_mean
        chunks.append(rng.normal(loc=mean, scale=sigma, size=per_regime))
    return np.concatenate(chunks)


def _sample_two_regime_prices(
    *,
    low_slope: float = -0.5,
    high_slope: float = 0.5,
    noise: float = 0.05,
    per_regime: int = 120,
    n_blocks: int = 4,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prices = []
    current = 0.0
    for block in range(n_blocks):
        slope = low_slope if block % 2 == 0 else high_slope
        x_local = np.arange(per_regime, dtype=float)
        noise_local = rng.normal(scale=noise, size=per_regime)
        block_values = current + slope * x_local + noise_local
        prices.append(block_values)
        current = float(block_values[-1])
    return np.concatenate(prices)


def test_wrapper_recovers_two_regime_means():
    returns = _sample_two_regime_returns(seed=1)
    wrapper = GaussianHMMWrapper(n_states=2, random_state=42, n_iter=200)
    result = wrapper.fit(returns)
    low, high = result.means
    assert low < 0 < high
    assert abs(low - -1.0) < 0.1
    assert abs(high - 1.0) < 0.1


def test_result_arrays_have_correct_shapes_and_normalization():
    returns = _sample_two_regime_returns(seed=2)
    wrapper = GaussianHMMWrapper(n_states=2, random_state=42)
    result = wrapper.fit(returns)

    assert isinstance(result, GaussianHMMResult)
    assert result.means.shape == (2,)
    assert result.variances.shape == (2,)
    assert result.transition_matrix.shape == (2, 2)
    assert result.initial_distribution.shape == (2,)
    assert np.allclose(result.transition_matrix.sum(axis=1), 1.0)
    assert np.isclose(result.initial_distribution.sum(), 1.0)
    assert np.all(result.variances > 0.0)


def test_result_state_grid_means_ascending():
    returns = _sample_two_regime_returns(seed=3)
    wrapper = GaussianHMMWrapper(n_states=2, random_state=42)
    result = wrapper.fit(returns)
    assert np.all(np.diff(result.state_grid.means) > 0)
    assert np.array_equal(result.state_grid.means, result.means)


def test_predict_proba_rows_sum_to_one():
    returns = _sample_two_regime_returns(seed=4)
    wrapper = GaussianHMMWrapper(n_states=2, random_state=42)
    wrapper.fit(returns)
    posteriors = wrapper.predict_proba(returns)
    assert posteriors.shape == (returns.shape[0], 2)
    assert np.allclose(posteriors.sum(axis=1), 1.0)


def test_predict_returns_valid_state_indices():
    returns = _sample_two_regime_returns(seed=5)
    wrapper = GaussianHMMWrapper(n_states=2, random_state=42)
    wrapper.fit(returns)
    states = wrapper.predict(returns)
    assert states.shape == (returns.shape[0],)
    assert states.min() >= 0
    assert states.max() < 2


def test_fit_is_deterministic_under_fixed_random_state():
    returns = _sample_two_regime_returns(seed=6)
    first = GaussianHMMWrapper(n_states=2, random_state=7).fit(returns)
    second = GaussianHMMWrapper(n_states=2, random_state=7).fit(returns)
    np.testing.assert_allclose(first.means, second.means)
    np.testing.assert_allclose(first.variances, second.variances)
    np.testing.assert_allclose(first.transition_matrix, second.transition_matrix)
    np.testing.assert_allclose(first.initial_distribution, second.initial_distribution)


def test_init_from_plr_produces_reproducible_fit():
    prices = _sample_two_regime_prices(seed=11)
    plr = fit_piecewise_linear_regression(prices, n_segments=2)
    returns = np.diff(prices)
    first = GaussianHMMWrapper(n_states=2).fit(returns, init_from_plr=plr)
    second = GaussianHMMWrapper(n_states=2).fit(returns, init_from_plr=plr)
    np.testing.assert_allclose(first.means, second.means)
    np.testing.assert_allclose(first.variances, second.variances)
    np.testing.assert_allclose(first.transition_matrix, second.transition_matrix)
    np.testing.assert_allclose(first.initial_distribution, second.initial_distribution)


def test_init_from_plr_rejects_mismatched_k():
    prices = _sample_two_regime_prices(seed=12)
    plr = fit_piecewise_linear_regression(prices, n_segments=2)
    wrapper = GaussianHMMWrapper(n_states=3)
    with pytest.raises(ValueError, match="does not match wrapper n_states"):
        wrapper.fit(np.diff(prices), init_from_plr=plr)


def test_predict_before_fit_raises():
    wrapper = GaussianHMMWrapper(n_states=2)
    with pytest.raises(RuntimeError, match="Call fit"):
        wrapper.predict(np.zeros(5))


def test_predict_proba_before_fit_raises():
    wrapper = GaussianHMMWrapper(n_states=2)
    with pytest.raises(RuntimeError, match="Call fit"):
        wrapper.predict_proba(np.zeros(5))


def test_fit_rejects_nan_input():
    wrapper = GaussianHMMWrapper(n_states=2, random_state=0)
    returns = np.array([0.0, 0.1, np.nan, 0.2])
    with pytest.raises(ValueError, match="finite"):
        wrapper.fit(returns)


def test_fit_rejects_multidim_input():
    wrapper = GaussianHMMWrapper(n_states=2, random_state=0)
    returns = np.zeros((5, 2))
    with pytest.raises(ValueError, match="one-dimensional"):
        wrapper.fit(returns)


def test_fit_rejects_empty_input():
    wrapper = GaussianHMMWrapper(n_states=2, random_state=0)
    with pytest.raises(ValueError, match="at least one observation"):
        wrapper.fit(np.array([], dtype=float))


@pytest.mark.parametrize(
    "kwargs",
    [{"n_states": 1}, {"n_states": 2, "n_iter": 0}, {"n_states": 2, "tol": 0.0}],
)
def test_constructor_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        GaussianHMMWrapper(**kwargs)


def test_result_is_frozen_and_arrays_read_only():
    returns = _sample_two_regime_returns(seed=9)
    wrapper = GaussianHMMWrapper(n_states=2, random_state=42)
    result = wrapper.fit(returns)

    with pytest.raises(AttributeError):
        result.log_likelihood = 0.0  # type: ignore[misc]

    with pytest.raises(ValueError):
        result.means[0] = 0.0
    with pytest.raises(ValueError):
        result.transition_matrix[0, 0] = 0.0
