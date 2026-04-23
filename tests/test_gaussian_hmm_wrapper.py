"""Tests for the GaussianHMMWrapper and init_from_plr seeding path."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from hft_hmm.data import load_csv_market_data
from hft_hmm.models.gaussian_hmm import GaussianHMMResult, GaussianHMMWrapper
from hft_hmm.models.plr_baseline import fit_piecewise_linear_regression
from hft_hmm.preprocessing import compute_log_returns

TRACKED_ES_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "es_1min_sample.csv"


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


def test_fit_records_monotone_em_log_likelihood_history():
    returns = _sample_two_regime_returns(seed=10)
    result = GaussianHMMWrapper(n_states=2, random_state=42, n_iter=200).fit(returns)

    assert result.n_iter == result.em_log_likelihood_history.shape[0]
    assert result.em_log_likelihood_history.shape[0] >= 2
    assert result.em_log_likelihood_is_monotone is True


def _tight_two_regime_returns() -> np.ndarray:
    """Returns whose natural EM variance sits far below the floor in the test."""
    rng = np.random.default_rng(0)
    return np.concatenate(
        [
            rng.normal(loc=-1e-5, scale=1e-6, size=500),
            rng.normal(loc=1e-5, scale=1e-6, size=500),
        ]
    )


def test_fit_clamps_variances_and_logs_affected_states(caplog):
    returns = _tight_two_regime_returns()
    min_variance = 1e-4

    with caplog.at_level(logging.WARNING, logger="hft_hmm.models.gaussian_hmm"):
        result = GaussianHMMWrapper(
            n_states=2,
            random_state=0,
            min_variance=min_variance,
        ).fit(returns)

    assert result.min_variance == pytest.approx(min_variance)
    assert np.all(result.variances >= min_variance)

    clamp_records = [r for r in caplog.records if "Clamping fitted variances" in r.getMessage()]
    assert clamp_records, "clamping must emit a logging.warning"
    message = clamp_records[0].getMessage()
    assert f"min_variance={min_variance:g}" in message
    # The payload lists the affected state indices — must be non-empty and parseable.
    bracket_open = message.index("[")
    bracket_close = message.index("]", bracket_open)
    raw_indices = message[bracket_open + 1 : bracket_close].split(",")
    indices = [int(part) for part in raw_indices if part.strip()]
    assert indices, "warning must name at least one affected state index"
    assert all(0 <= idx < 2 for idx in indices)


def test_fit_raises_when_variance_floor_policy_is_raise():
    returns = _tight_two_regime_returns()
    min_variance = 1e-4

    wrapper = GaussianHMMWrapper(
        n_states=2,
        random_state=0,
        min_variance=min_variance,
        variance_floor_policy="raise",
    )
    with pytest.raises(ValueError, match=r"fitted variance.*state index/indices \[\d"):
        wrapper.fit(returns)


def test_em_log_likelihood_is_monotone_on_tracked_es_fixture():
    prices = load_csv_market_data(str(TRACKED_ES_FIXTURE))
    returns = compute_log_returns(prices.set_index("timestamp")["price"]).dropna()

    result = GaussianHMMWrapper(n_states=2, random_state=0, n_iter=200).fit(returns)

    # The history must exist, cover at least two iterations (otherwise
    # "monotone" is vacuous), and never decrease beyond numerical noise.
    history = result.em_log_likelihood_history
    assert history.shape[0] >= 2
    assert result.em_log_likelihood_is_monotone is True
    assert np.all(np.diff(history) >= -np.sqrt(np.finfo(float).eps))


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
    [
        {"n_states": 1},
        {"n_states": 2, "n_iter": 0},
        {"n_states": 2, "tol": 0.0},
        {"n_states": 2, "min_variance": 0.0},
    ],
)
def test_constructor_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        GaussianHMMWrapper(**kwargs)


@pytest.mark.parametrize("min_variance", [float("nan"), float("inf"), float("-inf")])
def test_constructor_rejects_non_finite_min_variance(min_variance: float):
    with pytest.raises(ValueError, match="min_variance"):
        GaussianHMMWrapper(n_states=2, min_variance=min_variance)


def test_constructor_rejects_unknown_variance_floor_policy():
    with pytest.raises(ValueError, match="variance_floor_policy"):
        GaussianHMMWrapper(n_states=2, variance_floor_policy="quiet")  # type: ignore[arg-type]


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
