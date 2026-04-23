"""Tests for the volatility-ratio side-information predictor."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from hft_hmm.core.references import PAPER_FAITHFUL, module_category
from hft_hmm.features.volatility_ratio import (
    DEFAULT_DECAY,
    DEFAULT_FAST_WINDOW,
    DEFAULT_SLOW_WINDOW,
    VOLATILITY_RATIO_REFERENCE,
    VolatilityRatioConfig,
    ewma_volatility,
    volatility_ratio,
)

vr_module = importlib.import_module("hft_hmm.features.volatility_ratio")


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="UTC")


def _series(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=_index(values.shape[0]), name="ret")


# --- module taxonomy / reference wiring ---


def test_module_is_paper_faithful() -> None:
    assert module_category(vr_module) == PAPER_FAITHFUL


def test_paper_reference_points_to_section_4_2() -> None:
    assert VOLATILITY_RATIO_REFERENCE.section == "§4.2"
    assert "volatility" in VOLATILITY_RATIO_REFERENCE.topic.lower()


# --- ewma_volatility ---


def test_ewma_volatility_matches_closed_form_on_constant_returns() -> None:
    # Δy² ≡ c ⇒ σ² = (1 - λ) · c · Σ_{τ=0}^{ψ} λ^τ = c · (1 - λ^{ψ+1}).
    decay = 0.79
    window = 50
    c = 0.0004  # squared return magnitude
    returns = _series(np.full(window + 5, np.sqrt(c)))

    sigma = ewma_volatility(returns, decay=decay, window=window)

    assert sigma.iloc[:window].isna().all()
    expected_variance = c * (1.0 - decay ** (window + 1))
    np.testing.assert_allclose(
        sigma.iloc[window:].to_numpy(),
        np.sqrt(expected_variance),
        rtol=1e-12,
    )


def test_ewma_volatility_matches_brute_force_on_random_returns() -> None:
    rng = np.random.default_rng(42)
    decay = 0.79
    window = 10
    values = rng.normal(scale=1e-3, size=window + 25)
    returns = _series(values)

    sigma = ewma_volatility(returns, decay=decay, window=window)

    weights = decay ** np.arange(window + 1)
    expected = np.full(values.shape[0], np.nan)
    for t in range(window, values.shape[0]):
        tau_squared = values[t - window : t + 1][::-1] ** 2
        expected[t] = np.sqrt((1.0 - decay) * np.dot(weights, tau_squared))

    np.testing.assert_allclose(sigma.to_numpy(), expected, rtol=1e-12, equal_nan=True)


def test_ewma_volatility_preserves_index_and_name() -> None:
    returns = _series(np.linspace(-1e-3, 1e-3, 20))
    sigma = ewma_volatility(returns, decay=0.5, window=5)
    pd.testing.assert_index_equal(sigma.index, returns.index)
    assert sigma.name == returns.name


def test_ewma_volatility_returns_all_nan_when_input_shorter_than_window() -> None:
    returns = _series(np.array([0.001, -0.001]))
    sigma = ewma_volatility(returns, decay=0.5, window=10)
    assert sigma.isna().all()
    assert len(sigma) == len(returns)


def test_ewma_volatility_rejects_nan_input() -> None:
    returns = _series(np.array([0.001, np.nan, 0.002, 0.0]))
    with pytest.raises(ValueError, match="NaN"):
        ewma_volatility(returns, decay=0.5, window=2)


def test_ewma_volatility_rejects_non_series_input() -> None:
    with pytest.raises(TypeError, match="Series"):
        ewma_volatility(np.array([0.001, 0.002, 0.003]), decay=0.5, window=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_decay", [-0.1, 0.0, 1.0, 1.5, float("nan"), float("inf")])
def test_ewma_volatility_rejects_invalid_decay(bad_decay: float) -> None:
    returns = _series(np.array([0.001, 0.002, 0.003]))
    with pytest.raises(ValueError, match="decay"):
        ewma_volatility(returns, decay=bad_decay, window=1)


@pytest.mark.parametrize("bad_window", [0, -1, -10])
def test_ewma_volatility_rejects_non_positive_window(bad_window: int) -> None:
    returns = _series(np.array([0.001, 0.002, 0.003]))
    with pytest.raises(ValueError, match="window"):
        ewma_volatility(returns, decay=0.5, window=bad_window)


def test_ewma_volatility_rejects_non_integer_window() -> None:
    returns = _series(np.array([0.001, 0.002, 0.003]))
    with pytest.raises(TypeError, match="integer"):
        ewma_volatility(returns, decay=0.5, window=1.5)  # type: ignore[arg-type]


# --- volatility_ratio ---


def test_volatility_ratio_uses_paper_defaults() -> None:
    assert DEFAULT_DECAY == 0.79
    assert DEFAULT_FAST_WINDOW == 50
    assert DEFAULT_SLOW_WINDOW == 100


def test_volatility_ratio_matches_closed_form_on_constant_squared_returns() -> None:
    # Constant Δy² = c gives σ² = c · (1 - λ^{ψ+1}), so the ratio is
    # sqrt((1 - λ^{ψf+1}) / (1 - λ^{ψs+1})) — a fixed value independent of c.
    returns = _series(np.full(200, 0.01))
    ratio = volatility_ratio(returns)
    valid = ratio.dropna()
    assert len(valid) == 200 - DEFAULT_SLOW_WINDOW

    expected = np.sqrt(
        (1.0 - DEFAULT_DECAY ** (DEFAULT_FAST_WINDOW + 1))
        / (1.0 - DEFAULT_DECAY ** (DEFAULT_SLOW_WINDOW + 1))
    )
    np.testing.assert_allclose(valid.to_numpy(), expected, rtol=1e-12)


def test_volatility_ratio_drops_below_one_after_high_to_low_shift() -> None:
    # Use a near-unit decay so ψ actually controls effective memory and the
    # slow window can still "see" the high-vol prefix when the fast window no
    # longer does. With the paper's default λ = 0.79 the effective memory is
    # roughly 1/(1 - λ) ≈ 5 observations, so ψ_fast = 50 vs ψ_slow = 100 make
    # indistinguishable estimates once both truncated sums have converged.
    rng = np.random.default_rng(7)
    high_vol = rng.normal(scale=1e-2, size=200)
    low_vol = rng.normal(scale=1e-4, size=200)
    returns = _series(np.concatenate([high_vol, low_vol]))

    ratio = volatility_ratio(returns, fast_window=30, slow_window=150, decay=0.99)

    # Well after the shift the fast window sits entirely inside the low-vol
    # regime while the slow window still retains high-vol history.
    at_shift_plus_fast = ratio.iloc[200 + 40]
    assert at_shift_plus_fast < 0.5


def test_volatility_ratio_aligns_and_masks_slow_window_prefix() -> None:
    returns = _series(np.linspace(-1e-3, 1e-3, DEFAULT_SLOW_WINDOW + 20))
    ratio = volatility_ratio(returns)

    pd.testing.assert_index_equal(ratio.index, returns.index)
    assert ratio.iloc[:DEFAULT_SLOW_WINDOW].isna().all()
    assert ratio.iloc[DEFAULT_SLOW_WINDOW:].notna().all()


def test_volatility_ratio_is_deterministic() -> None:
    rng = np.random.default_rng(123)
    returns = _series(rng.normal(scale=1e-3, size=300))
    first = volatility_ratio(returns)
    second = volatility_ratio(returns)
    pd.testing.assert_series_equal(first, second)


def test_volatility_ratio_rejects_fast_ge_slow() -> None:
    returns = _series(np.array([0.001, 0.002, 0.003, 0.004]))
    with pytest.raises(ValueError, match="fast_window"):
        volatility_ratio(returns, fast_window=5, slow_window=5)
    with pytest.raises(ValueError, match="fast_window"):
        volatility_ratio(returns, fast_window=10, slow_window=5)


def test_volatility_ratio_config_validates_on_construction() -> None:
    with pytest.raises(ValueError, match="decay"):
        VolatilityRatioConfig(decay=1.5)
    with pytest.raises(ValueError, match="fast_window"):
        VolatilityRatioConfig(fast_window=100, slow_window=50)
    with pytest.raises(ValueError, match="slow_window"):
        VolatilityRatioConfig(slow_window=0)


def test_volatility_ratio_config_exposes_paper_defaults() -> None:
    config = VolatilityRatioConfig()
    assert config.decay == DEFAULT_DECAY
    assert config.fast_window == DEFAULT_FAST_WINDOW
    assert config.slow_window == DEFAULT_SLOW_WINDOW
