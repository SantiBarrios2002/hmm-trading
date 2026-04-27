"""Tests for the spline-based predictor."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from hft_hmm.core.references import ENGINEERING_APPROXIMATION, module_category
from hft_hmm.features.splines import (
    DEFAULT_MIN_OBS,
    DEFAULT_N_KNOTS,
    SPLINE_PREDICTOR_REFERENCE,
    SplinePredictorConfig,
    SplinePredictorResult,
    fit_spline_predictor,
)

splines_module = importlib.import_module("hft_hmm.features.splines")


# --- helpers ---


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="UTC")


def _feature(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=_index(len(values)), name="feature")


def _returns(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=_index(len(values)), name="returns")


def _linear_dataset(n: int = 200) -> tuple[pd.Series, pd.Series]:
    """Feature on [-1, 1], returns shifted so future_return[t] ≈ feature[t]."""
    x = np.linspace(-1.0, 1.0, n)
    # returns[t] = x[t-1], so future_returns[t] = returns[t+1] = x[t].
    r = np.empty(n)
    r[0] = 0.0
    r[1:] = x[:-1]
    return _feature(x), _returns(r)


# --- module taxonomy / reference wiring ---


def test_module_is_engineering_approximation() -> None:
    assert module_category(splines_module) == ENGINEERING_APPROXIMATION


def test_paper_reference_points_to_section_4_2() -> None:
    assert SPLINE_PREDICTOR_REFERENCE.section == "§4.2"
    assert "spline" in SPLINE_PREDICTOR_REFERENCE.topic.lower()


# --- config validation ---


def test_config_default_values() -> None:
    config = SplinePredictorConfig()
    assert config.n_knots == DEFAULT_N_KNOTS
    assert config.min_obs == DEFAULT_MIN_OBS
    assert config.degree == 3
    assert config.demean is False


@pytest.mark.parametrize("bad", [0, -1, -10])
def test_config_rejects_non_positive_n_knots(bad: int) -> None:
    with pytest.raises(ValueError, match="n_knots"):
        SplinePredictorConfig(n_knots=bad)


def test_config_rejects_non_integer_n_knots() -> None:
    with pytest.raises(TypeError, match="integer"):
        SplinePredictorConfig(n_knots=2.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1])
def test_config_rejects_non_positive_min_obs(bad: int) -> None:
    with pytest.raises(ValueError, match="min_obs"):
        SplinePredictorConfig(min_obs=bad)


def test_config_rejects_non_bool_demean() -> None:
    with pytest.raises(TypeError, match="demean"):
        SplinePredictorConfig(demean="yes")  # type: ignore[arg-type]


# --- fit: basic contract ---


def test_fit_returns_result_object() -> None:
    feature, returns = _linear_dataset()
    result = fit_spline_predictor(feature, returns)
    assert isinstance(result, SplinePredictorResult)


def test_fit_is_deterministic() -> None:
    feature, returns = _linear_dataset()
    r1 = fit_spline_predictor(feature, returns)
    r2 = fit_spline_predictor(feature, returns)
    x_eval = np.linspace(-0.9, 0.9, 50)
    np.testing.assert_array_equal(r1.evaluate(x_eval), r2.evaluate(x_eval))


def test_result_metadata() -> None:
    n = 200
    feature, returns = _linear_dataset(n)
    result = fit_spline_predictor(feature, returns)
    assert result.n_obs == n - 1  # last row NaN from shift(-1)
    # x_min/x_max span only the observations that had a valid future return
    # (all but the last, which has no shift(-1) partner).
    assert result.x_min == pytest.approx(feature.iloc[0], rel=1e-10)
    assert result.x_max == pytest.approx(feature.iloc[-2], rel=1e-10)
    assert 1 <= result.n_knots_effective <= DEFAULT_N_KNOTS
    assert result.prediction_mean is None  # demean=False default


# --- one-step-ahead alignment ---


def test_one_step_ahead_alignment() -> None:
    # feature[t] = t, returns[t] = t.  After shift(-1):
    # paired as (x=t, y=t+1), so the fitted spline ≈ y = x + 1.
    # If alignment used returns[t] instead, the spline would ≈ y = x.
    n = 100
    vals = np.arange(n, dtype=float)
    feature = _feature(vals)
    returns = _returns(vals)
    config = SplinePredictorConfig(n_knots=3)
    result = fit_spline_predictor(feature, returns, config=config)

    # Interior evaluation points — away from the tail-0 boundary artefact.
    np.testing.assert_allclose(result.evaluate(5.0), 6.0, atol=0.05)
    np.testing.assert_allclose(result.evaluate(50.0), 51.0, atol=0.05)


# --- NaN handling ---


def test_nan_prefix_dropped_silently() -> None:
    # Simulate the slow-window NaN prefix produced by volatility_ratio.
    n = 200
    rng = np.random.default_rng(1)
    feature_vals = np.concatenate([np.full(100, np.nan), np.linspace(0.5, 2.0, 100)])
    feature = _feature(feature_vals)
    returns = _returns(rng.normal(scale=0.01, size=n))
    result = fit_spline_predictor(feature, returns)
    # 100 NaN-feature rows + 1 NaN future-return at the end = 99 valid pairs.
    assert result.n_obs == 99


def test_nan_in_returns_row_is_dropped() -> None:
    n = 150
    rng = np.random.default_rng(2)
    feature_vals = np.concatenate([np.full(30, np.nan), np.linspace(1.0, 2.0, 120)])
    returns_vals = rng.normal(scale=0.01, size=n)
    returns_vals[80] = np.nan  # future_returns[79] becomes NaN after shift(-1)
    feature = _feature(feature_vals)
    returns = _returns(returns_vals)
    result = fit_spline_predictor(feature, returns)
    assert result.n_obs < n


# --- rejection of non-finite values ---


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_rejects_infinite_feature(bad: float) -> None:
    n = 100
    rng = np.random.default_rng(3)
    feature_vals = np.linspace(0.5, 2.0, n)
    feature_vals[50] = bad
    feature = _feature(feature_vals)
    returns = _returns(rng.normal(scale=0.01, size=n))
    with pytest.raises(ValueError, match="non-finite"):
        fit_spline_predictor(feature, returns)


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_rejects_infinite_return(bad: float) -> None:
    n = 100
    rng = np.random.default_rng(4)
    feature = _feature(np.linspace(0.5, 2.0, n))
    returns_vals = rng.normal(scale=0.01, size=n)
    returns_vals[50] = bad
    returns = _returns(returns_vals)
    with pytest.raises(ValueError, match="non-finite"):
        fit_spline_predictor(feature, returns)


# --- too few observations ---


def test_too_few_observations_raises() -> None:
    config = SplinePredictorConfig(min_obs=50)
    feature = _feature(np.linspace(0.0, 1.0, 30))
    returns = _returns(np.random.default_rng(5).normal(size=30))
    with pytest.raises(ValueError, match="valid"):
        fit_spline_predictor(feature, returns, config=config)


def test_too_few_unique_values_raises() -> None:
    # 5 unique x values, but n_knots=5 needs n_unique >= 5+3+1=9.
    n = 100
    x_vals = np.tile(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), 20)
    feature = _feature(x_vals)
    returns = _returns(np.random.default_rng(6).normal(scale=0.01, size=n))
    config = SplinePredictorConfig(n_knots=5)
    with pytest.raises(ValueError, match="unique"):
        fit_spline_predictor(feature, returns, config=config)


# --- evaluate: return types ---


def test_evaluate_scalar_returns_float() -> None:
    feature, returns = _linear_dataset()
    result = fit_spline_predictor(feature, returns)
    out = result.evaluate(0.0)
    assert isinstance(out, float)


def test_evaluate_array_returns_ndarray() -> None:
    feature, returns = _linear_dataset()
    result = fit_spline_predictor(feature, returns)
    x_eval = np.linspace(-0.9, 0.9, 20)
    out = result.evaluate(x_eval)
    assert isinstance(out, np.ndarray)
    assert out.shape == (20,)


def test_evaluate_series_preserves_index() -> None:
    feature, returns = _linear_dataset()
    result = fit_spline_predictor(feature, returns)
    idx = pd.date_range("2024-06-01", periods=10, freq="1h", tz="UTC")
    x_series = pd.Series(np.linspace(-0.5, 0.5, 10), index=idx)
    out = result.evaluate(x_series)
    assert isinstance(out, pd.Series)
    pd.testing.assert_index_equal(out.index, idx)
    assert out.name == "spline_prediction"


# --- demean ---


def test_demean_centers_predictions_over_support() -> None:
    feature, returns = _linear_dataset(300)
    config = SplinePredictorConfig(demean=True)
    result = fit_spline_predictor(feature, returns, config=config)
    assert result.prediction_mean is not None
    # Mean of evaluate() over the unique support should be ~0.
    _, y_grid = result.evaluation_grid(n=500)
    assert abs(np.mean(y_grid)) < 1e-10


def test_demean_false_leaves_prediction_mean_none() -> None:
    feature, returns = _linear_dataset()
    result = fit_spline_predictor(feature, returns, config=SplinePredictorConfig(demean=False))
    assert result.prediction_mean is None


# --- synthetic monotonic relation ---


def test_predictions_preserve_direction_on_monotone_input() -> None:
    # future_returns[t] = feature[t], so the spline ≈ y = x (monotone).
    feature, returns = _linear_dataset(300)
    config = SplinePredictorConfig(n_knots=4)
    result = fit_spline_predictor(feature, returns, config=config)
    _, y_grid = result.evaluation_grid(n=100)
    diffs = np.diff(y_grid)
    assert np.all(diffs > -1e-6), "Predictions should be non-decreasing for a monotone input."


# --- evaluation_grid ---


def test_evaluation_grid_shape_and_bounds() -> None:
    feature, returns = _linear_dataset()
    result = fit_spline_predictor(feature, returns)
    x_grid, y_grid = result.evaluation_grid(n=50)
    assert len(x_grid) == 50
    assert len(y_grid) == 50
    assert x_grid[0] == pytest.approx(result.x_min)
    assert x_grid[-1] == pytest.approx(result.x_max)


# --- duplicate feature values (seasonality-like) ---


def test_discrete_feature_values_fit_cleanly() -> None:
    # Simulate intraday seasonality: 10 unique bucket values, ~50 obs each.
    n = 500
    rng = np.random.default_rng(11)
    x_vals = np.tile(np.arange(10, dtype=float), 50)
    returns_vals = rng.normal(scale=0.01, size=n)
    feature = _feature(x_vals)
    returns = _returns(returns_vals)
    config = SplinePredictorConfig(n_knots=3)
    result = fit_spline_predictor(feature, returns, config=config)
    assert result.n_knots_effective >= 1
    assert result.n_obs == n - 1


# --- input type validation ---


def test_rejects_non_series_feature() -> None:
    returns = _returns(np.ones(50))
    with pytest.raises(TypeError, match="pd.Series"):
        fit_spline_predictor(np.ones(50), returns)  # type: ignore[arg-type]


def test_rejects_non_series_returns() -> None:
    feature = _feature(np.ones(50))
    with pytest.raises(TypeError, match="pd.Series"):
        fit_spline_predictor(feature, np.ones(50))  # type: ignore[arg-type]
