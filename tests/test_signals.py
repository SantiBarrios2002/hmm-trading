"""Tests for signal generation and alignment helpers."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from hft_hmm.core import EVALUATION_LAYER, StateGrid, module_category
from hft_hmm.inference import forward_filter
from hft_hmm.models.gaussian_hmm import GaussianHMMResult
from hft_hmm.strategy import (
    align_signal_with_future_return,
    sign_signal,
    signal_from_filter_result,
    thresholded_signal,
)

signals_module = importlib.import_module("hft_hmm.strategy.signals")


def _toy_model() -> GaussianHMMResult:
    means = np.array([-1.0, 1.0], dtype=float)
    variances = np.array([0.1, 0.1], dtype=float)
    transition_matrix = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)
    initial_distribution = np.array([0.5, 0.5], dtype=float)
    return GaussianHMMResult(
        state_grid=StateGrid(k=2, means=means, labels=("down", "up")),
        means=means,
        variances=variances,
        transition_matrix=transition_matrix,
        initial_distribution=initial_distribution,
        log_likelihood=0.0,
        n_observations=4,
        converged=True,
        n_iter=10,
        random_state=0,
    )


def test_signals_module_declares_evaluation_layer_category() -> None:
    assert module_category(signals_module) == EVALUATION_LAYER


def test_sign_signal_maps_sign_on_ndarray() -> None:
    values = np.array([-0.1, 0.0, 0.05, -0.02, 1e-9], dtype=float)

    signal = sign_signal(values)

    assert isinstance(signal, pd.Series)
    assert signal.dtype == np.int8
    assert signal.name == "signal"
    assert isinstance(signal.index, pd.RangeIndex)
    np.testing.assert_array_equal(signal.to_numpy(), np.array([-1, 0, 1, -1, 1]))


def test_sign_signal_preserves_series_index() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="1min")
    series = pd.Series([0.01, -0.02, 0.0, 0.03], index=index)

    signal = sign_signal(series)

    assert signal.index.equals(index)
    np.testing.assert_array_equal(signal.to_numpy(), np.array([1, -1, 0, 1]))


def test_sign_signal_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        sign_signal(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="at least one"):
        sign_signal(np.array([], dtype=float))
    with pytest.raises(ValueError, match="finite"):
        sign_signal(np.array([0.1, np.nan, 0.2]))


def test_thresholded_signal_matches_sign_at_zero_threshold() -> None:
    values = np.array([-0.1, 0.0, 0.05, -0.02], dtype=float)

    thresholded = thresholded_signal(values, threshold=0.0)
    sign = sign_signal(values)

    np.testing.assert_array_equal(thresholded.to_numpy(), sign.to_numpy())


def test_thresholded_signal_enforces_dead_zone() -> None:
    values = np.array([-0.05, -0.01, 0.0, 0.01, 0.05, 0.15], dtype=float)

    signal = thresholded_signal(values, threshold=0.02)

    np.testing.assert_array_equal(signal.to_numpy(), np.array([-1, 0, 0, 0, 1, 1]))


def test_thresholded_signal_boundary_is_flat() -> None:
    values = np.array([-0.02, -0.019, 0.019, 0.02], dtype=float)

    signal = thresholded_signal(values, threshold=0.02)

    np.testing.assert_array_equal(signal.to_numpy(), np.array([0, 0, 0, 0]))


def test_thresholded_signal_rejects_negative_or_non_finite_threshold() -> None:
    values = np.array([0.01, -0.02, 0.03], dtype=float)

    with pytest.raises(ValueError, match="non-negative"):
        thresholded_signal(values, threshold=-0.01)
    with pytest.raises(ValueError, match="finite"):
        thresholded_signal(values, threshold=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        thresholded_signal(values, threshold=float("nan"))


def test_align_signal_with_future_return_uses_previous_signal() -> None:
    index = pd.RangeIndex(4)
    signal = pd.Series([0, 1, -1, 1], index=index)
    returns = pd.Series([10.0, 20.0, 30.0, 40.0], index=index)

    aligned = align_signal_with_future_return(signal, returns)

    assert aligned.name == "strategy_return"
    np.testing.assert_array_equal(aligned.index.to_numpy(), np.array([1, 2, 3]))
    np.testing.assert_allclose(aligned.to_numpy(), np.array([0.0, 30.0, -40.0]))


def test_align_rejects_mismatched_index() -> None:
    signal = pd.Series([1, -1, 1], index=[0, 1, 2])
    returns = pd.Series([0.1, 0.2, 0.3], index=[10, 11, 12])

    with pytest.raises(ValueError, match="same index"):
        align_signal_with_future_return(signal, returns)


def test_align_rejects_mismatched_length() -> None:
    signal = pd.Series([1, -1], index=[0, 1])
    returns = pd.Series([0.1, 0.2, 0.3], index=[0, 1, 2])

    with pytest.raises(ValueError, match="same length"):
        align_signal_with_future_return(signal, returns)


def test_align_rejects_non_series_inputs() -> None:
    returns = pd.Series([0.1, 0.2], index=[0, 1])

    with pytest.raises(TypeError, match="signal"):
        align_signal_with_future_return([1, -1], returns)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="realized_returns"):
        align_signal_with_future_return(returns, np.array([0.1, 0.2]))  # type: ignore[arg-type]


def test_align_rejects_single_observation() -> None:
    signal = pd.Series([1], index=[0])
    returns = pd.Series([0.1], index=[0])

    with pytest.raises(ValueError, match="at least two"):
        align_signal_with_future_return(signal, returns)


def test_align_rejects_non_finite_inputs() -> None:
    index = pd.RangeIndex(3)
    signal = pd.Series([1, -1, 1], index=index)
    returns_nan = pd.Series([0.1, np.nan, 0.3], index=index)
    signal_inf = pd.Series([1.0, float("inf"), 1.0], index=index)
    returns = pd.Series([0.1, 0.2, 0.3], index=index)

    with pytest.raises(ValueError, match="realized_returns must contain only finite"):
        align_signal_with_future_return(signal, returns_nan)
    with pytest.raises(ValueError, match="signal must contain only finite"):
        align_signal_with_future_return(signal_inf, returns)


def test_align_leakage_probe_does_not_peek_into_future() -> None:
    """The helper must compute signal[t-1]*returns[t], never signal[t]*returns[t+1]."""
    index = pd.RangeIndex(5)
    signal = pd.Series([1, -1, 1, -1, 1], index=index)
    returns = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0], index=index)

    aligned = align_signal_with_future_return(signal, returns)

    # Correct: signal[t-1] * returns[t]
    expected_correct = np.array(
        [
            signal.iloc[0] * returns.iloc[1],
            signal.iloc[1] * returns.iloc[2],
            signal.iloc[2] * returns.iloc[3],
            signal.iloc[3] * returns.iloc[4],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(aligned.to_numpy(), expected_correct)

    # Wrong (same-bar) and wrong (future-peek) must not match the output.
    wrong_same_bar = (signal * returns).iloc[1:].to_numpy().astype(float)
    wrong_future_peek = (signal.iloc[1:].to_numpy() * returns.iloc[:-1].to_numpy()).astype(float)
    canonical_prev_signal = (signal.iloc[:-1].to_numpy() * returns.iloc[1:].to_numpy()).astype(
        float
    )
    assert not np.allclose(aligned.to_numpy(), wrong_same_bar)
    assert not np.allclose(aligned.to_numpy(), wrong_future_peek)
    # Sanity: the helper matches the canonical vectorized expression.
    np.testing.assert_allclose(aligned.to_numpy(), canonical_prev_signal)


def test_signal_from_filter_result_matches_sign_on_expected_returns() -> None:
    model = _toy_model()
    returns = np.array([-1.2, -0.8, 0.9, 1.1], dtype=float)

    filter_result = forward_filter(returns, model)
    via_wrapper = signal_from_filter_result(filter_result)
    via_sign = sign_signal(filter_result.expected_next_returns)

    np.testing.assert_array_equal(via_wrapper.to_numpy(), via_sign.to_numpy())
    assert isinstance(via_wrapper.index, pd.RangeIndex)


def test_signal_from_filter_result_preserves_supplied_index() -> None:
    model = _toy_model()
    returns = np.array([-1.2, -0.8, 0.9, 1.1], dtype=float)
    index = pd.date_range("2024-01-01 09:30", periods=4, freq="1min")

    filter_result = forward_filter(returns, model)
    signal = signal_from_filter_result(filter_result, index=index)

    assert signal.index.equals(index)


def test_signal_from_filter_result_rejects_mismatched_index() -> None:
    model = _toy_model()
    returns = np.array([-1.2, -0.8, 0.9, 1.1], dtype=float)
    bad_index = pd.RangeIndex(3)  # wrong length

    filter_result = forward_filter(returns, model)
    with pytest.raises(ValueError, match="index length"):
        signal_from_filter_result(filter_result, index=bad_index)


def test_signal_from_filter_result_applies_threshold() -> None:
    model = _toy_model()
    returns = np.array([-1.2, -0.8, 0.9, 1.1], dtype=float)

    filter_result = forward_filter(returns, model)
    thresholded = signal_from_filter_result(filter_result, threshold=10.0)

    # Threshold above any plausible expected return collapses to all-zero.
    np.testing.assert_array_equal(thresholded.to_numpy(), np.zeros(4, dtype=np.int8))


def test_sign_signal_tracks_toy_hmm_regime() -> None:
    model = _toy_model()
    returns = np.array([-1.2, -0.8, 0.9, 1.1], dtype=float)

    filter_result = forward_filter(returns, model)
    signal = sign_signal(filter_result.expected_next_returns)

    # Strongly persistent two-state model: down observations → short;
    # up observations → long.
    assert signal.iloc[0] == -1
    assert signal.iloc[1] == -1
    assert signal.iloc[-1] == 1


def test_align_integrates_with_forward_filter_pipeline() -> None:
    model = _toy_model()
    index = pd.date_range("2024-01-01 09:30", periods=4, freq="1min")
    returns_series = pd.Series([-1.2, -0.8, 0.9, 1.1], index=index)

    filter_result = forward_filter(returns_series.to_numpy(), model)
    signal = sign_signal(pd.Series(filter_result.expected_next_returns, index=index))
    strategy = align_signal_with_future_return(signal, returns_series)

    assert strategy.index.equals(index[1:])
    # strategy[t] = signal[t-1] * returns[t]
    expected = np.array(
        [
            signal.iloc[0] * returns_series.iloc[1],
            signal.iloc[1] * returns_series.iloc[2],
            signal.iloc[2] * returns_series.iloc[3],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(strategy.to_numpy(), expected)
