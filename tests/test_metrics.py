"""Tests for backtest metrics and report summaries."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from hft_hmm.core import EVALUATION_LAYER, module_category
from hft_hmm.evaluation import (
    apply_turnover_cost,
    cumulative_return,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    signal_turnover,
    summarize_backtest,
)

metrics_module = importlib.import_module("hft_hmm.evaluation.metrics")


def test_metrics_module_declares_evaluation_layer_category() -> None:
    assert module_category(metrics_module) == EVALUATION_LAYER


def test_signal_turnover_tracks_absolute_position_change() -> None:
    index = pd.date_range("2024-01-01 09:30", periods=4, freq="1min")
    signal = pd.Series([0, 1, -1, -1], index=index)

    turnover = signal_turnover(signal)

    assert turnover.index.equals(index[1:])
    assert turnover.name == "turnover"
    np.testing.assert_allclose(turnover.to_numpy(), np.array([1.0, 2.0, 0.0]))


def test_signal_turnover_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="pd.Series"):
        signal_turnover(np.array([0, 1, -1]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two"):
        signal_turnover(pd.Series([1.0], index=[0]))
    with pytest.raises(ValueError, match="finite"):
        signal_turnover(pd.Series([1.0, np.nan], index=[0, 1]))


def test_apply_turnover_cost_subtracts_bps_per_turnover() -> None:
    index = pd.RangeIndex(1, 4)
    strategy_returns = pd.Series([0.02, -0.01, 0.03], index=index)
    turnover = pd.Series([1.0, 2.0, 0.0], index=index)

    post_cost = apply_turnover_cost(
        strategy_returns,
        turnover,
        cost_bps_per_turnover=5.0,
    )

    np.testing.assert_allclose(post_cost.to_numpy(), np.array([0.0195, -0.011, 0.03]))
    assert post_cost.index.equals(index)


def test_apply_turnover_cost_rejects_invalid_alignment_and_cost() -> None:
    strategy_returns = pd.Series([0.1, 0.2], index=[1, 2])
    turnover = pd.Series([1.0, 0.0], index=[10, 11])

    with pytest.raises(ValueError, match="same index"):
        apply_turnover_cost(strategy_returns, turnover, cost_bps_per_turnover=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        apply_turnover_cost(strategy_returns, strategy_returns, cost_bps_per_turnover=-1.0)


def test_cumulative_return_compounds_log_returns() -> None:
    strategy_returns = np.log(np.array([1.10, 0.95], dtype=float))

    cumulative = cumulative_return(strategy_returns)

    assert cumulative == pytest.approx(0.045)


def test_sharpe_ratio_matches_sample_mean_over_std() -> None:
    strategy_returns = np.array([0.1, 0.2, 0.3], dtype=float)

    assert sharpe_ratio(strategy_returns) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "strategy_returns",
    [
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([0.01], dtype=float),
    ],
)
def test_sharpe_ratio_returns_zero_for_zero_variance_or_single_period(strategy_returns) -> None:
    assert sharpe_ratio(strategy_returns) == 0.0


def test_max_drawdown_uses_initial_capital_as_starting_peak() -> None:
    strategy_returns = np.log(np.array([0.9, 1.2, 0.8], dtype=float))

    drawdown = max_drawdown(strategy_returns)

    assert drawdown == pytest.approx(-0.2)


def test_hit_rate_counts_strictly_positive_periods() -> None:
    strategy_returns = np.array([0.1, 0.0, -0.2, 0.3], dtype=float)

    assert hit_rate(strategy_returns) == pytest.approx(0.5)


def test_summarize_backtest_returns_pre_and_post_cost_rows() -> None:
    index = pd.RangeIndex(4)
    signal = pd.Series([0, 1, -1, -1], index=index)
    realized_returns = pd.Series([0.01, 0.02, -0.03, 0.04], index=index)

    summary = summarize_backtest(signal, realized_returns, cost_bps_per_turnover=10.0)

    assert summary.index.tolist() == ["pre-cost", "post-cost"]
    assert summary.index.name == "mode"
    assert summary.loc["pre-cost", "n_periods"] == 3
    assert summary.loc["pre-cost", "cost_bps_per_turnover"] == pytest.approx(0.0)
    assert summary.loc["post-cost", "cost_bps_per_turnover"] == pytest.approx(10.0)
    assert summary.loc["post-cost", "cumulative_return"] < summary.loc[
        "pre-cost", "cumulative_return"
    ]


def test_summarize_backtest_post_cost_sharpe_equals_pre_cost_when_cost_is_zero() -> None:
    index = pd.RangeIndex(5)
    signal = pd.Series([1, 1, -1, -1, 1], index=index)
    realized_returns = pd.Series([0.0, 0.02, -0.01, 0.03, -0.02], index=index)

    summary = summarize_backtest(signal, realized_returns, cost_bps_per_turnover=0.0)

    assert summary.loc["post-cost", "sharpe_ratio"] == pytest.approx(
        summary.loc["pre-cost", "sharpe_ratio"]
    )


def test_summarize_backtest_constant_flat_signal_has_no_cost_drag() -> None:
    index = pd.RangeIndex(4)
    signal = pd.Series([0, 0, 0, 0], index=index)
    realized_returns = pd.Series([0.01, -0.02, 0.03, -0.04], index=index)

    summary = summarize_backtest(signal, realized_returns, cost_bps_per_turnover=25.0)

    metric_columns = ["cumulative_return", "sharpe_ratio", "max_drawdown", "hit_rate"]
    np.testing.assert_allclose(
        summary.loc["pre-cost", metric_columns].to_numpy(dtype=float),
        summary.loc["post-cost", metric_columns].to_numpy(dtype=float),
    )


def test_summarize_backtest_handles_single_period() -> None:
    index = pd.RangeIndex(2)
    signal = pd.Series([1, 1], index=index)
    realized_returns = pd.Series([0.0, 0.02], index=index)

    summary = summarize_backtest(signal, realized_returns, cost_bps_per_turnover=5.0)

    assert summary.loc["pre-cost", "n_periods"] == 1
    assert summary.loc["pre-cost", "sharpe_ratio"] == 0.0
    assert summary.loc["pre-cost", "hit_rate"] == pytest.approx(1.0)
