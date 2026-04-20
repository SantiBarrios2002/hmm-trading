"""Tests for the walk-forward retraining loop."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from hft_hmm.core import EVALUATION_LAYER, module_category
from hft_hmm.evaluation import (
    cumulative_return,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    summarize_backtest,
)
from hft_hmm.experiments import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
    walk_forward,
)

walk_forward_module = importlib.import_module("hft_hmm.experiments.walk_forward")


def _regime_switching_returns(
    *,
    n_days: int,
    bars_per_day: int,
    seed: int = 0,
) -> pd.Series:
    """Sample a two-state regime-switching return series with a tz-aware index."""
    rng = np.random.default_rng(seed)
    means = np.array([-0.02, 0.02], dtype=float)
    stds = np.array([0.01, 0.01], dtype=float)
    transition = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)

    total_bars = n_days * bars_per_day
    returns = np.empty(total_bars, dtype=float)
    state = 0
    for i in range(total_bars):
        returns[i] = rng.normal(means[state], stds[state])
        state = int(rng.choice(2, p=transition[state]))

    dates = pd.bdate_range("2024-01-02", periods=n_days, tz="UTC")
    timestamps = [
        date + pd.Timedelta(minutes=minute) for date in dates for minute in range(bars_per_day)
    ]
    index = pd.DatetimeIndex(timestamps, tz="UTC")
    return pd.Series(returns, index=index, name="log_return")


def _example_summary(*, cost_bps_per_turnover: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_periods": 1,
                "cost_bps_per_turnover": 0.0,
                "cumulative_return": 0.01,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "hit_rate": 1.0,
            },
            {
                "n_periods": 1,
                "cost_bps_per_turnover": cost_bps_per_turnover,
                "cumulative_return": 0.009,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "hit_rate": 1.0,
            },
        ],
        index=pd.Index(["pre-cost", "post-cost"], name="mode"),
    )


def _summary_from_return_modes(
    pre_cost_returns: pd.Series,
    post_cost_returns: pd.Series,
    *,
    cost_bps_per_turnover: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_periods": int(pre_cost_returns.shape[0]),
                "cost_bps_per_turnover": 0.0,
                "cumulative_return": cumulative_return(pre_cost_returns),
                "sharpe_ratio": sharpe_ratio(pre_cost_returns),
                "max_drawdown": max_drawdown(pre_cost_returns),
                "hit_rate": hit_rate(pre_cost_returns),
            },
            {
                "n_periods": int(post_cost_returns.shape[0]),
                "cost_bps_per_turnover": float(cost_bps_per_turnover),
                "cumulative_return": cumulative_return(post_cost_returns),
                "sharpe_ratio": sharpe_ratio(post_cost_returns),
                "max_drawdown": max_drawdown(post_cost_returns),
                "hit_rate": hit_rate(post_cost_returns),
            },
        ],
        index=pd.Index(["pre-cost", "post-cost"], name="mode"),
    )


def test_walk_forward_module_declares_evaluation_layer_category() -> None:
    assert module_category(walk_forward_module) == EVALUATION_LAYER


def test_walk_forward_config_defaults_match_paper() -> None:
    config = WalkForwardConfig()
    assert config.h_days == 23
    assert config.t_days == 1
    assert config.retrain_every_days == 1
    assert config.k_values == (2,)
    assert config.random_state == 0


def test_walk_forward_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="h_days"):
        WalkForwardConfig(h_days=0)
    with pytest.raises(ValueError, match="t_days"):
        WalkForwardConfig(t_days=0)
    with pytest.raises(ValueError, match="retrain_every_days"):
        WalkForwardConfig(retrain_every_days=0)
    with pytest.raises(ValueError, match=">= t_days"):
        WalkForwardConfig(t_days=2, retrain_every_days=1)
    with pytest.raises(ValueError, match="n_iter"):
        WalkForwardConfig(n_iter=0)
    with pytest.raises(ValueError, match="tol"):
        WalkForwardConfig(tol=0.0)
    with pytest.raises(ValueError, match="k_values must contain"):
        WalkForwardConfig(k_values=())
    with pytest.raises(ValueError, match="unique"):
        WalkForwardConfig(k_values=(2, 2))
    with pytest.raises(ValueError, match=">= 2"):
        WalkForwardConfig(k_values=(1,))
    with pytest.raises(TypeError, match="int"):
        WalkForwardConfig(k_values=(2.0,))  # type: ignore[arg-type]


def test_walk_forward_config_defaults_retrain_every_days_to_forecast_horizon() -> None:
    config = WalkForwardConfig(t_days=3)
    assert config.retrain_every_days == 3


def test_walk_forward_rejects_non_series_input() -> None:
    with pytest.raises(TypeError, match="pd.Series"):
        walk_forward(np.zeros(100), WalkForwardConfig())  # type: ignore[arg-type]


def test_walk_forward_rejects_non_datetime_index() -> None:
    returns = pd.Series(np.zeros(100), index=pd.RangeIndex(100))
    with pytest.raises(TypeError, match="DatetimeIndex"):
        walk_forward(returns, WalkForwardConfig())


def test_walk_forward_rejects_tz_naive_index() -> None:
    returns = pd.Series(
        np.zeros(100),
        index=pd.date_range("2024-01-02", periods=100, freq="1min"),
    )
    with pytest.raises(ValueError, match="tz-aware"):
        walk_forward(returns, WalkForwardConfig(h_days=2, t_days=1))


def test_walk_forward_rejects_insufficient_dates() -> None:
    returns = _regime_switching_returns(n_days=3, bars_per_day=10, seed=1)
    config = WalkForwardConfig(h_days=5, t_days=2, k_values=(2,))
    with pytest.raises(ValueError, match="at least h_days"):
        walk_forward(returns, config)


def test_walk_forward_rejects_non_finite_values() -> None:
    returns = _regime_switching_returns(n_days=10, bars_per_day=10, seed=0)
    returns.iloc[5] = np.nan
    config = WalkForwardConfig(h_days=5, t_days=2, k_values=(2,))
    with pytest.raises(ValueError, match="finite"):
        walk_forward(returns, config)


def test_walk_forward_covers_at_least_two_windows_on_fixture() -> None:
    returns = _regime_switching_returns(n_days=10, bars_per_day=20, seed=42)
    config = WalkForwardConfig(h_days=5, t_days=2, k_values=(2,), random_state=0)

    result = walk_forward(returns, config)

    assert isinstance(result, WalkForwardResult)
    assert len(result.windows) >= 2
    for window in result.windows:
        assert isinstance(window, WalkForwardWindow)
        assert window.chosen_k == 2
        assert window.forecast_start > window.train_end
        assert isinstance(window.summary, pd.DataFrame)


def test_walk_forward_signal_spans_full_forecast_horizon() -> None:
    returns = _regime_switching_returns(n_days=10, bars_per_day=20, seed=42)
    config = WalkForwardConfig(h_days=5, t_days=2, k_values=(2,), random_state=0)

    result = walk_forward(returns, config)

    total_forecast_obs = sum(window.n_forecast_obs for window in result.windows)
    total_aligned_obs = sum(window.n_forecast_obs - 1 for window in result.windows)
    assert result.signal.shape[0] == total_forecast_obs
    assert result.pre_cost_returns.shape[0] == total_aligned_obs
    assert result.post_cost_returns.shape[0] == total_aligned_obs
    forecast_index = result.signal.index
    assert forecast_index.is_monotonic_increasing
    assert not forecast_index.has_duplicates


def test_walk_forward_drops_the_first_bar_of_each_forecast_window() -> None:
    returns = _regime_switching_returns(n_days=10, bars_per_day=20, seed=42)
    config = WalkForwardConfig(h_days=5, t_days=2, k_values=(2,), random_state=0)

    result = walk_forward(returns, config)

    expected_index = pd.DatetimeIndex([], tz="UTC")
    for window in result.windows:
        window_signal = result.signal.loc[window.forecast_start : window.forecast_end]
        expected_index = expected_index.append(window_signal.index[1:])

    pd.testing.assert_index_equal(result.pre_cost_returns.index, expected_index)
    pd.testing.assert_index_equal(result.post_cost_returns.index, expected_index)


def test_walk_forward_boundary_assert_blocks_future_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe the in-loop leakage guard by short-circuiting forecast slicing."""
    returns = _regime_switching_returns(n_days=8, bars_per_day=10, seed=0)
    config = WalkForwardConfig(h_days=3, t_days=2, k_values=(2,), random_state=0)

    original_loc = pd.Series.loc
    call_counter = {"n": 0}

    class _OverlapLoc:
        def __init__(self, series: pd.Series) -> None:
            self._series = series

        def __getitem__(self, key):  # type: ignore[no-untyped-def]
            call_counter["n"] += 1
            if call_counter["n"] == 2:
                return self._series.iloc[: config.h_days * 10 + 1]
            return original_loc.__get__(self._series, pd.Series)[key]

    def _patched_loc(self: pd.Series):  # type: ignore[no-untyped-def]
        return _OverlapLoc(self)

    monkeypatch.setattr(pd.Series, "loc", property(_patched_loc))
    with pytest.raises(AssertionError, match="leakage guard"):
        walk_forward(returns, config)


def test_walk_forward_fixes_K_when_single_candidate() -> None:
    returns = _regime_switching_returns(n_days=10, bars_per_day=20, seed=7)
    config = WalkForwardConfig(h_days=5, t_days=2, k_values=(3,), random_state=0)

    result = walk_forward(returns, config)

    assert all(window.chosen_k == 3 for window in result.windows)


def test_walk_forward_selects_K_per_window_when_sweep_given() -> None:
    returns = _regime_switching_returns(n_days=10, bars_per_day=20, seed=3)
    config = WalkForwardConfig(h_days=5, t_days=2, k_values=(2, 3), random_state=0)

    result = walk_forward(returns, config)

    assert {window.chosen_k for window in result.windows} <= {2, 3}


def test_walk_forward_retrain_frequency_is_configurable() -> None:
    returns = _regime_switching_returns(n_days=10, bars_per_day=20, seed=5)
    config = WalkForwardConfig(
        h_days=5,
        t_days=1,
        retrain_every_days=2,
        k_values=(2,),
        random_state=0,
    )

    result = walk_forward(returns, config)

    forecast_dates = [window.forecast_start.date() for window in result.windows]
    all_dates = sorted(set(returns.index.date))
    assert forecast_dates == [all_dates[5], all_dates[7], all_dates[9]]


def test_walk_forward_summary_matches_return_series_metrics() -> None:
    returns = _regime_switching_returns(n_days=10, bars_per_day=20, seed=42)
    config = WalkForwardConfig(h_days=5, t_days=2, k_values=(2,), random_state=0)

    result = walk_forward(returns, config, cost_bps_per_turnover=1.0)

    expected_summary = _summary_from_return_modes(
        result.pre_cost_returns,
        result.post_cost_returns,
        cost_bps_per_turnover=1.0,
    )
    pd.testing.assert_frame_equal(result.summary, expected_summary)


def test_walk_forward_window_summaries_match_per_window_backtests() -> None:
    returns = _regime_switching_returns(n_days=10, bars_per_day=20, seed=42)
    config = WalkForwardConfig(h_days=5, t_days=2, k_values=(2,), random_state=0)

    result = walk_forward(returns, config, cost_bps_per_turnover=1.0)

    for window in result.windows:
        window_signal = result.signal.loc[window.forecast_start : window.forecast_end]
        realized = returns.loc[window_signal.index]
        expected_summary = summarize_backtest(window_signal, realized, cost_bps_per_turnover=1.0)
        pd.testing.assert_frame_equal(window.summary, expected_summary)


def test_walk_forward_deterministic_with_fixed_seed() -> None:
    returns = _regime_switching_returns(n_days=10, bars_per_day=20, seed=42)
    config = WalkForwardConfig(h_days=5, t_days=2, k_values=(2,), random_state=0)

    first = walk_forward(returns, config)
    second = walk_forward(returns, config)

    np.testing.assert_array_equal(first.signal.to_numpy(), second.signal.to_numpy())
    np.testing.assert_allclose(
        first.pre_cost_returns.to_numpy(), second.pre_cost_returns.to_numpy()
    )
    assert [w.log_likelihood for w in first.windows] == [w.log_likelihood for w in second.windows]


def test_walk_forward_rejects_unsorted_index() -> None:
    returns = _regime_switching_returns(n_days=10, bars_per_day=10, seed=0)
    shuffled = pd.Series(returns.to_numpy(), index=returns.index[::-1])
    config = WalkForwardConfig(h_days=5, t_days=2, k_values=(2,))
    with pytest.raises(ValueError, match="monotonically"):
        walk_forward(shuffled, config)


def test_walk_forward_rejects_duplicate_timestamps() -> None:
    returns = _regime_switching_returns(n_days=10, bars_per_day=10, seed=0)
    dup_index = returns.index.tolist()
    dup_index[1] = dup_index[0]
    duplicated = pd.Series(returns.to_numpy(), index=pd.DatetimeIndex(dup_index, tz="UTC"))
    config = WalkForwardConfig(h_days=5, t_days=2, k_values=(2,))
    with pytest.raises(ValueError, match="duplicates"):
        walk_forward(duplicated, config)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {
                "index": -1,
                "train_start": pd.Timestamp("2024-01-02", tz="UTC"),
                "train_end": pd.Timestamp("2024-01-03", tz="UTC"),
                "forecast_start": pd.Timestamp("2024-01-04", tz="UTC"),
                "forecast_end": pd.Timestamp("2024-01-05", tz="UTC"),
                "chosen_k": 2,
                "log_likelihood": 0.0,
                "n_train_obs": 1,
                "n_forecast_obs": 1,
                "summary": _example_summary(),
            },
            "index must be non-negative",
        ),
        (
            {
                "index": 0,
                "train_start": pd.Timestamp("2024-01-02", tz="UTC"),
                "train_end": pd.Timestamp("2024-01-03", tz="UTC"),
                "forecast_start": pd.Timestamp("2024-01-04", tz="UTC"),
                "forecast_end": pd.Timestamp("2024-01-05", tz="UTC"),
                "chosen_k": 1,
                "log_likelihood": 0.0,
                "n_train_obs": 1,
                "n_forecast_obs": 1,
                "summary": _example_summary(),
            },
            "chosen_k",
        ),
        (
            {
                "index": 0,
                "train_start": pd.Timestamp("2024-01-02", tz="UTC"),
                "train_end": pd.Timestamp("2024-01-03", tz="UTC"),
                "forecast_start": pd.Timestamp("2024-01-04", tz="UTC"),
                "forecast_end": pd.Timestamp("2024-01-05", tz="UTC"),
                "chosen_k": 2,
                "log_likelihood": 0.0,
                "n_train_obs": 0,
                "n_forecast_obs": 1,
                "summary": _example_summary(),
            },
            "n_train_obs",
        ),
        (
            {
                "index": 0,
                "train_start": pd.Timestamp("2024-01-02", tz="UTC"),
                "train_end": pd.Timestamp("2024-01-03", tz="UTC"),
                "forecast_start": pd.Timestamp("2024-01-04", tz="UTC"),
                "forecast_end": pd.Timestamp("2024-01-05", tz="UTC"),
                "chosen_k": 2,
                "log_likelihood": 0.0,
                "n_train_obs": 1,
                "n_forecast_obs": 0,
                "summary": _example_summary(),
            },
            "n_forecast_obs",
        ),
        (
            {
                "index": 0,
                "train_start": pd.Timestamp("2024-01-03", tz="UTC"),
                "train_end": pd.Timestamp("2024-01-02", tz="UTC"),
                "forecast_start": pd.Timestamp("2024-01-04", tz="UTC"),
                "forecast_end": pd.Timestamp("2024-01-05", tz="UTC"),
                "chosen_k": 2,
                "log_likelihood": 0.0,
                "n_train_obs": 1,
                "n_forecast_obs": 1,
                "summary": _example_summary(),
            },
            "train_end must not precede",
        ),
        (
            {
                "index": 0,
                "train_start": pd.Timestamp("2024-01-02", tz="UTC"),
                "train_end": pd.Timestamp("2024-01-03", tz="UTC"),
                "forecast_start": pd.Timestamp("2024-01-05", tz="UTC"),
                "forecast_end": pd.Timestamp("2024-01-04", tz="UTC"),
                "chosen_k": 2,
                "log_likelihood": 0.0,
                "n_train_obs": 1,
                "n_forecast_obs": 1,
                "summary": _example_summary(),
            },
            "forecast_end must not precede",
        ),
        (
            {
                "index": 0,
                "train_start": pd.Timestamp("2024-01-02", tz="UTC"),
                "train_end": pd.Timestamp("2024-01-05", tz="UTC"),
                "forecast_start": pd.Timestamp("2024-01-05", tz="UTC"),
                "forecast_end": pd.Timestamp("2024-01-06", tz="UTC"),
                "chosen_k": 2,
                "log_likelihood": 0.0,
                "n_train_obs": 1,
                "n_forecast_obs": 1,
                "summary": _example_summary(),
            },
            "strictly after",
        ),
    ],
)
def test_walk_forward_window_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        WalkForwardWindow(**kwargs)


def test_walk_forward_result_validates_fields() -> None:
    returns = _regime_switching_returns(n_days=10, bars_per_day=20, seed=42)
    config = WalkForwardConfig(h_days=5, t_days=2, k_values=(2,), random_state=0)
    result = walk_forward(returns, config)

    with pytest.raises(ValueError, match="non-negative"):
        WalkForwardResult(
            config=result.config,
            windows=result.windows,
            signal=result.signal,
            pre_cost_returns=result.pre_cost_returns,
            post_cost_returns=result.post_cost_returns,
            summary=result.summary,
            cost_bps_per_turnover=-1.0,
        )
    with pytest.raises(ValueError, match="zero windows"):
        WalkForwardResult(
            config=result.config,
            windows=(),
            signal=result.signal,
            pre_cost_returns=result.pre_cost_returns,
            post_cost_returns=result.post_cost_returns,
            summary=result.summary,
        )
    with pytest.raises(TypeError, match="signal"):
        WalkForwardResult(
            config=result.config,
            windows=result.windows,
            signal=result.signal.to_numpy(),  # type: ignore[arg-type]
            pre_cost_returns=result.pre_cost_returns,
            post_cost_returns=result.post_cost_returns,
            summary=result.summary,
        )
    with pytest.raises(TypeError, match="pre_cost_returns"):
        WalkForwardResult(
            config=result.config,
            windows=result.windows,
            signal=result.signal,
            pre_cost_returns=result.pre_cost_returns.to_numpy(),  # type: ignore[arg-type]
            post_cost_returns=result.post_cost_returns,
            summary=result.summary,
        )
    with pytest.raises(TypeError, match="post_cost_returns"):
        WalkForwardResult(
            config=result.config,
            windows=result.windows,
            signal=result.signal,
            pre_cost_returns=result.pre_cost_returns,
            post_cost_returns=result.post_cost_returns.to_numpy(),  # type: ignore[arg-type]
            summary=result.summary,
        )
    with pytest.raises(TypeError, match="summary"):
        WalkForwardResult(
            config=result.config,
            windows=result.windows,
            signal=result.signal,
            pre_cost_returns=result.pre_cost_returns,
            post_cost_returns=result.post_cost_returns,
            summary=result.summary.to_dict(),  # type: ignore[arg-type]
        )
