"""Backtest metrics and compact summaries for HMM trading signals.

This module evaluates the aligned strategy returns produced from the sign-based
signal path. Inputs are treated as per-bar log returns, so cumulative return
and drawdown are computed from the compounded wealth curve
``exp(cumsum(log_returns))``.

Cost model
----------
- Pre-cost strategy returns are ``signal[t-1] * realized_returns[t]`` via
  ``hft_hmm.strategy.align_signal_with_future_return``.
- Turnover at bar ``t`` is the absolute position change
  ``|signal[t] - signal[t-1]|``.
- A transaction cost of ``c`` basis points per unit turnover subtracts
  ``c * 1e-4 * |signal[t] - signal[t-1]|`` from the aligned pre-cost return.
- Under this convention, a flat-to-long rebalance costs ``c`` basis points and
  a long-to-short flip costs ``2c`` basis points.

Sharpe ratio is reported as sample mean divided by sample standard deviation.
For fewer than two observations, or for zero sample variance, the function
returns ``0.0`` as a stable evaluation-layer fallback.

References: §8 sign-based trading signal (evaluation layer)
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from hft_hmm.core import EVALUATION_LAYER, PaperReference, reference
from hft_hmm.strategy import align_signal_with_future_return

__category__: Final[str] = EVALUATION_LAYER
BACKTEST_METRICS_REFERENCE: Final[PaperReference] = reference(
    "§8", "backtest metrics and cost-aware evaluation"
)


def signal_turnover(signal: pd.Series) -> pd.Series:
    """Return absolute position changes indexed at the rebalance bar."""
    if not isinstance(signal, pd.Series):
        raise TypeError(f"signal must be a pd.Series, got {type(signal).__name__}.")
    if len(signal) < 2:
        raise ValueError("at least two observations are required to compute turnover.")

    signal_values = np.asarray(signal, dtype=float)
    if not np.all(np.isfinite(signal_values)):
        raise ValueError("signal must contain only finite values.")

    turnover = np.abs(np.diff(signal_values))
    return pd.Series(turnover, index=signal.index[1:], name="turnover")


def apply_turnover_cost(
    strategy_returns: pd.Series,
    turnover: pd.Series,
    *,
    cost_bps_per_turnover: float,
) -> pd.Series:
    """Subtract linear turnover costs from aligned strategy returns."""
    if not isinstance(strategy_returns, pd.Series):
        raise TypeError(
            "strategy_returns must be a pd.Series, " f"got {type(strategy_returns).__name__}."
        )
    if not isinstance(turnover, pd.Series):
        raise TypeError(f"turnover must be a pd.Series, got {type(turnover).__name__}.")
    if len(strategy_returns) != len(turnover):
        raise ValueError(
            "strategy_returns and turnover must have the same length; "
            f"got {len(strategy_returns)} vs {len(turnover)}."
        )
    if not strategy_returns.index.equals(turnover.index):
        raise ValueError(
            "strategy_returns and turnover must share the same index; "
            "align them to the same rebalance timeline before applying costs."
        )
    if len(strategy_returns) < 1:
        raise ValueError("strategy_returns must contain at least one observation.")
    if not np.isfinite(cost_bps_per_turnover) or cost_bps_per_turnover < 0.0:
        raise ValueError(
            "cost_bps_per_turnover must be a finite non-negative float, "
            f"got {cost_bps_per_turnover!r}."
        )

    return_values = np.asarray(strategy_returns, dtype=float)
    turnover_values = np.asarray(turnover, dtype=float)
    if not np.all(np.isfinite(return_values)):
        raise ValueError("strategy_returns must contain only finite values.")
    if not np.all(np.isfinite(turnover_values)):
        raise ValueError("turnover must contain only finite values.")

    cost_in_return_units = turnover_values * (cost_bps_per_turnover * 1e-4)
    return pd.Series(
        return_values - cost_in_return_units,
        index=strategy_returns.index,
        name="strategy_return_post_cost",
    )


def cumulative_return(strategy_returns: pd.Series | np.ndarray) -> float:
    """Return compounded cumulative return from per-period log returns."""
    values = _coerce_metric_returns(strategy_returns, name="strategy_returns")
    return float(np.expm1(values.sum()))


def sharpe_ratio(strategy_returns: pd.Series | np.ndarray) -> float:
    """Return sample mean divided by sample standard deviation."""
    values = _coerce_metric_returns(strategy_returns, name="strategy_returns")
    if values.shape[0] < 2:
        return 0.0

    std = float(values.std(ddof=1))
    if std <= np.finfo(float).eps:
        return 0.0
    return float(values.mean() / std)


def max_drawdown(strategy_returns: pd.Series | np.ndarray) -> float:
    """Return the worst peak-to-trough drawdown on the compounded wealth curve."""
    values = _coerce_metric_returns(strategy_returns, name="strategy_returns")
    cumulative_log_returns = np.cumsum(values)
    running_peak = np.maximum.accumulate(
        np.concatenate((np.array([0.0], dtype=float), cumulative_log_returns))
    )[1:]
    drawdowns = np.expm1(cumulative_log_returns - running_peak)
    return float(drawdowns.min())


def hit_rate(strategy_returns: pd.Series | np.ndarray) -> float:
    """Return the fraction of periods with strictly positive strategy return."""
    values = _coerce_metric_returns(strategy_returns, name="strategy_returns")
    return float(np.mean(values > 0.0))


def summarize_backtest(
    signal: pd.Series,
    realized_returns: pd.Series,
    *,
    cost_bps_per_turnover: float = 0.0,
) -> pd.DataFrame:
    """Return a compact pre-cost and post-cost summary table."""
    pre_cost_returns = align_signal_with_future_return(signal, realized_returns)
    turnover = signal_turnover(signal)
    post_cost_returns = apply_turnover_cost(
        pre_cost_returns,
        turnover,
        cost_bps_per_turnover=cost_bps_per_turnover,
    )

    summary = pd.DataFrame(
        [
            _summary_row(pre_cost_returns, cost_bps_per_turnover=0.0),
            _summary_row(post_cost_returns, cost_bps_per_turnover=cost_bps_per_turnover),
        ],
        index=pd.Index(["pre-cost", "post-cost"], name="mode"),
    )
    return summary


def _summary_row(
    strategy_returns: pd.Series,
    *,
    cost_bps_per_turnover: float,
) -> dict[str, float | int]:
    return {
        "n_periods": int(strategy_returns.shape[0]),
        "cost_bps_per_turnover": float(cost_bps_per_turnover),
        "cumulative_return": cumulative_return(strategy_returns),
        "sharpe_ratio": sharpe_ratio(strategy_returns),
        "max_drawdown": max_drawdown(strategy_returns),
        "hit_rate": hit_rate(strategy_returns),
    }


def _coerce_metric_returns(
    strategy_returns: pd.Series | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    values = np.asarray(strategy_returns, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {values.shape}.")
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values
