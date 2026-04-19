"""Evaluation helpers for signal backtests and report summaries."""

from hft_hmm.evaluation.metrics import (
    BACKTEST_METRICS_REFERENCE,
    apply_turnover_cost,
    cumulative_return,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    signal_turnover,
    summarize_backtest,
)

__all__ = [
    "BACKTEST_METRICS_REFERENCE",
    "apply_turnover_cost",
    "cumulative_return",
    "hit_rate",
    "max_drawdown",
    "sharpe_ratio",
    "signal_turnover",
    "summarize_backtest",
]
