"""Evaluation helpers for signal backtests and report summaries."""

from hft_hmm.evaluation.metrics import (
    BACKTEST_METRICS_REFERENCE,
    TurnoverDiagnostics,
    annualized_sharpe_ratio,
    apply_turnover_cost,
    cumulative_return,
    daily_annualized_sharpe_ratio,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    signal_turnover,
    summarize_backtest,
    turnover_diagnostics,
)

__all__ = [
    "BACKTEST_METRICS_REFERENCE",
    "TurnoverDiagnostics",
    "annualized_sharpe_ratio",
    "apply_turnover_cost",
    "cumulative_return",
    "daily_annualized_sharpe_ratio",
    "hit_rate",
    "max_drawdown",
    "sharpe_ratio",
    "signal_turnover",
    "summarize_backtest",
    "turnover_diagnostics",
]
