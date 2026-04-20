"""Walk-forward retraining loop for the Gaussian HMM momentum strategy.

This module reproduces the paper's retraining scheme (§2.3: *"parameter
estimation can be done using the previous H days of market data, when the
market is shut"*). The default window length ``h_days = 23`` matches §3.1
(one-month rolling window) and the forecast horizon ``t_days = 1`` matches
the paper's daily cadence — fit during the overnight break, forecast the next
trading day, advance the window by one day, retrain.

Algorithm
---------
1. Group the input return series by calendar date from its tz-aware
   ``DatetimeIndex``. The distinct sorted dates define the walk-forward grid.
2. For each window ``i`` in ``0..n_windows - 1``:
   - train slice  = bars on dates[i*t_days : i*t_days + h_days]
   - forecast slice = bars on dates[i*t_days + h_days : i*t_days + h_days + t_days]
   - assert ``train.index.max() < forecast.index.min()`` as the no-leakage
     guard.
   - Fit a :class:`~hft_hmm.models.gaussian_hmm.GaussianHMMWrapper`. When the
     config carries multiple candidate ``K`` values, pick per window via
     :func:`~hft_hmm.selection.compare_state_counts` and its ``best_by_bic``
     choice; otherwise use the single candidate directly.
   - Run :func:`~hft_hmm.inference.forward_filter` on the forecast slice and
     emit a sign-based signal via
     :func:`~hft_hmm.strategy.signal_from_filter_result`.
3. Concatenate per-window signals into one series spanning the full
   out-of-sample forecast horizon and summarize via
   :func:`~hft_hmm.evaluation.summarize_backtest`.

References: §2.3 rolling-window overnight retraining scheme (evaluation layer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from hft_hmm.core import EVALUATION_LAYER, PaperReference, reference
from hft_hmm.evaluation import summarize_backtest
from hft_hmm.inference import forward_filter
from hft_hmm.models.gaussian_hmm import GaussianHMMWrapper
from hft_hmm.selection import compare_state_counts
from hft_hmm.strategy import align_signal_with_future_return, signal_from_filter_result

__category__: Final[str] = EVALUATION_LAYER
WALK_FORWARD_REFERENCE: Final[PaperReference] = reference(
    "§2.3", "rolling-window overnight retraining scheme"
)


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward window geometry and per-fit HMM hyperparameters.

    Paper-anchored defaults: ``h_days = 23`` trading days (§3.1 rolling
    window), ``t_days = 1`` day forecast horizon (paper retrains nightly),
    ``k_values = (2,)`` (paper's cross-validation default K=2).
    """

    h_days: int = 23
    t_days: int = 1
    k_values: tuple[int, ...] = (2,)
    random_state: int = 0
    n_iter: int = 100
    tol: float = 1e-4

    def __post_init__(self) -> None:
        if self.h_days < 1:
            raise ValueError(f"h_days must be >= 1, got {self.h_days}.")
        if self.t_days < 1:
            raise ValueError(f"t_days must be >= 1, got {self.t_days}.")
        if self.n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {self.n_iter}.")
        if self.tol <= 0.0:
            raise ValueError(f"tol must be strictly positive, got {self.tol}.")
        k_tuple = tuple(self.k_values)
        if not k_tuple:
            raise ValueError("k_values must contain at least one candidate.")
        if len(set(k_tuple)) != len(k_tuple):
            raise ValueError(f"k_values must be unique, got {k_tuple}.")
        for k in k_tuple:
            if isinstance(k, bool) or not isinstance(k, int):
                raise TypeError(f"k_values entries must be int, got {type(k).__name__}.")
            if k < 2:
                raise ValueError(f"k_values entries must be >= 2, got {k}.")
        object.__setattr__(self, "k_values", k_tuple)


@dataclass(frozen=True)
class WalkForwardWindow:
    """Per-window record of a single train/forecast iteration."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    chosen_k: int
    log_likelihood: float
    n_train_obs: int
    n_forecast_obs: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must be non-negative, got {self.index}.")
        if self.chosen_k < 2:
            raise ValueError(f"chosen_k must be >= 2, got {self.chosen_k}.")
        if self.n_train_obs < 1:
            raise ValueError(f"n_train_obs must be >= 1, got {self.n_train_obs}.")
        if self.n_forecast_obs < 1:
            raise ValueError(f"n_forecast_obs must be >= 1, got {self.n_forecast_obs}.")
        if self.train_end < self.train_start:
            raise ValueError("train_end must not precede train_start.")
        if self.forecast_end < self.forecast_start:
            raise ValueError("forecast_end must not precede forecast_start.")
        if self.forecast_start <= self.train_end:
            raise ValueError(
                "forecast_start must be strictly after train_end; "
                f"got train_end={self.train_end} forecast_start={self.forecast_start}."
            )


@dataclass(frozen=True)
class WalkForwardResult:
    """Immutable snapshot of a walk-forward run."""

    config: WalkForwardConfig
    windows: tuple[WalkForwardWindow, ...]
    signal: pd.Series
    pre_cost_returns: pd.Series
    post_cost_returns: pd.Series
    summary: pd.DataFrame
    cost_bps_per_turnover: float = field(default=0.0)

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("walk-forward run produced zero windows.")
        if not isinstance(self.signal, pd.Series):
            raise TypeError("signal must be a pd.Series.")
        if not isinstance(self.summary, pd.DataFrame):
            raise TypeError("summary must be a pd.DataFrame.")
        if self.cost_bps_per_turnover < 0.0 or not np.isfinite(self.cost_bps_per_turnover):
            raise ValueError(
                "cost_bps_per_turnover must be a finite non-negative float, "
                f"got {self.cost_bps_per_turnover!r}."
            )


def walk_forward(
    returns: pd.Series,
    config: WalkForwardConfig,
    *,
    cost_bps_per_turnover: float = 0.0,
) -> WalkForwardResult:
    """Run the paper's rolling-window retraining scheme on a return series.

    The input ``returns`` must be a ``pd.Series`` with a tz-aware monotonic
    ``DatetimeIndex`` so that calendar-date grouping is unambiguous. The
    ``cost_bps_per_turnover`` parameter is threaded into the final
    :func:`~hft_hmm.evaluation.summarize_backtest` call; it does not change
    the signal path itself.

    References: §2.3 rolling-window overnight retraining scheme (evaluation layer)
    """
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a pd.Series, got {type(returns).__name__}.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns.index must be a pd.DatetimeIndex.")
    if returns.index.tz is None:
        raise ValueError("returns.index must be tz-aware; localize to UTC before calling.")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns.index must be monotonically increasing.")
    if returns.index.has_duplicates:
        raise ValueError("returns.index must not contain duplicates.")
    if not np.all(np.isfinite(np.asarray(returns, dtype=float))):
        raise ValueError("returns must contain only finite values; drop NaN/inf first.")

    sorted_dates = np.array(sorted(set(returns.index.date)), dtype=object)
    n_dates = sorted_dates.size
    if n_dates < config.h_days + config.t_days:
        raise ValueError(
            "returns must span at least h_days + t_days distinct calendar dates; "
            f"got {n_dates} dates for h_days={config.h_days}, t_days={config.t_days}."
        )

    n_windows = (n_dates - config.h_days) // config.t_days
    if n_windows < 1:  # pragma: no cover - guarded by the distinct-dates check above
        raise ValueError("walk-forward produced zero windows; check h_days / t_days.")

    windows: list[WalkForwardWindow] = []
    signal_parts: list[pd.Series] = []
    bar_dates = returns.index.date

    for window_index in range(n_windows):
        train_start_date = sorted_dates[window_index * config.t_days]
        train_end_date = sorted_dates[window_index * config.t_days + config.h_days - 1]
        forecast_start_date = sorted_dates[window_index * config.t_days + config.h_days]
        forecast_end_date = sorted_dates[
            window_index * config.t_days + config.h_days + config.t_days - 1
        ]

        train_mask = (bar_dates >= train_start_date) & (bar_dates <= train_end_date)
        forecast_mask = (bar_dates >= forecast_start_date) & (bar_dates <= forecast_end_date)
        train_slice = returns.loc[train_mask]
        forecast_slice = returns.loc[forecast_mask]

        assert train_slice.index.max() < forecast_slice.index.min(), (
            "walk-forward leakage guard tripped: "
            f"train ends at {train_slice.index.max()} but forecast starts at "
            f"{forecast_slice.index.min()}."
        )

        chosen_k = _select_k(train_slice, config)
        wrapper = GaussianHMMWrapper(
            n_states=chosen_k,
            random_state=config.random_state,
            n_iter=config.n_iter,
            tol=config.tol,
        )
        fitted = wrapper.fit(train_slice)

        filter_result = forward_filter(forecast_slice, fitted)
        window_signal = signal_from_filter_result(
            filter_result,
            threshold=0.0,
            index=forecast_slice.index,
        )
        signal_parts.append(window_signal)

        windows.append(
            WalkForwardWindow(
                index=window_index,
                train_start=train_slice.index.min(),
                train_end=train_slice.index.max(),
                forecast_start=forecast_slice.index.min(),
                forecast_end=forecast_slice.index.max(),
                chosen_k=chosen_k,
                log_likelihood=fitted.log_likelihood,
                n_train_obs=int(train_slice.shape[0]),
                n_forecast_obs=int(forecast_slice.shape[0]),
            )
        )

    combined_signal = pd.concat(signal_parts).astype(np.int8)
    combined_signal.name = "signal"
    realized_on_forecast = returns.loc[combined_signal.index]

    pre_cost_returns = align_signal_with_future_return(combined_signal, realized_on_forecast)
    summary = summarize_backtest(
        combined_signal,
        realized_on_forecast,
        cost_bps_per_turnover=cost_bps_per_turnover,
    )

    # Recompute post-cost returns for the result object (summarize_backtest
    # already builds them internally, but does not expose the series).
    turnover_values = np.abs(np.diff(np.asarray(combined_signal, dtype=float)))
    cost_in_return_units = turnover_values * (cost_bps_per_turnover * 1e-4)
    post_cost_values = pre_cost_returns.to_numpy() - cost_in_return_units
    post_cost_returns = pd.Series(
        post_cost_values,
        index=pre_cost_returns.index,
        name="strategy_return_post_cost",
    )

    return WalkForwardResult(
        config=config,
        windows=tuple(windows),
        signal=combined_signal,
        pre_cost_returns=pre_cost_returns,
        post_cost_returns=post_cost_returns,
        summary=summary,
        cost_bps_per_turnover=float(cost_bps_per_turnover),
    )


def _select_k(train_slice: pd.Series, config: WalkForwardConfig) -> int:
    """Return the chosen K for a single training window.

    With a single candidate, use it directly. With multiple candidates, run
    the model-selection sweep and take ``best_by_bic`` — BIC penalizes
    complexity more aggressively than AIC, which matches the paper's
    preference for parsimonious 2–3 state models in §3.1.
    """
    if len(config.k_values) == 1:
        return int(config.k_values[0])
    selection = compare_state_counts(
        train_slice,
        k_values=config.k_values,
        random_state=config.random_state,
        n_iter=config.n_iter,
        tol=config.tol,
    )
    return int(selection.best_by_bic)
