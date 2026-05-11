"""Empirical calibration helper for the thresholded_hold signal policy.

Loads the data referenced by a side-info comparison YAML, fits the baseline
Gaussian HMM on consecutive walk-forward windows, runs the forward filter on
the matching forecast slices, and prints the quantile distribution of
``|E[Δy_{t+1}]|`` together with a small post-cost sweep table over a
hand-picked threshold grid. The sweep uses the same per-window signal/return
alignment as the run artifacts. The output is meant to inform the
``signal_threshold`` choice in the YAML config; it is not a substitute for the
full side-information comparison run.

Usage:
    python scripts/calibrate_threshold.py configs/example_es_databento_side_info_comparison.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hft_hmm.evaluation import (
    apply_turnover_cost,
    daily_annualized_sharpe_ratio,
    signal_turnover,
)
from hft_hmm.experiments._data_loading import load_returns_from_source
from hft_hmm.experiments.side_info_comparison import SideInfoComparisonConfig
from hft_hmm.inference import forward_filter
from hft_hmm.models.gaussian_hmm import GaussianHMMWrapper
from hft_hmm.strategy import align_signal_with_future_return, build_signal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--n-windows",
        type=int,
        default=12,
        help="Number of consecutive walk-forward windows to use for calibration.",
    )
    args = parser.parse_args(argv)

    cfg = SideInfoComparisonConfig.from_yaml(args.config)
    returns = load_returns_from_source(cfg.data, frequency=cfg.frequency)
    print(f"loaded {returns.shape[0]:,} bars from {returns.index.min()} to {returns.index.max()}")

    wf = cfg.walk_forward
    sorted_dates = np.array(sorted(set(returns.index.date)), dtype=object)
    bar_dates = returns.index.date

    expected_parts: list[pd.Series] = []
    realized_parts: list[pd.Series] = []
    chosen_k = sorted(wf.k_values)[0]

    available_windows = ((sorted_dates.size - wf.h_days - wf.t_days) // wf.retrain_every_days) + 1
    n_windows = min(args.n_windows, available_windows)
    for w in range(n_windows):
        day_offset = w * wf.retrain_every_days
        train_start = sorted_dates[day_offset]
        train_end = sorted_dates[day_offset + wf.h_days - 1]
        forecast_start = sorted_dates[day_offset + wf.h_days]
        forecast_end = sorted_dates[day_offset + wf.h_days + wf.t_days - 1]

        train_slice = returns.loc[(bar_dates >= train_start) & (bar_dates <= train_end)]
        forecast_slice = returns.loc[(bar_dates >= forecast_start) & (bar_dates <= forecast_end)]

        wrapper = GaussianHMMWrapper(
            n_states=chosen_k,
            random_state=wf.random_state,
            n_iter=wf.n_iter,
            tol=wf.tol,
            min_variance=wf.min_variance,
            variance_floor_policy=wf.variance_floor_policy,
        )
        fitted = wrapper.fit(train_slice)
        filt = forward_filter(forecast_slice, fitted)
        expected_parts.append(pd.Series(filt.expected_next_returns, index=forecast_slice.index))
        realized_parts.append(forecast_slice)
        print(
            f"window {w}: train [{train_start}, {train_end}] "
            f"forecast [{forecast_start}, {forecast_end}] "
            f"n_train={train_slice.shape[0]} n_forecast={forecast_slice.shape[0]}"
        )

    expected = pd.concat(expected_parts)
    abs_expected = np.abs(expected)

    print()
    print("|E[Δy]| quantile distribution:")
    for q in (0.50, 0.75, 0.90, 0.95, 0.99, 0.995):
        print(f"  q{int(q*1000)/10}: {np.quantile(abs_expected, q):.3e}")

    print()
    print(
        f"{'threshold':>12} {'n_active':>10} {'turnover':>10} "
        f"{'pre_sharpe':>11} {'post_sharpe':>12} {'post/pre':>9}"
    )
    cost_bps = cfg.cost_bps_per_turnover

    grid = sorted(
        {
            0.0,
            float(cfg.signal_threshold),
            *(float(np.quantile(abs_expected, q)) for q in (0.25, 0.5, 0.75, 0.9, 0.95, 0.99)),
        }
    )
    for threshold in grid:
        policy = "sign" if threshold == 0.0 else "thresholded_hold"
        initial_position = 0
        signal_parts: list[pd.Series] = []
        pre_parts: list[pd.Series] = []
        post_parts: list[pd.Series] = []
        applied_turnover = 0.0
        for expected_window, realized_window in zip(expected_parts, realized_parts, strict=True):
            signal = build_signal(
                expected_window,
                policy=policy,
                threshold=threshold,
                initial_position=initial_position,
            )
            initial_position = int(signal.iloc[-1])
            pre_window = align_signal_with_future_return(signal, realized_window)
            turnover_window = signal_turnover(signal)
            post_window = apply_turnover_cost(
                pre_window,
                turnover_window,
                cost_bps_per_turnover=cost_bps,
            )
            signal_parts.append(signal)
            pre_parts.append(pre_window)
            post_parts.append(post_window)
            applied_turnover += float(turnover_window.sum())

        signal = pd.concat(signal_parts)
        pre = pd.concat(pre_parts)
        post = pd.concat(post_parts)
        n_active = int(np.sum(signal.to_numpy() != 0))
        try:
            pre_sharpe = daily_annualized_sharpe_ratio(pre)
            post_sharpe = daily_annualized_sharpe_ratio(post)
        except Exception as exc:  # noqa: BLE001 - calibration helper only
            pre_sharpe = float("nan")
            post_sharpe = float("nan")
            print(f"sharpe failed: {exc}")
        ratio = (
            post_sharpe / pre_sharpe
            if np.isfinite(pre_sharpe) and pre_sharpe != 0.0
            else float("nan")
        )
        print(
            f"{threshold:>12.3e} {n_active:>10d} {applied_turnover:>10.0f} "
            f"{pre_sharpe:>11.4f} {post_sharpe:>12.4f} {ratio:>9.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
