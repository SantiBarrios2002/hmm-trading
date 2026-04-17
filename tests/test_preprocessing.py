"""Tests for return preprocessing and sampling frequency utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hft_hmm.preprocessing import compute_log_returns, resample_prices, train_test_split_time


def _make_frame(prices: list[float], timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "price": prices,
        }
    )


# --- compute_log_returns ---


def test_compute_log_returns_values() -> None:
    prices = pd.Series([100.0, 110.0, 99.0])
    returns = compute_log_returns(prices)
    assert np.isnan(returns.iloc[0])
    assert returns.iloc[1] == pytest.approx(np.log(110.0 / 100.0))
    assert returns.iloc[2] == pytest.approx(np.log(99.0 / 110.0))


def test_compute_log_returns_preserves_length() -> None:
    prices = pd.Series([1.0, 2.0, 4.0, 8.0])
    assert len(compute_log_returns(prices)) == len(prices)


def test_compute_log_returns_first_is_nan() -> None:
    returns = compute_log_returns(pd.Series([50.0, 51.0]))
    assert np.isnan(returns.iloc[0])


def test_compute_log_returns_constant_prices_give_zero() -> None:
    returns = compute_log_returns(pd.Series([100.0, 100.0, 100.0]))
    assert returns.dropna().tolist() == pytest.approx([0.0, 0.0])


# --- resample_prices ---


def test_resample_prices_daily_from_intraday() -> None:
    frame = _make_frame(
        [100.0, 101.0, 102.0, 103.0],
        [
            "2024-01-02 09:30:00",
            "2024-01-02 15:00:00",
            "2024-01-03 09:30:00",
            "2024-01-03 15:00:00",
        ],
    )
    result = resample_prices(frame, "1D")
    assert len(result) == 2
    assert result["price"].tolist() == pytest.approx([101.0, 103.0])


def test_resample_prices_preserves_ascending_order() -> None:
    frame = _make_frame(
        [100.0, 101.0, 102.0],
        ["2024-01-04 09:30:00", "2024-01-02 09:30:00", "2024-01-03 09:30:00"],
    )
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    result = resample_prices(frame, "1D")
    timestamps = result["timestamp"].tolist()
    assert timestamps == sorted(timestamps)


def test_resample_prices_drops_empty_bars() -> None:
    frame = _make_frame(
        [100.0, 105.0],
        ["2024-01-02 09:30:00", "2024-01-05 09:30:00"],
    )
    result = resample_prices(frame, "1D")
    assert len(result) == 2


def test_resample_prices_sums_volume_within_bar() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02 09:30:00", "2024-01-02 15:00:00"], utc=True),
            "price": [100.0, 101.0],
            "volume": [10.0, 20.0],
        }
    )
    result = resample_prices(frame, "1D")
    assert result["volume"].iloc[0] == pytest.approx(30.0)


def test_resample_prices_single_tick_per_bar() -> None:
    frame = _make_frame([200.0, 210.0], ["2024-01-02 09:30:00", "2024-01-03 09:30:00"])
    result = resample_prices(frame, "1D")
    assert result["price"].tolist() == pytest.approx([200.0, 210.0])


# --- train_test_split_time ---


def test_train_test_split_no_leakage() -> None:
    frame = _make_frame(
        list(range(1, 11)),
        [f"2024-01-{i:02d} 09:30:00" for i in range(1, 11)],
    )
    train, test = train_test_split_time(frame, test_fraction=0.3)
    assert train["timestamp"].max() < test["timestamp"].min()


def test_train_test_split_sizes() -> None:
    frame = _make_frame(
        [float(i) for i in range(10)],
        [f"2024-01-{i + 1:02d} 09:30:00" for i in range(10)],
    )
    train, test = train_test_split_time(frame, test_fraction=0.2)
    assert len(train) == 8
    assert len(test) == 2


def test_train_test_split_combined_covers_all_rows() -> None:
    frame = _make_frame(
        [float(i) for i in range(20)],
        [f"2024-01-{i + 1:02d} 09:30:00" for i in range(20)],
    )
    train, test = train_test_split_time(frame, test_fraction=0.25)
    assert len(train) + len(test) == len(frame)


def test_train_test_split_invalid_fraction_raises() -> None:
    frame = _make_frame([100.0], ["2024-01-01 09:30:00"])
    with pytest.raises(ValueError):
        train_test_split_time(frame, test_fraction=0.0)
    with pytest.raises(ValueError):
        train_test_split_time(frame, test_fraction=1.0)
    with pytest.raises(ValueError):
        train_test_split_time(frame, test_fraction=1.5)


def test_train_test_split_rejects_insufficient_rows() -> None:
    frame = _make_frame([100.0], ["2024-01-01 09:30:00"])
    with pytest.raises(ValueError, match="at least 2 rows"):
        train_test_split_time(frame, test_fraction=0.5)


def test_train_test_split_rejects_empty_partition() -> None:
    frame = _make_frame([100.0, 101.0], ["2024-01-01 09:30:00", "2024-01-02 09:30:00"])
    with pytest.raises(ValueError, match="non-empty train and test splits"):
        train_test_split_time(frame, test_fraction=0.9)
