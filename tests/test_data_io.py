"""Tests for the market-data loading contract."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from hft_hmm.data import (
    MarketDataSpec,
    MarketDataValidationError,
    load_csv_market_data,
    load_yfinance_market_data,
    validate_market_data,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_csv_market_data_normalizes_fixture() -> None:
    frame = load_csv_market_data(FIXTURES_DIR / "sample_prices.csv")

    assert list(frame.columns) == ["timestamp", "price", "volume"]
    assert str(frame["timestamp"].dtype).startswith("datetime64[")
    assert str(frame["timestamp"].dtype).endswith(", UTC]")
    assert frame["price"].tolist() == [100.0, 100.4, 100.1]


def test_validate_market_data_supports_custom_column_names() -> None:
    raw = pd.DataFrame(
        {
            "time": ["2024-01-02 09:30:00", "2024-01-02 09:35:00"],
            "close": [100.0, 101.0],
            "shares": [10, 20],
        }
    )

    frame = validate_market_data(
        raw,
        spec=MarketDataSpec(timestamp_column="time", price_column="close", volume_column="shares"),
    )

    assert list(frame.columns) == ["timestamp", "price", "volume"]
    assert frame["volume"].tolist() == [10, 20]


def test_validate_market_data_rejects_missing_required_columns() -> None:
    raw = pd.DataFrame({"timestamp": ["2024-01-02 09:30:00"]})

    with pytest.raises(MarketDataValidationError, match="Missing required columns: price"):
        validate_market_data(raw)


def test_validate_market_data_rejects_bad_timestamps() -> None:
    raw = pd.DataFrame({"timestamp": ["not-a-date"], "price": [100.0]})

    with pytest.raises(MarketDataValidationError, match="Invalid timestamp values detected."):
        validate_market_data(raw)


def test_validate_market_data_rejects_duplicate_timestamps() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-02 09:30:00", "2024-01-02 09:30:00"],
            "price": [100.0, 100.2],
        }
    )

    with pytest.raises(MarketDataValidationError, match="Duplicate timestamps are not allowed."):
        validate_market_data(raw)


def test_validate_market_data_rejects_non_numeric_price() -> None:
    raw = pd.DataFrame({"timestamp": ["2024-01-02 09:30:00"], "price": ["bad"]})

    with pytest.raises(
        MarketDataValidationError,
        match="Price column must contain numeric values.",
    ):
        validate_market_data(raw)


def test_validate_market_data_rejects_zero_or_negative_price() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-02 09:30:00", "2024-01-02 09:35:00"],
            "price": [100.0, 0.0],
        }
    )

    with pytest.raises(
        MarketDataValidationError,
        match="Price column must contain strictly positive values.",
    ):
        validate_market_data(raw)


def test_validate_market_data_sorts_by_timestamp() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-02 09:35:00", "2024-01-02 09:30:00"],
            "price": [101.0, 100.0],
        }
    )

    frame = validate_market_data(raw)
    assert frame["price"].tolist() == [100.0, 101.0]


def test_validate_market_data_rejects_non_numeric_volume_when_provided() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-02 09:30:00"],
            "price": [100.0],
            "volume": ["bad"],
        }
    )

    with pytest.raises(
        MarketDataValidationError,
        match="Volume column must contain numeric values when provided.",
    ):
        validate_market_data(raw)


def test_validate_market_data_drops_volume_when_it_is_entirely_missing() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-02 09:30:00", "2024-01-02 09:35:00"],
            "price": [100.0, 101.0],
            "volume": [None, None],
        }
    )

    frame = validate_market_data(raw)

    assert list(frame.columns) == ["timestamp", "price"]


def test_validate_market_data_rejects_mixed_numeric_and_missing_volume() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-02 09:30:00", "2024-01-02 09:35:00"],
            "price": [100.0, 101.0],
            "volume": [10, None],
        }
    )

    with pytest.raises(
        MarketDataValidationError,
        match="Volume column must contain numeric values when provided.",
    ):
        validate_market_data(raw)


def test_validate_market_data_rejects_canonical_column_collisions() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-02 09:30:00"],
            "close": [100.0],
            "price": [999.0],
        }
    )

    with pytest.raises(
        MarketDataValidationError,
        match="would collide during renaming: price",
    ):
        validate_market_data(raw, spec=MarketDataSpec(price_column="close"))


def test_validate_market_data_rejects_cross_canonical_collision() -> None:
    raw = pd.DataFrame(
        {
            "time": ["2024-01-02 09:30:00"],
            "price": [100.0],
            "timestamp": [123],
        }
    )

    with pytest.raises(
        MarketDataValidationError,
        match="would collide during renaming: timestamp",
    ):
        validate_market_data(
            raw,
            spec=MarketDataSpec(
                timestamp_column="time",
                price_column="price",
                volume_column="timestamp",
            ),
        )


def test_load_yfinance_market_data_normalizes_history_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_frame = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "Volume": [10, 20],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True),
    )
    history_frame.index.name = "Date"

    captured: dict[str, object] = {}

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            captured["symbol"] = symbol

        def history(
            self,
            *,
            start: str | None,
            end: str | None,
            interval: str,
            auto_adjust: bool,
        ) -> pd.DataFrame:
            captured["start"] = start
            captured["end"] = end
            captured["interval"] = interval
            captured["auto_adjust"] = auto_adjust
            return history_frame

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=FakeTicker))

    frame = load_yfinance_market_data("SPY", start="2024-01-02", end="2024-01-04")

    assert captured["symbol"] == "SPY"
    assert captured["start"] == "2024-01-02"
    assert captured["end"] == "2024-01-04"
    assert captured["interval"] == "1d"
    assert captured["auto_adjust"] is True
    assert list(frame.columns) == ["timestamp", "price", "volume"]
    assert frame["price"].tolist() == [100.0, 101.0]


def test_load_yfinance_market_data_prefers_adj_close_when_auto_adjust_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_frame = pd.DataFrame(
        {
            "Close": [200.0, 201.0],
            "Adj Close": [100.0, 101.0],
            "Volume": [10, 20],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True),
    )
    history_frame.index.name = "Date"

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(
            self,
            *,
            start: str | None,
            end: str | None,
            interval: str,
            auto_adjust: bool,
        ) -> pd.DataFrame:
            assert auto_adjust is False
            return history_frame

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=FakeTicker))

    frame = load_yfinance_market_data(
        "SPY",
        start="2024-01-02",
        end="2024-01-04",
        auto_adjust=False,
    )

    assert list(frame.columns) == ["timestamp", "price", "volume"]
    assert frame["price"].tolist() == [100.0, 101.0]
