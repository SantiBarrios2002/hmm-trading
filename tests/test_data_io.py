"""Tests for the market-data loading contract."""

from pathlib import Path

import pandas as pd
import pytest

from hft_hmm.data import (
    MarketDataSpec,
    MarketDataValidationError,
    load_csv_market_data,
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
