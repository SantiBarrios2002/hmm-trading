"""Data loading and validation utilities for market time series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

REQUIRED_COLUMNS: Final[tuple[str, str]] = ("timestamp", "price")
OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ("volume",)
CANONICAL_COLUMNS: Final[tuple[str, ...]] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


class MarketDataValidationError(ValueError):
    """Raised when raw market data does not satisfy the repository contract."""


@dataclass(frozen=True)
class MarketDataSpec:
    """Typed contract for downstream modules that consume price data."""

    timestamp_column: str = "timestamp"
    price_column: str = "price"
    volume_column: str = "volume"


def load_csv_market_data(path: str | Path, spec: MarketDataSpec | None = None) -> pd.DataFrame:
    """Load a CSV file and normalize it to the repository market-data contract."""

    frame = pd.read_csv(path)
    return validate_market_data(frame, spec=spec)


def load_yfinance_market_data(
    symbol: str,
    *,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
    spec: MarketDataSpec | None = None,
) -> pd.DataFrame:
    """Fetch market data from Yahoo Finance and normalize the output schema.

    When ``start`` and ``end`` are both ``None``, ``yfinance.Ticker.history`` falls
    back to ``period="1mo"``, so an unbounded call still returns roughly one month
    of data. With ``auto_adjust=False`` and both ``Close`` and ``Adj Close`` present,
    the adjusted series is mapped to the normalized ``price`` column; otherwise the
    raw ``Close`` series is used. When a ``volume`` column is present, callers should
    ensure it is numeric and pre-clean or drop rows with missing values before
    normalization.
    """

    import yfinance as yf

    downloaded = yf.Ticker(symbol).history(
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
    )
    if downloaded.empty:
        raise MarketDataValidationError(f"No market data returned for symbol '{symbol}'.")

    frame = downloaded.reset_index()
    rename_map: dict[str, str] = {
        "Date": "timestamp",
        "Datetime": "timestamp",
        "Volume": "volume",
    }

    # Avoid duplicate "price" columns when both Close and Adj Close are present.
    has_adj_close = "Adj Close" in frame.columns
    if not auto_adjust and has_adj_close:
        rename_map["Adj Close"] = "price"
    else:
        rename_map["Close"] = "price"

    renamed = frame.rename(columns=rename_map)
    return validate_market_data(renamed, spec=spec)


def validate_market_data(
    frame: pd.DataFrame,
    spec: MarketDataSpec | None = None,
) -> pd.DataFrame:
    """Validate and normalize raw market data into canonical repository columns.

    The normalized DataFrame requires numeric ``price`` values and, when the
    ``volume`` column is meaningfully provided, numeric ``volume`` values as well.
    If the ``volume`` column exists but is entirely missing, it is treated as not
    provided and dropped from the normalized DataFrame. Mixed numeric and missing
    ``volume`` values raise ``MarketDataValidationError``; callers should pre-clean
    or drop rows with missing ``volume`` entries before calling this function.
    """

    resolved_spec = spec or MarketDataSpec()
    source_columns = [
        resolved_spec.timestamp_column,
        resolved_spec.price_column,
        resolved_spec.volume_column,
    ]
    if len(set(source_columns)) != len(source_columns):
        raise MarketDataValidationError("MarketDataSpec columns must be distinct.")

    canonical_to_source = dict(zip(CANONICAL_COLUMNS, source_columns, strict=True))
    incoming_columns = set(frame.columns)
    canonical_set = set(CANONICAL_COLUMNS)
    conflicting_canonical_columns = sorted(
        canonical
        for canonical in incoming_columns & canonical_set
        if canonical_to_source[canonical] != canonical
    )
    if conflicting_canonical_columns:
        joined = ", ".join(conflicting_canonical_columns)
        raise MarketDataValidationError(
            "Input contains canonical columns that would collide during renaming: " f"{joined}"
        )

    renamed = frame.rename(
        columns={
            resolved_spec.timestamp_column: "timestamp",
            resolved_spec.price_column: "price",
            resolved_spec.volume_column: "volume",
        }
    ).copy()

    missing = [column for column in REQUIRED_COLUMNS if column not in renamed.columns]
    if missing:
        joined = ", ".join(missing)
        raise MarketDataValidationError(f"Missing required columns: {joined}")

    renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], utc=True, errors="coerce")
    if renamed["timestamp"].isna().any():
        raise MarketDataValidationError("Invalid timestamp values detected.")

    if renamed["timestamp"].duplicated().any():
        raise MarketDataValidationError("Duplicate timestamps are not allowed.")

    renamed["price"] = pd.to_numeric(renamed["price"], errors="coerce")
    if renamed["price"].isna().any():
        raise MarketDataValidationError("Price column must contain numeric values.")

    if (renamed["price"] <= 0).any():
        raise MarketDataValidationError("Price column must contain strictly positive values.")

    selected_columns = [column for column in CANONICAL_COLUMNS if column in renamed.columns]
    normalized = renamed.loc[:, selected_columns].sort_values("timestamp").reset_index(drop=True)

    if "volume" in normalized.columns:
        if normalized["volume"].isna().all():
            normalized = normalized.drop(columns="volume")
            return normalized

        numeric_volume = pd.to_numeric(normalized["volume"], errors="coerce")
        if numeric_volume.isna().any():
            raise MarketDataValidationError(
                "Volume column must contain numeric values when provided."
            )
        normalized["volume"] = numeric_volume

    return normalized
