"""Shared data loading and reproducibility helpers for experiment runners.

Both baseline HMM experiments and standalone predictor experiments load market
data through the same source contract, resampling, and return preprocessing.
Keeping that path here prevents drift between comparable experiment runners.

References: §4.4 reproducible simulation artifacts (evaluation layer)
"""

from __future__ import annotations

import warnings
from typing import Final, Protocol

import pandas as pd

from hft_hmm.config.experiment_config import DataSourceConfig, compute_file_sha256
from hft_hmm.data import (
    load_csv_market_data,
    load_databento_parquet,
    load_yfinance_market_data,
)
from hft_hmm.preprocessing import compute_log_returns, resample_prices

NON_REPRODUCIBLE_WARNING: Final[str] = (
    "yfinance data may drift across vendor updates; re-runs may not match bit-for-bit."
)
DATA_FINGERPRINT_MISMATCH_WARNING: Final[str] = (
    "Configured sha256 does not match file contents at {path}; "
    "the run will be marked non-reproducible."
)

_YFINANCE_INTERVAL: Final[dict[str, str]] = {
    "1min": "1m",
    "5min": "5m",
    "1D": "1d",
}


class ReproducibleExperimentConfig(Protocol):
    """Minimal config protocol needed for reproducibility validation."""

    @property
    def data(self) -> DataSourceConfig: ...

    @property
    def sha256(self) -> str | None: ...

    @property
    def is_reproducible(self) -> bool: ...


def load_returns_from_source(data: DataSourceConfig, *, frequency: str) -> pd.Series:
    """Load raw market data, resample, and return tz-aware log returns.

    ``data.kind`` selects the source loader; all paths then route through the
    same resampling and log-return preprocessing. The output name is always
    ``"log_return"``.

    References: §4.4 reproducible simulation artifacts (evaluation layer)
    """
    if frequency not in _YFINANCE_INTERVAL:
        raise ValueError(
            f"frequency must be one of {tuple(_YFINANCE_INTERVAL)}, got {frequency!r}."
        )

    if data.kind == "csv":
        assert data.path is not None  # validated in DataSourceConfig.__post_init__
        frame = load_csv_market_data(data.path)
    elif data.kind == "databento_parquet":
        assert data.path is not None
        assert data.symbol is not None
        frame = load_databento_parquet(data.path, symbol=data.symbol)
    else:
        assert data.symbol is not None
        assert data.start is not None and data.end is not None
        frame = load_yfinance_market_data(
            data.symbol,
            start=data.start,
            end=data.end,
            interval=_YFINANCE_INTERVAL[frequency],
        )

    resampled = resample_prices(frame, freq=frequency)
    prices = resampled.set_index("timestamp")["price"]
    returns = compute_log_returns(prices).dropna()
    returns.name = "log_return"
    return returns


def validate_data_reproducibility(
    config: ReproducibleExperimentConfig,
    *,
    stacklevel: int = 2,
) -> bool:
    """Return whether a config is reproducible, emitting standard warnings."""
    if not config.is_reproducible:
        warnings.warn(NON_REPRODUCIBLE_WARNING, UserWarning, stacklevel=stacklevel)
        return False

    assert config.data.path is not None
    assert config.sha256 is not None
    actual_sha256 = compute_file_sha256(config.data.path)
    if actual_sha256 != config.sha256:
        warnings.warn(
            DATA_FINGERPRINT_MISMATCH_WARNING.format(path=config.data.path),
            UserWarning,
            stacklevel=stacklevel,
        )
        return False
    return True
