"""Core package for the HMM trading project."""

from hft_hmm._version import __version__
from hft_hmm.data import (
    MarketDataSpec,
    MarketDataValidationError,
    load_csv_market_data,
    load_yfinance_market_data,
    validate_market_data,
)
from hft_hmm.preprocessing import compute_log_returns, resample_prices, train_test_split_time
from hft_hmm.project import PROJECT_NAME, ProjectInfo, get_project_info

__all__ = [
    "PROJECT_NAME",
    "ProjectInfo",
    "MarketDataSpec",
    "MarketDataValidationError",
    "__version__",
    "compute_log_returns",
    "get_project_info",
    "load_csv_market_data",
    "load_yfinance_market_data",
    "resample_prices",
    "train_test_split_time",
    "validate_market_data",
]
