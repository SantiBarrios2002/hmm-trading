"""Core package for the HMM trading project."""

from hft_hmm._version import __version__
from hft_hmm.data import (
    MarketDataSpec,
    MarketDataValidationError,
    load_csv_market_data,
    load_yfinance_market_data,
    validate_market_data,
)
from hft_hmm.project import PROJECT_NAME, ProjectInfo, get_project_info

__all__ = [
    "PROJECT_NAME",
    "ProjectInfo",
    "MarketDataSpec",
    "MarketDataValidationError",
    "__version__",
    "get_project_info",
    "load_csv_market_data",
    "load_yfinance_market_data",
    "validate_market_data",
]
