"""Core package for the HMM trading project."""

from hft_hmm._version import __version__
from hft_hmm.core import (
    ALL_CATEGORIES,
    ENGINEERING_APPROXIMATION,
    EVALUATION_LAYER,
    PAPER_FAITHFUL,
    PaperReference,
    StateGrid,
    default_labels,
    linear_grid,
    module_category,
    reference,
)
from hft_hmm.data import (
    MarketDataSpec,
    MarketDataValidationError,
    load_csv_market_data,
    load_databento_parquet,
    load_yfinance_market_data,
    validate_market_data,
)
from hft_hmm.models import (
    GaussianHMMResult,
    GaussianHMMWrapper,
    PLRBaselineResult,
    PLRSegment,
    PLRStateSummary,
    fit_piecewise_linear_regression,
)
from hft_hmm.preprocessing import compute_log_returns, resample_prices, train_test_split_time
from hft_hmm.project import PROJECT_NAME, ProjectInfo, get_project_info

__all__ = [
    "ALL_CATEGORIES",
    "ENGINEERING_APPROXIMATION",
    "EVALUATION_LAYER",
    "PAPER_FAITHFUL",
    "PROJECT_NAME",
    "GaussianHMMResult",
    "GaussianHMMWrapper",
    "MarketDataSpec",
    "MarketDataValidationError",
    "PaperReference",
    "PLRBaselineResult",
    "PLRSegment",
    "PLRStateSummary",
    "ProjectInfo",
    "StateGrid",
    "__version__",
    "compute_log_returns",
    "default_labels",
    "fit_piecewise_linear_regression",
    "get_project_info",
    "linear_grid",
    "load_csv_market_data",
    "load_databento_parquet",
    "load_yfinance_market_data",
    "module_category",
    "reference",
    "resample_prices",
    "train_test_split_time",
    "validate_market_data",
]
