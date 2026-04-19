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
from hft_hmm.inference import ForwardFilterResult, filter_from_result, forward_filter
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
from hft_hmm.selection import (
    ModelSelectionResult,
    ModelSelectionRow,
    aic,
    bic,
    compare_state_counts,
    count_gaussian_hmm_parameters,
    plot_selection_curves,
)
from hft_hmm.strategy import (
    SIGNAL_REFERENCE,
    align_signal_with_future_return,
    sign_signal,
    signal_from_filter_result,
    thresholded_signal,
)

__all__ = [
    "ALL_CATEGORIES",
    "ENGINEERING_APPROXIMATION",
    "EVALUATION_LAYER",
    "PAPER_FAITHFUL",
    "PROJECT_NAME",
    "SIGNAL_REFERENCE",
    "ForwardFilterResult",
    "GaussianHMMResult",
    "GaussianHMMWrapper",
    "MarketDataSpec",
    "MarketDataValidationError",
    "ModelSelectionResult",
    "ModelSelectionRow",
    "PaperReference",
    "PLRBaselineResult",
    "PLRSegment",
    "PLRStateSummary",
    "ProjectInfo",
    "StateGrid",
    "__version__",
    "aic",
    "align_signal_with_future_return",
    "bic",
    "compare_state_counts",
    "compute_log_returns",
    "count_gaussian_hmm_parameters",
    "default_labels",
    "fit_piecewise_linear_regression",
    "filter_from_result",
    "forward_filter",
    "get_project_info",
    "linear_grid",
    "load_csv_market_data",
    "load_databento_parquet",
    "load_yfinance_market_data",
    "module_category",
    "plot_selection_curves",
    "reference",
    "resample_prices",
    "sign_signal",
    "signal_from_filter_result",
    "thresholded_signal",
    "train_test_split_time",
    "validate_market_data",
]
