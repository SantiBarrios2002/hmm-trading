"""Model-building primitives used by the HMM trading project."""

from hft_hmm.models.plr_baseline import (
    PLRBaselineResult,
    PLRSegment,
    PLRStateSummary,
    fit_piecewise_linear_regression,
)

__all__ = [
    "PLRBaselineResult",
    "PLRSegment",
    "PLRStateSummary",
    "fit_piecewise_linear_regression",
]
