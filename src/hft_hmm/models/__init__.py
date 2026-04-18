"""Model-building primitives used by the HMM trading project."""

from hft_hmm.models.gaussian_hmm import GaussianHMMResult, GaussianHMMWrapper
from hft_hmm.models.plr_baseline import (
    PLRBaselineResult,
    PLRSegment,
    PLRStateSummary,
    fit_piecewise_linear_regression,
)

from . import gaussian_hmm, plr_baseline

__all__ = [
    "GaussianHMMResult",
    "GaussianHMMWrapper",
    "PLRBaselineResult",
    "PLRSegment",
    "PLRStateSummary",
    "fit_piecewise_linear_regression",
    "gaussian_hmm",
    "plr_baseline",
]
