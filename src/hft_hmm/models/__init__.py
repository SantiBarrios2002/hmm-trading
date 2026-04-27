"""Model-building primitives used by the HMM trading project."""

from hft_hmm.models.gaussian_hmm import GaussianHMMResult, GaussianHMMWrapper
from hft_hmm.models.iohmm_approx import (
    BucketedTransitionConfig,
    BucketedTransitionResult,
    bucket_boundaries_from_spline_grid,
    fit_bucketed_transition_model,
)
from hft_hmm.models.plr_baseline import (
    PLRBaselineResult,
    PLRSegment,
    PLRStateSummary,
    fit_piecewise_linear_regression,
)

from . import gaussian_hmm, iohmm_approx, plr_baseline

__all__ = [
    "BucketedTransitionConfig",
    "BucketedTransitionResult",
    "GaussianHMMResult",
    "GaussianHMMWrapper",
    "PLRBaselineResult",
    "PLRSegment",
    "PLRStateSummary",
    "bucket_boundaries_from_spline_grid",
    "fit_bucketed_transition_model",
    "fit_piecewise_linear_regression",
    "gaussian_hmm",
    "iohmm_approx",
    "plr_baseline",
]
