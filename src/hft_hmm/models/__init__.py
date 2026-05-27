"""Model-building primitives used by the HMM trading project."""

from hft_hmm.models.gaussian_hmm import GaussianHMMResult, GaussianHMMWrapper
from hft_hmm.models.iohmm_approx import (
    BucketedTransitionConfig,
    BucketedTransitionResult,
    bucket_boundaries_from_quantiles,
    bucket_boundaries_from_spline_grid,
    fit_bucketed_transition_model,
)
from hft_hmm.models.iohmm_continuous import (
    ContinuousIOHMMConfig,
    ContinuousIOHMMResult,
    fit_continuous_iohmm,
    transition_probabilities_at,
)
from hft_hmm.models.plr_baseline import (
    PLRBaselineResult,
    PLRSegment,
    PLRStateSummary,
    fit_piecewise_linear_regression,
)

# Note: ``default_hmm`` is intentionally not re-exported here. It imports
# ``hft_hmm.inference.forward_filter``, which in turn imports
# ``hft_hmm.models.gaussian_hmm`` — eagerly exposing ``default_hmm`` at
# package-import time creates a circular import. Import it directly via
# ``from hft_hmm.models.default_hmm import fit_default_hmm``.
from . import gaussian_hmm, iohmm_approx, iohmm_continuous, plr_baseline

__all__ = [
    "BucketedTransitionConfig",
    "BucketedTransitionResult",
    "ContinuousIOHMMConfig",
    "ContinuousIOHMMResult",
    "GaussianHMMResult",
    "GaussianHMMWrapper",
    "PLRBaselineResult",
    "PLRSegment",
    "PLRStateSummary",
    "bucket_boundaries_from_quantiles",
    "bucket_boundaries_from_spline_grid",
    "fit_continuous_iohmm",
    "fit_bucketed_transition_model",
    "fit_piecewise_linear_regression",
    "gaussian_hmm",
    "iohmm_approx",
    "iohmm_continuous",
    "plr_baseline",
    "transition_probabilities_at",
]
