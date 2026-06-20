"""Model-selection utilities for HMM experiments."""

from hft_hmm.selection.cross_validation import (
    CROSS_VALIDATION_REFERENCE,
    CrossValidationFoldScore,
    CrossValidationResult,
    CrossValidationRow,
    select_k_by_cv,
)
from hft_hmm.selection.model_selection import (
    ModelSelectionResult,
    ModelSelectionRow,
    aic,
    bic,
    compare_state_counts,
    count_gaussian_hmm_parameters,
)
from hft_hmm.selection.plots import (
    plot_dmm_filtered_latent_trajectory,
    plot_selection_curves,
)

__all__ = [
    "CROSS_VALIDATION_REFERENCE",
    "CrossValidationFoldScore",
    "CrossValidationResult",
    "CrossValidationRow",
    "ModelSelectionResult",
    "ModelSelectionRow",
    "aic",
    "bic",
    "compare_state_counts",
    "count_gaussian_hmm_parameters",
    "plot_dmm_filtered_latent_trajectory",
    "plot_selection_curves",
    "select_k_by_cv",
]
