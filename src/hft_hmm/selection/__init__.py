"""Model-selection utilities for HMM experiments."""

from hft_hmm.selection.model_selection import (
    ModelSelectionResult,
    ModelSelectionRow,
    aic,
    bic,
    compare_state_counts,
    count_gaussian_hmm_parameters,
)
from hft_hmm.selection.plots import plot_selection_curves

__all__ = [
    "ModelSelectionResult",
    "ModelSelectionRow",
    "aic",
    "bic",
    "compare_state_counts",
    "count_gaussian_hmm_parameters",
    "plot_selection_curves",
]
