"""Cross-cutting primitives: paper references, taxonomy constants, state grids."""

from hft_hmm.core.references import (
    ALL_CATEGORIES,
    ENGINEERING_APPROXIMATION,
    EVALUATION_LAYER,
    PAPER_FAITHFUL,
    PaperReference,
    module_category,
    reference,
)
from hft_hmm.core.state_metadata import StateGrid, default_labels, linear_grid

__all__ = [
    "ALL_CATEGORIES",
    "ENGINEERING_APPROXIMATION",
    "EVALUATION_LAYER",
    "PAPER_FAITHFUL",
    "PaperReference",
    "StateGrid",
    "default_labels",
    "linear_grid",
    "module_category",
    "reference",
]
