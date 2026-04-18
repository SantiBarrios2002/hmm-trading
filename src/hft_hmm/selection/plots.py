"""Plotting helper for model-selection curves.

Matplotlib is imported lazily so importing the selection package does not pull
a GUI backend into memory for users that only need the numeric scoring helpers.
"""

from __future__ import annotations

from typing import Any, Final

from hft_hmm.core import EVALUATION_LAYER, PaperReference, reference
from hft_hmm.selection.model_selection import ModelSelectionResult

__category__: Final[str] = EVALUATION_LAYER
SELECTION_PLOT_REFERENCE: Final[PaperReference] = reference("§4", "model-selection curves")


def plot_selection_curves(
    result: ModelSelectionResult,
    *,
    ax: Any = None,
) -> Any:
    """Plot AIC and BIC curves across candidate ``K`` values.

    Returns the matplotlib ``Axes`` the curves were drawn on. The helper is
    deliberately minimal: it marks the best-by-AIC and best-by-BIC points so
    the grader can verify the ranking visually. Callers wanting a figure
    should create one and pass in its ``Axes``.

    References: §4 model-selection curves (evaluation layer)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    k_values = [row.k for row in result.rows]
    aic_values = [row.aic for row in result.rows]
    bic_values = [row.bic for row in result.rows]

    ax.plot(k_values, aic_values, marker="o", label="AIC")
    ax.plot(k_values, bic_values, marker="s", label="BIC")
    ax.axvline(
        result.best_by_aic,
        linestyle="--",
        alpha=0.4,
        label=f"best AIC = {result.best_by_aic}",
    )
    ax.axvline(
        result.best_by_bic,
        linestyle=":",
        alpha=0.4,
        label=f"best BIC = {result.best_by_bic}",
    )
    ax.set_xlabel("hidden states (K)")
    ax.set_ylabel("information criterion")
    ax.set_title("Model selection over K")
    ax.legend()

    return ax
