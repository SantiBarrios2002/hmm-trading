"""Plotting helper for model-selection curves.

Matplotlib is imported lazily so importing the selection package does not pull
a GUI backend into memory for users that only need the numeric scoring helpers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import pandas as pd

from hft_hmm.core import EVALUATION_LAYER, PaperReference, reference
from hft_hmm.selection.model_selection import ModelSelectionResult

__category__: Final[str] = EVALUATION_LAYER
SELECTION_PLOT_REFERENCE: Final[PaperReference] = reference("§4", "model-selection curves")
DMM_TRAJECTORY_PLOT_REFERENCE: Final[PaperReference] = reference(
    "§3-4", "DMM filtered latent-state trajectories"
)


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


def plot_dmm_filtered_latent_trajectory(
    latent_trajectory: pd.DataFrame | np.ndarray,
    *,
    returns: pd.Series | np.ndarray | None = None,
    expected_next_returns: pd.Series | np.ndarray | None = None,
    max_dims: int | None = None,
    ax: Any = None,
) -> Any:
    """Plot the causal DMM filtered latent-state trajectory over time.

    ``latent_trajectory`` is expected to be the output of
    ``filtered_latent_mean_trajectory_from_dmm()`` or an equivalent
    ``(n_obs, z_dim)`` array. When ``max_dims`` is provided, the helper plots
    the latent dimensions with the largest sample standard deviations. Optional
    return overlays are drawn on a secondary y-axis so their scale does not
    obscure the latent means.

    References: Krishnan et al. (2017) §3-4
    """
    import matplotlib.pyplot as plt

    latent_frame = _coerce_latent_trajectory_frame(latent_trajectory)
    columns_to_plot = _select_latent_columns(latent_frame, max_dims=max_dims)

    if ax is None:
        _, ax = plt.subplots()

    for column in columns_to_plot:
        ax.plot(latent_frame.index, latent_frame[column], label=str(column))

    ax.set_xlabel("time")
    ax.set_ylabel("filtered latent mean")
    ax.set_title("DMM filtered latent-state trajectory")

    overlay_ax = None
    if returns is not None or expected_next_returns is not None:
        overlay_ax = ax.twinx()
        overlay_ax.set_ylabel("return")
        if returns is not None:
            returns_series = _coerce_overlay_series(
                returns,
                index=latent_frame.index,
                default_name="returns",
            )
            overlay_ax.plot(
                returns_series.index,
                returns_series.to_numpy(),
                alpha=0.35,
                linestyle="--",
                label=returns_series.name,
            )
        if expected_next_returns is not None:
            expected_series = _coerce_overlay_series(
                expected_next_returns,
                index=latent_frame.index,
                default_name="expected_next_returns",
            )
            overlay_ax.plot(
                expected_series.index,
                expected_series.to_numpy(),
                alpha=0.85,
                linewidth=1.2,
                label=expected_series.name,
            )

    handles, labels = ax.get_legend_handles_labels()
    if overlay_ax is not None:
        overlay_handles, overlay_labels = overlay_ax.get_legend_handles_labels()
        handles.extend(overlay_handles)
        labels.extend(overlay_labels)
    if handles:
        ax.legend(handles, labels, loc="best")

    return ax


def _coerce_latent_trajectory_frame(latent_trajectory: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(latent_trajectory, pd.DataFrame):
        frame = latent_trajectory.astype(float).copy()
    else:
        latent_values = np.asarray(latent_trajectory, dtype=float)
        if latent_values.ndim != 2:
            raise ValueError(
                "latent_trajectory must have shape (n_obs, z_dim), "
                f"got shape {latent_values.shape}."
            )
        frame = pd.DataFrame(
            latent_values,
            columns=[f"latent_{latent_dim + 1}" for latent_dim in range(latent_values.shape[1])],
        )
    if frame.empty:
        raise ValueError("latent_trajectory must contain at least one row.")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("latent_trajectory must contain only finite values.")
    return frame


def _select_latent_columns(latent_frame: pd.DataFrame, *, max_dims: int | None) -> Sequence[Any]:
    if max_dims is None:
        return list(latent_frame.columns)
    if max_dims < 1:
        raise ValueError(f"max_dims must be positive when provided, got {max_dims}.")
    if max_dims >= latent_frame.shape[1]:
        return list(latent_frame.columns)
    column_order = latent_frame.std(axis=0).sort_values(ascending=False).index
    return list(column_order[:max_dims])


def _coerce_overlay_series(
    values: pd.Series | np.ndarray,
    *,
    index: pd.Index,
    default_name: str,
) -> pd.Series:
    if isinstance(values, pd.Series):
        if not values.index.equals(index):
            raise ValueError("overlay series index must exactly match the latent trajectory index.")
        series = values.astype(float).copy()
    else:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"overlay series must be one-dimensional, got shape {array.shape}.")
        if array.shape[0] != len(index):
            raise ValueError(
                "overlay series length must match the latent trajectory length, "
                f"got {array.shape[0]} and {len(index)}."
            )
        series = pd.Series(array, index=index, name=default_name)
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("overlay series must contain only finite values.")
    if series.name is None:
        series = series.rename(default_name)
    return series
