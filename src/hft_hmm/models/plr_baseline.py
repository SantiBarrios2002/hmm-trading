"""Piecewise linear regression baseline used for HMM initialization experiments.

This module implements a deterministic top-down segmentation baseline inspired
by the paper's PLR initialization idea. The implementation is an engineering
approximation rather than a literal reproduction of the paper's full procedure.
See ``PLR_BASELINE_REFERENCE`` for the paper pointer used in docstrings and
review notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Final

import numpy as np
import pandas as pd
from statsmodels.stats.stattools import durbin_watson

from hft_hmm.core import (
    ENGINEERING_APPROXIMATION,
    PaperReference,
    StateGrid,
    default_labels,
    reference,
)

__category__: Final[str] = ENGINEERING_APPROXIMATION
PLR_BASELINE_REFERENCE: Final[PaperReference] = reference("§3.1", "PLR baseline")


@dataclass(frozen=True)
class PLRSegment:
    """Chronological segment from a piecewise linear fit."""

    segment_index: int
    state_index: int
    start_idx: int
    end_idx: int
    slope: float
    intercept: float
    residual_variance: float

    def __post_init__(self) -> None:
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative.")
        if self.state_index < 0:
            raise ValueError("state_index must be non-negative.")
        if self.start_idx < 0:
            raise ValueError("start_idx must be non-negative.")
        if self.end_idx <= self.start_idx:
            raise ValueError("end_idx must be strictly greater than start_idx.")
        if self.residual_variance < 0.0:
            raise ValueError("residual_variance must be non-negative.")

    @property
    def n_obs(self) -> int:
        """Return the number of observations assigned to the segment."""
        return self.end_idx - self.start_idx


@dataclass(frozen=True)
class PLRStateSummary:
    """State-level summary derived from one chronological PLR segment."""

    state_index: int
    label: str
    segment_index: int
    start_idx: int
    end_idx: int
    slope: float
    residual_variance: float

    def __post_init__(self) -> None:
        if self.state_index < 0:
            raise ValueError("state_index must be non-negative.")
        if not self.label:
            raise ValueError("label must be non-empty.")
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative.")
        if self.end_idx <= self.start_idx:
            raise ValueError("end_idx must be strictly greater than start_idx.")
        if self.residual_variance < 0.0:
            raise ValueError("residual_variance must be non-negative.")

    @property
    def n_obs(self) -> int:
        """Return the number of observations assigned to the summarized state."""
        return self.end_idx - self.start_idx


@dataclass(frozen=True)
class PLRBaselineResult:
    """Immutable result of ``fit_piecewise_linear_regression``."""

    segments: tuple[PLRSegment, ...]
    state_summaries: tuple[PLRStateSummary, ...]
    state_grid: StateGrid
    breakpoints: tuple[int, ...]
    fitted_values: np.ndarray
    residuals: np.ndarray
    segment_assignments: np.ndarray
    state_sequence: np.ndarray
    durbin_watson: float | None = None

    def __post_init__(self) -> None:
        fitted = np.asarray(self.fitted_values, dtype=float).copy()
        residuals = np.asarray(self.residuals, dtype=float).copy()
        segment_assignments = np.asarray(self.segment_assignments, dtype=int).copy()
        state_sequence = np.asarray(self.state_sequence, dtype=int).copy()

        n_obs = fitted.shape[0]
        if fitted.ndim != 1:
            raise ValueError("fitted_values must be one-dimensional.")
        if residuals.shape != (n_obs,):
            raise ValueError("residuals must have the same shape as fitted_values.")
        if segment_assignments.shape != (n_obs,):
            raise ValueError("segment_assignments must have the same shape as fitted_values.")
        if state_sequence.shape != (n_obs,):
            raise ValueError("state_sequence must have the same shape as fitted_values.")
        if len(self.breakpoints) != len(self.segments):
            raise ValueError("breakpoints must align one-to-one with segments.")
        if self.breakpoints[-1] != n_obs:
            raise ValueError("the final breakpoint must equal the number of observations.")
        if len(self.state_summaries) != self.state_grid.k:
            raise ValueError("state_summaries must align with state_grid.k.")

        fitted.setflags(write=False)
        residuals.setflags(write=False)
        segment_assignments.setflags(write=False)
        state_sequence.setflags(write=False)

        object.__setattr__(self, "fitted_values", fitted)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "segment_assignments", segment_assignments)
        object.__setattr__(self, "state_sequence", state_sequence)

    @property
    def state_means(self) -> np.ndarray:
        """Return the ordered state-level slope proxies used for initialization."""
        return self.state_grid.means

    @property
    def state_variances(self) -> np.ndarray:
        """Return the ordered state-level residual variances."""
        variances = np.array(
            [summary.residual_variance for summary in self.state_summaries],
            dtype=float,
        )
        variances.setflags(write=False)
        return variances


@dataclass(frozen=True)
class _IntervalFit:
    slope: float
    intercept: float
    sse: float


def fit_piecewise_linear_regression(
    series: pd.Series | np.ndarray,
    n_segments: int,
    *,
    min_segment_length: int = 2,
    compute_durbin_watson: bool = False,
) -> PLRBaselineResult:
    """Fit a deterministic piecewise linear regression baseline to ``series``.

    The series is segmented chronologically via dynamic programming that
    minimizes total squared error under the ``min_segment_length`` constraint.
    The 2D DP tables ``cost`` and ``previous_start`` store optimal partial
    solutions and allow exact backtracking of globally optimal segment
    boundaries. Segment slopes and residual variances are exposed both
    chronologically and in return-ordered state space, so the result can seed
    later HMM initialization work. The DP routine runs in roughly
    :math:`O(k * n^2)` time for ``k = n_segments`` and ``n = len(series)``.

    References: §3.1 PLR baseline (engineering approximation)
    """

    values = _coerce_series(series)
    if n_segments < 2:
        raise ValueError(f"n_segments must be at least 2, got {n_segments}.")
    if min_segment_length < 2:
        raise ValueError(f"min_segment_length must be at least 2, got {min_segment_length}.")
    if len(values) < n_segments * min_segment_length:
        raise ValueError(
            "series is too short for the requested segmentation; "
            f"len(series)={len(values)}, n_segments={n_segments}, "
            f"min_segment_length={min_segment_length}."
        )

    n_obs = len(values)
    x = np.arange(n_obs, dtype=float)
    prefix_x = _prefix_sum(x)
    prefix_xx = _prefix_sum(x * x)
    prefix_y = _prefix_sum(values)
    prefix_yy = _prefix_sum(values * values)
    prefix_xy = _prefix_sum(x * values)

    @cache
    def interval_fit(start_idx: int, end_idx: int) -> _IntervalFit:
        n_interval = end_idx - start_idx
        sum_x = prefix_x[end_idx] - prefix_x[start_idx]
        sum_y = prefix_y[end_idx] - prefix_y[start_idx]
        sum_xx = prefix_xx[end_idx] - prefix_xx[start_idx]
        sum_yy = prefix_yy[end_idx] - prefix_yy[start_idx]
        sum_xy = prefix_xy[end_idx] - prefix_xy[start_idx]

        if n_interval == 1:
            intercept = float(values[start_idx])
            return _IntervalFit(slope=0.0, intercept=intercept, sse=0.0)

        denominator = n_interval * sum_xx - sum_x * sum_x
        if np.isclose(denominator, 0.0):
            slope = 0.0
            intercept = sum_y / n_interval
        else:
            slope = (n_interval * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n_interval

        sse = (
            sum_yy
            - 2.0 * intercept * sum_y
            - 2.0 * slope * sum_xy
            + n_interval * intercept * intercept
            + 2.0 * intercept * slope * sum_x
            + slope * slope * sum_xx
        )
        return _IntervalFit(
            slope=float(slope),
            intercept=float(intercept),
            sse=float(max(sse, 0.0)),
        )

    cost = np.full((n_segments + 1, n_obs + 1), np.inf, dtype=float)
    previous_start = np.full((n_segments + 1, n_obs + 1), -1, dtype=int)
    cost[0, 0] = 0.0

    for segment_count in range(1, n_segments + 1):
        min_end = segment_count * min_segment_length
        max_end = n_obs - (n_segments - segment_count) * min_segment_length
        for end_idx in range(min_end, max_end + 1):
            start_min = (segment_count - 1) * min_segment_length
            start_max = end_idx - min_segment_length
            best_cost = np.inf
            best_start = -1

            for start_idx in range(start_min, start_max + 1):
                previous_cost = cost[segment_count - 1, start_idx]
                if not np.isfinite(previous_cost):
                    continue
                candidate_cost = previous_cost + interval_fit(start_idx, end_idx).sse
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_start = start_idx

            cost[segment_count, end_idx] = best_cost
            previous_start[segment_count, end_idx] = best_start

    if not np.isfinite(cost[n_segments, n_obs]):
        raise ValueError(
            "Unable to segment the series with the requested configuration; "
            f"n_segments={n_segments}, min_segment_length={min_segment_length}."
        )

    segment_ranges: list[tuple[int, int]] = []
    end_idx = n_obs
    for segment_count in range(n_segments, 0, -1):
        start_idx = previous_start[segment_count, end_idx]
        if start_idx < 0:
            raise ValueError("Segmentation backtracking failed to recover segment boundaries.")
        segment_ranges.append((start_idx, end_idx))
        end_idx = start_idx
    segment_ranges.reverse()

    breakpoints = tuple(bp_end for _, bp_end in segment_ranges)
    slopes = np.array(
        [interval_fit(start_idx, end_idx).slope for start_idx, end_idx in segment_ranges],
        dtype=float,
    )
    order = np.argsort(slopes, kind="mergesort")
    labels = default_labels(n_segments)
    state_index_by_segment = {
        segment_index: state_index for state_index, segment_index in enumerate(order.tolist())
    }

    segments: list[PLRSegment] = []
    state_summaries: list[PLRStateSummary] = []
    fitted_values = np.empty(n_obs, dtype=float)
    residuals = np.empty(n_obs, dtype=float)
    segment_assignments = np.empty(n_obs, dtype=int)
    state_sequence = np.empty(n_obs, dtype=int)

    for segment_index, (start_idx, end_idx) in enumerate(segment_ranges):
        fit = interval_fit(start_idx, end_idx)
        state_index = state_index_by_segment[segment_index]
        x_segment = x[start_idx:end_idx]
        fitted_segment = fit.slope * x_segment + fit.intercept
        segment_residuals = values[start_idx:end_idx] - fitted_segment
        residual_variance = float(np.mean(segment_residuals**2))

        fitted_values[start_idx:end_idx] = fitted_segment
        residuals[start_idx:end_idx] = segment_residuals
        segment_assignments[start_idx:end_idx] = segment_index
        state_sequence[start_idx:end_idx] = state_index

        segments.append(
            PLRSegment(
                segment_index=segment_index,
                state_index=state_index,
                start_idx=start_idx,
                end_idx=end_idx,
                slope=fit.slope,
                intercept=fit.intercept,
                residual_variance=residual_variance,
            )
        )

    ordered_segments = [segments[segment_index] for segment_index in order.tolist()]
    state_grid = StateGrid(
        k=n_segments,
        means=np.array([segment.slope for segment in ordered_segments], dtype=float),
        labels=labels,
    )

    for state_index, segment in enumerate(ordered_segments):
        state_summaries.append(
            PLRStateSummary(
                state_index=state_index,
                label=labels[state_index],
                segment_index=segment.segment_index,
                start_idx=segment.start_idx,
                end_idx=segment.end_idx,
                slope=segment.slope,
                residual_variance=segment.residual_variance,
            )
        )

    dw_stat = float(durbin_watson(residuals)) if compute_durbin_watson else None

    return PLRBaselineResult(
        segments=tuple(segments),
        state_summaries=tuple(state_summaries),
        state_grid=state_grid,
        breakpoints=breakpoints,
        fitted_values=fitted_values,
        residuals=residuals,
        segment_assignments=segment_assignments,
        state_sequence=state_sequence,
        durbin_watson=dw_stat,
    )


def _coerce_series(series: pd.Series | np.ndarray) -> np.ndarray:
    """Coerce input values to a finite one-dimensional float array.

    If ``series`` is a pandas ``Series`` with a ``DatetimeIndex``, the index
    must be monotonic increasing to preserve chronological order for
    segmentation.
    """
    if (
        isinstance(series, pd.Series)
        and isinstance(series.index, pd.DatetimeIndex)
        and not series.index.is_monotonic_increasing
    ):
        raise ValueError(
            "series DatetimeIndex must be monotonic increasing; "
            "sort with series.sort_index() before calling fit_piecewise_linear_regression."
        )

    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"series must be one-dimensional, got shape {values.shape}.")
    if values.size == 0:
        raise ValueError("series must contain at least one observation.")
    invalid_positions = np.flatnonzero(~np.isfinite(values))
    if invalid_positions.size > 0:
        shown = ", ".join(str(position) for position in invalid_positions[:5])
        if invalid_positions.size > 5:
            shown = f"{shown}, ..."
        raise ValueError(
            "series must contain only finite values; "
            f"found {invalid_positions.size} invalid value(s) at position(s): {shown}."
        )
    return values


def _prefix_sum(values: np.ndarray) -> np.ndarray:
    return np.concatenate((np.array([0.0]), np.cumsum(values, dtype=float)))
