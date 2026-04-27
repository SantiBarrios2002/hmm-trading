"""Bucketed transition-matrix approximation for IOHMM-style side-information conditioning.

This module is an **engineering approximation** to the IOHMM transition model
described in §4 of Christensen, Turner & Godsill (2020).  The paper conditions
the HMM transition matrix on a side-information variable x_t through a
continuous parametric form; this module replaces that with a finite-bucket
scheme:

  1. Partition the support of x_t into ``n_buckets`` intervals.
  2. Assign each transition (s_t → s_{t+1}) to the bucket that contains x_t.
  3. Estimate one K × K row-stochastic transition matrix per bucket using
     additive smoothing toward a pooled baseline matrix.

The result is deterministic, interpretable, and directly usable in Gate H
experiments where a lookup ``A(x_t)`` replaces the fixed HMM transition matrix.

**This is NOT the exact IOHMM model from the paper.** It is a finite-bucket
engineering approximation intended for Gate H experiments only.  The paper's
continuous parametric conditioning is out of scope for the current coursework.

References: §4 side-information / IOHMM approximation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

import numpy as np

from hft_hmm.core import ENGINEERING_APPROXIMATION, PaperReference, reference

if TYPE_CHECKING:
    from hft_hmm.features.splines import SplinePredictorResult

__category__: Final[str] = ENGINEERING_APPROXIMATION
IOHMM_APPROX_REFERENCE: Final[PaperReference] = reference(
    "§4", "side-information / IOHMM approximation"
)


@dataclass(frozen=True)
class BucketedTransitionConfig:
    """Parameter bundle for the bucketed transition model.

    ``n_buckets`` is the number of side-information intervals (>= 2).
    ``smoothing`` is the additive prior weight toward the baseline matrix;
    empty rows in any bucket fall back exactly to the baseline when
    ``smoothing > 0``.
    ``grid_size`` is reserved for use by ``bucket_boundaries_from_spline_grid``
    when deriving evaluation-grid boundaries from a spline predictor.

    References: §4 side-information / IOHMM approximation
    """

    n_buckets: int = 3
    smoothing: float = 1.0
    grid_size: int = 200

    def __post_init__(self) -> None:
        if self.n_buckets < 2:
            raise ValueError(f"n_buckets must be at least 2, got {self.n_buckets}.")
        if self.smoothing <= 0.0:
            raise ValueError(f"smoothing must be positive, got {self.smoothing}.")
        if self.grid_size < 2:
            raise ValueError(f"grid_size must be at least 2, got {self.grid_size}.")


@dataclass(frozen=True)
class BucketedTransitionResult:
    """Fitted bucketed transition model.

    ``bucket_boundaries`` holds the n_buckets - 1 interior split points.
    ``transition_matrices`` has shape (n_buckets, K, K); every row sums to 1.
    ``baseline_transition_matrix`` is the pooled (or caller-supplied) K × K
    matrix used as the smoothing prior; empty bucket rows fall back to it.
    ``bucket_observation_counts`` counts raw transitions assigned to each bucket
    (before smoothing).

    References: §4 side-information / IOHMM approximation
    """

    config: BucketedTransitionConfig
    bucket_boundaries: np.ndarray  # shape (n_buckets - 1,)
    transition_matrices: np.ndarray  # shape (n_buckets, K, K)
    baseline_transition_matrix: np.ndarray  # shape (K, K)
    bucket_observation_counts: np.ndarray  # shape (n_buckets,), raw transition counts

    def __post_init__(self) -> None:
        n_buckets = self.config.n_buckets

        boundaries = np.asarray(self.bucket_boundaries, dtype=float).copy()
        matrices = np.asarray(self.transition_matrices, dtype=float).copy()
        baseline = np.asarray(self.baseline_transition_matrix, dtype=float).copy()
        counts = np.asarray(self.bucket_observation_counts, dtype=int).copy()

        if boundaries.ndim != 1 or len(boundaries) != n_buckets - 1:
            raise ValueError(
                f"bucket_boundaries must have shape ({n_buckets - 1},) for "
                f"n_buckets={n_buckets}; got {boundaries.shape}."
            )
        if (
            matrices.ndim != 3
            or matrices.shape[0] != n_buckets
            or matrices.shape[1] != matrices.shape[2]
        ):
            raise ValueError(
                f"transition_matrices must have shape ({n_buckets}, K, K); "
                f"got {matrices.shape}."
            )
        K = matrices.shape[1]
        if baseline.shape != (K, K):
            raise ValueError(
                f"baseline_transition_matrix must have shape ({K}, {K}); "
                f"got {baseline.shape}."
            )
        if counts.shape != (n_buckets,):
            raise ValueError(
                f"bucket_observation_counts must have shape ({n_buckets},); "
                f"got {counts.shape}."
            )

        boundaries.setflags(write=False)
        matrices.setflags(write=False)
        baseline.setflags(write=False)
        counts.setflags(write=False)

        object.__setattr__(self, "bucket_boundaries", boundaries)
        object.__setattr__(self, "transition_matrices", matrices)
        object.__setattr__(self, "baseline_transition_matrix", baseline)
        object.__setattr__(self, "bucket_observation_counts", counts)

    def bucket_index_for(self, x: float) -> int:
        """Return the bucket index (0-based) for side-information value ``x``."""
        return int(np.searchsorted(self.bucket_boundaries, x, side="right"))

    def transition_matrix_for(self, x: float) -> np.ndarray:
        """Return the K × K transition matrix for side-information value ``x``."""
        return cast(np.ndarray, self.transition_matrices[self.bucket_index_for(x)])


def fit_bucketed_transition_model(
    state_sequence: np.ndarray,
    side_information: np.ndarray,
    *,
    n_states: int | None = None,
    baseline_transition_matrix: np.ndarray | None = None,
    bucket_boundaries: np.ndarray | None = None,
    config: BucketedTransitionConfig | None = None,
) -> BucketedTransitionResult:
    """Estimate one transition matrix per side-information bucket.

    ``state_sequence`` is a 1-D integer array of length T.  ``side_information``
    is a 1-D finite float array of the same length T.  For each consecutive pair
    (s_t, s_{t+1}), the transition is assigned to the bucket determined by x_t.

    ``n_states`` overrides the inferred state-space size; must be >= 2.
    ``baseline_transition_matrix`` is used as the smoothing prior; if omitted it
    is estimated from pooled counts across all buckets.
    ``bucket_boundaries`` are the n_buckets - 1 interior split points; if omitted
    they are derived from quantiles of the side-information values over the
    transition indices.

    References: §4 side-information / IOHMM approximation
    """
    if config is None:
        config = BucketedTransitionConfig()

    states = _coerce_state_sequence(state_sequence)
    x = _coerce_side_information(side_information, len(states))
    K = _resolve_n_states(states, n_states)
    n_buckets = config.n_buckets

    # x_t and transition pairs use indices 0 .. T-2
    x_t = x[:-1]
    s_from = states[:-1]
    s_to = states[1:]

    resolved_boundaries = _resolve_bucket_boundaries(x_t, bucket_boundaries, n_buckets)

    # Raw transition counts per bucket
    raw_counts = np.zeros((n_buckets, K, K), dtype=float)
    bucket_indices = np.searchsorted(resolved_boundaries, x_t, side="right")
    for t_idx in range(len(s_from)):
        raw_counts[bucket_indices[t_idx], s_from[t_idx], s_to[t_idx]] += 1.0

    if baseline_transition_matrix is not None:
        baseline = _validate_baseline(baseline_transition_matrix, K)
    else:
        baseline = _estimate_baseline(raw_counts, K)

    transition_matrices = _smooth_and_normalize(raw_counts, baseline, config.smoothing)
    bucket_obs_counts = raw_counts.sum(axis=(1, 2)).astype(int)

    return BucketedTransitionResult(
        config=config,
        bucket_boundaries=resolved_boundaries,
        transition_matrices=transition_matrices,
        baseline_transition_matrix=baseline,
        bucket_observation_counts=bucket_obs_counts,
    )


def bucket_boundaries_from_spline_grid(
    spline_result: SplinePredictorResult,
    *,
    config: BucketedTransitionConfig | None = None,
) -> np.ndarray:
    """Derive deterministic bucket boundaries from a fitted spline predictor.

    Places n_buckets - 1 equally-spaced interior boundaries across the spline's
    observed support [x_min, x_max].  This is a deterministic approximation
    that partitions the feature space uniformly; callers may pass the result
    directly as ``bucket_boundaries`` to ``fit_bucketed_transition_model``.

    References: §4 side-information / IOHMM approximation
    """
    if config is None:
        config = BucketedTransitionConfig()
    endpoints = np.linspace(spline_result.x_min, spline_result.x_max, config.n_buckets + 1)
    return endpoints[1:-1]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _coerce_state_sequence(seq: np.ndarray) -> np.ndarray:
    try:
        states = np.asarray(seq, dtype=int)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"state_sequence must be integer-like; got {type(seq).__name__}."
        ) from exc
    if states.ndim != 1:
        raise ValueError(
            f"state_sequence must be one-dimensional, got shape {states.shape}."
        )
    if len(states) < 2:
        raise ValueError(
            f"state_sequence must have at least 2 observations to form a transition; "
            f"got {len(states)}."
        )
    if np.any(states < 0):
        raise ValueError("state_sequence must contain only non-negative state indices.")
    return states


def _coerce_side_information(x: np.ndarray, expected_length: int) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError(
            f"side_information must be one-dimensional, got shape {values.shape}."
        )
    if len(values) != expected_length:
        raise ValueError(
            f"side_information must have the same length as state_sequence; "
            f"got {len(values)}, expected {expected_length}."
        )
    if not np.all(np.isfinite(values)):
        n_invalid = int(np.sum(~np.isfinite(values)))
        raise ValueError(
            f"side_information must contain only finite values; "
            f"found {n_invalid} non-finite value(s)."
        )
    return values


def _resolve_n_states(states: np.ndarray, n_states: int | None) -> int:
    inferred = int(states.max()) + 1
    if n_states is not None:
        if n_states < 2:
            raise ValueError(f"n_states must be at least 2, got {n_states}.")
        if inferred > n_states:
            raise ValueError(
                f"state_sequence contains state index {inferred - 1} "
                f"but n_states={n_states}; all states must be in [0, n_states - 1]."
            )
        return n_states
    if inferred < 2:
        raise ValueError(
            f"At least 2 states are required; inferred K={inferred} from state_sequence. "
            "Pass n_states explicitly or include at least 2 distinct states."
        )
    return inferred


def _resolve_bucket_boundaries(
    x: np.ndarray,
    provided: np.ndarray | None,
    n_buckets: int,
) -> np.ndarray:
    if provided is not None:
        b = np.asarray(provided, dtype=float)
        if b.ndim != 1:
            raise ValueError("bucket_boundaries must be one-dimensional.")
        if len(b) != n_buckets - 1:
            raise ValueError(
                f"bucket_boundaries must have {n_buckets - 1} interior value(s) for "
                f"n_buckets={n_buckets}; got {len(b)}."
            )
        if not np.all(np.isfinite(b)):
            raise ValueError("bucket_boundaries must contain only finite values.")
        if n_buckets > 2 and not np.all(b[:-1] < b[1:]):
            raise ValueError("bucket_boundaries must be strictly increasing.")
        return b.copy()

    # Quantile-based boundaries; fall back to linspace if duplicates arise
    levels = np.linspace(0.0, 1.0, n_buckets + 1)[1:-1]
    boundaries = np.quantile(x, levels)
    if len(np.unique(boundaries)) < len(boundaries):
        lo, hi = float(x.min()), float(x.max())
        if np.isclose(lo, hi):
            lo, hi = lo - 1e-10, hi + 1e-10
        boundaries = np.linspace(lo, hi, n_buckets + 1)[1:-1]
    return cast(np.ndarray, boundaries)


def _estimate_baseline(raw_counts: np.ndarray, K: int) -> np.ndarray:
    pooled = raw_counts.sum(axis=0)  # (K, K)
    row_sums = pooled.sum(axis=1, keepdims=True)
    uniform = np.ones((1, K), dtype=float) / K
    with np.errstate(invalid="ignore", divide="ignore"):
        baseline = np.where(row_sums == 0, uniform, pooled / np.where(row_sums == 0, 1.0, row_sums))
    return baseline


def _validate_baseline(matrix: np.ndarray, K: int) -> np.ndarray:
    b = np.asarray(matrix, dtype=float)
    if b.shape != (K, K):
        raise ValueError(
            f"baseline_transition_matrix must have shape ({K}, {K}), got {b.shape}."
        )
    if not np.all(np.isfinite(b)):
        raise ValueError("baseline_transition_matrix must contain only finite values.")
    if not np.all(b >= 0):
        raise ValueError("baseline_transition_matrix must have nonnegative entries.")
    row_sums = b.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(
            "baseline_transition_matrix rows must each sum to 1.0 within tolerance; "
            f"got row sums: {row_sums.tolist()}."
        )
    return b.copy()


def _smooth_and_normalize(
    raw_counts: np.ndarray,
    baseline: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    # Broadcast baseline over bucket dimension: (n_buckets, K, K)
    smoothed = raw_counts + smoothing * baseline[np.newaxis, :, :]
    row_sums = smoothed.sum(axis=2, keepdims=True)
    # row_sums > 0 is guaranteed: smoothing > 0 and baseline rows sum to 1
    return cast(np.ndarray, smoothed / row_sums)
