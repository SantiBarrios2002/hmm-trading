"""Tests for the bucketed IOHMM-style transition approximation."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from hft_hmm.core.references import ENGINEERING_APPROXIMATION, module_category
from hft_hmm.models.iohmm_approx import (
    BucketedTransitionConfig,
    fit_bucketed_transition_model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_states_and_x(
    T: int = 200,
    n_states: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = rng.integers(0, n_states, size=T)
    x = rng.uniform(0.0, 1.0, size=T)
    return states, x


def _two_bucket_fixture(T: int = 600, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic fixture where x_t < 0.5 keeps state 0 sticky, x_t >= 0.5 keeps state 1 sticky."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=T)
    states = np.zeros(T, dtype=int)
    for t in range(T - 1):
        if x[t] < 0.5:
            states[t + 1] = 0 if rng.random() < 0.9 else 1
        else:
            states[t + 1] = 1 if rng.random() < 0.9 else 0
    return states, x


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


def test_module_declares_engineering_approximation() -> None:
    mod = importlib.import_module("hft_hmm.models.iohmm_approx")
    assert module_category(mod) == ENGINEERING_APPROXIMATION


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = BucketedTransitionConfig()
    assert cfg.n_buckets == 3
    assert cfg.smoothing == 1.0
    assert cfg.grid_size == 200


def test_config_rejects_n_buckets_below_2() -> None:
    with pytest.raises(ValueError, match="n_buckets must be at least 2"):
        BucketedTransitionConfig(n_buckets=1)


def test_config_rejects_nonpositive_smoothing() -> None:
    with pytest.raises(ValueError, match="smoothing must be positive"):
        BucketedTransitionConfig(smoothing=0.0)
    with pytest.raises(ValueError, match="smoothing must be positive"):
        BucketedTransitionConfig(smoothing=-1.0)


def test_config_rejects_grid_size_below_2() -> None:
    with pytest.raises(ValueError, match="grid_size must be at least 2"):
        BucketedTransitionConfig(grid_size=1)


# ---------------------------------------------------------------------------
# Result immutability
# ---------------------------------------------------------------------------


def test_result_is_frozen() -> None:
    states, x = _make_states_and_x()
    result = fit_bucketed_transition_model(states, x, config=BucketedTransitionConfig(n_buckets=2))
    with pytest.raises((AttributeError, TypeError)):
        result.bucket_boundaries = np.array([0.5])  # type: ignore[misc]


def test_result_arrays_are_read_only() -> None:
    states, x = _make_states_and_x()
    result = fit_bucketed_transition_model(states, x)
    with pytest.raises(ValueError):
        result.transition_matrices[0, 0, 0] = 999.0


# ---------------------------------------------------------------------------
# Bucket assignment determinism
# ---------------------------------------------------------------------------


def test_bucket_index_for_is_deterministic() -> None:
    states, x = _make_states_and_x()
    result = fit_bucketed_transition_model(states, x)
    assert result.bucket_index_for(0.25) == result.bucket_index_for(0.25)


def test_bucket_index_for_respects_boundaries() -> None:
    states, x = _make_states_and_x()
    boundaries = np.array([0.3, 0.7])
    result = fit_bucketed_transition_model(
        states, x, bucket_boundaries=boundaries, config=BucketedTransitionConfig(n_buckets=3)
    )
    assert result.bucket_index_for(0.0) == 0
    assert result.bucket_index_for(0.3) == 1  # searchsorted side="right"
    assert result.bucket_index_for(0.5) == 1
    assert result.bucket_index_for(0.7) == 2
    assert result.bucket_index_for(1.0) == 2


def test_bucket_index_for_two_bucket_case() -> None:
    states = np.array([0, 1, 0, 1, 0])
    x = np.array([0.2, 0.8, 0.1, 0.9, 0.6])
    result = fit_bucketed_transition_model(
        states, x, bucket_boundaries=np.array([0.5]), config=BucketedTransitionConfig(n_buckets=2)
    )
    assert result.bucket_index_for(0.0) == 0
    assert result.bucket_index_for(0.4999) == 0
    assert result.bucket_index_for(0.5) == 1
    assert result.bucket_index_for(1.0) == 1


# ---------------------------------------------------------------------------
# transition_matrix_for
# ---------------------------------------------------------------------------


def test_transition_matrix_for_returns_correct_slice() -> None:
    states, x = _make_states_and_x()
    boundaries = np.array([0.5])
    result = fit_bucketed_transition_model(
        states, x, bucket_boundaries=boundaries, config=BucketedTransitionConfig(n_buckets=2)
    )
    np.testing.assert_array_equal(
        result.transition_matrix_for(0.2), result.transition_matrices[0]
    )
    np.testing.assert_array_equal(
        result.transition_matrix_for(0.8), result.transition_matrices[1]
    )


def test_transition_matrix_for_differs_across_buckets_on_synthetic_fixture() -> None:
    states, x = _two_bucket_fixture()
    result = fit_bucketed_transition_model(
        states,
        x,
        bucket_boundaries=np.array([0.5]),
        config=BucketedTransitionConfig(n_buckets=2, smoothing=0.1),
    )
    # Low-x bucket: state 0 is sticky → A[0, 0, 0] should be notably higher than A[1, 0, 0]
    a_low = result.transition_matrix_for(0.2)
    a_high = result.transition_matrix_for(0.8)
    assert a_low[0, 0] > a_high[0, 0], (
        f"Expected A_low[0,0]={a_low[0,0]:.3f} > A_high[0,0]={a_high[0,0]:.3f}"
    )


# ---------------------------------------------------------------------------
# Shape / finiteness / nonnegativity / row-sum invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_buckets,n_states", [(2, 2), (3, 3), (4, 5)])
def test_all_matrices_have_correct_shape(n_buckets: int, n_states: int) -> None:
    states, x = _make_states_and_x(T=300, n_states=n_states)
    result = fit_bucketed_transition_model(
        states, x, config=BucketedTransitionConfig(n_buckets=n_buckets)
    )
    assert result.transition_matrices.shape == (n_buckets, n_states, n_states)


@pytest.mark.parametrize("n_buckets", [2, 3, 5])
def test_all_matrices_are_finite(n_buckets: int) -> None:
    states, x = _make_states_and_x(T=300)
    result = fit_bucketed_transition_model(
        states, x, config=BucketedTransitionConfig(n_buckets=n_buckets)
    )
    assert np.all(np.isfinite(result.transition_matrices))


@pytest.mark.parametrize("n_buckets", [2, 3, 5])
def test_all_matrices_are_nonnegative(n_buckets: int) -> None:
    states, x = _make_states_and_x(T=300)
    result = fit_bucketed_transition_model(
        states, x, config=BucketedTransitionConfig(n_buckets=n_buckets)
    )
    assert np.all(result.transition_matrices >= 0.0)


@pytest.mark.parametrize("n_buckets", [2, 3, 5])
def test_all_row_sums_are_one(n_buckets: int) -> None:
    states, x = _make_states_and_x(T=300)
    result = fit_bucketed_transition_model(
        states, x, config=BucketedTransitionConfig(n_buckets=n_buckets)
    )
    row_sums = result.transition_matrices.sum(axis=2)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Sparse / empty bucket smoothing
# ---------------------------------------------------------------------------


def test_empty_bucket_falls_back_to_baseline() -> None:
    # All x_t values are concentrated in bucket 0; bucket 1 will be empty.
    T = 100
    states = np.zeros(T, dtype=int)
    states[::2] = 1  # alternate 0,1,0,1,...
    # Force all x values below 0.1 so bucket 0 gets everything
    x = np.full(T, 0.05)
    boundaries = np.array([0.5])
    result = fit_bucketed_transition_model(
        states, x, bucket_boundaries=boundaries, config=BucketedTransitionConfig(n_buckets=2)
    )
    assert result.bucket_observation_counts[1] == 0
    np.testing.assert_allclose(
        result.transition_matrices[1],
        result.baseline_transition_matrix,
        atol=1e-12,
    )


def test_sparse_bucket_smoothed_toward_baseline_stays_normalized() -> None:
    # Bucket 1 gets only 1 observation; after smoothing it must still sum to 1.
    states = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    x = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    boundaries = np.array([0.5])
    result = fit_bucketed_transition_model(
        states, x, bucket_boundaries=boundaries, config=BucketedTransitionConfig(n_buckets=2)
    )
    # Bucket 0 has exactly 1 transition (x[0]=0.9 → bucket 1 index 1... wait)
    # Actually x[0]=0.9 > 0.5 → bucket index 1, so bucket 1 gets that transition
    # Bucket 0 gets all the remaining transitions where x < 0.5
    row_sums = result.transition_matrices.sum(axis=2)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)
    assert np.all(np.isfinite(result.transition_matrices))


def test_smoothing_strength_affects_sparse_bucket() -> None:
    # With very high smoothing, sparse bucket should be very close to baseline.
    T = 200
    states = np.zeros(T, dtype=int)
    states[::3] = 1
    x = np.concatenate([np.full(T - 2, 0.1), [0.9, 0.9]])  # nearly all in bucket 0
    boundaries = np.array([0.5])
    result_strong = fit_bucketed_transition_model(
        states, x, bucket_boundaries=boundaries,
        config=BucketedTransitionConfig(n_buckets=2, smoothing=1000.0),
    )
    # With extreme smoothing, bucket 1 (sparse) should be nearly identical to baseline
    diff = np.abs(result_strong.transition_matrices[1] - result_strong.baseline_transition_matrix)
    assert diff.max() < 0.05


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_n_states_raises() -> None:
    states = np.array([0, 1, 0, 1])
    x = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="n_states must be at least 2"):
        fit_bucketed_transition_model(states, x, n_states=1)


def test_state_sequence_out_of_range_raises() -> None:
    states = np.array([0, 1, 5, 0])  # state 5 but n_states=3
    x = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="state index"):
        fit_bucketed_transition_model(states, x, n_states=3)


def test_inferred_k_below_2_raises() -> None:
    states = np.array([0, 0, 0, 0])  # all zeros → K=1
    x = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="At least 2 states"):
        fit_bucketed_transition_model(states, x)


def test_state_sequence_must_be_1d() -> None:
    states = np.array([[0, 1], [0, 1]])
    x = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="one-dimensional"):
        fit_bucketed_transition_model(states, x)


def test_state_sequence_too_short_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        fit_bucketed_transition_model(np.array([0]), np.array([0.5]))


def test_side_information_length_mismatch_raises() -> None:
    states = np.array([0, 1, 0, 1])
    x = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="same length"):
        fit_bucketed_transition_model(states, x)


def test_side_information_with_nan_raises() -> None:
    states = np.array([0, 1, 0, 1])
    x = np.array([0.1, float("nan"), 0.3, 0.4])
    with pytest.raises(ValueError, match="non-finite"):
        fit_bucketed_transition_model(states, x)


def test_side_information_with_inf_raises() -> None:
    states = np.array([0, 1, 0, 1])
    x = np.array([0.1, float("inf"), 0.3, 0.4])
    with pytest.raises(ValueError, match="non-finite"):
        fit_bucketed_transition_model(states, x)


def test_bad_boundaries_unsorted_raises() -> None:
    states = np.array([0, 1, 0, 1, 0])
    x = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_bucketed_transition_model(
            states, x,
            bucket_boundaries=np.array([0.7, 0.3]),
            config=BucketedTransitionConfig(n_buckets=3),
        )


def test_bad_boundaries_wrong_length_raises() -> None:
    states = np.array([0, 1, 0, 1, 0])
    x = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    with pytest.raises(ValueError, match="interior value"):
        fit_bucketed_transition_model(
            states, x,
            bucket_boundaries=np.array([0.5]),  # need 2 for n_buckets=3
            config=BucketedTransitionConfig(n_buckets=3),
        )


def test_bad_boundaries_nonfinite_raises() -> None:
    states = np.array([0, 1, 0, 1, 0])
    x = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    with pytest.raises(ValueError, match="finite"):
        fit_bucketed_transition_model(
            states, x,
            bucket_boundaries=np.array([float("nan")]),
            config=BucketedTransitionConfig(n_buckets=2),
        )


def test_invalid_baseline_wrong_shape_raises() -> None:
    states = np.array([0, 1, 0, 1, 0])
    x = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    bad_baseline = np.eye(3)  # wrong shape for K=2
    with pytest.raises(ValueError, match="shape"):
        fit_bucketed_transition_model(states, x, baseline_transition_matrix=bad_baseline)


def test_invalid_baseline_rows_not_summing_raises() -> None:
    states = np.array([0, 1, 0, 1, 0])
    x = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    bad_baseline = np.array([[0.6, 0.6], [0.5, 0.5]])  # rows sum to 1.2
    with pytest.raises(ValueError, match="sum to 1"):
        fit_bucketed_transition_model(states, x, baseline_transition_matrix=bad_baseline)


# ---------------------------------------------------------------------------
# Synthetic two-bucket fixture — main functional test
# ---------------------------------------------------------------------------


def test_synthetic_two_bucket_fixture_produces_measurably_different_matrices() -> None:
    states, x = _two_bucket_fixture(T=1000, seed=42)
    result = fit_bucketed_transition_model(
        states,
        x,
        bucket_boundaries=np.array([0.5]),
        config=BucketedTransitionConfig(n_buckets=2, smoothing=0.5),
    )
    a_low = result.transition_matrices[0]   # x < 0.5: state 0 sticky
    a_high = result.transition_matrices[1]  # x >= 0.5: state 1 sticky

    # State 0 self-transition should be higher in the low-x bucket
    assert a_low[0, 0] > 0.7, f"A_low[0,0]={a_low[0,0]:.3f}"
    # State 1 self-transition should be higher in the high-x bucket
    assert a_high[1, 1] > 0.7, f"A_high[1,1]={a_high[1,1]:.3f}"
    # Difference in A[0, 0, 0] should be substantial
    assert a_low[0, 0] - a_high[0, 0] > 0.3, (
        f"Expected gap > 0.3; got {a_low[0,0]:.3f} vs {a_high[0,0]:.3f}"
    )


# ---------------------------------------------------------------------------
# Provided baseline round-trips
# ---------------------------------------------------------------------------


def test_custom_baseline_is_used_for_smoothing() -> None:
    states = np.array([0, 1, 0, 1, 0, 1, 0])
    x = np.full(7, 0.9)  # all in bucket 1, bucket 0 is empty
    uniform_baseline = np.array([[0.5, 0.5], [0.5, 0.5]])
    result = fit_bucketed_transition_model(
        states,
        x,
        bucket_boundaries=np.array([0.5]),
        baseline_transition_matrix=uniform_baseline,
        config=BucketedTransitionConfig(n_buckets=2),
    )
    # Bucket 0 is empty → must equal the provided baseline
    np.testing.assert_allclose(result.transition_matrices[0], uniform_baseline, atol=1e-12)
    np.testing.assert_allclose(result.baseline_transition_matrix, uniform_baseline, atol=1e-12)


# ---------------------------------------------------------------------------
# bucket_boundaries_from_spline_grid helper
# ---------------------------------------------------------------------------


def test_bucket_boundaries_from_spline_grid() -> None:
    from unittest.mock import MagicMock

    from hft_hmm.models.iohmm_approx import bucket_boundaries_from_spline_grid

    spline = MagicMock()
    spline.x_min = 0.0
    spline.x_max = 1.0

    cfg = BucketedTransitionConfig(n_buckets=3)
    boundaries = bucket_boundaries_from_spline_grid(spline, config=cfg)

    assert boundaries.shape == (2,)  # n_buckets - 1
    np.testing.assert_allclose(boundaries, [1 / 3, 2 / 3], atol=1e-10)


def test_bucket_boundaries_from_spline_grid_uses_default_config() -> None:
    from unittest.mock import MagicMock

    from hft_hmm.models.iohmm_approx import bucket_boundaries_from_spline_grid

    spline = MagicMock()
    spline.x_min = 0.0
    spline.x_max = 3.0

    boundaries = bucket_boundaries_from_spline_grid(spline)
    assert len(boundaries) == BucketedTransitionConfig().n_buckets - 1


# ---------------------------------------------------------------------------
# __init__.py re-exports
# ---------------------------------------------------------------------------


def test_exports_available_from_models_package() -> None:
    import hft_hmm.models as models

    assert hasattr(models, "BucketedTransitionConfig")
    assert hasattr(models, "BucketedTransitionResult")
    assert hasattr(models, "fit_bucketed_transition_model")
    assert hasattr(models, "iohmm_approx")
