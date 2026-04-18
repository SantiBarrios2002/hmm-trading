"""Tests for the piecewise linear regression baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hft_hmm.core import ENGINEERING_APPROXIMATION, module_category
from hft_hmm.models import fit_piecewise_linear_regression, plr_baseline


def _piecewise_trend(slopes: list[float], segment_length: int) -> pd.Series:
    values: list[float] = []
    current = 10.0
    for slope in slopes:
        for _ in range(segment_length):
            values.append(current)
            current += slope
    return pd.Series(values, dtype=float)


def test_plr_baseline_declares_engineering_category() -> None:
    assert module_category(plr_baseline) == ENGINEERING_APPROXIMATION


def test_fit_piecewise_linear_regression_segments_clean_synthetic_trend() -> None:
    series = _piecewise_trend([-0.02, 0.0, 0.03], segment_length=20)

    result = fit_piecewise_linear_regression(
        series,
        n_segments=3,
        compute_durbin_watson=True,
    )

    assert result.breakpoints == (20, 40, 60)
    assert [segment.start_idx for segment in result.segments] == [0, 20, 40]
    assert [segment.end_idx for segment in result.segments] == [20, 40, 60]
    assert [segment.slope for segment in result.segments] == pytest.approx([-0.02, 0.0, 0.03])
    assert result.state_grid.labels == ("down", "flat", "up")
    assert result.state_means.tolist() == pytest.approx([-0.02, 0.0, 0.03])
    assert result.state_variances.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert result.state_sequence[:20].tolist() == [0] * 20
    assert result.state_sequence[20:40].tolist() == [1] * 20
    assert result.state_sequence[40:].tolist() == [2] * 20
    assert result.durbin_watson is not None


def test_fit_piecewise_linear_regression_orders_states_by_slope() -> None:
    series = _piecewise_trend([0.02, -0.01, 0.03], segment_length=12)

    result = fit_piecewise_linear_regression(series, n_segments=3)

    assert [segment.slope for segment in result.segments] == pytest.approx([0.02, -0.01, 0.03])
    assert result.state_means.tolist() == pytest.approx([-0.01, 0.02, 0.03])
    assert [summary.segment_index for summary in result.state_summaries] == [1, 0, 2]
    for segment in result.segments:
        assigned_states = result.state_sequence[segment.start_idx : segment.end_idx]
        assert assigned_states.tolist() == [segment.state_index] * segment.n_obs


def test_fit_piecewise_linear_regression_is_deterministic_on_noisy_input() -> None:
    x = np.arange(45, dtype=float)
    base = _piecewise_trend([-0.01, 0.015, 0.005], segment_length=15).to_numpy()
    noisy_series = pd.Series(base + 0.001 * np.sin(x), dtype=float)

    first = fit_piecewise_linear_regression(noisy_series, n_segments=3)
    second = fit_piecewise_linear_regression(noisy_series, n_segments=3)

    assert first.breakpoints == second.breakpoints
    assert first.fitted_values.tolist() == pytest.approx(second.fitted_values.tolist())
    assert first.state_means.tolist() == pytest.approx(second.state_means.tolist())
    assert first.state_sequence.tolist() == second.state_sequence.tolist()
    assert first.segment_assignments.tolist() == second.segment_assignments.tolist()


def test_fit_piecewise_linear_regression_rejects_invalid_configuration() -> None:
    series = pd.Series(np.linspace(1.0, 2.0, num=5), dtype=float)

    with pytest.raises(ValueError, match="at least 2"):
        fit_piecewise_linear_regression(series, n_segments=1)
    with pytest.raises(ValueError, match="min_segment_length must be at least 2"):
        fit_piecewise_linear_regression(series, n_segments=2, min_segment_length=1)
    with pytest.raises(ValueError, match="too short for the requested segmentation"):
        fit_piecewise_linear_regression(series, n_segments=3, min_segment_length=2)


def test_fit_piecewise_linear_regression_rejects_non_finite_values() -> None:
    series = pd.Series([1.0, np.nan, 2.0], dtype=float)

    with pytest.raises(ValueError, match="only finite values"):
        fit_piecewise_linear_regression(series, n_segments=2)
