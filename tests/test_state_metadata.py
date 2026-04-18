"""Tests for state-grid metadata helpers."""

from __future__ import annotations

import numpy as np
import pytest

from hft_hmm.core.state_metadata import StateGrid, default_labels, linear_grid


def test_default_labels_k2():
    assert default_labels(2) == ("down", "up")


def test_default_labels_k3():
    assert default_labels(3) == ("down", "flat", "up")


@pytest.mark.parametrize("k", [4, 5, 8])
def test_default_labels_generic(k):
    labels = default_labels(k)
    assert labels == tuple(f"state_{i}" for i in range(k))
    assert len(labels) == k
    assert len(set(labels)) == k


def test_default_labels_rejects_small_k():
    with pytest.raises(ValueError, match="k >= 2"):
        default_labels(1)


@pytest.mark.parametrize("k", [2, 3, 5])
def test_linear_grid_monotone_increasing(k):
    grid = linear_grid(k, -0.01, 0.01)
    assert grid.k == k
    assert grid.means.shape == (k,)
    assert np.all(np.diff(grid.means) > 0)
    assert grid.means[0] == pytest.approx(-0.01)
    assert grid.means[-1] == pytest.approx(0.01)


def test_linear_grid_round_trip_label_index():
    grid = linear_grid(3, -0.01, 0.01)
    for i, expected_label in enumerate(("down", "flat", "up")):
        assert grid.label(i) == expected_label
        assert grid.index(expected_label) == i


def test_linear_grid_uses_default_labels():
    assert linear_grid(2, -0.01, 0.01).labels == ("down", "up")
    assert linear_grid(3, -0.01, 0.01).labels == ("down", "flat", "up")
    assert linear_grid(5, -0.02, 0.02).labels == tuple(f"state_{i}" for i in range(5))


def test_linear_grid_rejects_small_k():
    with pytest.raises(ValueError, match="k >= 2"):
        linear_grid(1, -0.01, 0.01)


def test_linear_grid_rejects_inverted_range():
    with pytest.raises(ValueError, match="strictly less than"):
        linear_grid(3, 0.01, -0.01)


def test_linear_grid_rejects_degenerate_range():
    with pytest.raises(ValueError, match="strictly less than"):
        linear_grid(3, 0.0, 0.0)


def test_state_grid_is_frozen():
    grid = linear_grid(2, -0.01, 0.01)
    with pytest.raises(AttributeError):
        grid.k = 3  # type: ignore[misc]


def test_state_grid_label_out_of_range():
    grid = linear_grid(2, -0.01, 0.01)
    with pytest.raises(IndexError):
        grid.label(2)
    with pytest.raises(IndexError):
        grid.label(-1)


def test_state_grid_index_unknown_label():
    grid = linear_grid(2, -0.01, 0.01)
    with pytest.raises(KeyError):
        grid.index("sideways")


def test_state_grid_rejects_mismatched_means_shape():
    with pytest.raises(ValueError, match="means must have shape"):
        StateGrid(k=3, means=np.array([-0.01, 0.01]), labels=("a", "b", "c"))


def test_state_grid_rejects_mismatched_labels_length():
    with pytest.raises(ValueError, match="labels must have length"):
        StateGrid(k=3, means=np.array([-0.01, 0.0, 0.01]), labels=("a", "b"))


def test_state_grid_rejects_duplicate_labels():
    with pytest.raises(ValueError, match="labels must be unique"):
        StateGrid(k=3, means=np.array([-0.01, 0.0, 0.01]), labels=("a", "a", "b"))


def test_state_grid_rejects_small_k():
    with pytest.raises(ValueError, match="k >= 2"):
        StateGrid(k=1, means=np.array([0.0]), labels=("x",))
