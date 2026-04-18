"""State-grid metadata used by the HMM baseline and downstream evaluators.

A ``StateGrid`` fixes an ordering over hidden states together with their
representative mean returns. Downstream modules (filtering, signal generation,
plotting) can then refer to states by label rather than by integer index, and a
single source of truth governs the ordering used for signed expected returns.

References: Engineering utility
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateGrid:
    """Immutable mapping from hidden-state index to label and mean return."""

    k: int
    means: np.ndarray
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError(f"StateGrid requires k >= 2, got {self.k}.")
        if self.means.shape != (self.k,):
            raise ValueError(f"means must have shape ({self.k},), got {self.means.shape}.")
        if len(self.labels) != self.k:
            raise ValueError(f"labels must have length {self.k}, got {len(self.labels)}.")
        if len(set(self.labels)) != self.k:
            raise ValueError("labels must be unique.")

    def label(self, index: int) -> str:
        """Return the label for state ``index``."""
        if not 0 <= index < self.k:
            raise IndexError(f"state index {index} out of range for k={self.k}.")
        return self.labels[index]

    def index(self, label: str) -> int:
        """Return the state index for ``label``."""
        try:
            return self.labels.index(label)
        except ValueError as exc:
            raise KeyError(f"unknown state label {label!r}.") from exc


def default_labels(k: int) -> tuple[str, ...]:
    """Return semantic labels for small ``k`` or generic ``state_i`` labels.

    ``k == 2`` maps to ``("down", "up")``, ``k == 3`` maps to
    ``("down", "flat", "up")``, and larger grids use ``("state_0", ...,
    "state_{k-1}")`` to avoid implying structure the model has not recovered.
    """
    if k < 2:
        raise ValueError(f"default_labels requires k >= 2, got {k}.")
    if k == 2:
        return ("down", "up")
    if k == 3:
        return ("down", "flat", "up")
    return tuple(f"state_{i}" for i in range(k))


def linear_grid(k: int, min_return: float, max_return: float) -> StateGrid:
    """Construct a ``StateGrid`` with means evenly spaced over a return range.

    The grid ordering is monotonically increasing in the mean return, so the
    lowest-index state is the most bearish and the highest-index state is the
    most bullish. Default semantic labels are assigned via ``default_labels``.
    """
    if k < 2:
        raise ValueError(f"linear_grid requires k >= 2, got {k}.")
    if not min_return < max_return:
        raise ValueError(
            f"min_return must be strictly less than max_return; "
            f"got min_return={min_return}, max_return={max_return}."
        )
    means = np.linspace(min_return, max_return, num=k)
    labels = default_labels(k)
    return StateGrid(k=k, means=means, labels=labels)
