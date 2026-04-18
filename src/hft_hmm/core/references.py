"""Paper reference helpers and module taxonomy constants.

Every module in this repository belongs to one of three categories declared via
the module-level ``__category__`` attribute:

- ``PAPER_FAITHFUL`` — direct reproduction of a construction from
  Christensen/Turner/Godsill (2020).
- ``ENGINEERING_APPROXIMATION`` — pragmatic substitute where an exact
  reproduction is out of scope for the coursework.
- ``EVALUATION_LAYER`` — metrics, plotting, or experiment utilities that do not
  model the data-generating process directly.

The ``PaperReference`` dataclass and ``reference()`` factory attach structured
pointers to a paper section alongside a short topic label, so module docstrings
and tests can cite the paper without free-form strings drifting out of sync.

References: Engineering utility
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Final, cast

PAPER_FAITHFUL: Final[str] = "paper-faithful"
ENGINEERING_APPROXIMATION: Final[str] = "engineering-approximation"
EVALUATION_LAYER: Final[str] = "evaluation-layer"

ALL_CATEGORIES: Final[frozenset[str]] = frozenset(
    {PAPER_FAITHFUL, ENGINEERING_APPROXIMATION, EVALUATION_LAYER}
)


@dataclass(frozen=True)
class PaperReference:
    """Structured pointer to a section of the replicated paper."""

    section: str
    topic: str

    def __str__(self) -> str:
        return f"{self.section} — {self.topic}"


def reference(section: str, topic: str) -> PaperReference:
    """Create a ``PaperReference`` after validating both fields are non-empty."""
    if not section or not section.strip():
        raise ValueError("section must be a non-empty string.")
    if not topic or not topic.strip():
        raise ValueError("topic must be a non-empty string.")
    return PaperReference(section=section.strip(), topic=topic.strip())


def module_category(module: ModuleType) -> str | None:
    """Return the declared ``__category__`` of ``module`` or ``None`` if absent.

    Raises ``ValueError`` when the module declares a category outside
    ``ALL_CATEGORIES``, so typos surface at import-test time rather than during
    later review.
    """
    category = getattr(module, "__category__", None)
    if category is None:
        return None
    if category not in ALL_CATEGORIES:
        raise ValueError(
            f"Module {module.__name__} declares unknown __category__ {category!r}; "
            f"expected one of {sorted(ALL_CATEGORIES)}."
        )
    return cast(str, category)
