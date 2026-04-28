"""Walk-forward and cross-experiment utilities.

The runner (:mod:`hft_hmm.experiments.runner`) depends on
:mod:`hft_hmm.config`, which in turn depends on :mod:`walk_forward`. To keep
that dependency chain acyclic at package-init time, this package only
re-exports the walk-forward primitives. Runner symbols (and the side-info
comparison runner, which itself imports config) are re-exported at the
top-level :mod:`hft_hmm` package instead.
"""

from hft_hmm.experiments.walk_forward import (
    WALK_FORWARD_REFERENCE,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
    walk_forward,
)

__all__ = [
    "WALK_FORWARD_REFERENCE",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardWindow",
    "walk_forward",
]
