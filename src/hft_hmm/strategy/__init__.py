"""Signal generation and trading policy utilities."""

from hft_hmm.strategy.signals import (
    SIGNAL_REFERENCE,
    align_signal_with_future_return,
    sign_signal,
    signal_from_filter_result,
    thresholded_signal,
)

__all__ = [
    "SIGNAL_REFERENCE",
    "align_signal_with_future_return",
    "sign_signal",
    "signal_from_filter_result",
    "thresholded_signal",
]
