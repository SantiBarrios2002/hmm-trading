"""Signal generation and trading policy utilities."""

from hft_hmm.strategy.signals import (
    SIGNAL_REFERENCE,
    SignalPolicy,
    align_signal_with_future_return,
    build_signal,
    sign_signal,
    signal_from_filter_result,
    thresholded_hold_signal,
    thresholded_signal,
)

__all__ = [
    "SIGNAL_REFERENCE",
    "SignalPolicy",
    "align_signal_with_future_return",
    "build_signal",
    "sign_signal",
    "signal_from_filter_result",
    "thresholded_hold_signal",
    "thresholded_signal",
]
