"""Signal generation and alignment for HMM-driven trading.

Evaluation layer — turns the one-step-ahead expected returns produced by the
forward filter into a simple long/short/flat position and provides the
alignment helper that maps a signal to its realized next-bar return without
leaking future information.

Conventions
-----------
- ``signal[t]`` is the position chosen using information available at bar
  ``t`` (``Δy_{1:t}``) and held from bar ``t`` to bar ``t+1``.
- Realized per-bar strategy return at bar ``t`` is
  ``signal[t-1] * realized_returns[t]``; ``align_signal_with_future_return``
  enforces this alignment and drops the first bar.
- ``ForwardFilterResult.expected_next_returns[t]`` is ``E[Δy_{t+1} | Δy_{1:t}]``
  (see ``hft_hmm.inference.forward_filter``), so feeding that array straight
  into ``sign_signal`` is consistent with the convention above; no ``shift``
  is required.

Signal policies
---------------
- ``sign`` — discrete ±1/0 positions following ``np.sign(E[Δy_{t+1}])``.
  This is the paper's policy and the primary path for the academic comparison.
- ``thresholded_hold`` — sign-with-deadzone variant that keeps the previous
  position inside ``|E[Δy_{t+1}]| <= threshold``. Used as a turnover-aware
  diagnostic against the conservative cost convention; **not** a paper-matching
  policy.
- ``conviction_weighted`` — engineering extension. Continuous position in
  ``[-1, +1]`` proportional to ``E[Δy_{t+1}]`` standardized by a per-window
  scale (typically the std of training-side predicted expected returns).
  Small-magnitude predictions still trade, just with smaller positions. Labeled
  evaluation-layer rather than paper-faithful because the paper specifies a
  ±1 sign rule.

Evaluation modes
----------------
- **pre-cost**: aggregate metrics are computed on
  ``signal[t-1] * returns[t]`` with no execution-cost adjustment. This is the
  primary academic signal-quality mode used for paper comparison.
- **post-cost diagnostic**: a turnover cost ``c`` (basis points per position
  change) is subtracted after signal alignment. The cost model itself lives in
  the Gate E metrics module (Issue 10); this module only emits signals and does
  not apply costs. Post-cost results are useful stress tests, but are not a
  paper-matching target because the paper leaves execution costs unspecified.

References: §8 sign-based trading signal (evaluation layer)
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
import pandas as pd

from hft_hmm.core import EVALUATION_LAYER, PaperReference, reference
from hft_hmm.inference import ForwardFilterResult

__category__: Final[str] = EVALUATION_LAYER
SIGNAL_REFERENCE: Final[PaperReference] = reference(
    "§8", "sign-based trading signal from expected return"
)

SignalPolicy = Literal["sign", "thresholded_hold", "conviction_weighted"]
_VALID_SIGNAL_POLICIES: Final[tuple[str, ...]] = (
    "sign",
    "thresholded_hold",
    "conviction_weighted",
)
_DISCRETE_SIGNAL_POLICIES: Final[tuple[str, ...]] = ("sign", "thresholded_hold")


def sign_signal(expected_next_returns: pd.Series | np.ndarray) -> pd.Series:
    """Return +1/-1/0 positions from one-step-ahead expected returns.

    ``signal[t]`` is the sign of ``E[Δy_{t+1} | Δy_{1:t}]``: ``+1`` when the
    expected next return is strictly positive, ``-1`` when strictly negative,
    and ``0`` when exactly zero. The output is a ``pd.Series`` of ``int8``
    preserving the input index, or a positional ``RangeIndex`` for raw numpy
    input.

    References: §8 sign-based trading signal (evaluation layer)
    """
    values, index = _coerce_expected_returns(expected_next_returns)
    signal = np.sign(values).astype(np.int8, copy=False)
    return pd.Series(signal, index=index, name="signal")


def thresholded_signal(
    expected_next_returns: pd.Series | np.ndarray,
    *,
    threshold: float,
) -> pd.Series:
    """Sign-with-deadzone signal over absolute expected return.

    ``signal[t]`` is ``+1`` when ``E[Δy_{t+1}] >  threshold``, ``-1`` when
    ``E[Δy_{t+1}] < -threshold``, and ``0`` inside the dead-zone
    ``|E[Δy_{t+1}]| <= threshold``. ``threshold`` is expressed in log-return
    units, must be a finite non-negative float, and ``threshold=0`` reproduces
    ``sign_signal``.

    References: §8 sign-based trading signal (evaluation layer)
    """
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"threshold must be a finite non-negative float, got {threshold!r}.")
    values, index = _coerce_expected_returns(expected_next_returns)
    signal = np.zeros_like(values, dtype=np.int8)
    signal[values > threshold] = 1
    signal[values < -threshold] = -1
    return pd.Series(signal, index=index, name="signal")


def thresholded_hold_signal(
    expected_next_returns: pd.Series | np.ndarray,
    *,
    threshold: float,
    initial_position: int = 0,
) -> pd.Series:
    """Thresholded signal that holds the prior position inside the dead-zone.

    Values above ``threshold`` set the position to ``+1`` and values below
    ``-threshold`` set it to ``-1``. Values inside the no-trade band keep the
    previous position instead of forcing a flat rebalance, which makes this
    policy suitable for turnover-aware threshold sweeps.

    References: §8 sign-based trading signal (evaluation layer)
    """
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"threshold must be a finite non-negative float, got {threshold!r}.")
    if initial_position not in (-1, 0, 1):
        raise ValueError(f"initial_position must be one of -1, 0, 1; got {initial_position!r}.")
    values, index = _coerce_expected_returns(expected_next_returns)
    signal = np.empty(values.shape[0], dtype=np.int8)
    position = int(initial_position)
    for i, value in enumerate(values):
        if value > threshold:
            position = 1
        elif value < -threshold:
            position = -1
        signal[i] = position
    return pd.Series(signal, index=index, name="signal")


def conviction_weighted_signal(
    expected_next_returns: pd.Series | np.ndarray,
    *,
    scale: float,
    clip: float = 1.0,
) -> pd.Series:
    """Continuous-position policy: ``position[t] = clip(E[Δy_{t+1}] / scale, -clip, +clip)``.

    Unlike ``sign_signal`` and ``thresholded_hold_signal``, the output is a
    float position in ``[-clip, +clip]`` rather than a discrete ``{-1, 0, +1}``
    integer. Larger ``|E[Δy_{t+1}]|`` means a larger position; small predictions
    get small (but non-zero) positions instead of being discarded by a dead
    zone.

    ``scale`` must be a strictly positive float in the same units as the
    expected returns (log-return per bar). The walk-forward driver computes it
    per window from training-side predictions to keep the signal path
    leakage-free, then passes it here.

    References: §8 sign-based trading signal (evaluation layer)
    """
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be a finite strictly positive float, got {scale!r}.")
    if not np.isfinite(clip) or clip <= 0.0:
        raise ValueError(f"clip must be a finite strictly positive float, got {clip!r}.")
    values, index = _coerce_expected_returns(expected_next_returns)
    positions = np.clip(values / scale, -clip, clip).astype(float)
    return pd.Series(positions, index=index, name="signal")


def build_signal(
    expected_next_returns: pd.Series | np.ndarray,
    *,
    policy: SignalPolicy,
    threshold: float = 0.0,
    initial_position: int = 0,
    scale: float | None = None,
    clip: float = 1.0,
) -> pd.Series:
    """Dispatch helper that turns expected returns into a signal under ``policy``.

    ``policy="sign"`` ignores every other parameter and reproduces
    :func:`sign_signal`. ``policy="thresholded_hold"`` calls
    :func:`thresholded_hold_signal`, which holds the previous position inside
    the dead-zone — the turnover-aware variant used for cost-sensitivity
    sweeps. ``policy="conviction_weighted"`` calls
    :func:`conviction_weighted_signal` and requires ``scale`` to be supplied
    (the per-window standardizer computed by the walk-forward driver from
    training-only predictions).

    References: §8 sign-based trading signal (evaluation layer)
    """
    if policy not in _VALID_SIGNAL_POLICIES:
        raise ValueError(
            f"policy must be one of {_VALID_SIGNAL_POLICIES}, got {policy!r}."
        )
    if policy == "sign":
        return sign_signal(expected_next_returns)
    if policy == "thresholded_hold":
        return thresholded_hold_signal(
            expected_next_returns,
            threshold=threshold,
            initial_position=initial_position,
        )
    if scale is None:
        raise ValueError(
            "policy='conviction_weighted' requires the per-window scale to be supplied; "
            "the walk-forward driver computes it from training-side predicted returns."
        )
    return conviction_weighted_signal(expected_next_returns, scale=scale, clip=clip)


def signal_from_filter_result(
    result: ForwardFilterResult,
    *,
    threshold: float = 0.0,
    index: pd.Index | None = None,
) -> pd.Series:
    """Convenience wrapper that builds a thresholded signal from a filter result.

    ``result.expected_next_returns`` is a plain numpy array, so callers that
    want to preserve their return-series index should pass it via ``index``.
    With ``threshold=0`` this reduces to ``sign_signal``.

    References: §8 sign-based trading signal (evaluation layer)
    """
    source: pd.Series | np.ndarray = result.expected_next_returns
    if index is not None:
        if len(index) != result.expected_next_returns.shape[0]:
            raise ValueError(
                "index length must match expected_next_returns; "
                f"got {len(index)} vs {result.expected_next_returns.shape[0]}."
            )
        source = pd.Series(result.expected_next_returns, index=index)
    return thresholded_signal(source, threshold=threshold)


def align_signal_with_future_return(
    signal: pd.Series,
    realized_returns: pd.Series,
) -> pd.Series:
    """Return ``signal[t-1] * realized_returns[t]`` indexed at bar ``t``.

    The two Series must share an index; sharing the index is the primary
    guard against accidental shift mistakes upstream. The output is indexed
    at the realization bar and drops the first observation (which has no
    preceding signal), so the caller can compute strategy returns without
    further shifting. Inputs must be fully finite — drop NaN from realized
    returns before calling.

    References: §8 sign-based trading signal (evaluation layer)
    """
    if not isinstance(signal, pd.Series):
        raise TypeError(f"signal must be a pd.Series, got {type(signal).__name__}.")
    if not isinstance(realized_returns, pd.Series):
        raise TypeError(
            f"realized_returns must be a pd.Series, got {type(realized_returns).__name__}."
        )
    if len(signal) != len(realized_returns):
        raise ValueError(
            "signal and realized_returns must have the same length; "
            f"got {len(signal)} vs {len(realized_returns)}."
        )
    if not signal.index.equals(realized_returns.index):
        raise ValueError(
            "signal and realized_returns must share the same index; "
            "align both to the same timeline before calling this helper."
        )
    if len(signal) < 2:
        raise ValueError("at least two observations are required to align signal with returns.")

    signal_values = np.asarray(signal, dtype=float)
    return_values = np.asarray(realized_returns, dtype=float)
    if not np.all(np.isfinite(signal_values)):
        raise ValueError("signal must contain only finite values.")
    if not np.all(np.isfinite(return_values)):
        raise ValueError("realized_returns must contain only finite values; drop NaN/inf first.")

    realized = signal_values[:-1] * return_values[1:]
    return pd.Series(
        realized,
        index=realized_returns.index[1:],
        name="strategy_return",
    )


def _coerce_expected_returns(
    expected_next_returns: pd.Series | np.ndarray,
) -> tuple[np.ndarray, pd.Index]:
    if isinstance(expected_next_returns, pd.Series):
        # pd.Series is inherently 1-D; only non-Series inputs need an explicit ndim check.
        values = np.asarray(expected_next_returns, dtype=float)
        index: pd.Index = expected_next_returns.index
    else:
        values = np.asarray(expected_next_returns, dtype=float)
        if values.ndim != 1:
            raise ValueError(
                "expected_next_returns must be one-dimensional, " f"got shape {values.shape}."
            )
        index = pd.RangeIndex(values.shape[0])
    if values.size == 0:
        raise ValueError("expected_next_returns must contain at least one observation.")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "expected_next_returns must contain only finite values; drop NaN/inf first."
        )
    return values, index
