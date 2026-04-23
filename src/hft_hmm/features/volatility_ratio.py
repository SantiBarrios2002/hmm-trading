"""Volatility-ratio side-information predictor (Christensen et al. §4.2).

The paper's Predictor I is the ratio of a short-window EWMA volatility estimate
to a long-window EWMA volatility estimate. Both estimators use the J.P. Morgan
RiskMetrics IGARCH(1,1) formulation of Eq. (4):

    σ_{t+1|t} = sqrt( (1 - λ) · Σ_{τ=0}^{ψ} λ^τ · Δy²_{t-τ} )

with decay ``λ = 0.79`` for 1-minute data and window sizes ``ψ_fast = 50`` and
``ψ_slow = 100``. The predictor is ``X_t = σ_{t+1|t}(ψ_fast) / σ_{t+1|t}(ψ_slow)``.

This module is paper-faithful: the finite-sum EWMA form and the default
parameter values follow the paper directly. Downstream consumers (spline
predictor, IOHMM bucketing) treat the output ``pd.Series`` as their input ``x_t``.

References: §4.2 volatility ratio
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from hft_hmm.core.references import PAPER_FAITHFUL, PaperReference, reference

__category__: Final[str] = PAPER_FAITHFUL
VOLATILITY_RATIO_REFERENCE: Final[PaperReference] = reference("§4.2", "volatility ratio")

DEFAULT_DECAY: Final[float] = 0.79
DEFAULT_FAST_WINDOW: Final[int] = 50
DEFAULT_SLOW_WINDOW: Final[int] = 100


@dataclass(frozen=True)
class VolatilityRatioConfig:
    """Typed parameter bundle for the volatility-ratio predictor.

    ``decay`` is the EWMA variance decay λ shared by both estimators.
    ``fast_window`` and ``slow_window`` are the paper's ψ_fast and ψ_slow, i.e.
    the number of historical observations retained in each truncated sum.

    References: §4.2 volatility ratio
    """

    decay: float = DEFAULT_DECAY
    fast_window: int = DEFAULT_FAST_WINDOW
    slow_window: int = DEFAULT_SLOW_WINDOW

    def __post_init__(self) -> None:
        _validate_decay(self.decay)
        _validate_window(self.fast_window, name="fast_window")
        _validate_window(self.slow_window, name="slow_window")
        if self.fast_window >= self.slow_window:
            raise ValueError(
                "fast_window must be strictly less than slow_window; "
                f"got fast_window={self.fast_window}, slow_window={self.slow_window}."
            )


def ewma_volatility(
    returns: pd.Series,
    *,
    decay: float,
    window: int,
) -> pd.Series:
    """Compute the paper's truncated EWMA volatility forecast σ_{t+1|t}.

    At index ``t`` the output is ``sqrt((1 - decay) · Σ_{τ=0}^{window} decay^τ ·
    returns²_{t-τ})``. The first ``window`` positions are NaN because the
    truncated sum requires ``window + 1`` observations. The input ``returns``
    must be numeric and free of NaN; callers should ``.dropna()`` log returns
    before calling this function.

    References: §4.2, Eq. (4)
    """
    _validate_decay(decay)
    _validate_window(window, name="window")
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a pandas Series, got {type(returns).__name__}.")

    values = returns.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("returns must not contain NaN; drop missing values before calling.")

    squared = values**2
    sigma = np.full(values.shape[0], np.nan, dtype=float)
    if values.shape[0] >= window + 1:
        weights = decay ** np.arange(window + 1, dtype=float)
        reversed_weights = weights[::-1]
        windows = sliding_window_view(squared, window + 1)
        variance_tail = (1.0 - decay) * (windows @ reversed_weights)
        sigma[window:] = np.sqrt(variance_tail)

    return pd.Series(sigma, index=returns.index, name=returns.name)


def volatility_ratio(
    returns: pd.Series,
    *,
    fast_window: int = DEFAULT_FAST_WINDOW,
    slow_window: int = DEFAULT_SLOW_WINDOW,
    decay: float = DEFAULT_DECAY,
) -> pd.Series:
    """Compute the paper's volatility-ratio predictor σ_fast / σ_slow.

    Both estimators share ``decay`` and differ only in the truncation window.
    The first ``slow_window`` positions are NaN. The output is a pandas Series
    aligned with the input index.

    References: §4.2 volatility ratio
    """
    config = VolatilityRatioConfig(
        decay=decay,
        fast_window=fast_window,
        slow_window=slow_window,
    )
    sigma_fast = ewma_volatility(returns, decay=config.decay, window=config.fast_window)
    sigma_slow = ewma_volatility(returns, decay=config.decay, window=config.slow_window)
    ratio = sigma_fast / sigma_slow
    return ratio.rename(returns.name)


def _validate_decay(decay: float) -> None:
    if not np.isfinite(decay) or not (0.0 < decay < 1.0):
        raise ValueError(f"decay must be a finite value in (0, 1); got {decay!r}.")


def _validate_window(window: int, *, name: str) -> None:
    if not isinstance(window, (int, np.integer)) or isinstance(window, bool):
        raise TypeError(f"{name} must be an integer; got {type(window).__name__}.")
    if window < 1:
        raise ValueError(f"{name} must be a positive integer; got {window}.")
