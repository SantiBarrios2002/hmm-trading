"""Intraday seasonality side-information predictor (Christensen et al. §4.2).

The paper's second predictor is defined over exchange-local clock time rather
than the repository's canonical UTC timestamps. This module converts a
UTC-indexed price series into a deterministic time-of-day coordinate aligned to
the input index.

The UTC -> exchange-local conversion is paper-faithful. The final scalar
encoding emitted by ``intraday_seasonality`` is an engineering approximation:
local clock time is mapped into fixed-width buckets and optionally normalized to
the unit interval so the spline fitter in Issue 15 can consume a one-dimensional
predictor.

References: §4.2 intraday seasonality
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from hft_hmm.core.references import ENGINEERING_APPROXIMATION, PaperReference, reference

__category__: Final[str] = ENGINEERING_APPROXIMATION
INTRADAY_SEASONALITY_REFERENCE: Final[PaperReference] = reference("§4.2", "intraday seasonality")

DEFAULT_EXCHANGE_TZ: Final[str] = "America/Chicago"
DEFAULT_BUCKET_MINUTES: Final[int] = 1
_MINUTES_PER_DAY: Final[int] = 24 * 60


@dataclass(frozen=True)
class SeasonalityConfig:
    """Typed parameter bundle for the intraday seasonality predictor."""

    exchange_tz: str = DEFAULT_EXCHANGE_TZ
    bucket_minutes: int = DEFAULT_BUCKET_MINUTES
    normalize: bool = True

    def __post_init__(self) -> None:
        _validate_exchange_tz(self.exchange_tz)
        _validate_bucket_minutes(self.bucket_minutes)
        if not isinstance(self.normalize, bool):
            raise TypeError(f"normalize must be a bool; got {type(self.normalize).__name__}.")


def intraday_seasonality(
    prices: pd.Series,
    config: SeasonalityConfig | None = None,
    *,
    exchange_tz: str = DEFAULT_EXCHANGE_TZ,
    bucket_minutes: int = DEFAULT_BUCKET_MINUTES,
    normalize: bool = True,
) -> pd.Series:
    """Map UTC timestamps to exchange-local time-of-day buckets.

    The input ``prices`` must be a ``pd.Series`` indexed by a tz-aware UTC
    ``DatetimeIndex``. Each timestamp is converted to ``exchange_tz`` and then
    mapped to a fixed-width clock-time bucket. When ``normalize=True`` the
    returned values lie on ``[0, 1)`` via ``bucket / n_buckets``; otherwise the
    integer bucket ids are returned directly.

    Pass ``config`` to reuse a validated ``SeasonalityConfig``; when omitted,
    one is built from the keyword parameters. Non-default keyword overrides are
    rejected when ``config`` is provided to avoid silent precedence bugs.

    References: §4.2 intraday seasonality
    """
    kwargs_given = (
        exchange_tz != DEFAULT_EXCHANGE_TZ
        or bucket_minutes != DEFAULT_BUCKET_MINUTES
        or normalize is not True
    )
    if config is not None and kwargs_given:
        raise TypeError(
            "intraday_seasonality does not accept keyword parameters when a config is provided."
        )
    if config is None:
        config = SeasonalityConfig(
            exchange_tz=exchange_tz,
            bucket_minutes=bucket_minutes,
            normalize=normalize,
        )
    index = _validate_utc_index(prices)

    local_index = index.tz_convert(config.exchange_tz)
    minute_of_day = local_index.hour * 60 + local_index.minute
    bucket = (minute_of_day // config.bucket_minutes).to_numpy(dtype=np.int64)

    if config.normalize:
        n_buckets = int(np.ceil(_MINUTES_PER_DAY / config.bucket_minutes))
        values = bucket.astype(float) / float(n_buckets)
    else:
        values = bucket

    return pd.Series(values, index=prices.index, name="seasonality")


def _validate_utc_index(prices: pd.Series) -> pd.DatetimeIndex:
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pd.Series, got {type(prices).__name__}.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices.index must be a pd.DatetimeIndex.")
    if prices.index.tz is None:
        raise ValueError("prices.index must be tz-aware; localize to UTC before calling.")
    if not _is_utc_timezone(prices.index):
        raise ValueError("prices.index must be in UTC; convert to UTC before calling.")
    return prices.index


def _is_utc_timezone(index: pd.DatetimeIndex) -> bool:
    tz = index.tz
    return getattr(tz, "key", None) == "UTC" or str(tz) == "UTC"


def _validate_exchange_tz(exchange_tz: str) -> None:
    if not isinstance(exchange_tz, str):
        raise TypeError(f"exchange_tz must be a string; got {type(exchange_tz).__name__}.")
    if not exchange_tz.strip():
        raise ValueError("exchange_tz must be a non-empty string.")
    try:
        ZoneInfo(exchange_tz)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown exchange timezone: {exchange_tz!r}.") from exc


def _validate_bucket_minutes(bucket_minutes: int) -> None:
    if not isinstance(bucket_minutes, (int, np.integer)) or isinstance(bucket_minutes, bool):
        raise TypeError(f"bucket_minutes must be an integer; got {type(bucket_minutes).__name__}.")
    if bucket_minutes < 1 or bucket_minutes > _MINUTES_PER_DAY:
        raise ValueError(
            "bucket_minutes must be between 1 and 1440 inclusive; " f"got {bucket_minutes}."
        )
