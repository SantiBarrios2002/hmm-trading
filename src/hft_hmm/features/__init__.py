"""Side-information feature modules used to bias HMM transition dynamics."""

from hft_hmm.features.seasonality import (
    DEFAULT_BUCKET_MINUTES,
    DEFAULT_EXCHANGE_TZ,
    INTRADAY_SEASONALITY_REFERENCE,
    SeasonalityConfig,
    intraday_seasonality,
)
from hft_hmm.features.volatility_ratio import (
    VOLATILITY_RATIO_REFERENCE,
    VolatilityRatioConfig,
    ewma_volatility,
    volatility_ratio,
)

__all__ = [
    "DEFAULT_BUCKET_MINUTES",
    "DEFAULT_EXCHANGE_TZ",
    "INTRADAY_SEASONALITY_REFERENCE",
    "SeasonalityConfig",
    "intraday_seasonality",
    "VOLATILITY_RATIO_REFERENCE",
    "VolatilityRatioConfig",
    "ewma_volatility",
    "volatility_ratio",
]
