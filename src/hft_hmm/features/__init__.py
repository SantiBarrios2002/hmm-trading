"""Side-information feature modules used to bias HMM transition dynamics."""

from hft_hmm.features.seasonality import (
    DEFAULT_BUCKET_MINUTES,
    DEFAULT_EXCHANGE_TZ,
    INTRADAY_SEASONALITY_REFERENCE,
    SeasonalityConfig,
    intraday_seasonality,
)
from hft_hmm.features.splines import (
    DEFAULT_MIN_OBS,
    DEFAULT_N_KNOTS,
    SPLINE_PREDICTOR_REFERENCE,
    SplinePredictorConfig,
    SplinePredictorResult,
    fit_spline_predictor,
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
    "DEFAULT_MIN_OBS",
    "DEFAULT_N_KNOTS",
    "SPLINE_PREDICTOR_REFERENCE",
    "SplinePredictorConfig",
    "SplinePredictorResult",
    "fit_spline_predictor",
    "VOLATILITY_RATIO_REFERENCE",
    "VolatilityRatioConfig",
    "ewma_volatility",
    "volatility_ratio",
]
