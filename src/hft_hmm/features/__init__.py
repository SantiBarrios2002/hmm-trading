"""Side-information feature modules used to bias HMM transition dynamics."""

from hft_hmm.features.volatility_ratio import (
    VOLATILITY_RATIO_REFERENCE,
    VolatilityRatioConfig,
    ewma_volatility,
    volatility_ratio,
)

__all__ = [
    "VOLATILITY_RATIO_REFERENCE",
    "VolatilityRatioConfig",
    "ewma_volatility",
    "volatility_ratio",
]
