"""Return preprocessing and sampling frequency utilities.

Engineering utility — these functions are not paper-faithful reproductions but
support data ingestion and feature engineering pipelines.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from hft_hmm.core.references import ENGINEERING_APPROXIMATION

__category__: Final[str] = ENGINEERING_APPROXIMATION


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute continuously compounded log returns: ln(p_t / p_{t-1}).

    The first element is NaN because there is no preceding price. Call
    ``.dropna()`` before passing returns to model-fitting functions.

    References: Engineering utility
    """
    non_positive = prices <= 0
    if non_positive.any():
        bad_positions = np.flatnonzero(non_positive.to_numpy())
        if len(bad_positions) > 5:
            shown = ", ".join(str(pos) for pos in bad_positions[:5])
            location_hint = f"{shown}, ..."
        else:
            location_hint = ", ".join(str(pos) for pos in bad_positions)
        raise ValueError(
            "prices must be strictly positive to compute log returns; "
            f"found {len(bad_positions)} non-positive value(s) at position(s): {location_hint}."
        )
    return np.log(prices / prices.shift(1))


def resample_prices(
    frame: pd.DataFrame,
    freq: str,
    *,
    price_column: str = "price",
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Resample market data to ``freq``, keeping the last price in each bar.

    ``freq`` follows pandas offset alias notation (e.g. ``"1D"``, ``"5min"``,
    ``"1h"``). Bars with no observations are dropped. When a ``volume`` column
    is present it is summed within each bar. Output is sorted by timestamp.

    References: Engineering utility
    """
    indexed = frame.set_index(timestamp_column).sort_index()
    agg: dict[str, str] = {price_column: "last"}
    if "volume" in indexed.columns and price_column != "volume":
        agg["volume"] = "sum"
    resampled = indexed[list(agg.keys())].resample(freq).agg(agg).dropna(subset=[price_column])
    return resampled.reset_index()


def train_test_split_time(
    frame: pd.DataFrame,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame chronologically without shuffling.

    ``test_fraction`` controls the proportion of rows assigned to the test set.
    The test set always follows the training set in time, so no future data
    leaks into the training set.

    Raises ``ValueError`` for ``test_fraction`` outside the open interval (0, 1),
    or when the input is too short to produce non-empty train and test splits.

    References: Engineering utility
    """
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction!r}.")
    if len(frame) < 2:
        raise ValueError("frame must contain at least 2 rows for a chronological split.")
    split = int(len(frame) * (1.0 - test_fraction))
    if split == 0:
        raise ValueError(
            "test_fraction and frame length must yield non-empty train and test splits."
        )
    train = frame.iloc[:split].reset_index(drop=True)
    test = frame.iloc[split:].reset_index(drop=True)
    return train, test
