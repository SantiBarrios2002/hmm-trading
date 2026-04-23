"""Tests for the intraday seasonality side-information predictor."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from hft_hmm.core.references import ENGINEERING_APPROXIMATION, module_category
from hft_hmm.features.seasonality import (
    DEFAULT_BUCKET_MINUTES,
    DEFAULT_EXCHANGE_TZ,
    INTRADAY_SEASONALITY_REFERENCE,
    SeasonalityConfig,
    intraday_seasonality,
)

seasonality_module = importlib.import_module("hft_hmm.features.seasonality")


def _series(timestamps: list[str], *, tz: str | None = "UTC") -> pd.Series:
    parsed = pd.to_datetime(timestamps)
    if tz is None:
        index = pd.DatetimeIndex(parsed)
    elif parsed.tz is None:
        index = pd.DatetimeIndex(parsed).tz_localize(tz)
    else:
        index = pd.DatetimeIndex(parsed).tz_convert(tz)
    values = np.linspace(100.0, 100.0 + index.shape[0] - 1, index.shape[0])
    return pd.Series(values, index=index, name="price")


def test_module_is_engineering_approximation() -> None:
    assert module_category(seasonality_module) == ENGINEERING_APPROXIMATION


def test_paper_reference_points_to_section_4_2() -> None:
    assert INTRADAY_SEASONALITY_REFERENCE.section == "§4.2"
    assert "seasonality" in INTRADAY_SEASONALITY_REFERENCE.topic.lower()


def test_intraday_seasonality_uses_expected_defaults() -> None:
    assert DEFAULT_EXCHANGE_TZ == "America/Chicago"
    assert DEFAULT_BUCKET_MINUTES == 1


def test_intraday_seasonality_maps_930_chicago_consistently_across_dst() -> None:
    prices = _series(
        [
            "2024-01-15 15:30:00+00:00",  # 09:30 CST
            "2024-07-15 14:30:00+00:00",  # 09:30 CDT
        ]
    )

    bucket = intraday_seasonality(prices, normalize=False)
    normalized = intraday_seasonality(prices)

    assert bucket.tolist() == [570, 570]
    np.testing.assert_allclose(normalized.to_numpy(), np.array([570.0 / 1440.0] * 2))


def test_intraday_seasonality_supports_coarser_bucket_sizes() -> None:
    prices = _series(
        [
            "2024-01-02 15:30:00+00:00",  # 09:30 CST
            "2024-01-02 15:59:00+00:00",  # 09:59 CST
            "2024-01-02 16:00:00+00:00",  # 10:00 CST
        ]
    )

    bucket = intraday_seasonality(prices, bucket_minutes=30, normalize=False)
    normalized = intraday_seasonality(prices, bucket_minutes=30)

    assert bucket.tolist() == [19, 19, 20]
    np.testing.assert_allclose(
        normalized.to_numpy(),
        np.array([19.0 / 48.0, 19.0 / 48.0, 20.0 / 48.0]),
    )


def test_intraday_seasonality_preserves_index_and_sets_feature_name() -> None:
    prices = _series(["2024-01-02 15:30:00+00:00", "2024-01-02 15:31:00+00:00"])

    feature = intraday_seasonality(prices)

    pd.testing.assert_index_equal(feature.index, prices.index)
    assert feature.name == "seasonality"


def test_intraday_seasonality_is_deterministic() -> None:
    prices = _series(
        [
            "2024-01-02 15:30:00+00:00",
            "2024-01-02 15:31:00+00:00",
            "2024-01-02 15:32:00+00:00",
        ]
    )

    first = intraday_seasonality(prices, bucket_minutes=5)
    second = intraday_seasonality(prices, bucket_minutes=5)

    pd.testing.assert_series_equal(first, second)


def test_intraday_seasonality_accepts_config_object() -> None:
    prices = _series(
        [
            "2024-01-02 15:30:00+00:00",  # 10:30 EST
            "2024-01-02 15:59:00+00:00",  # 10:59 EST
            "2024-01-02 16:00:00+00:00",  # 11:00 EST
        ]
    )
    config = SeasonalityConfig(
        exchange_tz="America/New_York",
        bucket_minutes=30,
        normalize=False,
    )

    feature = intraday_seasonality(prices, config=config)

    assert feature.tolist() == [21, 21, 22]


def test_intraday_seasonality_rejects_non_default_kwargs_with_config() -> None:
    prices = _series(["2024-01-02 15:30:00+00:00"])
    config = SeasonalityConfig(bucket_minutes=5)

    with pytest.raises(TypeError, match="does not accept keyword parameters"):
        intraday_seasonality(prices, config=config, normalize=False)


def test_intraday_seasonality_rejects_non_series_input() -> None:
    with pytest.raises(TypeError, match="pd.Series"):
        intraday_seasonality(np.array([1.0, 2.0, 3.0]))  # type: ignore[arg-type]


def test_intraday_seasonality_rejects_non_datetime_index() -> None:
    prices = pd.Series([100.0, 101.0], index=[0, 1], name="price")

    with pytest.raises(TypeError, match="DatetimeIndex"):
        intraday_seasonality(prices)


def test_intraday_seasonality_rejects_tz_naive_index() -> None:
    prices = _series(["2024-01-02 09:30:00", "2024-01-02 09:31:00"], tz=None)

    with pytest.raises(ValueError, match="tz-aware"):
        intraday_seasonality(prices)


def test_intraday_seasonality_rejects_non_utc_index() -> None:
    prices = _series(
        ["2024-01-02 09:30:00", "2024-01-02 09:31:00"],
        tz="America/New_York",
    )

    with pytest.raises(ValueError, match="must be in UTC"):
        intraday_seasonality(prices)


@pytest.mark.parametrize("bad_bucket_minutes", [0, -1, 1441])
def test_intraday_seasonality_rejects_invalid_bucket_minutes(bad_bucket_minutes: int) -> None:
    prices = _series(["2024-01-02 15:30:00+00:00"])

    with pytest.raises(ValueError, match="bucket_minutes"):
        intraday_seasonality(prices, bucket_minutes=bad_bucket_minutes)


def test_intraday_seasonality_rejects_non_integer_bucket_minutes() -> None:
    prices = _series(["2024-01-02 15:30:00+00:00"])

    with pytest.raises(TypeError, match="integer"):
        intraday_seasonality(prices, bucket_minutes=1.5)  # type: ignore[arg-type]


def test_intraday_seasonality_rejects_unknown_exchange_timezone() -> None:
    prices = _series(["2024-01-02 15:30:00+00:00"])

    with pytest.raises(ValueError, match="Unknown exchange timezone"):
        intraday_seasonality(prices, exchange_tz="Mars/Olympus")


def test_seasonality_config_validates_on_construction() -> None:
    with pytest.raises(ValueError, match="bucket_minutes"):
        SeasonalityConfig(bucket_minutes=0)
    with pytest.raises(ValueError, match="Unknown exchange timezone"):
        SeasonalityConfig(exchange_tz="Mars/Olympus")
    with pytest.raises(TypeError, match="normalize"):
        SeasonalityConfig(normalize="yes")  # type: ignore[arg-type]


def test_seasonality_config_exposes_defaults() -> None:
    config = SeasonalityConfig()
    assert config.exchange_tz == DEFAULT_EXCHANGE_TZ
    assert config.bucket_minutes == DEFAULT_BUCKET_MINUTES
    assert config.normalize is True
