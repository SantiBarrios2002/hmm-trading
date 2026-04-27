"""Tests for standalone predictor walk-forward backtests."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hft_hmm.core import EVALUATION_LAYER, module_category
from hft_hmm.experiments.standalone_predictor import (
    StandaloneExperimentConfig,
    StandalonePredictorConfig,
    StandalonePredictorResult,
    StandalonePredictorWindow,
    StandaloneWalkForwardConfig,
    run_standalone_experiment,
    standalone_predictor_backtest,
    standalone_run_id,
)
from hft_hmm.features.seasonality import SeasonalityConfig
from hft_hmm.features.splines import SplinePredictorConfig
from hft_hmm.features.volatility_ratio import VolatilityRatioConfig

standalone_module = importlib.import_module("hft_hmm.experiments.standalone_predictor")

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "es_1min_month.csv"
FIXTURE_SHA256 = "c81161b1932361e119483a37fa27b2e16ce39020bcfcc3e871812c5cb7a9ca34"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(
    *,
    n_days: int,
    bars_per_day: int,
    seed: int = 0,
) -> pd.Series:
    """Deterministic UTC-aware return series."""
    rng = np.random.default_rng(seed)
    total = n_days * bars_per_day
    values = rng.normal(0.0, 0.01, total)
    dates = pd.bdate_range("2024-01-02", periods=n_days, tz="UTC")
    timestamps = [date + pd.Timedelta(minutes=m) for date in dates for m in range(bars_per_day)]
    index = pd.DatetimeIndex(timestamps, tz="UTC")
    return pd.Series(values, index=index, name="log_return")


def _vol_ratio_config(
    h_days: int = 5,
    t_days: int = 2,
    retrain_every_days: int = 2,
) -> StandalonePredictorConfig:
    """Small-window vol-ratio config suitable for synthetic fixtures."""
    return StandalonePredictorConfig(
        predictor="volatility_ratio",
        walk_forward=StandaloneWalkForwardConfig(
            h_days=h_days, t_days=t_days, retrain_every_days=retrain_every_days
        ),
        spline=SplinePredictorConfig(n_knots=3, min_obs=10),
        vol_ratio=VolatilityRatioConfig(fast_window=10, slow_window=20),
    )


def _seasonality_config(
    h_days: int = 5,
    t_days: int = 2,
    retrain_every_days: int = 2,
) -> StandalonePredictorConfig:
    """Seasonality config using 30-min buckets for synthetic fixtures."""
    return StandalonePredictorConfig(
        predictor="seasonality",
        walk_forward=StandaloneWalkForwardConfig(
            h_days=h_days, t_days=t_days, retrain_every_days=retrain_every_days
        ),
        spline=SplinePredictorConfig(n_knots=3, min_obs=10),
        seasonality=SeasonalityConfig(bucket_minutes=30),
    )


@pytest.fixture
def es_returns() -> pd.Series:
    """Log returns from the tracked ES 1-min CSV fixture."""
    from hft_hmm.data import load_csv_market_data
    from hft_hmm.preprocessing import compute_log_returns, resample_prices

    frame = load_csv_market_data(str(FIXTURE_CSV))
    resampled = resample_prices(frame, freq="1min")
    prices = resampled.set_index("timestamp")["price"]
    returns = compute_log_returns(prices).dropna()
    returns.name = "log_return"
    return returns


# ---------------------------------------------------------------------------
# Module taxonomy
# ---------------------------------------------------------------------------


def test_standalone_module_is_evaluation_layer() -> None:
    assert module_category(standalone_module) == EVALUATION_LAYER


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_standalone_wf_config_defaults() -> None:
    cfg = StandaloneWalkForwardConfig()
    assert cfg.h_days == 23
    assert cfg.t_days == 1
    assert cfg.retrain_every_days == 1


def test_standalone_wf_config_normalizes_retrain_every_days() -> None:
    cfg = StandaloneWalkForwardConfig(t_days=3)
    assert cfg.retrain_every_days == 3


def test_standalone_wf_config_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="h_days"):
        StandaloneWalkForwardConfig(h_days=0)
    with pytest.raises(ValueError, match="t_days"):
        StandaloneWalkForwardConfig(t_days=0)
    with pytest.raises(ValueError, match="retrain_every_days"):
        StandaloneWalkForwardConfig(retrain_every_days=0)


def test_standalone_predictor_config_rejects_unknown_predictor() -> None:
    with pytest.raises(ValueError, match="predictor"):
        StandalonePredictorConfig(
            predictor="unknown",  # type: ignore[arg-type]
            walk_forward=StandaloneWalkForwardConfig(),
        )


def test_standalone_predictor_config_rejects_invalid_subconfigs() -> None:
    with pytest.raises(TypeError, match="spline"):
        StandalonePredictorConfig(
            predictor="volatility_ratio",
            walk_forward=StandaloneWalkForwardConfig(),
            spline={"n_knots": 3},  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="vol_ratio"):
        StandalonePredictorConfig(
            predictor="volatility_ratio",
            walk_forward=StandaloneWalkForwardConfig(),
            vol_ratio={"fast_window": 10},  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="seasonality"):
        StandalonePredictorConfig(
            predictor="seasonality",
            walk_forward=StandaloneWalkForwardConfig(),
            seasonality={"bucket_minutes": 5},  # type: ignore[arg-type]
        )


def test_standalone_experiment_config_rejects_non_string_sha256() -> None:
    from hft_hmm.config.experiment_config import DataSourceConfig

    with pytest.raises(ValueError, match="sha256 must be a string"):
        StandaloneExperimentConfig(
            data=DataSourceConfig(kind="csv", path=str(FIXTURE_CSV)),
            frequency="1min",
            predictor="volatility_ratio",
            walk_forward=StandaloneWalkForwardConfig(h_days=10, t_days=2, retrain_every_days=2),
            sha256=123,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Two-window walk-forward on synthetic fixture
# ---------------------------------------------------------------------------


def test_vol_ratio_standalone_covers_two_windows() -> None:
    returns = _make_returns(n_days=10, bars_per_day=300, seed=0)
    config = _vol_ratio_config()

    result = standalone_predictor_backtest(returns, config)

    assert isinstance(result, StandalonePredictorResult)
    assert len(result.windows) >= 2
    for w in result.windows:
        assert isinstance(w, StandalonePredictorWindow)
        assert w.n_forecast_obs >= 2
        assert isinstance(w.summary, pd.DataFrame)


def test_backtest_rejects_non_config() -> None:
    returns = _make_returns(n_days=10, bars_per_day=300, seed=0)
    with pytest.raises(TypeError, match="StandalonePredictorConfig"):
        standalone_predictor_backtest(returns, {"predictor": "volatility_ratio"})  # type: ignore[arg-type]


def test_backtest_rejects_invalid_cost() -> None:
    returns = _make_returns(n_days=10, bars_per_day=300, seed=0)
    with pytest.raises(ValueError, match="cost_bps_per_turnover"):
        standalone_predictor_backtest(
            returns,
            _vol_ratio_config(),
            cost_bps_per_turnover=-1.0,
        )


def test_seasonality_standalone_covers_two_windows() -> None:
    returns = _make_returns(n_days=10, bars_per_day=300, seed=1)
    config = _seasonality_config()

    result = standalone_predictor_backtest(returns, config)

    assert isinstance(result, StandalonePredictorResult)
    assert len(result.windows) >= 2
    for w in result.windows:
        assert isinstance(w, StandalonePredictorWindow)
        assert w.n_forecast_obs >= 2


# ---------------------------------------------------------------------------
# No-leakage boundary checks
# ---------------------------------------------------------------------------


def test_no_leakage_training_ends_before_forecast_starts() -> None:
    returns = _make_returns(n_days=10, bars_per_day=300, seed=2)
    config = _vol_ratio_config()

    result = standalone_predictor_backtest(returns, config)

    for w in result.windows:
        assert (
            w.train_end < w.forecast_start
        ), f"window {w.index}: train_end={w.train_end} >= forecast_start={w.forecast_start}"


def test_leakage_guard_assertion_trips_on_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch Series.loc so the forecast slice overlaps training; the assert must fire."""
    returns = _make_returns(n_days=8, bars_per_day=100, seed=0)
    config = _vol_ratio_config(h_days=3, t_days=2, retrain_every_days=2)

    original_loc = pd.Series.loc
    call_counter = {"n": 0}

    class _OverlapLoc:
        def __init__(self, s: pd.Series) -> None:
            self._s = s

        def __getitem__(self, key):  # type: ignore[no-untyped-def]
            call_counter["n"] += 1
            if call_counter["n"] == 2:
                # Return a slice that overlaps the training region.
                return self._s.iloc[: 3 * 100 + 1]
            return original_loc.__get__(self._s, pd.Series)[key]

    def _patched(self: pd.Series):  # type: ignore[no-untyped-def]
        return _OverlapLoc(self)

    monkeypatch.setattr(pd.Series, "loc", property(_patched))
    with pytest.raises(AssertionError, match="leakage guard"):
        standalone_predictor_backtest(returns, config)


# ---------------------------------------------------------------------------
# HMM exclusion
# ---------------------------------------------------------------------------


def test_standalone_does_not_instantiate_gaussian_hmm(monkeypatch: pytest.MonkeyPatch) -> None:
    import hft_hmm.models.gaussian_hmm as hmm_module

    instantiated: list[bool] = []
    original_init = hmm_module.GaussianHMMWrapper.__init__

    def tracking_init(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        instantiated.append(True)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(hmm_module.GaussianHMMWrapper, "__init__", tracking_init)

    returns = _make_returns(n_days=10, bars_per_day=300, seed=0)
    config = _vol_ratio_config()
    standalone_predictor_backtest(returns, config)

    assert (
        len(instantiated) == 0
    ), "GaussianHMMWrapper was instantiated inside the standalone predictor path"


# ---------------------------------------------------------------------------
# Signal and return alignment (strategy contract)
# ---------------------------------------------------------------------------


def test_signal_alignment_follows_strategy_contract() -> None:
    """pre_cost_returns index must be the tail of each window's forecast index."""
    returns = _make_returns(n_days=10, bars_per_day=300, seed=3)
    config = _vol_ratio_config()

    result = standalone_predictor_backtest(returns, config)

    expected_parts: list[pd.Index] = []
    for w in result.windows:
        window_signal = result.signal.loc[w.forecast_start : w.forecast_end]
        # The aligned return series drops the first bar of each window.
        expected_parts.append(window_signal.index[1:])

    expected_index = expected_parts[0]
    for part in expected_parts[1:]:
        expected_index = expected_index.append(part)

    pd.testing.assert_index_equal(result.pre_cost_returns.index, expected_index)
    pd.testing.assert_index_equal(result.post_cost_returns.index, expected_index)


def test_signal_and_returns_have_consistent_lengths() -> None:
    returns = _make_returns(n_days=10, bars_per_day=300, seed=4)
    config = _seasonality_config()

    result = standalone_predictor_backtest(returns, config)

    n_windows = len(result.windows)
    total_signal = result.signal.shape[0]
    total_returns = result.pre_cost_returns.shape[0]
    # Each window drops its first bar when aligning signal with future return.
    assert total_returns == total_signal - n_windows


# ---------------------------------------------------------------------------
# Finite metrics — integration tests on the tracked ES fixture
# ---------------------------------------------------------------------------


def test_vol_ratio_standalone_on_fixture_produces_finite_metrics(
    es_returns: pd.Series,
) -> None:
    config = StandalonePredictorConfig(
        predictor="volatility_ratio",
        walk_forward=StandaloneWalkForwardConfig(h_days=10, t_days=2, retrain_every_days=2),
        spline=SplinePredictorConfig(n_knots=5, min_obs=20),
        vol_ratio=VolatilityRatioConfig(fast_window=50, slow_window=100),
    )

    result = standalone_predictor_backtest(es_returns, config, cost_bps_per_turnover=1.0)

    assert len(result.windows) >= 2
    for col in ("cumulative_return", "sharpe_ratio", "max_drawdown", "hit_rate"):
        for mode in ("pre-cost", "post-cost"):
            val = result.summary.loc[mode, col]
            assert np.isfinite(float(val)), f"{mode} {col} is not finite: {val}"


def test_seasonality_standalone_on_fixture_produces_finite_metrics(
    es_returns: pd.Series,
) -> None:
    config = StandalonePredictorConfig(
        predictor="seasonality",
        walk_forward=StandaloneWalkForwardConfig(h_days=10, t_days=2, retrain_every_days=2),
        spline=SplinePredictorConfig(n_knots=5, min_obs=20),
        seasonality=SeasonalityConfig(bucket_minutes=1, exchange_tz="America/Chicago"),
    )

    result = standalone_predictor_backtest(es_returns, config, cost_bps_per_turnover=1.0)

    assert len(result.windows) >= 2
    for col in ("cumulative_return", "sharpe_ratio", "max_drawdown", "hit_rate"):
        for mode in ("pre-cost", "post-cost"):
            val = result.summary.loc[mode, col]
            assert np.isfinite(float(val)), f"{mode} {col} is not finite: {val}"


# ---------------------------------------------------------------------------
# StandaloneExperimentConfig YAML round-trip
# ---------------------------------------------------------------------------


def test_standalone_experiment_config_yaml_round_trip(tmp_path: Path) -> None:
    from hft_hmm.config.experiment_config import DataSourceConfig

    cfg = StandaloneExperimentConfig(
        data=DataSourceConfig(kind="csv", path="tests/fixtures/es_1min_month.csv"),
        frequency="1min",
        predictor="volatility_ratio",
        walk_forward=StandaloneWalkForwardConfig(h_days=10, t_days=2, retrain_every_days=2),
        spline=SplinePredictorConfig(n_knots=5),
        vol_ratio=VolatilityRatioConfig(fast_window=50, slow_window=100),
        cost_bps_per_turnover=1.0,
        notes="test",
        sha256=FIXTURE_SHA256,
    )

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_bytes(cfg.to_yaml_bytes())
    loaded = StandaloneExperimentConfig.from_yaml(yaml_path)

    assert loaded.predictor == cfg.predictor
    assert loaded.walk_forward.h_days == cfg.walk_forward.h_days
    assert loaded.vol_ratio.fast_window == cfg.vol_ratio.fast_window
    assert loaded.sha256 == cfg.sha256
    assert standalone_run_id(loaded) == standalone_run_id(cfg)


def test_run_standalone_experiment_writes_expected_artifact_layout(tmp_path: Path) -> None:
    from hft_hmm.config.experiment_config import DataSourceConfig

    cfg = StandaloneExperimentConfig(
        data=DataSourceConfig(kind="csv", path=str(FIXTURE_CSV)),
        frequency="1min",
        predictor="volatility_ratio",
        walk_forward=StandaloneWalkForwardConfig(h_days=10, t_days=2, retrain_every_days=2),
        spline=SplinePredictorConfig(n_knots=5, min_obs=20),
        vol_ratio=VolatilityRatioConfig(fast_window=50, slow_window=100),
        cost_bps_per_turnover=1.0,
        notes="artifact-test",
        sha256=FIXTURE_SHA256,
    )

    artifacts = run_standalone_experiment(cfg, runs_root=tmp_path)

    assert artifacts.directory == tmp_path / artifacts.run_id
    assert (artifacts.directory / "config.yaml").is_file()
    assert (artifacts.directory / "metrics.json").is_file()
    assert (artifacts.directory / "log.jsonl").is_file()
    assert (artifacts.directory / "figures").is_dir()
    metrics = json.loads((artifacts.directory / "metrics.json").read_text())
    assert metrics["run_id"] == artifacts.run_id
    assert metrics["predictor"] == "volatility_ratio"
    assert metrics["reproducible"] is True
    assert metrics["n_windows"] == len(artifacts.result.windows)


def test_standalone_configs_are_loadable() -> None:
    """Both tracked YAML configs round-trip through StandaloneExperimentConfig."""
    configs_root = Path(__file__).parent.parent / "configs"
    for name in (
        "example_es_vol_ratio_standalone.yaml",
        "example_es_seasonality_standalone.yaml",
    ):
        cfg = StandaloneExperimentConfig.from_yaml(configs_root / name)
        assert cfg.predictor in ("volatility_ratio", "seasonality")
        assert cfg.sha256 == FIXTURE_SHA256
