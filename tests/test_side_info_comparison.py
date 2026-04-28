"""Tests for the Gate H side-information comparison runner."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hft_hmm.config.experiment_config import DataSourceConfig
from hft_hmm.core import EVALUATION_LAYER, StateGrid, module_category
from hft_hmm.experiments.side_info_comparison import (
    BASELINE_VARIANT,
    EXPECTED_VARIANTS,
    SideInfoComparisonConfig,
    comparison_id,
    run_side_info_comparison,
)
from hft_hmm.experiments.walk_forward import WalkForwardConfig, walk_forward
from hft_hmm.features.seasonality import SeasonalityConfig
from hft_hmm.features.splines import SplinePredictorConfig
from hft_hmm.features.volatility_ratio import VolatilityRatioConfig
from hft_hmm.models.gaussian_hmm import GaussianHMMResult
from hft_hmm.models.iohmm_approx import BucketedTransitionConfig, BucketedTransitionResult

side_info_module = importlib.import_module("hft_hmm.experiments.side_info_comparison")

REPO_ROOT = Path(__file__).parent.parent
FIXTURE_CSV = REPO_ROOT / "tests" / "fixtures" / "es_1min_month.csv"
FIXTURE_SHA256 = "c81161b1932361e119483a37fa27b2e16ce39020bcfcc3e871812c5cb7a9ca34"
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "example_es_side_info_comparison.yaml"


def _make_config(*, fixture_path: str = str(FIXTURE_CSV)) -> SideInfoComparisonConfig:
    return SideInfoComparisonConfig(
        data=DataSourceConfig(kind="csv", path=fixture_path),
        frequency="1min",
        walk_forward=WalkForwardConfig(
            h_days=10,
            t_days=2,
            retrain_every_days=2,
            k_values=(2,),
            random_state=0,
            n_iter=100,
            tol=1e-4,
            min_variance=1e-8,
            variance_floor_policy="clamp",
        ),
        spline=SplinePredictorConfig(n_knots=5, min_obs=20),
        bucketed_transition=BucketedTransitionConfig(n_buckets=3, smoothing=1.0),
        vol_ratio=VolatilityRatioConfig(fast_window=50, slow_window=100),
        seasonality=SeasonalityConfig(bucket_minutes=1, exchange_tz="America/Chicago"),
        cost_bps_per_turnover=1.0,
        notes="test",
        sha256=FIXTURE_SHA256,
    )


# ---------------------------------------------------------------------------
# Module taxonomy
# ---------------------------------------------------------------------------


def test_side_info_comparison_module_is_evaluation_layer() -> None:
    assert module_category(side_info_module) == EVALUATION_LAYER


# ---------------------------------------------------------------------------
# Config round-trip and deterministic id
# ---------------------------------------------------------------------------


def test_config_yaml_round_trip_and_deterministic_id(tmp_path: Path) -> None:
    cfg = _make_config()
    yaml_path = tmp_path / "comparison.yaml"
    yaml_path.write_bytes(cfg.to_yaml_bytes())
    loaded = SideInfoComparisonConfig.from_yaml(yaml_path)

    assert loaded.frequency == cfg.frequency
    assert loaded.walk_forward.h_days == cfg.walk_forward.h_days
    assert loaded.bucketed_transition.n_buckets == cfg.bucketed_transition.n_buckets
    assert loaded.vol_ratio.fast_window == cfg.vol_ratio.fast_window
    assert loaded.seasonality.bucket_minutes == cfg.seasonality.bucket_minutes
    assert loaded.spline.n_knots == cfg.spline.n_knots
    assert loaded.sha256 == cfg.sha256
    assert comparison_id(loaded) == comparison_id(cfg)
    assert len(comparison_id(cfg)) == 12


def test_example_config_loads() -> None:
    cfg = SideInfoComparisonConfig.from_yaml(EXAMPLE_CONFIG)
    assert cfg.frequency == "1min"
    assert cfg.sha256 == FIXTURE_SHA256
    assert cfg.walk_forward.k_values == (2,)


def test_config_rejects_invalid_subconfigs() -> None:
    base = dict(
        data=DataSourceConfig(kind="csv", path=str(FIXTURE_CSV)),
        frequency="1min",
        walk_forward=WalkForwardConfig(h_days=10, t_days=2, retrain_every_days=2),
        sha256=FIXTURE_SHA256,
    )
    with pytest.raises(TypeError, match="bucketed_transition"):
        SideInfoComparisonConfig(**base, bucketed_transition={"n_buckets": 3})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="spline"):
        SideInfoComparisonConfig(**base, spline={"n_knots": 5})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


@pytest.fixture
def comparison_artifacts(tmp_path: Path):
    cfg = _make_config()
    return run_side_info_comparison(cfg, runs_root=tmp_path)


def test_runner_produces_all_expected_variants(comparison_artifacts) -> None:
    variants = comparison_artifacts.result.variants
    assert set(variants.keys()) == set(EXPECTED_VARIANTS)
    for name in EXPECTED_VARIANTS:
        assert variants[name].variant == name
        assert len(variants[name].windows) >= 2


def test_summary_metrics_are_finite_or_null(comparison_artifacts) -> None:
    summary_path = comparison_artifacts.directory / "summary.json"
    payload = json.loads(summary_path.read_text())
    for variant in EXPECTED_VARIANTS:
        entry = payload["variants"][variant]
        for mode in ("pre-cost", "post-cost"):
            for column in ("cumulative_return", "sharpe_ratio", "max_drawdown", "hit_rate"):
                value = entry["summary"][mode][column]
                assert value is None or np.isfinite(
                    value
                ), f"{variant}.{mode}.{column} is neither finite nor null: {value!r}"


def test_baseline_summary_matches_direct_walk_forward(comparison_artifacts) -> None:
    cfg = comparison_artifacts.config
    from hft_hmm.experiments._data_loading import load_returns_from_source

    returns = load_returns_from_source(cfg.data, frequency=cfg.frequency)
    direct = walk_forward(
        returns, cfg.walk_forward, cost_bps_per_turnover=cfg.cost_bps_per_turnover
    )

    baseline_variant = comparison_artifacts.result.variants[BASELINE_VARIANT]
    pd.testing.assert_frame_equal(baseline_variant.summary, direct.summary)
    pd.testing.assert_series_equal(baseline_variant.signal, direct.signal)
    pd.testing.assert_series_equal(baseline_variant.pre_cost_returns, direct.pre_cost_returns)
    pd.testing.assert_series_equal(baseline_variant.post_cost_returns, direct.post_cost_returns)


def test_variants_do_not_mutate_config(tmp_path: Path) -> None:
    cfg = _make_config()
    snapshot = deepcopy(cfg.to_dict())
    run_side_info_comparison(cfg, runs_root=tmp_path)
    assert cfg.to_dict() == snapshot


def test_artifact_layout_is_written(comparison_artifacts) -> None:
    directory = comparison_artifacts.directory
    assert directory == comparison_artifacts.directory
    assert directory.name == comparison_artifacts.comparison_id
    assert (directory / "config.yaml").is_file()
    assert (directory / "summary.json").is_file()
    assert (directory / "figures").is_dir()
    for variant in EXPECTED_VARIANTS:
        assert (directory / f"{variant}.log.jsonl").is_file()


def test_force_overwrites_existing_directory(tmp_path: Path) -> None:
    cfg = _make_config()
    first = run_side_info_comparison(cfg, runs_root=tmp_path)
    with pytest.raises(FileExistsError):
        run_side_info_comparison(cfg, runs_root=tmp_path)
    second = run_side_info_comparison(cfg, runs_root=tmp_path, force=True)
    assert first.directory == second.directory


def test_summary_includes_required_metric_fields(comparison_artifacts) -> None:
    payload = json.loads((comparison_artifacts.directory / "summary.json").read_text())
    assert payload["comparison_id"] == comparison_artifacts.comparison_id
    for variant in EXPECTED_VARIANTS:
        entry = payload["variants"][variant]
        assert entry["variant"] == variant
        assert entry["comparison_id"] == comparison_artifacts.comparison_id
        assert entry["n_windows"] >= 2
        assert entry["n_forecast_obs"] >= 2
        assert entry["cost_bps_per_turnover"] == 1.0
        assert isinstance(entry["chosen_k_per_window"], list)
        assert all(isinstance(k, int) and k >= 2 for k in entry["chosen_k_per_window"])
        assert "start" in entry["sample_window"] and "end" in entry["sample_window"]


def test_summary_payload_uses_aligned_return_sample_window(comparison_artifacts) -> None:
    payload = json.loads((comparison_artifacts.directory / "summary.json").read_text())
    for variant in EXPECTED_VARIANTS:
        result = comparison_artifacts.result.variants[variant]
        entry = payload["variants"][variant]
        assert entry["n_forecast_obs"] == len(result.pre_cost_returns)
        assert entry["sample_window"] == {
            "start": result.pre_cost_returns.index.min().isoformat(),
            "end": result.pre_cost_returns.index.max().isoformat(),
        }


def test_dynamic_filter_uses_supplied_training_posterior_seed() -> None:
    means = np.array([-1.0, 1.0])
    fitted = GaussianHMMResult(
        state_grid=StateGrid(k=2, means=means, labels=("down", "up")),
        means=means,
        variances=np.array([1.0, 1.0]),
        transition_matrix=np.array([[0.9, 0.1], [0.1, 0.9]]),
        initial_distribution=np.array([1.0, 0.0]),
        log_likelihood=-1.0,
        n_observations=3,
        converged=True,
        n_iter=1,
        random_state=0,
    )
    bucketed = BucketedTransitionResult(
        config=BucketedTransitionConfig(n_buckets=2, smoothing=1.0),
        bucket_boundaries=np.array([0.0]),
        transition_matrices=np.array(
            [
                [[0.9, 0.1], [0.1, 0.9]],
                [[0.9, 0.1], [0.1, 0.9]],
            ]
        ),
        baseline_transition_matrix=np.array([[0.9, 0.1], [0.1, 0.9]]),
        bucket_observation_counts=np.array([1, 1]),
    )

    expected = side_info_module._dynamic_forward_expected_returns(
        forecast_returns=np.array([0.0]),
        forecast_features=np.array([1.0]),
        fitted=fitted,
        bucketed=bucketed,
        initial_state_distribution=np.array([0.0, 1.0]),
    )

    assert expected[0] > 0.0


# ---------------------------------------------------------------------------
# CLI subprocess
# ---------------------------------------------------------------------------


def test_cli_runs_from_repo_root(tmp_path: Path) -> None:
    runs_root = tmp_path / "cli-runs"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_side_info_comparison.py",
            str(EXAMPLE_CONFIG.relative_to(REPO_ROOT)),
            "--runs-root",
            str(runs_root),
            "--force",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    printed = Path(completed.stdout.strip())
    assert printed.exists()
    assert printed.parent == runs_root
    assert (printed / "summary.json").is_file()
    for variant in EXPECTED_VARIANTS:
        assert (printed / f"{variant}.log.jsonl").is_file()
