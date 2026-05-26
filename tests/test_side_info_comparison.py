"""Tests for the Gate H side-information comparison runner."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hft_hmm.config.experiment_config import DataSourceConfig
from hft_hmm.core import EVALUATION_LAYER, StateGrid, module_category
from hft_hmm.experiments.side_info_comparison import (
    BASELINE_VARIANT,
    DEFAULT_HMM_VARIANT,
    EXPECTED_VARIANTS,
    SEASONALITY_HMC_CONTINUOUS_VARIANT,
    SEASONALITY_VARIANT,
    VOLATILITY_RATIO_HMC_CONTINUOUS_VARIANT,
    VOLATILITY_RATIO_VARIANT,
    SideInfoComparisonConfig,
    _CHECKPOINT_BASELINE_WF,
    comparison_id,
    run_side_info_comparison,
)
from hft_hmm.experiments.walk_forward import WalkForwardConfig, walk_forward
from hft_hmm.features.seasonality import SeasonalityConfig
from hft_hmm.features.splines import SplinePredictorConfig
from hft_hmm.features.volatility_ratio import VolatilityRatioConfig
from hft_hmm.models.gaussian_hmm import GaussianHMMResult
from hft_hmm.models.iohmm_approx import BucketedTransitionConfig, BucketedTransitionResult
from hft_hmm.models.iohmm_continuous import ContinuousIOHMMConfig

side_info_module = importlib.import_module("hft_hmm.experiments.side_info_comparison")

REPO_ROOT = Path(__file__).parent.parent
FIXTURE_CSV = REPO_ROOT / "tests" / "fixtures" / "es_1min_month.csv"
FIXTURE_SHA256 = "c81161b1932361e119483a37fa27b2e16ce39020bcfcc3e871812c5cb7a9ca34"
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "example_es_side_info_comparison.yaml"
EXAMPLE_HMC_CONFIG = REPO_ROOT / "configs" / "example_es_databento_side_info_comparison_hmc.yaml"
HMC_VARIANTS = {
    VOLATILITY_RATIO_HMC_CONTINUOUS_VARIANT,
    SEASONALITY_HMC_CONTINUOUS_VARIANT,
}
BUCKETED_VARIANTS = {
    VOLATILITY_RATIO_VARIANT,
    SEASONALITY_VARIANT,
}


def _make_config(*, fixture_path: str = str(FIXTURE_CSV)) -> SideInfoComparisonConfig:
    return SideInfoComparisonConfig(
        data=DataSourceConfig(kind="csv", path=fixture_path),
        frequency="1min",
        walk_forward=WalkForwardConfig(
            h_days=10,
            t_days=5,
            retrain_every_days=5,
            k_values=(2,),
            random_state=0,
            n_iter=100,
            tol=1e-4,
            min_variance=1e-8,
            variance_floor_policy="clamp",
        ),
        spline=SplinePredictorConfig(n_knots=5, min_obs=20),
        bucketed_transition=BucketedTransitionConfig(n_buckets=3, smoothing=1.0),
        continuous_iohmm=ContinuousIOHMMConfig(
            num_warmup=5,
            num_samples=8,
            seed=0,
            rhat_threshold=10.0,
            ess_bulk_threshold=1,
        ),
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
    assert loaded.bucketed_transition.boundary_mode == cfg.bucketed_transition.boundary_mode
    assert loaded.continuous_iohmm.num_samples == cfg.continuous_iohmm.num_samples
    assert SideInfoComparisonConfig.from_dict(loaded.to_dict()) == loaded
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
    assert cfg.continuous_iohmm.num_samples == 12


def test_hmc_example_config_round_trips() -> None:
    cfg = SideInfoComparisonConfig.from_yaml(EXAMPLE_HMC_CONFIG)
    raw = cfg.to_dict()

    assert "continuous_iohmm" in raw
    assert cfg.continuous_iohmm.num_chains == 2
    assert cfg.continuous_iohmm.num_samples == 1000
    assert SideInfoComparisonConfig.from_dict(raw) == cfg


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
    with pytest.raises(TypeError, match="continuous_iohmm"):
        SideInfoComparisonConfig(**base, continuous_iohmm={"num_samples": 8})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def comparison_artifacts(tmp_path_factory):
    cfg = _make_config()
    return run_side_info_comparison(cfg, runs_root=tmp_path_factory.mktemp("side-info-runs"))


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
        returns,
        cfg.walk_forward,
        cost_bps_per_turnover=cfg.cost_bps_per_turnover,
        signal_policy=cfg.signal_policy,
        signal_threshold=cfg.signal_threshold,
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
        log_path = directory / f"{variant}.log.jsonl"
        assert log_path.is_file()
        first_record = json.loads(log_path.read_text().splitlines()[0])
        if variant in BUCKETED_VARIANTS:
            assert first_record["boundary_mode"] == "grid"
        elif variant in HMC_VARIANTS:
            assert "converged" in first_record
            assert "rhat_max" in first_record
            assert "ess_bulk_min" in first_record


@pytest.mark.parametrize(
    ("feature_shape", "expected_mode"),
    [
        ("unique", "quantile"),
        ("duplicate_quantiles", "grid"),
    ],
)
def test_quantile_boundary_mode_logs_effective_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feature_shape: str,
    expected_mode: str,
) -> None:
    def fake_feature(
        variant: str,
        series: pd.Series,
        config: SideInfoComparisonConfig,
    ) -> pd.Series:
        del variant, config
        n_obs = len(series)
        if feature_shape == "unique":
            values = np.linspace(0.0, 1.0, n_obs)
        else:
            # 60% concentration at zero forces duplicate interior quantiles for
            # the n_buckets=4 quantile mode (q25 and q50 both fall in the zero
            # mass), driving the per-window fallback to grid. Smaller training
            # windows under module-scoped fixtures need this tighter ratio than
            # the 0.70 used historically — the module-scoped runner exposes a
            # narrower spread of window sizes, and 0.70 no longer triggered
            # the duplicate-quantile path on every window.
            split = int(0.60 * n_obs)
            values = np.concatenate(
                [
                    np.zeros(split),
                    np.linspace(1.0, 2.0, n_obs - split),
                ]
            )
        return pd.Series(values, index=series.index)

    monkeypatch.setattr(side_info_module, "_build_feature", fake_feature)
    cfg = replace(
        _make_config(),
        bucketed_transition=BucketedTransitionConfig(
            n_buckets=4,
            smoothing=1.0,
            boundary_mode="quantile",
        ),
    )

    artifacts = run_side_info_comparison(cfg, runs_root=tmp_path)

    _no_bucket_variants = {BASELINE_VARIANT, DEFAULT_HMM_VARIANT} | HMC_VARIANTS
    for variant in EXPECTED_VARIANTS:
        records = [
            json.loads(line)
            for line in (artifacts.directory / f"{variant}.log.jsonl").read_text().splitlines()
        ]
        if variant in _no_bucket_variants:
            assert all("boundary_mode" not in record for record in records)
        else:
            assert {record["boundary_mode"] for record in records} == {expected_mode}


def test_hmc_variant_logs_diagnostics_without_bucket_fields(comparison_artifacts) -> None:
    for variant in HMC_VARIANTS:
        records = [
            json.loads(line)
            for line in (comparison_artifacts.directory / f"{variant}.log.jsonl")
            .read_text()
            .splitlines()
        ]
        assert records
        for record in records:
            assert isinstance(record["converged"], bool)
            assert np.isfinite(record["rhat_max"])
            assert isinstance(record["ess_bulk_min"], int)
            assert isinstance(record["posterior_mean_W"], list)
            assert isinstance(record["posterior_mean_b"], list)
            assert "bucket_observation_counts" not in record
            assert "boundary_mode" not in record


def test_hmc_variant_persists_posterior_samples_npz(comparison_artifacts) -> None:
    for variant in HMC_VARIANTS:
        posterior_dir = comparison_artifacts.directory / f"{variant}.posterior"
        assert posterior_dir.is_dir(), f"missing posterior dir for {variant}"
        npz_files = sorted(posterior_dir.glob("window_*.npz"))
        assert npz_files, f"no per-window posterior npz files for {variant}"
        log_records = [
            json.loads(line)
            for line in (comparison_artifacts.directory / f"{variant}.log.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(npz_files) == len(log_records)
        for path in npz_files:
            with np.load(path) as data:
                w = data["W"]
                b = data["b"]
            assert w.ndim == 4 and w.shape[2] == w.shape[3] >= 2
            assert b.shape == w.shape
            assert np.all(np.isfinite(w))
            assert np.all(np.isfinite(b))


def test_non_hmc_variants_skip_posterior_dir(comparison_artifacts) -> None:
    for variant in EXPECTED_VARIANTS:
        if variant in HMC_VARIANTS:
            continue
        assert not (comparison_artifacts.directory / f"{variant}.posterior").exists()


def test_force_overwrites_existing_directory(tmp_path: Path) -> None:
    cfg = _make_config()
    first = run_side_info_comparison(cfg, runs_root=tmp_path)
    with pytest.raises(FileExistsError):
        run_side_info_comparison(cfg, runs_root=tmp_path)
    second = run_side_info_comparison(cfg, runs_root=tmp_path, force=True)
    assert first.directory == second.directory


# ---------------------------------------------------------------------------
# Checkpointing (per-variant resume)
# ---------------------------------------------------------------------------


def _summary_payload(directory: Path) -> dict:
    return json.loads((directory / "summary.json").read_text())


def test_checkpointing_produces_identical_artifacts_and_cleans_up(tmp_path: Path) -> None:
    cfg = _make_config()
    reference = run_side_info_comparison(cfg, runs_root=tmp_path / "ref")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpointed = run_side_info_comparison(
        cfg,
        runs_root=tmp_path / "ckpt-runs",
        checkpoint_dir=checkpoint_dir,
    )
    assert _summary_payload(reference.directory) == _summary_payload(checkpointed.directory)
    assert not checkpoint_dir.exists(), "checkpoint dir must be removed after a successful run"


def test_checkpointing_resumes_after_simulated_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_config()
    reference = run_side_info_comparison(cfg, runs_root=tmp_path / "ref")

    checkpoint_dir = tmp_path / "checkpoints"
    runs_root = tmp_path / "ckpt-runs"

    real_default = side_info_module._run_default_hmm_variant

    def crash(**kwargs):
        raise RuntimeError("simulated crash inside default_hmm variant")

    monkeypatch.setattr(side_info_module, "_run_default_hmm_variant", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_side_info_comparison(cfg, runs_root=runs_root, checkpoint_dir=checkpoint_dir)
    monkeypatch.setattr(side_info_module, "_run_default_hmm_variant", real_default)

    # Earlier stages should have been pickled; default_hmm should not be.
    assert (checkpoint_dir / f"{_CHECKPOINT_BASELINE_WF}.pkl").is_file()
    for variant in EXPECTED_VARIANTS:
        path = checkpoint_dir / f"{variant}.pkl"
        if variant == DEFAULT_HMM_VARIANT:
            assert not path.exists()
        else:
            assert path.is_file()

    # The first attempt aborted before any run_dir was written.
    cmp_id = comparison_id(cfg)
    assert not (runs_root / cmp_id).exists()

    # Spy on factories: nothing already-pickled should be recomputed on resume.
    walk_forward_calls = 0
    real_walk_forward = side_info_module.walk_forward

    def spy_walk_forward(*args, **kwargs):
        nonlocal walk_forward_calls
        walk_forward_calls += 1
        return real_walk_forward(*args, **kwargs)

    monkeypatch.setattr(side_info_module, "walk_forward", spy_walk_forward)

    side_info_variant_calls: list[str] = []
    real_run_side_info_variant = side_info_module._run_side_info_variant

    def spy_run_side_info_variant(variant, **kwargs):
        side_info_variant_calls.append(variant)
        return real_run_side_info_variant(variant, **kwargs)

    monkeypatch.setattr(side_info_module, "_run_side_info_variant", spy_run_side_info_variant)

    default_hmm_calls = 0

    def spy_run_default_hmm_variant(**kwargs):
        nonlocal default_hmm_calls
        default_hmm_calls += 1
        return real_default(**kwargs)

    monkeypatch.setattr(side_info_module, "_run_default_hmm_variant", spy_run_default_hmm_variant)

    resumed = run_side_info_comparison(
        cfg, runs_root=runs_root, checkpoint_dir=checkpoint_dir
    )

    assert walk_forward_calls == 0, "baseline walk_forward must be loaded from checkpoint"
    assert side_info_variant_calls == [], (
        f"no side-info variant should be recomputed on resume, got {side_info_variant_calls!r}"
    )
    assert default_hmm_calls == 1, "default_hmm was the only missing stage and must be recomputed"
    assert not checkpoint_dir.exists()
    assert _summary_payload(reference.directory) == _summary_payload(resumed.directory)


def test_checkpointing_tolerates_corrupt_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_config()
    reference = run_side_info_comparison(cfg, runs_root=tmp_path / "ref")

    checkpoint_dir = tmp_path / "checkpoints"
    runs_root = tmp_path / "ckpt-runs"

    real_default = side_info_module._run_default_hmm_variant
    monkeypatch.setattr(
        side_info_module,
        "_run_default_hmm_variant",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    with pytest.raises(RuntimeError, match="stop"):
        run_side_info_comparison(cfg, runs_root=runs_root, checkpoint_dir=checkpoint_dir)
    monkeypatch.setattr(side_info_module, "_run_default_hmm_variant", real_default)

    # Corrupt one variant checkpoint — the runner should detect, drop, and recompute it.
    (checkpoint_dir / f"{BASELINE_VARIANT}.pkl").write_bytes(b"not a real pickle")

    resumed = run_side_info_comparison(
        cfg, runs_root=runs_root, checkpoint_dir=checkpoint_dir
    )
    assert not checkpoint_dir.exists()
    assert _summary_payload(reference.directory) == _summary_payload(resumed.directory)


def test_summary_includes_required_metric_fields(comparison_artifacts) -> None:
    payload = json.loads((comparison_artifacts.directory / "summary.json").read_text())
    assert payload["comparison_id"] == comparison_artifacts.comparison_id
    assert "primary academic" in payload["metric_interpretation"]["pre-cost"]
    assert "diagnostic" in payload["metric_interpretation"]["post-cost"]
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
        assert entry["daily_annualized_sharpe"]["trading_days_per_year"] == 258.0
        assert "UTC date" in entry["daily_annualized_sharpe"]["method"]
        assert np.isfinite(entry["daily_annualized_sharpe"]["pre-cost"])
        assert np.isfinite(entry["daily_annualized_sharpe"]["post-cost"])
        turnover = entry["turnover_diagnostics"]
        assert turnover["total_turnover"] >= 0.0
        assert turnover["mean_turnover_per_period"] >= 0.0
        assert turnover["position_change_count"] >= 0
        assert turnover["mean_holding_periods"] > 0.0
        assert turnover["cost_drag_cumulative_return"] is not None


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
