"""Tests for the experiment-config dataclasses, YAML I/O, and run_id hashing."""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from hft_hmm.config import DataSourceConfig, ExperimentConfig, compute_file_sha256, run_id
from hft_hmm.config.experiment_config import EXPERIMENT_CONFIG_REFERENCE
from hft_hmm.core import EVALUATION_LAYER
from hft_hmm.experiments.walk_forward import WalkForwardConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "es_1min_sample.csv"
SAMPLE_SHA256 = compute_file_sha256(SAMPLE_FIXTURE)
PLACEHOLDER_SHA256 = "0" * 64


def _wf() -> WalkForwardConfig:
    return WalkForwardConfig(h_days=5, t_days=1, k_values=(2,), random_state=7)


def _csv_data() -> DataSourceConfig:
    return DataSourceConfig(kind="csv", path="tests/fixtures/es_1min_sample.csv")


def _parquet_data() -> DataSourceConfig:
    return DataSourceConfig(
        kind="databento_parquet",
        path="data/databento/databento/ES_c_0_ohlcv-1m_2024-01-01_2024-02-01.parquet",
        symbol="ES.c.0",
    )


def _yfinance_data() -> DataSourceConfig:
    return DataSourceConfig(
        kind="yfinance",
        symbol="SPY",
        start="2024-01-01",
        end="2024-02-01",
    )


def test_experiment_config_module_declares_evaluation_layer_category() -> None:
    module = importlib.import_module("hft_hmm.config.experiment_config")
    assert module.__category__ == EVALUATION_LAYER
    assert EXPERIMENT_CONFIG_REFERENCE.section == "§4.4"


@pytest.mark.parametrize(
    "data_factory",
    [_csv_data, _parquet_data, _yfinance_data],
    ids=["csv", "databento_parquet", "yfinance"],
)
def test_experiment_config_roundtrip_preserves_fields(data_factory, tmp_path: Path) -> None:
    sha256 = None if data_factory is _yfinance_data else SAMPLE_SHA256
    if data_factory is _parquet_data:
        sha256 = PLACEHOLDER_SHA256
    original = ExperimentConfig(
        data=data_factory(),
        frequency="1min",
        walk_forward=_wf(),
        cost_bps_per_turnover=1.25,
        notes="roundtrip",
        sha256=sha256,
    )
    path = tmp_path / "cfg.yaml"
    original.to_yaml(path)
    loaded = ExperimentConfig.from_yaml(path)
    assert loaded == original


def test_experiment_config_yaml_is_deterministic(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        data=_csv_data(),
        frequency="1min",
        walk_forward=_wf(),
        cost_bps_per_turnover=0.5,
        notes="alpha",
        sha256=SAMPLE_SHA256,
    )
    assert cfg.to_yaml_bytes() == cfg.to_yaml_bytes()


def test_experiment_config_yaml_round_trip_is_fixed_point(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        data=_csv_data(),
        frequency="1min",
        walk_forward=_wf(),
        notes="fixed-point",
        sha256=SAMPLE_SHA256,
    )
    first = tmp_path / "a.yaml"
    second = tmp_path / "b.yaml"
    cfg.to_yaml(first)
    ExperimentConfig.from_yaml(first).to_yaml(second)
    assert first.read_bytes() == second.read_bytes()


def test_experiment_config_roundtrip_preserves_min_variance(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        data=_csv_data(),
        frequency="1min",
        walk_forward=WalkForwardConfig(
            h_days=5,
            t_days=1,
            k_values=(2,),
            random_state=7,
            min_variance=2.5e-8,
        ),
        sha256=SAMPLE_SHA256,
    )

    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    loaded = ExperimentConfig.from_yaml(path)

    assert loaded.walk_forward.min_variance == pytest.approx(2.5e-8)


def test_data_source_config_csv_requires_path() -> None:
    with pytest.raises(ValueError, match="requires path"):
        DataSourceConfig(kind="csv")


def test_data_source_config_databento_requires_path_and_symbol() -> None:
    with pytest.raises(ValueError, match="requires path"):
        DataSourceConfig(kind="databento_parquet", symbol="ES.c.0")
    with pytest.raises(ValueError, match="requires symbol"):
        DataSourceConfig(kind="databento_parquet", path="data.parquet")


def test_data_source_config_yfinance_requires_symbol_start_end() -> None:
    with pytest.raises(ValueError, match="requires symbol"):
        DataSourceConfig(kind="yfinance", start="2024-01-01", end="2024-02-01")
    with pytest.raises(ValueError, match="start and end"):
        DataSourceConfig(kind="yfinance", symbol="SPY")
    with pytest.raises(ValueError, match="start and end"):
        DataSourceConfig(kind="yfinance", symbol="SPY", start="2024-01-01")


def test_data_source_config_yfinance_rejects_path() -> None:
    with pytest.raises(ValueError, match="must not set path"):
        DataSourceConfig(
            kind="yfinance",
            path="ignored.csv",
            symbol="SPY",
            start="2024-01-01",
            end="2024-02-01",
        )


def test_data_source_config_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        DataSourceConfig(kind="bloomberg")  # type: ignore[arg-type]


def test_data_source_config_is_reproducible_flag() -> None:
    assert _csv_data().is_reproducible is True
    assert _parquet_data().is_reproducible is True
    assert _yfinance_data().is_reproducible is False


def test_experiment_config_is_reproducible_flag() -> None:
    assert (
        ExperimentConfig(
            data=_csv_data(),
            frequency="1min",
            walk_forward=_wf(),
            sha256=SAMPLE_SHA256,
        ).is_reproducible
        is True
    )
    assert (
        ExperimentConfig(
            data=_yfinance_data(),
            frequency="1min",
            walk_forward=_wf(),
        ).is_reproducible
        is False
    )


def test_experiment_config_rejects_bad_frequency() -> None:
    with pytest.raises(ValueError, match="frequency must be one of"):
        ExperimentConfig(
            data=_csv_data(),
            frequency="2min",  # type: ignore[arg-type]
            walk_forward=_wf(),
            sha256=SAMPLE_SHA256,
        )


def test_experiment_config_rejects_negative_cost() -> None:
    with pytest.raises(ValueError, match="cost_bps_per_turnover must be non-negative"):
        ExperimentConfig(
            data=_csv_data(),
            frequency="1min",
            walk_forward=_wf(),
            cost_bps_per_turnover=-0.1,
            sha256=SAMPLE_SHA256,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_experiment_config_rejects_non_finite_cost(value: float) -> None:
    with pytest.raises(ValueError, match="cost_bps_per_turnover must be finite"):
        ExperimentConfig(
            data=_csv_data(),
            frequency="1min",
            walk_forward=_wf(),
            cost_bps_per_turnover=value,
            sha256=SAMPLE_SHA256,
        )


def test_experiment_config_requires_sha256_for_file_backed_data() -> None:
    with pytest.raises(ValueError, match="requires sha256"):
        ExperimentConfig(
            data=_csv_data(),
            frequency="1min",
            walk_forward=_wf(),
        )


@pytest.mark.parametrize("sha256", ["abc", "g" * 64])
def test_experiment_config_rejects_invalid_sha256(sha256: str) -> None:
    with pytest.raises(ValueError, match="64-character hexadecimal digest"):
        ExperimentConfig(
            data=_csv_data(),
            frequency="1min",
            walk_forward=_wf(),
            sha256=sha256,
        )


def test_experiment_config_rejects_sha256_for_yfinance() -> None:
    with pytest.raises(ValueError, match="must not set sha256"):
        ExperimentConfig(
            data=_yfinance_data(),
            frequency="1min",
            walk_forward=_wf(),
            sha256=SAMPLE_SHA256,
        )


def test_experiment_config_rejects_wrong_types() -> None:
    with pytest.raises(TypeError, match="data must be a DataSourceConfig"):
        ExperimentConfig(
            data={"kind": "csv", "path": "x.csv"},  # type: ignore[arg-type]
            frequency="1min",
            walk_forward=_wf(),
            sha256=SAMPLE_SHA256,
        )
    with pytest.raises(TypeError, match="walk_forward must be a WalkForwardConfig"):
        ExperimentConfig(
            data=_csv_data(),
            frequency="1min",
            walk_forward={"h_days": 5},  # type: ignore[arg-type]
            sha256=SAMPLE_SHA256,
        )
    with pytest.raises(TypeError, match="cost_bps_per_turnover must be a real number"):
        ExperimentConfig(
            data=_csv_data(),
            frequency="1min",
            walk_forward=_wf(),
            cost_bps_per_turnover=True,  # type: ignore[arg-type]
            sha256=SAMPLE_SHA256,
        )
    with pytest.raises(TypeError, match="notes must be a str"):
        ExperimentConfig(
            data=_csv_data(),
            frequency="1min",
            walk_forward=_wf(),
            notes=42,  # type: ignore[arg-type]
            sha256=SAMPLE_SHA256,
        )


def test_run_id_is_12_hex_chars_and_stable() -> None:
    cfg = ExperimentConfig(
        data=_csv_data(),
        frequency="1min",
        walk_forward=_wf(),
        cost_bps_per_turnover=0.5,
        notes="stable",
        sha256=SAMPLE_SHA256,
    )
    first = run_id(cfg)
    assert len(first) == 12
    assert all(c in "0123456789abcdef" for c in first)
    assert run_id(cfg) == first


def test_run_id_stable_across_subprocess(tmp_path: Path) -> None:
    script = tmp_path / "compute.py"
    script.write_text(textwrap.dedent("""
            import sys
            from hft_hmm.config import (
                DataSourceConfig,
                ExperimentConfig,
                compute_file_sha256,
                run_id,
            )
            from hft_hmm.experiments.walk_forward import WalkForwardConfig

            cfg = ExperimentConfig(
                data=DataSourceConfig(kind="csv", path="tests/fixtures/es_1min_sample.csv"),
                frequency="1min",
                walk_forward=WalkForwardConfig(h_days=5, t_days=1, k_values=(2,), random_state=7),
                cost_bps_per_turnover=0.5,
                notes="stable",
                sha256=compute_file_sha256("tests/fixtures/es_1min_sample.csv"),
            )
            sys.stdout.write(run_id(cfg))
            """))
    completed = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    in_process_id = run_id(
        ExperimentConfig(
            data=_csv_data(),
            frequency="1min",
            walk_forward=_wf(),
            cost_bps_per_turnover=0.5,
            notes="stable",
            sha256=SAMPLE_SHA256,
        )
    )
    assert completed.stdout == in_process_id


def test_run_id_changes_when_random_state_changes() -> None:
    base = ExperimentConfig(
        data=_csv_data(),
        frequency="1min",
        walk_forward=_wf(),
        cost_bps_per_turnover=0.5,
        sha256=SAMPLE_SHA256,
    )
    perturbed = ExperimentConfig(
        data=_csv_data(),
        frequency="1min",
        walk_forward=WalkForwardConfig(h_days=5, t_days=1, k_values=(2,), random_state=8),
        cost_bps_per_turnover=0.5,
        sha256=SAMPLE_SHA256,
    )
    assert run_id(base) != run_id(perturbed)


def test_run_id_changes_when_cost_changes() -> None:
    base = ExperimentConfig(
        data=_csv_data(),
        frequency="1min",
        walk_forward=_wf(),
        sha256=SAMPLE_SHA256,
    )
    perturbed = ExperimentConfig(
        data=_csv_data(),
        frequency="1min",
        walk_forward=_wf(),
        cost_bps_per_turnover=1.0,
        sha256=SAMPLE_SHA256,
    )
    assert run_id(base) != run_id(perturbed)


def test_run_id_changes_when_notes_change() -> None:
    base = ExperimentConfig(
        data=_csv_data(),
        frequency="1min",
        walk_forward=_wf(),
        notes="A",
        sha256=SAMPLE_SHA256,
    )
    perturbed = ExperimentConfig(
        data=_csv_data(),
        frequency="1min",
        walk_forward=_wf(),
        notes="B",
        sha256=SAMPLE_SHA256,
    )
    assert run_id(base) != run_id(perturbed)


def test_run_id_changes_when_sha256_changes() -> None:
    base = ExperimentConfig(
        data=_csv_data(),
        frequency="1min",
        walk_forward=_wf(),
        sha256=SAMPLE_SHA256,
    )
    perturbed = ExperimentConfig(
        data=_csv_data(),
        frequency="1min",
        walk_forward=_wf(),
        sha256=PLACEHOLDER_SHA256,
    )
    assert run_id(base) != run_id(perturbed)


def test_from_dict_rejects_legacy_walk_forward_without_min_variance() -> None:
    payload = {
        "data": {
            "kind": "csv",
            "path": "tests/fixtures/es_1min_sample.csv",
            "symbol": None,
            "start": None,
            "end": None,
        },
        "frequency": "1min",
        "cost_bps_per_turnover": 0.0,
        "sha256": SAMPLE_SHA256,
        "walk_forward": {
            "h_days": 3,
            "t_days": 1,
            "retrain_every_days": 1,
            "k_values": [2],
            "random_state": 0,
            "n_iter": 50,
            "tol": 1e-4,
            "variance_floor_policy": "clamp",
        },
        "notes": "",
    }
    with pytest.raises(ValueError, match="walk_forward.min_variance is required"):
        ExperimentConfig.from_dict(payload)


def test_from_dict_rejects_legacy_walk_forward_without_variance_floor_policy() -> None:
    payload = {
        "data": {
            "kind": "csv",
            "path": "tests/fixtures/es_1min_sample.csv",
            "symbol": None,
            "start": None,
            "end": None,
        },
        "frequency": "1min",
        "cost_bps_per_turnover": 0.0,
        "sha256": SAMPLE_SHA256,
        "walk_forward": {
            "h_days": 3,
            "t_days": 1,
            "retrain_every_days": 1,
            "k_values": [2],
            "random_state": 0,
            "n_iter": 50,
            "tol": 1e-4,
            "min_variance": 1e-8,
        },
        "notes": "",
    }
    with pytest.raises(ValueError, match="walk_forward.variance_floor_policy is required"):
        ExperimentConfig.from_dict(payload)


def test_run_id_changes_when_variance_floor_policy_changes() -> None:
    base = ExperimentConfig(
        data=_csv_data(),
        frequency="1min",
        walk_forward=_wf(),
        sha256=SAMPLE_SHA256,
    )
    perturbed = ExperimentConfig(
        data=_csv_data(),
        frequency="1min",
        walk_forward=WalkForwardConfig(
            h_days=5,
            t_days=1,
            k_values=(2,),
            random_state=7,
            variance_floor_policy="raise",
        ),
        sha256=SAMPLE_SHA256,
    )
    assert run_id(base) != run_id(perturbed)


def test_run_id_changes_when_min_variance_changes() -> None:
    base = ExperimentConfig(
        data=_csv_data(),
        frequency="1min",
        walk_forward=_wf(),
        sha256=SAMPLE_SHA256,
    )
    perturbed = ExperimentConfig(
        data=_csv_data(),
        frequency="1min",
        walk_forward=WalkForwardConfig(
            h_days=5,
            t_days=1,
            k_values=(2,),
            random_state=7,
            min_variance=2e-8,
        ),
        sha256=SAMPLE_SHA256,
    )
    assert run_id(base) != run_id(perturbed)


def test_run_id_rejects_non_experiment_config() -> None:
    with pytest.raises(TypeError, match="must be an ExperimentConfig"):
        run_id({"frequency": "1min"})  # type: ignore[arg-type]


def test_from_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="decode to a mapping"):
        ExperimentConfig.from_yaml(bad)


def test_from_yaml_reads_yaml_written_externally(tmp_path: Path) -> None:
    path = tmp_path / "external.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "kind": "csv",
                    "path": "tests/fixtures/es_1min_sample.csv",
                    "symbol": None,
                    "start": None,
                    "end": None,
                },
                "frequency": "1min",
                "cost_bps_per_turnover": 2.0,
                "sha256": SAMPLE_SHA256,
                "walk_forward": {
                    "h_days": 3,
                    "t_days": 1,
                    "retrain_every_days": 1,
                    "k_values": [2, 3],
                    "random_state": 0,
                    "n_iter": 50,
                    "tol": 1e-4,
                    "min_variance": 1e-8,
                    "variance_floor_policy": "clamp",
                },
                "notes": "external writer",
            },
            sort_keys=True,
        )
    )
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.walk_forward.k_values == (2, 3)
    assert cfg.walk_forward.min_variance == pytest.approx(1e-8)
    assert cfg.walk_forward.variance_floor_policy == "clamp"
    assert cfg.cost_bps_per_turnover == 2.0
    assert cfg.notes == "external writer"
    assert cfg.sha256 == SAMPLE_SHA256
