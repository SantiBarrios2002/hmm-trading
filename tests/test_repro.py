"""Tests for the experiment runner and `scripts/repro.py` CLI."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from hft_hmm.config import DataSourceConfig, ExperimentConfig, run_id
from hft_hmm.core import EVALUATION_LAYER
from hft_hmm.experiments.runner import NON_REPRODUCIBLE_WARNING, run_experiment
from hft_hmm.experiments.walk_forward import WalkForwardConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "es_1min_sample.csv"
MONTH_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "es_1min_month.csv"
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "example_es_csv.yaml"
REPRO_SCRIPT = REPO_ROOT / "scripts" / "repro.py"


def _csv_config(path: Path = SAMPLE_FIXTURE, **wf_overrides) -> ExperimentConfig:
    wf = WalkForwardConfig(
        h_days=wf_overrides.pop("h_days", 2),
        t_days=wf_overrides.pop("t_days", 1),
        k_values=wf_overrides.pop("k_values", (2,)),
        random_state=wf_overrides.pop("random_state", 0),
    )
    return ExperimentConfig(
        data=DataSourceConfig(kind="csv", path=str(path)),
        frequency="1min",
        walk_forward=wf,
        cost_bps_per_turnover=0.5,
        notes="repro-test",
    )


def test_runner_module_declares_evaluation_layer_category() -> None:
    module = importlib.import_module("hft_hmm.experiments.runner")
    assert module.__category__ == EVALUATION_LAYER


def test_run_experiment_writes_expected_artifact_layout(tmp_path: Path) -> None:
    config = _csv_config()
    artifacts = run_experiment(config, runs_root=tmp_path)

    assert artifacts.directory == tmp_path / artifacts.run_id
    assert (artifacts.directory / "config.yaml").is_file()
    assert (artifacts.directory / "metrics.json").is_file()
    assert (artifacts.directory / "log.jsonl").is_file()
    assert (artifacts.directory / "figures").is_dir()

    # run_id is sha256(config.yaml bytes)[:12]
    written = (artifacts.directory / "config.yaml").read_bytes()
    assert hashlib.sha256(written).hexdigest()[:12] == artifacts.run_id

    metrics = json.loads((artifacts.directory / "metrics.json").read_text())
    assert metrics["run_id"] == artifacts.run_id
    assert metrics["reproducible"] is True
    assert metrics["n_windows"] == len(artifacts.walk_forward.windows)
    assert metrics["n_windows"] >= 2

    log_lines = (artifacts.directory / "log.jsonl").read_text().strip().splitlines()
    assert len(log_lines) == len(artifacts.walk_forward.windows)
    first = json.loads(log_lines[0])
    assert set(first).issuperset(
        {
            "index",
            "train_start",
            "train_end",
            "forecast_start",
            "forecast_end",
            "chosen_k",
            "log_likelihood",
            "summary",
        }
    )


def test_run_experiment_raises_file_exists_without_force(tmp_path: Path) -> None:
    config = _csv_config()
    run_experiment(config, runs_root=tmp_path)
    with pytest.raises(FileExistsError, match="force=True"):
        run_experiment(config, runs_root=tmp_path)


def test_run_experiment_overwrites_with_force(tmp_path: Path) -> None:
    config = _csv_config()
    first = run_experiment(config, runs_root=tmp_path)
    marker = first.directory / "marker.txt"
    marker.write_text("stale")
    run_experiment(config, runs_root=tmp_path, force=True)
    assert not marker.exists()


def test_run_experiment_rejects_non_config(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="config must be an ExperimentConfig"):
        run_experiment({"frequency": "1min"}, runs_root=tmp_path)  # type: ignore[arg-type]


def test_run_experiment_emits_non_reproducible_warning_for_yfinance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hft_hmm.data import load_csv_market_data

    sample = load_csv_market_data(SAMPLE_FIXTURE)

    def fake_yfinance_loader(symbol, *, start, end, interval, auto_adjust=True, spec=None):
        return sample.copy()

    monkeypatch.setattr(
        "hft_hmm.experiments.runner.load_yfinance_market_data",
        fake_yfinance_loader,
    )
    cfg = ExperimentConfig(
        data=DataSourceConfig(
            kind="yfinance",
            symbol="ES=F",
            start="2024-01-02",
            end="2024-01-09",
        ),
        frequency="1min",
        walk_forward=WalkForwardConfig(h_days=2, t_days=1, k_values=(2,)),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifacts = run_experiment(cfg, runs_root=tmp_path)

    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert NON_REPRODUCIBLE_WARNING in messages

    metrics = json.loads((artifacts.directory / "metrics.json").read_text())
    assert metrics["reproducible"] is False


def test_run_experiment_is_deterministic(tmp_path: Path) -> None:
    first = run_experiment(_csv_config(), runs_root=tmp_path / "a")
    second = run_experiment(_csv_config(), runs_root=tmp_path / "b")
    assert first.run_id == second.run_id
    assert (first.directory / "config.yaml").read_bytes() == (
        second.directory / "config.yaml"
    ).read_bytes()
    assert (first.directory / "metrics.json").read_bytes() == (
        second.directory / "metrics.json"
    ).read_bytes()
    assert (first.directory / "log.jsonl").read_bytes() == (
        second.directory / "log.jsonl"
    ).read_bytes()


def test_repro_cli_bit_for_bit(tmp_path: Path) -> None:
    config = _csv_config()
    expected_id = run_id(config)

    config_yaml = tmp_path / "cfg.yaml"
    config.to_yaml(config_yaml)
    runs_root = tmp_path / "runs"

    completed = subprocess.run(
        [sys.executable, str(REPRO_SCRIPT), str(config_yaml), "--runs-root", str(runs_root)],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    printed = Path(completed.stdout.strip())
    assert printed == runs_root / expected_id

    in_process = run_experiment(config, runs_root=tmp_path / "in_process")
    assert (printed / "metrics.json").read_bytes() == (
        in_process.directory / "metrics.json"
    ).read_bytes()
    assert (printed / "log.jsonl").read_bytes() == (in_process.directory / "log.jsonl").read_bytes()


def test_repro_cli_rejects_missing_config(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    completed = subprocess.run(
        [sys.executable, str(REPRO_SCRIPT), str(missing)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 2
    assert "Config not found" in completed.stderr


def test_repro_cli_runs_example_config_end_to_end(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    completed = subprocess.run(
        [sys.executable, str(REPRO_SCRIPT), str(EXAMPLE_CONFIG), "--runs-root", str(runs_root)],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    printed = Path(completed.stdout.strip())
    assert printed.is_dir()
    metrics = json.loads((printed / "metrics.json").read_text())
    assert metrics["reproducible"] is True
    assert metrics["n_windows"] >= 1
    assert np.isfinite(metrics["summary"]["post-cost"]["sharpe_ratio"])


def test_repro_cli_force_overwrite(tmp_path: Path) -> None:
    config = _csv_config()
    config_yaml = tmp_path / "cfg.yaml"
    config.to_yaml(config_yaml)
    runs_root = tmp_path / "runs"

    subprocess.run(
        [sys.executable, str(REPRO_SCRIPT), str(config_yaml), "--runs-root", str(runs_root)],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )
    # Second run without --force must fail
    blocked = subprocess.run(
        [sys.executable, str(REPRO_SCRIPT), str(config_yaml), "--runs-root", str(runs_root)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert blocked.returncode != 0
    # With --force it succeeds
    subprocess.run(
        [
            sys.executable,
            str(REPRO_SCRIPT),
            str(config_yaml),
            "--runs-root",
            str(runs_root),
            "--force",
        ],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )
