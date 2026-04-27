"""End-to-end experiment runner: config → data → walk-forward → ``runs/<run_id>/``.

This module turns an :class:`~hft_hmm.config.ExperimentConfig` into a fully
reproducible artifact directory. The dispatch on ``config.data.kind`` routes to
the appropriate loader; the rest of the pipeline is identical. When the data
source is not reproducible (currently ``yfinance``) the runner emits a
``UserWarning`` and tags the written ``metrics.json`` with ``"reproducible":
false`` so downstream consumers can filter by trust level.

References: §4.4 reproducible simulation artifacts (evaluation layer)
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from hft_hmm.config.experiment_config import ExperimentConfig, run_id
from hft_hmm.core import EVALUATION_LAYER
from hft_hmm.experiments._data_loading import (
    DATA_FINGERPRINT_MISMATCH_WARNING,  # noqa: F401 -- re-exported compatibility constant
    NON_REPRODUCIBLE_WARNING,  # noqa: F401 -- re-exported compatibility constant
    load_returns_from_source,
    validate_data_reproducibility,
)
from hft_hmm.experiments.walk_forward import WalkForwardResult, walk_forward

__category__: Final[str] = EVALUATION_LAYER


@dataclass(frozen=True)
class RunArtifacts:
    """Handle returned by :func:`run_experiment` summarizing the written directory."""

    run_id: str
    directory: Path
    config: ExperimentConfig
    walk_forward: WalkForwardResult


def run_experiment(
    config: ExperimentConfig,
    *,
    runs_root: Path | str = Path("runs"),
    force: bool = False,
) -> RunArtifacts:
    """Run the walk-forward experiment described by ``config`` and write artifacts.

    The target directory is ``runs_root / run_id(config)``. If it already
    exists, ``force=True`` wipes it before writing; otherwise a
    ``FileExistsError`` is raised so prior results cannot be silently overwritten.
    """
    if not isinstance(config, ExperimentConfig):
        raise TypeError(f"config must be an ExperimentConfig, got {type(config).__name__}.")

    runs_root_path = Path(runs_root)
    experiment_id = run_id(config)
    run_dir = runs_root_path / experiment_id

    if run_dir.exists():
        if not force:
            raise FileExistsError(
                f"Run directory already exists at {run_dir}; pass force=True to overwrite."
            )
    reproducible = _validate_reproducibility(config)

    returns = _load_returns(config)
    wf_result = walk_forward(
        returns,
        config.walk_forward,
        cost_bps_per_turnover=config.cost_bps_per_turnover,
    )

    runs_root_path.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f"{experiment_id}.tmp-", dir=runs_root_path))
    backup_dir: Path | None = None
    try:
        _write_artifacts(staging_dir, config, experiment_id, wf_result, reproducible=reproducible)

        if run_dir.exists():
            backup_dir = run_dir.with_name(f"{run_dir.name}.backup-{uuid.uuid4().hex}")
            os.replace(run_dir, backup_dir)

        try:
            os.replace(staging_dir, run_dir)
        except Exception:
            if backup_dir is not None and backup_dir.exists():
                os.replace(backup_dir, run_dir)
            raise

        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return RunArtifacts(
        run_id=experiment_id,
        directory=run_dir,
        config=config,
        walk_forward=wf_result,
    )


def _load_returns(config: ExperimentConfig) -> pd.Series:
    """Load raw market data, resample, and return tz-aware log returns."""
    return load_returns_from_source(config.data, frequency=config.frequency)


def _validate_reproducibility(config: ExperimentConfig) -> bool:
    return validate_data_reproducibility(config, stacklevel=3)


def _write_artifacts(
    run_dir: Path,
    config: ExperimentConfig,
    experiment_id: str,
    result: WalkForwardResult,
    *,
    reproducible: bool,
) -> None:
    (run_dir / "figures").mkdir()
    (run_dir / "config.yaml").write_bytes(config.to_yaml_bytes())
    _write_metrics(
        run_dir / "metrics.json", config, experiment_id, result, reproducible=reproducible
    )
    _write_log(run_dir / "log.jsonl", result)


def _write_metrics(
    path: Path,
    config: ExperimentConfig,
    experiment_id: str,
    result: WalkForwardResult,
    *,
    reproducible: bool,
) -> None:
    payload: dict[str, Any] = {
        "run_id": experiment_id,
        "reproducible": reproducible,
        "cost_bps_per_turnover": float(config.cost_bps_per_turnover),
        "n_windows": len(result.windows),
        "n_forecast_obs": int(result.signal.shape[0]),
        "summary": _summary_to_payload(result.summary),
    }
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_log(path: Path, result: WalkForwardResult) -> None:
    lines = []
    for window in result.windows:
        lines.append(
            json.dumps(
                {
                    "index": int(window.index),
                    "train_start": window.train_start.isoformat(),
                    "train_end": window.train_end.isoformat(),
                    "forecast_start": window.forecast_start.isoformat(),
                    "forecast_end": window.forecast_end.isoformat(),
                    "chosen_k": int(window.chosen_k),
                    "log_likelihood": _json_safe(window.log_likelihood),
                    "n_train_obs": int(window.n_train_obs),
                    "n_forecast_obs": int(window.n_forecast_obs),
                    "summary": _summary_to_payload(window.summary),
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_to_payload(summary: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Convert a per-mode summary DataFrame into a JSON-safe nested dict."""
    payload: dict[str, dict[str, float | None]] = {}
    for mode, row in summary.iterrows():
        payload[str(mode)] = {str(col): _json_safe(row[col]) for col in summary.columns}
    return payload


def _json_safe(value: Any) -> float | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    # Round serialized metrics to 8 decimals so artifacts compare bit-for-bit
    # across subprocess and in-process runs. BLAS thread-ordering produces
    # sub-ULP drift at full float64 precision that would otherwise break
    # Gate F's reproducibility contract without materially affecting any
    # reported Sharpe / return figure.
    return round(numeric, 8)
