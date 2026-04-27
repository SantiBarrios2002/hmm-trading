"""Standalone walk-forward backtests for spline-based side-information predictors.

Evaluation layer — each predictor (volatility ratio, intraday seasonality) is
evaluated in isolation before being folded into the IOHMM (Issue 16). The
signal is sign(f(x_t)) where f(·) is a cubic spline fitted on the training
window only. No HMM state is used.

The walk-forward geometry, cost model, and metric definitions mirror the
baseline HMM walk-forward (§2.3) so the results are directly comparable.
This mirrors the paper's §4 structure: each predictor is evaluated standalone
before being incorporated into the IOHMM.

Classification: evaluation layer.
References: §4 side-information predictors (evaluation layer)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
import yaml

from hft_hmm.config.experiment_config import (
    DataSourceConfig,
    Frequency,
)
from hft_hmm.core import EVALUATION_LAYER, PaperReference, reference
from hft_hmm.evaluation import (
    apply_turnover_cost,
    cumulative_return,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    signal_turnover,
)
from hft_hmm.experiments._data_loading import (
    load_returns_from_source,
    validate_data_reproducibility,
)
from hft_hmm.features.seasonality import SeasonalityConfig, intraday_seasonality
from hft_hmm.features.splines import SplinePredictorConfig, fit_spline_predictor
from hft_hmm.features.volatility_ratio import VolatilityRatioConfig, volatility_ratio
from hft_hmm.strategy import align_signal_with_future_return, sign_signal

__category__: Final[str] = EVALUATION_LAYER
STANDALONE_PREDICTOR_REFERENCE: Final[PaperReference] = reference(
    "§4", "side-information predictor standalone evaluation"
)

PredictorKind = Literal["volatility_ratio", "seasonality"]
_PREDICTOR_KINDS: Final[tuple[str, ...]] = ("volatility_ratio", "seasonality")
_VALID_FREQUENCIES: Final[tuple[str, ...]] = ("1min", "5min", "1D")
_SHA256_HEX_LENGTH: Final[int] = 64


# ---------------------------------------------------------------------------
# Config types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StandaloneWalkForwardConfig:
    """Walk-forward window geometry for standalone predictor backtests.

    Mirrors :class:`~hft_hmm.experiments.walk_forward.WalkForwardConfig` but
    omits all HMM-specific parameters (k_values, min_variance, etc.).

    References: §2.3 rolling-window overnight retraining scheme (evaluation layer)
    """

    h_days: int = 23
    t_days: int = 1
    retrain_every_days: int | None = None

    def __post_init__(self) -> None:
        if self.h_days < 1:
            raise ValueError(f"h_days must be >= 1, got {self.h_days}.")
        if self.t_days < 1:
            raise ValueError(f"t_days must be >= 1, got {self.t_days}.")
        retrain = self.t_days if self.retrain_every_days is None else self.retrain_every_days
        if retrain < 1:
            raise ValueError(f"retrain_every_days must be >= 1, got {retrain}.")
        object.__setattr__(self, "retrain_every_days", int(retrain))


@dataclass(frozen=True)
class StandalonePredictorConfig:
    """Complete configuration for a standalone spline-predictor backtest.

    ``predictor`` selects which feature to build. ``vol_ratio`` and
    ``seasonality`` carry feature-specific parameters; the one not matching
    ``predictor`` is ignored at runtime.

    References: §4 side-information predictor standalone evaluation
    """

    predictor: PredictorKind
    walk_forward: StandaloneWalkForwardConfig
    spline: SplinePredictorConfig = field(default_factory=SplinePredictorConfig)
    vol_ratio: VolatilityRatioConfig = field(default_factory=VolatilityRatioConfig)
    seasonality: SeasonalityConfig = field(default_factory=SeasonalityConfig)

    def __post_init__(self) -> None:
        if self.predictor not in _PREDICTOR_KINDS:
            raise ValueError(
                f"predictor must be one of {_PREDICTOR_KINDS}, got {self.predictor!r}."
            )
        if not isinstance(self.walk_forward, StandaloneWalkForwardConfig):
            raise TypeError(
                f"walk_forward must be a StandaloneWalkForwardConfig, "
                f"got {type(self.walk_forward).__name__}."
            )
        if not isinstance(self.spline, SplinePredictorConfig):
            raise TypeError(
                f"spline must be a SplinePredictorConfig, got {type(self.spline).__name__}."
            )
        if not isinstance(self.vol_ratio, VolatilityRatioConfig):
            raise TypeError(
                "vol_ratio must be a VolatilityRatioConfig, "
                f"got {type(self.vol_ratio).__name__}."
            )
        if not isinstance(self.seasonality, SeasonalityConfig):
            raise TypeError(
                "seasonality must be a SeasonalityConfig, "
                f"got {type(self.seasonality).__name__}."
            )


# ---------------------------------------------------------------------------
# Per-window and full-result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StandalonePredictorWindow:
    """Per-window record from a standalone predictor walk-forward run.

    References: §4 side-information predictor standalone evaluation
    """

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    n_train_obs: int
    n_forecast_obs: int
    n_knots_effective: int
    summary: pd.DataFrame

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must be non-negative, got {self.index}.")
        if self.n_train_obs < 1:
            raise ValueError(f"n_train_obs must be >= 1, got {self.n_train_obs}.")
        if self.n_forecast_obs < 1:
            raise ValueError(f"n_forecast_obs must be >= 1, got {self.n_forecast_obs}.")
        if self.train_end < self.train_start:
            raise ValueError("train_end must not precede train_start.")
        if self.forecast_end < self.forecast_start:
            raise ValueError("forecast_end must not precede forecast_start.")
        if self.forecast_start <= self.train_end:
            raise ValueError(
                "forecast_start must be strictly after train_end; "
                f"got train_end={self.train_end} forecast_start={self.forecast_start}."
            )
        if not isinstance(self.summary, pd.DataFrame):
            raise TypeError("summary must be a pd.DataFrame.")


@dataclass(frozen=True)
class StandalonePredictorResult:
    """Immutable snapshot of a standalone predictor walk-forward run.

    Field layout mirrors
    :class:`~hft_hmm.experiments.walk_forward.WalkForwardResult` so downstream
    reporting code can handle both identically.

    References: §4 side-information predictor standalone evaluation
    """

    config: StandalonePredictorConfig
    windows: tuple[StandalonePredictorWindow, ...]
    signal: pd.Series
    pre_cost_returns: pd.Series
    post_cost_returns: pd.Series
    summary: pd.DataFrame
    cost_bps_per_turnover: float = field(default=0.0)

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("standalone predictor run produced zero windows.")
        if not isinstance(self.signal, pd.Series):
            raise TypeError("signal must be a pd.Series.")
        if not isinstance(self.pre_cost_returns, pd.Series):
            raise TypeError("pre_cost_returns must be a pd.Series.")
        if not isinstance(self.post_cost_returns, pd.Series):
            raise TypeError("post_cost_returns must be a pd.Series.")
        if not isinstance(self.summary, pd.DataFrame):
            raise TypeError("summary must be a pd.DataFrame.")
        if self.cost_bps_per_turnover < 0.0 or not np.isfinite(self.cost_bps_per_turnover):
            raise ValueError(
                "cost_bps_per_turnover must be a finite non-negative float, "
                f"got {self.cost_bps_per_turnover!r}."
            )
        if not np.all(np.isfinite(np.asarray(self.signal, dtype=float))):
            raise ValueError("signal must contain only finite values.")
        if not np.all(np.isfinite(np.asarray(self.pre_cost_returns, dtype=float))):
            raise ValueError("pre_cost_returns must contain only finite values.")
        if not np.all(np.isfinite(np.asarray(self.post_cost_returns, dtype=float))):
            raise ValueError("post_cost_returns must contain only finite values.")
        if not self.pre_cost_returns.index.equals(self.post_cost_returns.index):
            raise ValueError("pre_cost_returns and post_cost_returns must share the same index.")
        if len(self.pre_cost_returns) != len(self.signal) - len(self.windows):
            raise ValueError(
                "pre_cost_returns length must equal len(signal) - len(windows); "
                f"got len(pre_cost_returns)={len(self.pre_cost_returns)}, "
                f"len(signal)={len(self.signal)}, len(windows)={len(self.windows)}."
            )


# ---------------------------------------------------------------------------
# Main backtest function
# ---------------------------------------------------------------------------


def standalone_predictor_backtest(
    returns: pd.Series,
    config: StandalonePredictorConfig,
    *,
    cost_bps_per_turnover: float = 0.0,
) -> StandalonePredictorResult:
    """Run a walk-forward standalone spline-predictor backtest.

    For each window a spline f(x_t) ≈ E[r_{t+1} | x_t] is fitted on the
    training slice only. The signal is sign(f(x_t)) evaluated on out-of-sample
    feature values. No HMM state is used.

    Feature computation uses the concatenated train+forecast slice for rolling
    features (volatility ratio) so the rolling window is fully initialized at
    the start of the forecast period. The spline is fitted on the training
    portion only, preserving the out-of-sample boundary.

    The ``returns`` series must be tz-aware UTC with a monotonic DatetimeIndex,
    identical to the requirements of
    :func:`~hft_hmm.experiments.walk_forward.walk_forward`.

    References: §4 side-information predictor standalone evaluation (evaluation layer)
    """
    if not isinstance(config, StandalonePredictorConfig):
        raise TypeError(
            f"config must be a StandalonePredictorConfig, got {type(config).__name__}."
        )
    if not np.isfinite(cost_bps_per_turnover) or cost_bps_per_turnover < 0.0:
        raise ValueError(
            "cost_bps_per_turnover must be a finite non-negative float, "
            f"got {cost_bps_per_turnover!r}."
        )
    _validate_returns(returns)

    sorted_dates = np.array(sorted(set(returns.index.date)), dtype=object)
    n_dates = sorted_dates.size
    wf = config.walk_forward
    assert wf.retrain_every_days is not None  # normalized in __post_init__
    retrain_every_days = wf.retrain_every_days

    if n_dates < wf.h_days + wf.t_days:
        raise ValueError(
            "returns must span at least h_days + t_days distinct calendar dates; "
            f"got {n_dates} dates for h_days={wf.h_days}, t_days={wf.t_days}."
        )

    max_window_start = n_dates - wf.h_days - wf.t_days
    bar_dates = returns.index.date

    windows: list[StandalonePredictorWindow] = []
    signal_parts: list[pd.Series] = []
    pre_cost_parts: list[pd.Series] = []
    post_cost_parts: list[pd.Series] = []

    for window_index, day_offset in enumerate(range(0, max_window_start + 1, retrain_every_days)):
        train_start_date = sorted_dates[day_offset]
        train_end_date = sorted_dates[day_offset + wf.h_days - 1]
        forecast_start_date = sorted_dates[day_offset + wf.h_days]
        forecast_end_date = sorted_dates[day_offset + wf.h_days + wf.t_days - 1]

        train_mask = (bar_dates >= train_start_date) & (bar_dates <= train_end_date)
        forecast_mask = (bar_dates >= forecast_start_date) & (bar_dates <= forecast_end_date)
        train_slice = returns.loc[train_mask]
        forecast_slice = returns.loc[forecast_mask]

        # No-leakage boundary check — identical guard to walk_forward.py.
        assert train_slice.index.max() < forecast_slice.index.min(), (
            "standalone predictor leakage guard tripped: "
            f"train ends at {train_slice.index.max()} but forecast starts at "
            f"{forecast_slice.index.min()}."
        )

        next_day_offset = day_offset + retrain_every_days
        if next_day_offset <= max_window_start:
            next_forecast_start_date = sorted_dates[next_day_offset + wf.h_days]
            effective_forecast = forecast_slice.loc[
                forecast_slice.index.date < next_forecast_start_date
            ]
        else:
            effective_forecast = forecast_slice

        if effective_forecast.shape[0] < 2:
            raise ValueError(
                "each effective forecast window must contain at least 2 observations so "
                "one-step-ahead signals can be aligned with realized returns; "
                f"window {window_index} has {effective_forecast.shape[0]} observation(s) "
                f"for t_days={wf.t_days} and retrain_every_days={retrain_every_days}."
            )

        # Compute feature on concat(train, forecast) so rolling features are
        # fully initialized at the forecast boundary.
        combined = pd.concat([train_slice, effective_forecast])
        feature_full = _build_feature(combined, config)
        feature_train = feature_full.loc[train_slice.index]
        feature_forecast = feature_full.loc[effective_forecast.index]

        if feature_forecast.isna().any():
            raise ValueError(
                f"forecast feature contains NaN in window {window_index}; "
                "increase h_days so the rolling window is fully initialized "
                "before the forecast slice."
            )

        # Fit spline on training data only. fit_spline_predictor drops NaN
        # pairs internally (e.g. the vol_ratio startup region at train start).
        spline_result = fit_spline_predictor(feature_train, train_slice, config=config.spline)

        predicted = spline_result.evaluate(feature_forecast)
        # evaluate() returns pd.Series when given a pd.Series input.
        assert isinstance(predicted, pd.Series)

        window_signal = sign_signal(predicted)
        window_pre_cost = align_signal_with_future_return(window_signal, effective_forecast)
        window_post_cost = apply_turnover_cost(
            window_pre_cost,
            signal_turnover(window_signal),
            cost_bps_per_turnover=cost_bps_per_turnover,
        )
        window_summary = _summarize_return_modes(
            window_pre_cost, window_post_cost, cost_bps_per_turnover=cost_bps_per_turnover
        )

        signal_parts.append(window_signal)
        pre_cost_parts.append(window_pre_cost)
        post_cost_parts.append(window_post_cost)
        windows.append(
            StandalonePredictorWindow(
                index=window_index,
                train_start=train_slice.index.min(),
                train_end=train_slice.index.max(),
                forecast_start=effective_forecast.index.min(),
                forecast_end=effective_forecast.index.max(),
                n_train_obs=int(train_slice.shape[0]),
                n_forecast_obs=int(effective_forecast.shape[0]),
                n_knots_effective=spline_result.n_knots_effective,
                summary=window_summary,
            )
        )

    combined_signal = pd.concat(signal_parts).astype(np.int8)
    combined_signal.name = "signal"
    pre_cost_returns = pd.concat(pre_cost_parts)
    post_cost_returns = pd.concat(post_cost_parts)
    summary = _summarize_return_modes(
        pre_cost_returns, post_cost_returns, cost_bps_per_turnover=cost_bps_per_turnover
    )

    return StandalonePredictorResult(
        config=config,
        windows=tuple(windows),
        signal=combined_signal,
        pre_cost_returns=pre_cost_returns,
        post_cost_returns=post_cost_returns,
        summary=summary,
        cost_bps_per_turnover=float(cost_bps_per_turnover),
    )


# ---------------------------------------------------------------------------
# Experiment config (data source + standalone predictor config)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StandaloneExperimentConfig:
    """Full recipe for one standalone-predictor experiment run.

    Mirrors :class:`~hft_hmm.config.experiment_config.ExperimentConfig` but
    replaces the HMM walk-forward config with :class:`StandaloneWalkForwardConfig`
    and adds predictor-specific sub-configs.

    References: §4 side-information predictor standalone evaluation (evaluation layer)
    """

    data: DataSourceConfig
    frequency: Frequency
    predictor: PredictorKind
    walk_forward: StandaloneWalkForwardConfig
    spline: SplinePredictorConfig = field(default_factory=SplinePredictorConfig)
    vol_ratio: VolatilityRatioConfig = field(default_factory=VolatilityRatioConfig)
    seasonality: SeasonalityConfig = field(default_factory=SeasonalityConfig)
    cost_bps_per_turnover: float = 0.0
    notes: str = ""
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, DataSourceConfig):
            raise TypeError(f"data must be a DataSourceConfig, got {type(self.data).__name__}.")
        if self.frequency not in _VALID_FREQUENCIES:
            raise ValueError(
                f"frequency must be one of {_VALID_FREQUENCIES}, got {self.frequency!r}."
            )
        if self.predictor not in _PREDICTOR_KINDS:
            raise ValueError(
                f"predictor must be one of {_PREDICTOR_KINDS}, got {self.predictor!r}."
            )
        if not isinstance(self.walk_forward, StandaloneWalkForwardConfig):
            raise TypeError(
                "walk_forward must be a StandaloneWalkForwardConfig, "
                f"got {type(self.walk_forward).__name__}."
            )
        if not isinstance(self.spline, SplinePredictorConfig):
            raise TypeError(
                f"spline must be a SplinePredictorConfig, got {type(self.spline).__name__}."
            )
        if not isinstance(self.vol_ratio, VolatilityRatioConfig):
            raise TypeError(
                "vol_ratio must be a VolatilityRatioConfig, "
                f"got {type(self.vol_ratio).__name__}."
            )
        if not isinstance(self.seasonality, SeasonalityConfig):
            raise TypeError(
                "seasonality must be a SeasonalityConfig, "
                f"got {type(self.seasonality).__name__}."
            )
        cost = float(self.cost_bps_per_turnover)
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(
                f"cost_bps_per_turnover must be a finite non-negative float, got {cost!r}."
            )
        object.__setattr__(self, "cost_bps_per_turnover", cost)

        if self.data.is_reproducible:
            if self.sha256 is None:
                raise ValueError("StandaloneExperimentConfig for file-backed data requires sha256.")
            if not isinstance(self.sha256, str):
                raise ValueError(
                    "sha256 must be a string for file-backed standalone experiments; "
                    f"got {type(self.sha256).__name__}."
                )
            normalized = self.sha256.lower()
            if len(normalized) != _SHA256_HEX_LENGTH or any(
                c not in "0123456789abcdef" for c in normalized
            ):
                raise ValueError(
                    f"sha256 must be a {_SHA256_HEX_LENGTH}-character hex digest, "
                    f"got {self.sha256!r}."
                )
            object.__setattr__(self, "sha256", normalized)
        elif self.sha256 is not None:
            raise ValueError("sha256 must not be set for non-file-backed data.")

    @property
    def is_reproducible(self) -> bool:
        """``True`` when the run is file-backed and pinned to a specific file digest."""
        return self.data.is_reproducible and self.sha256 is not None

    def to_predictor_config(self) -> StandalonePredictorConfig:
        """Return the inner predictor config for passing to ``standalone_predictor_backtest``."""
        return StandalonePredictorConfig(
            predictor=self.predictor,
            walk_forward=self.walk_forward,
            spline=self.spline,
            vol_ratio=self.vol_ratio,
            seasonality=self.seasonality,
        )

    def to_dict(self) -> dict[str, Any]:
        wf = self.walk_forward
        assert wf.retrain_every_days is not None
        return {
            "cost_bps_per_turnover": float(self.cost_bps_per_turnover),
            "data": self.data.to_dict(),
            "frequency": self.frequency,
            "notes": self.notes,
            "predictor": self.predictor,
            "seasonality": {
                "bucket_minutes": int(self.seasonality.bucket_minutes),
                "exchange_tz": str(self.seasonality.exchange_tz),
                "normalize": bool(self.seasonality.normalize),
            },
            "sha256": self.sha256,
            "spline": {
                "degree": int(self.spline.degree),
                "demean": bool(self.spline.demean),
                "demean_grid_size": int(self.spline.demean_grid_size),
                "min_obs": int(self.spline.min_obs),
                "n_knots": int(self.spline.n_knots),
            },
            "vol_ratio": {
                "decay": float(self.vol_ratio.decay),
                "fast_window": int(self.vol_ratio.fast_window),
                "slow_window": int(self.vol_ratio.slow_window),
            },
            "walk_forward": {
                "h_days": int(wf.h_days),
                "retrain_every_days": int(wf.retrain_every_days),
                "t_days": int(wf.t_days),
            },
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StandaloneExperimentConfig:
        wf_raw = raw["walk_forward"]
        walk_forward = StandaloneWalkForwardConfig(
            h_days=int(wf_raw["h_days"]),
            t_days=int(wf_raw["t_days"]),
            retrain_every_days=int(wf_raw["retrain_every_days"])
            if "retrain_every_days" in wf_raw
            else None,
        )
        sp_raw = raw.get("spline", {})
        spline = SplinePredictorConfig(
            n_knots=int(sp_raw.get("n_knots", 5)),
            degree=int(sp_raw.get("degree", 3)),
            min_obs=int(sp_raw.get("min_obs", 20)),
            demean=bool(sp_raw.get("demean", False)),
            demean_grid_size=int(sp_raw.get("demean_grid_size", 1000)),
        )
        vr_raw = raw.get("vol_ratio", {})
        vol_ratio = VolatilityRatioConfig(
            decay=float(vr_raw.get("decay", 0.79)),
            fast_window=int(vr_raw.get("fast_window", 50)),
            slow_window=int(vr_raw.get("slow_window", 100)),
        )
        s_raw = raw.get("seasonality", {})
        seasonality = SeasonalityConfig(
            exchange_tz=str(s_raw.get("exchange_tz", "America/Chicago")),
            bucket_minutes=int(s_raw.get("bucket_minutes", 1)),
            normalize=bool(s_raw.get("normalize", True)),
        )
        return cls(
            data=DataSourceConfig.from_dict(raw["data"]),
            frequency=raw["frequency"],
            predictor=raw["predictor"],
            walk_forward=walk_forward,
            spline=spline,
            vol_ratio=vol_ratio,
            seasonality=seasonality,
            cost_bps_per_turnover=float(raw.get("cost_bps_per_turnover", 0.0)),
            notes=str(raw.get("notes", "")),
            sha256=raw.get("sha256"),
        )

    def to_yaml_bytes(self) -> bytes:
        """Canonical UTF-8 YAML bytes — the exact input to :func:`standalone_run_id`."""
        return yaml.safe_dump(
            self.to_dict(),
            sort_keys=True,
            default_flow_style=False,
            allow_unicode=False,
            width=1_000_000,
        ).encode("utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> StandaloneExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(
                f"YAML at {path} must decode to a mapping, got {type(raw).__name__}."
            )
        return cls.from_dict(raw)


def standalone_run_id(config: StandaloneExperimentConfig) -> str:
    """Return the 12-char hex run id derived from the config's canonical YAML bytes.

    Two configs hash to the same id iff they produce the same canonical YAML
    output, mirroring :func:`~hft_hmm.config.experiment_config.run_id`.

    References: §4.4 reproducible simulation artifacts (evaluation layer)
    """
    if not isinstance(config, StandaloneExperimentConfig):
        raise TypeError(
            f"config must be a StandaloneExperimentConfig, got {type(config).__name__}."
        )
    return hashlib.sha256(config.to_yaml_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StandaloneRunArtifacts:
    """Handle returned by :func:`run_standalone_experiment`."""

    run_id: str
    directory: Path
    config: StandaloneExperimentConfig
    result: StandalonePredictorResult


def run_standalone_experiment(
    config: StandaloneExperimentConfig,
    *,
    runs_root: Path | str = Path("runs"),
    force: bool = False,
) -> StandaloneRunArtifacts:
    """Run the standalone predictor experiment described by ``config`` and write artifacts.

    The target directory is ``runs_root / standalone_run_id(config)``. If it
    already exists, ``force=True`` wipes it before writing; otherwise a
    :exc:`FileExistsError` is raised.

    References: §4 side-information predictor standalone evaluation (evaluation layer)
    """
    if not isinstance(config, StandaloneExperimentConfig):
        raise TypeError(
            f"config must be a StandaloneExperimentConfig, got {type(config).__name__}."
        )
    runs_root_path = Path(runs_root)
    experiment_id = standalone_run_id(config)
    run_dir = runs_root_path / experiment_id

    if run_dir.exists() and not force:
        raise FileExistsError(
            f"Run directory already exists at {run_dir}; pass force=True to overwrite."
        )

    reproducible = _validate_standalone_reproducibility(config)
    returns = _load_returns(config)
    predictor_config = config.to_predictor_config()
    result = standalone_predictor_backtest(
        returns,
        predictor_config,
        cost_bps_per_turnover=config.cost_bps_per_turnover,
    )

    runs_root_path.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f"{experiment_id}.tmp-", dir=runs_root_path))
    backup_dir: Path | None = None
    try:
        _write_artifacts(staging_dir, config, experiment_id, result, reproducible=reproducible)
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

    return StandaloneRunArtifacts(
        run_id=experiment_id,
        directory=run_dir,
        config=config,
        result=result,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_feature(
    returns: pd.Series,
    config: StandalonePredictorConfig,
) -> pd.Series:
    """Return the side-information feature series aligned to ``returns.index``."""
    if config.predictor == "volatility_ratio":
        return volatility_ratio(
            returns,
            fast_window=config.vol_ratio.fast_window,
            slow_window=config.vol_ratio.slow_window,
            decay=config.vol_ratio.decay,
        )
    # seasonality only looks at the DatetimeIndex, not the values.
    return intraday_seasonality(returns, config=config.seasonality)


def _validate_returns(returns: pd.Series) -> None:
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a pd.Series, got {type(returns).__name__}.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns.index must be a pd.DatetimeIndex.")
    if returns.index.tz is None:
        raise ValueError("returns.index must be tz-aware; localize to UTC before calling.")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns.index must be monotonically increasing.")
    if returns.index.has_duplicates:
        raise ValueError("returns.index must not contain duplicates.")
    if not np.all(np.isfinite(np.asarray(returns, dtype=float))):
        raise ValueError("returns must contain only finite values; drop NaN/inf first.")


def _summarize_return_modes(
    pre_cost_returns: pd.Series,
    post_cost_returns: pd.Series,
    *,
    cost_bps_per_turnover: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _summary_row(pre_cost_returns, cost_bps_per_turnover=0.0),
            _summary_row(post_cost_returns, cost_bps_per_turnover=cost_bps_per_turnover),
        ],
        index=pd.Index(["pre-cost", "post-cost"], name="mode"),
    )


def _summary_row(
    strategy_returns: pd.Series,
    *,
    cost_bps_per_turnover: float,
) -> dict[str, float | int]:
    return {
        "n_periods": int(strategy_returns.shape[0]),
        "cost_bps_per_turnover": float(cost_bps_per_turnover),
        "cumulative_return": cumulative_return(strategy_returns),
        "sharpe_ratio": sharpe_ratio(strategy_returns),
        "max_drawdown": max_drawdown(strategy_returns),
        "hit_rate": hit_rate(strategy_returns),
    }


def _load_returns(config: StandaloneExperimentConfig) -> pd.Series:
    return load_returns_from_source(config.data, frequency=config.frequency)


def _validate_standalone_reproducibility(config: StandaloneExperimentConfig) -> bool:
    return validate_data_reproducibility(config, stacklevel=3)


def _write_artifacts(
    run_dir: Path,
    config: StandaloneExperimentConfig,
    experiment_id: str,
    result: StandalonePredictorResult,
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
    config: StandaloneExperimentConfig,
    experiment_id: str,
    result: StandalonePredictorResult,
    *,
    reproducible: bool,
) -> None:
    payload: dict[str, Any] = {
        "run_id": experiment_id,
        "reproducible": reproducible,
        "predictor": config.predictor,
        "cost_bps_per_turnover": float(config.cost_bps_per_turnover),
        "n_windows": len(result.windows),
        "n_forecast_obs": int(result.signal.shape[0]),
        "summary": _summary_to_payload(result.summary),
    }
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_log(path: Path, result: StandalonePredictorResult) -> None:
    lines = []
    for w in result.windows:
        lines.append(
            json.dumps(
                {
                    "index": int(w.index),
                    "train_start": w.train_start.isoformat(),
                    "train_end": w.train_end.isoformat(),
                    "forecast_start": w.forecast_start.isoformat(),
                    "forecast_end": w.forecast_end.isoformat(),
                    "n_knots_effective": int(w.n_knots_effective),
                    "n_train_obs": int(w.n_train_obs),
                    "n_forecast_obs": int(w.n_forecast_obs),
                    "summary": _summary_to_payload(w.summary),
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_to_payload(summary: pd.DataFrame) -> dict[str, dict[str, float | None]]:
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
    return round(numeric, 8)
