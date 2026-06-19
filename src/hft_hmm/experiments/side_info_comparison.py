"""Gate H/K/Q side-information comparison experiment.

Runs side-information walk-forward variants on the same return series with the same
window geometry, K selection, cost model, and reporting units, so the
variants are directly comparable on pre-cost signal quality:

1. ``baseline`` — Gaussian HMM walk-forward (delegates to
   :func:`hft_hmm.experiments.walk_forward.walk_forward` so the summary
   matches the existing baseline path exactly).
2. ``volatility_ratio_conditioned`` — same walk-forward geometry, but the
   transition matrix is replaced per-bar by an Issue-16 bucketed
   approximation conditioned on the volatility-ratio side information.
3. ``seasonality_conditioned`` — same as above with the intraday seasonality
   feature.
4. ``volatility_ratio_hmc_continuous`` and
   ``seasonality_hmc_continuous`` — Gate K continuous-parametric IOHMM
   transition functions, fit per window with NumPyro NUTS on decoded training
   states while holding Baum-Welch emissions fixed.
5. ``vol_ratio_seasonality_hmc_continuous`` — Gate Q combined-predictor
   continuous-parametric IOHMM using volatility-ratio and intraday-seasonality
   side information jointly.

The IOHMM-style conditioning is the engineering approximation from
``hft_hmm.models.iohmm_approx``: a bucket index is derived from the
side-information value, and a separate K × K transition matrix is fitted per
bucket on the training slice, with the per-window MLE matrix as the smoothing
prior.

Pre-cost Sharpe is the primary paper-comparison metric. Post-cost results are
reported separately as turnover-cost diagnostics under the repo's fixed linear
cost convention; they are not calibrated to the paper's unspecified execution
model.

References: §4 side-information / IOHMM evaluation (evaluation layer)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, TypeVar

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import logsumexp

from hft_hmm.config.experiment_config import (
    DataSourceConfig,
    Frequency,
)
from hft_hmm.core import EVALUATION_LAYER, PaperReference, reference
from hft_hmm.evaluation import (
    apply_turnover_cost,
    cumulative_return,
    daily_annualized_sharpe_ratio,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    signal_turnover,
    turnover_diagnostics,
)
from hft_hmm.experiments._data_loading import (
    load_returns_from_source,
    validate_data_reproducibility,
)
from hft_hmm.experiments.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
    conviction_scale_from_train_predictions,
    walk_forward,
)
from hft_hmm.features.seasonality import SeasonalityConfig, intraday_seasonality
from hft_hmm.features.splines import (
    SplinePredictorConfig,
    SplinePredictorResult,
    fit_spline_predictor,
)
from hft_hmm.features.volatility_ratio import VolatilityRatioConfig, volatility_ratio
from hft_hmm.inference import filter_from_result
from hft_hmm.inference.forward_filter import forward_filter
from hft_hmm.models.default_hmm import fit_default_hmm
from hft_hmm.models.dmm import DMMConfig, expected_next_returns_from_dmm, fit_dmm
from hft_hmm.models.gaussian_hmm import GaussianHMMResult, GaussianHMMWrapper
from hft_hmm.models.iohmm_approx import (
    BoundaryMode,
    BucketedTransitionConfig,
    BucketedTransitionResult,
    bucket_boundaries_from_quantiles,
    bucket_boundaries_from_spline_grid,
    fit_bucketed_transition_model,
)
from hft_hmm.models.iohmm_continuous import (
    ContinuousIOHMMConfig,
    ContinuousIOHMMResult,
    fit_continuous_iohmm,
    transition_probabilities_at,
)
from hft_hmm.strategy import (
    SignalPolicy,
    align_signal_with_future_return,
    build_signal,
)
from hft_hmm.strategy.signals import (
    _DISCRETE_SIGNAL_POLICIES,
    _VALID_SIGNAL_POLICIES,
)

__category__: Final[str] = EVALUATION_LAYER
SIDE_INFO_COMPARISON_REFERENCE: Final[PaperReference] = reference(
    "§4", "side-information IOHMM comparison (Gate H)"
)

_VALID_FREQUENCIES: Final[tuple[str, ...]] = ("1min", "5min", "1D")
_SHA256_HEX_LENGTH: Final[int] = 64

BASELINE_VARIANT: Final[str] = "baseline"
VOLATILITY_RATIO_VARIANT: Final[str] = "volatility_ratio_conditioned"
SEASONALITY_VARIANT: Final[str] = "seasonality_conditioned"
VOLATILITY_RATIO_HMC_CONTINUOUS_VARIANT: Final[str] = "volatility_ratio_hmc_continuous"
SEASONALITY_HMC_CONTINUOUS_VARIANT: Final[str] = "seasonality_hmc_continuous"
VOL_RATIO_SEASONALITY_HMC_CONTINUOUS_VARIANT: Final[str] = "vol_ratio_seasonality_hmc_continuous"
DEFAULT_HMM_VARIANT: Final[str] = "default_hmm"
DMM_VARIANT: Final[str] = "dmm"
EXPECTED_VARIANTS: Final[tuple[str, ...]] = (
    BASELINE_VARIANT,
    VOLATILITY_RATIO_VARIANT,
    SEASONALITY_VARIANT,
    VOLATILITY_RATIO_HMC_CONTINUOUS_VARIANT,
    SEASONALITY_HMC_CONTINUOUS_VARIANT,
    VOL_RATIO_SEASONALITY_HMC_CONTINUOUS_VARIANT,
    DMM_VARIANT,
    DEFAULT_HMM_VARIANT,
)
_SIDE_INFO_VARIANTS: Final[tuple[str, ...]] = (
    VOLATILITY_RATIO_VARIANT,
    SEASONALITY_VARIANT,
    VOLATILITY_RATIO_HMC_CONTINUOUS_VARIANT,
    SEASONALITY_HMC_CONTINUOUS_VARIANT,
    VOL_RATIO_SEASONALITY_HMC_CONTINUOUS_VARIANT,
)
_DMM_FEATURE_IDENTITY: Final[str] = "volatility_ratio"
_MIN_STANDARDIZATION_STD: Final[float] = 1e-12
_HMC_CONTINUOUS_VARIANTS: Final[tuple[str, ...]] = (
    VOLATILITY_RATIO_HMC_CONTINUOUS_VARIANT,
    SEASONALITY_HMC_CONTINUOUS_VARIANT,
    VOL_RATIO_SEASONALITY_HMC_CONTINUOUS_VARIANT,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SideInfoComparisonConfig:
    """Recipe for one Gate H side-information comparison run.

    The ``walk_forward`` block is the same :class:`WalkForwardConfig` consumed
    by the baseline path so the baseline summary is bit-for-bit identical to a
    direct ``walk_forward`` invocation. ``spline``, ``bucketed_transition``,
    ``vol_ratio``, and ``seasonality`` parameterise the side-information
    variants only — the baseline ignores them.

    References: §4 side-information IOHMM comparison (evaluation layer)
    """

    data: DataSourceConfig
    frequency: Frequency
    walk_forward: WalkForwardConfig
    spline: SplinePredictorConfig = field(default_factory=SplinePredictorConfig)
    bucketed_transition: BucketedTransitionConfig = field(default_factory=BucketedTransitionConfig)
    continuous_iohmm: ContinuousIOHMMConfig = field(default_factory=ContinuousIOHMMConfig)
    dmm: DMMConfig = field(default_factory=DMMConfig)
    vol_ratio: VolatilityRatioConfig = field(default_factory=VolatilityRatioConfig)
    seasonality: SeasonalityConfig = field(default_factory=SeasonalityConfig)
    cost_bps_per_turnover: float = 0.0
    signal_policy: SignalPolicy = "sign"
    signal_threshold: float = 0.0
    notes: str = ""
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, DataSourceConfig):
            raise TypeError(f"data must be a DataSourceConfig, got {type(self.data).__name__}.")
        if self.frequency not in _VALID_FREQUENCIES:
            raise ValueError(
                f"frequency must be one of {_VALID_FREQUENCIES}, got {self.frequency!r}."
            )
        if not isinstance(self.walk_forward, WalkForwardConfig):
            raise TypeError(
                "walk_forward must be a WalkForwardConfig, "
                f"got {type(self.walk_forward).__name__}."
            )
        if not isinstance(self.spline, SplinePredictorConfig):
            raise TypeError(
                f"spline must be a SplinePredictorConfig, got {type(self.spline).__name__}."
            )
        if not isinstance(self.bucketed_transition, BucketedTransitionConfig):
            raise TypeError(
                "bucketed_transition must be a BucketedTransitionConfig, "
                f"got {type(self.bucketed_transition).__name__}."
            )
        if not isinstance(self.continuous_iohmm, ContinuousIOHMMConfig):
            raise TypeError(
                "continuous_iohmm must be a ContinuousIOHMMConfig, "
                f"got {type(self.continuous_iohmm).__name__}."
            )
        if not isinstance(self.dmm, DMMConfig):
            raise TypeError(f"dmm must be a DMMConfig, got {type(self.dmm).__name__}.")
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
        if self.signal_policy not in _VALID_SIGNAL_POLICIES:
            raise ValueError(
                f"signal_policy must be one of {_VALID_SIGNAL_POLICIES}, "
                f"got {self.signal_policy!r}."
            )
        signal_threshold = float(self.signal_threshold)
        if not math.isfinite(signal_threshold) or signal_threshold < 0.0:
            raise ValueError(
                "signal_threshold must be a finite non-negative float, "
                f"got {self.signal_threshold!r}."
            )
        object.__setattr__(self, "signal_threshold", signal_threshold)

        if self.data.is_reproducible:
            if self.sha256 is None:
                raise ValueError("SideInfoComparisonConfig for file-backed data requires sha256.")
            if not isinstance(self.sha256, str):
                raise TypeError(
                    "sha256 must be a string for file-backed comparisons; "
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
        return self.data.is_reproducible and self.sha256 is not None

    def to_dict(self) -> dict[str, Any]:
        wf = self.walk_forward
        retrain_every_days = wf.retrain_every_days
        assert retrain_every_days is not None  # normalized in WalkForwardConfig.__post_init__
        return {
            "bucketed_transition": {
                "boundary_mode": self.bucketed_transition.boundary_mode,
                "grid_size": int(self.bucketed_transition.grid_size),
                "n_buckets": int(self.bucketed_transition.n_buckets),
                "smoothing": float(self.bucketed_transition.smoothing),
            },
            "continuous_iohmm": {
                "ess_bulk_threshold": int(self.continuous_iohmm.ess_bulk_threshold),
                "num_chains": int(self.continuous_iohmm.num_chains),
                "num_samples": int(self.continuous_iohmm.num_samples),
                "num_warmup": int(self.continuous_iohmm.num_warmup),
                "rhat_threshold": float(self.continuous_iohmm.rhat_threshold),
                "seed": int(self.continuous_iohmm.seed),
                "target_accept_prob": float(self.continuous_iohmm.target_accept_prob),
            },
            "cost_bps_per_turnover": float(self.cost_bps_per_turnover),
            "data": self.data.to_dict(),
            "dmm": {
                "annealing_epochs": int(self.dmm.annealing_epochs),
                "beta1": float(self.dmm.beta1),
                "beta2": float(self.dmm.beta2),
                "clip_norm": float(self.dmm.clip_norm),
                "emission_dim": int(self.dmm.emission_dim),
                "learning_rate": float(self.dmm.learning_rate),
                "lr_decay": float(self.dmm.lr_decay),
                "min_annealing_factor": float(self.dmm.min_annealing_factor),
                "min_scale": float(self.dmm.min_scale),
                "mini_batch_size": int(self.dmm.mini_batch_size),
                "num_epochs": int(self.dmm.num_epochs),
                "num_layers": int(self.dmm.num_layers),
                "obs_dim": int(self.dmm.obs_dim),
                "rnn_dim": int(self.dmm.rnn_dim),
                "rnn_dropout_rate": float(self.dmm.rnn_dropout_rate),
                "seed": int(self.dmm.seed),
                "side_info_dim": int(self.dmm.side_info_dim),
                "transition_dim": int(self.dmm.transition_dim),
                "weight_decay": float(self.dmm.weight_decay),
                "z_dim": int(self.dmm.z_dim),
            },
            "frequency": self.frequency,
            "notes": self.notes,
            "seasonality": {
                "bucket_minutes": int(self.seasonality.bucket_minutes),
                "exchange_tz": str(self.seasonality.exchange_tz),
                "normalize": bool(self.seasonality.normalize),
            },
            "sha256": self.sha256,
            "signal_policy": self.signal_policy,
            "signal_threshold": float(self.signal_threshold),
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
                "k_values": list(wf.k_values),
                "min_variance": float(wf.min_variance),
                "n_iter": int(wf.n_iter),
                "random_state": int(wf.random_state),
                "retrain_every_days": int(retrain_every_days),
                "t_days": int(wf.t_days),
                "tol": float(wf.tol),
                "variance_floor_policy": str(wf.variance_floor_policy),
            },
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SideInfoComparisonConfig:
        wf_raw = raw["walk_forward"]
        for required_key in ("min_variance", "variance_floor_policy"):
            if required_key not in wf_raw:
                raise ValueError(
                    f"walk_forward.{required_key} is required so the EM-stability knobs "
                    "are explicit in the comparison_id hash."
                )
        walk_forward_cfg = WalkForwardConfig(
            h_days=int(wf_raw["h_days"]),
            t_days=int(wf_raw["t_days"]),
            retrain_every_days=(
                int(wf_raw["retrain_every_days"])
                if wf_raw.get("retrain_every_days") is not None
                else None
            ),
            k_values=tuple(int(k) for k in wf_raw["k_values"]),
            random_state=int(wf_raw["random_state"]),
            n_iter=int(wf_raw["n_iter"]),
            tol=float(wf_raw["tol"]),
            min_variance=float(wf_raw["min_variance"]),
            variance_floor_policy=wf_raw["variance_floor_policy"],
        )
        sp_raw = raw.get("spline", {})
        spline = SplinePredictorConfig(
            n_knots=int(sp_raw.get("n_knots", 5)),
            degree=int(sp_raw.get("degree", 3)),
            min_obs=int(sp_raw.get("min_obs", 20)),
            demean=bool(sp_raw.get("demean", False)),
            demean_grid_size=int(sp_raw.get("demean_grid_size", 1000)),
        )
        bt_raw = raw.get("bucketed_transition", {})
        bucketed = BucketedTransitionConfig(
            n_buckets=int(bt_raw.get("n_buckets", 3)),
            smoothing=float(bt_raw.get("smoothing", 1.0)),
            grid_size=int(bt_raw.get("grid_size", 200)),
            boundary_mode=bt_raw.get("boundary_mode", "grid"),
        )
        ci_raw = raw.get("continuous_iohmm", {})
        if not isinstance(ci_raw, Mapping):
            raise ValueError("continuous_iohmm must be a mapping; " f"got {type(ci_raw).__name__}.")
        continuous_iohmm = ContinuousIOHMMConfig(
            num_chains=int(ci_raw.get("num_chains", 2)),
            num_warmup=int(ci_raw.get("num_warmup", 500)),
            num_samples=int(ci_raw.get("num_samples", 1000)),
            target_accept_prob=float(ci_raw.get("target_accept_prob", 0.8)),
            seed=int(ci_raw.get("seed", 0)),
            rhat_threshold=float(ci_raw.get("rhat_threshold", 1.05)),
            ess_bulk_threshold=int(ci_raw.get("ess_bulk_threshold", 200)),
        )
        dmm_raw = raw.get("dmm", {})
        if not isinstance(dmm_raw, Mapping):
            raise ValueError(f"dmm must be a mapping; got {type(dmm_raw).__name__}.")
        dmm = DMMConfig(
            obs_dim=int(dmm_raw.get("obs_dim", 1)),
            z_dim=int(dmm_raw.get("z_dim", 16)),
            emission_dim=int(dmm_raw.get("emission_dim", 32)),
            transition_dim=int(dmm_raw.get("transition_dim", 32)),
            rnn_dim=int(dmm_raw.get("rnn_dim", 32)),
            side_info_dim=int(dmm_raw.get("side_info_dim", 0)),
            num_layers=int(dmm_raw.get("num_layers", 1)),
            rnn_dropout_rate=float(dmm_raw.get("rnn_dropout_rate", 0.1)),
            min_scale=float(dmm_raw.get("min_scale", 1e-4)),
            num_epochs=int(dmm_raw.get("num_epochs", 150)),
            mini_batch_size=int(dmm_raw.get("mini_batch_size", 20)),
            learning_rate=float(dmm_raw.get("learning_rate", 3e-4)),
            beta1=float(dmm_raw.get("beta1", 0.96)),
            beta2=float(dmm_raw.get("beta2", 0.999)),
            clip_norm=float(dmm_raw.get("clip_norm", 10.0)),
            lr_decay=float(dmm_raw.get("lr_decay", 0.99996)),
            weight_decay=float(dmm_raw.get("weight_decay", 2.0)),
            min_annealing_factor=float(dmm_raw.get("min_annealing_factor", 0.2)),
            annealing_epochs=int(dmm_raw.get("annealing_epochs", 1000)),
            seed=int(dmm_raw.get("seed", 54)),
        )
        vr_raw = raw.get("vol_ratio", {})
        vol_ratio_cfg = VolatilityRatioConfig(
            decay=float(vr_raw.get("decay", 0.79)),
            fast_window=int(vr_raw.get("fast_window", 50)),
            slow_window=int(vr_raw.get("slow_window", 100)),
        )
        s_raw = raw.get("seasonality", {})
        seasonality_cfg = SeasonalityConfig(
            exchange_tz=str(s_raw.get("exchange_tz", "America/Chicago")),
            bucket_minutes=int(s_raw.get("bucket_minutes", 1)),
            normalize=bool(s_raw.get("normalize", True)),
        )
        return cls(
            data=DataSourceConfig.from_dict(raw["data"]),
            frequency=raw["frequency"],
            walk_forward=walk_forward_cfg,
            spline=spline,
            bucketed_transition=bucketed,
            continuous_iohmm=continuous_iohmm,
            dmm=dmm,
            vol_ratio=vol_ratio_cfg,
            seasonality=seasonality_cfg,
            cost_bps_per_turnover=float(raw.get("cost_bps_per_turnover", 0.0)),
            signal_policy=raw.get("signal_policy", "sign"),
            signal_threshold=float(raw.get("signal_threshold", 0.0)),
            notes=str(raw.get("notes", "")),
            sha256=raw.get("sha256"),
        )

    def to_yaml_bytes(self) -> bytes:
        return yaml.safe_dump(
            self.to_dict(),
            sort_keys=True,
            default_flow_style=False,
            allow_unicode=False,
            width=1_000_000,
        ).encode("utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> SideInfoComparisonConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"YAML at {path} must decode to a mapping, got {type(raw).__name__}.")
        return cls.from_dict(raw)


def comparison_id(config: SideInfoComparisonConfig) -> str:
    """Return the 12-char hex comparison id derived from the canonical YAML bytes."""
    if not isinstance(config, SideInfoComparisonConfig):
        raise TypeError(
            "config must be a SideInfoComparisonConfig, " f"got {type(config).__name__}."
        )
    return hashlib.sha256(config.to_yaml_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Per-window and per-variant result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SideInfoVariantWindow:
    """Per-window record for a side-information variant."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    chosen_k: int
    n_train_obs: int
    n_forecast_obs: int
    bucket_observation_counts: tuple[int, ...]
    boundary_mode: str
    summary: pd.DataFrame

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must be non-negative, got {self.index}.")
        if self.chosen_k < 2:
            raise ValueError(f"chosen_k must be >= 2, got {self.chosen_k}.")
        if self.n_train_obs < 1:
            raise ValueError(f"n_train_obs must be >= 1, got {self.n_train_obs}.")
        if self.n_forecast_obs < 2:
            raise ValueError(
                f"n_forecast_obs must be >= 2 to align signals; got {self.n_forecast_obs}."
            )
        if self.train_end < self.train_start:
            raise ValueError("train_end must not precede train_start.")
        if self.forecast_end < self.forecast_start:
            raise ValueError("forecast_end must not precede forecast_start.")
        if self.forecast_start <= self.train_end:
            raise ValueError(
                "forecast_start must be strictly after train_end; "
                f"got train_end={self.train_end} forecast_start={self.forecast_start}."
            )
        if self.boundary_mode not in ("grid", "quantile"):
            raise ValueError(
                "boundary_mode must be one of ('grid', 'quantile'), " f"got {self.boundary_mode!r}."
            )
        if not isinstance(self.summary, pd.DataFrame):
            raise TypeError("summary must be a pd.DataFrame.")


@dataclass(frozen=True)
class ContinuousSideInfoVariantWindow:
    """Per-window record for a continuous-parametric HMC side-information variant."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    chosen_k: int
    n_train_obs: int
    n_forecast_obs: int
    converged: bool
    rhat_max: float
    ess_bulk_min: int
    posterior_mean_W: np.ndarray
    posterior_mean_b: np.ndarray
    posterior_samples: dict[str, np.ndarray]
    summary: pd.DataFrame

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must be non-negative, got {self.index}.")
        if self.chosen_k < 2:
            raise ValueError(f"chosen_k must be >= 2, got {self.chosen_k}.")
        if self.n_train_obs < 1:
            raise ValueError(f"n_train_obs must be >= 1, got {self.n_train_obs}.")
        if self.n_forecast_obs < 2:
            raise ValueError(
                f"n_forecast_obs must be >= 2 to align signals; got {self.n_forecast_obs}."
            )
        if self.train_end < self.train_start:
            raise ValueError("train_end must not precede train_start.")
        if self.forecast_end < self.forecast_start:
            raise ValueError("forecast_end must not precede forecast_start.")
        if self.forecast_start <= self.train_end:
            raise ValueError(
                "forecast_start must be strictly after train_end; "
                f"got train_end={self.train_end} forecast_start={self.forecast_start}."
            )
        rhat_max = float(self.rhat_max)
        if not math.isfinite(rhat_max) or rhat_max < 0.0:
            raise ValueError(f"rhat_max must be finite and non-negative, got {rhat_max!r}.")
        if not isinstance(self.ess_bulk_min, int) or isinstance(self.ess_bulk_min, bool):
            raise TypeError(
                f"ess_bulk_min must be an integer, got {type(self.ess_bulk_min).__name__}."
            )
        if self.ess_bulk_min < 0:
            raise ValueError(f"ess_bulk_min must be non-negative, got {self.ess_bulk_min}.")

        mean_w = np.asarray(self.posterior_mean_W, dtype=float).copy()
        mean_b = np.asarray(self.posterior_mean_b, dtype=float).copy()
        for attr_name, values in (
            ("posterior_mean_W", mean_w),
            ("posterior_mean_b", mean_b),
        ):
            expected_shape = (
                (self.chosen_k, self.chosen_k, values.shape[2])
                if attr_name == "posterior_mean_W" and values.ndim == 3
                else (self.chosen_k, self.chosen_k)
            )
            if values.shape != expected_shape:
                raise ValueError(
                    f"{attr_name} must have shape {expected_shape}, " f"got {values.shape}."
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{attr_name} must contain only finite values.")
            values.setflags(write=False)

        if not isinstance(self.posterior_samples, dict):
            raise TypeError(
                "posterior_samples must be a dict of parameter samples, "
                f"got {type(self.posterior_samples).__name__}."
            )
        missing = {"W", "b"} - set(self.posterior_samples)
        if missing:
            raise ValueError(f"posterior_samples is missing keys {sorted(missing)!r}.")
        frozen_samples: dict[str, np.ndarray] = {}
        sample_prefix: tuple[int, int] | None = None
        feature_dim: int | None = None
        for param_name in ("W", "b"):
            arr = np.asarray(self.posterior_samples[param_name], dtype=float).copy()
            expected_ndim = 5 if param_name == "W" else 4
            if arr.ndim != expected_ndim:
                raise ValueError(
                    f"posterior_samples[{param_name!r}] must be {expected_ndim}-D; "
                    f"got shape {arr.shape}."
                )
            if arr.shape[2] != self.chosen_k or arr.shape[3] != self.chosen_k:
                raise ValueError(
                    f"posterior_samples[{param_name!r}] trailing shape must be "
                    f"({self.chosen_k}, {self.chosen_k}); got {arr.shape}."
                )
            if param_name == "W":
                feature_dim = int(arr.shape[4])
                if feature_dim < 1:
                    raise ValueError("posterior_samples['W'] must have at least one feature.")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"posterior_samples[{param_name!r}] must be finite.")
            if sample_prefix is None:
                sample_prefix = (arr.shape[0], arr.shape[1])
            elif (arr.shape[0], arr.shape[1]) != sample_prefix:
                raise ValueError(
                    "posterior_samples['W'] and posterior_samples['b'] must share "
                    f"chain/sample dimensions; got {sample_prefix} and {arr.shape[:2]}."
                )
            arr.setflags(write=False)
            frozen_samples[param_name] = arr
        if mean_w.ndim != 3 or feature_dim is None or mean_w.shape[2] != feature_dim:
            raise ValueError(
                "posterior_mean_W feature dimension must match posterior_samples['W']; "
                f"got {mean_w.shape} and {feature_dim}."
            )

        if not isinstance(self.summary, pd.DataFrame):
            raise TypeError("summary must be a pd.DataFrame.")

        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "rhat_max", rhat_max)
        object.__setattr__(self, "posterior_mean_W", mean_w)
        object.__setattr__(self, "posterior_mean_b", mean_b)
        object.__setattr__(self, "posterior_samples", frozen_samples)


@dataclass(frozen=True)
class DmmVariantWindow:
    """Per-window record for the DMM side-information benchmark."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    chosen_k: int
    n_train_obs: int
    n_forecast_obs: int
    feature_identity: str
    side_info_dim: int
    num_epochs: int
    mini_batch_size: int
    learning_rate: float
    beta1: float
    beta2: float
    clip_norm: float
    lr_decay: float
    weight_decay: float
    min_annealing_factor: float
    annealing_epochs: int
    seed: int
    final_loss: float
    summary: pd.DataFrame

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must be non-negative, got {self.index}.")
        if self.chosen_k < 2:
            raise ValueError(f"chosen_k must be >= 2, got {self.chosen_k}.")
        if self.n_train_obs < 1:
            raise ValueError(f"n_train_obs must be >= 1, got {self.n_train_obs}.")
        if self.n_forecast_obs < 2:
            raise ValueError(
                f"n_forecast_obs must be >= 2 to align signals; got {self.n_forecast_obs}."
            )
        if self.train_end < self.train_start:
            raise ValueError("train_end must not precede train_start.")
        if self.forecast_end < self.forecast_start:
            raise ValueError("forecast_end must not precede forecast_start.")
        if self.forecast_start <= self.train_end:
            raise ValueError(
                "forecast_start must be strictly after train_end; "
                f"got train_end={self.train_end} forecast_start={self.forecast_start}."
            )
        if not isinstance(self.feature_identity, str) or not self.feature_identity:
            raise ValueError("feature_identity must be a non-empty string.")
        if self.side_info_dim < 1:
            raise ValueError(f"side_info_dim must be >= 1, got {self.side_info_dim}.")
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}.")
        if self.mini_batch_size < 1:
            raise ValueError(f"mini_batch_size must be >= 1, got {self.mini_batch_size}.")
        if self.annealing_epochs < 0:
            raise ValueError(f"annealing_epochs must be >= 0, got {self.annealing_epochs}.")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}.")
        for field_name, value in (
            ("learning_rate", self.learning_rate),
            ("clip_norm", self.clip_norm),
            ("lr_decay", self.lr_decay),
            ("weight_decay", self.weight_decay),
            ("min_annealing_factor", self.min_annealing_factor),
            ("beta1", self.beta1),
            ("beta2", self.beta2),
            ("final_loss", self.final_loss),
        ):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{field_name} must be finite, got {numeric!r}.")
        if not isinstance(self.summary, pd.DataFrame):
            raise TypeError("summary must be a pd.DataFrame.")


@dataclass(frozen=True)
class SideInfoVariantResult:
    """Per-variant result of a side-information comparison."""

    variant: str
    chosen_k_per_window: tuple[int, ...]
    windows: tuple[
        SideInfoVariantWindow
        | ContinuousSideInfoVariantWindow
        | DmmVariantWindow
        | WalkForwardWindow,
        ...,
    ]
    signal: pd.Series
    pre_cost_returns: pd.Series
    post_cost_returns: pd.Series
    summary: pd.DataFrame
    cost_bps_per_turnover: float

    def __post_init__(self) -> None:
        if self.variant not in EXPECTED_VARIANTS:
            raise ValueError(f"variant must be one of {EXPECTED_VARIANTS}, got {self.variant!r}.")
        if not self.windows:
            raise ValueError(f"variant {self.variant!r} produced zero windows.")
        if not isinstance(self.signal, pd.Series):
            raise TypeError("signal must be a pd.Series.")
        if not isinstance(self.pre_cost_returns, pd.Series):
            raise TypeError("pre_cost_returns must be a pd.Series.")
        if not isinstance(self.post_cost_returns, pd.Series):
            raise TypeError("post_cost_returns must be a pd.Series.")
        if not isinstance(self.summary, pd.DataFrame):
            raise TypeError("summary must be a pd.DataFrame.")
        if not self.pre_cost_returns.index.equals(self.post_cost_returns.index):
            raise ValueError("pre_cost_returns and post_cost_returns must share the same index.")
        if self.cost_bps_per_turnover < 0.0 or not np.isfinite(self.cost_bps_per_turnover):
            raise ValueError(
                "cost_bps_per_turnover must be a finite non-negative float, "
                f"got {self.cost_bps_per_turnover!r}."
            )


@dataclass(frozen=True)
class SideInfoComparisonResult:
    """Bundle of variant results from a single comparison run."""

    config: SideInfoComparisonConfig
    comparison_id: str
    variants: dict[str, SideInfoVariantResult]

    def __post_init__(self) -> None:
        missing = [v for v in EXPECTED_VARIANTS if v not in self.variants]
        if missing:
            raise ValueError(f"variants is missing {missing!r}; got {list(self.variants)!r}.")


@dataclass(frozen=True)
class SideInfoComparisonArtifacts:
    """Handle returned by :func:`run_side_info_comparison`."""

    comparison_id: str
    directory: Path
    config: SideInfoComparisonConfig
    result: SideInfoComparisonResult


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


_CHECKPOINT_BASELINE_WF: Final[str] = "baseline_wf"
_T = TypeVar("_T")


def run_side_info_comparison(
    config: SideInfoComparisonConfig,
    *,
    runs_root: Path | str = Path("runs"),
    force: bool = False,
    checkpoint_dir: Path | str | None = None,
) -> SideInfoComparisonArtifacts:
    """Execute the comparison described by ``config`` and write artifacts.

    Target directory is ``runs_root / comparison_id(config)``; ``force=True``
    replaces an existing directory atomically (with rollback on failure).

    When ``checkpoint_dir`` is provided, each stage (baseline walk-forward plus
    each variant) is pickled to that directory on completion. On a
    subsequent invocation with the same ``checkpoint_dir``, already-completed
    stages are loaded from disk instead of recomputed, so a crash mid-run only
    costs the in-progress variant. The directory is removed once the final
    artifact write succeeds.

    References: §4 side-information IOHMM comparison (evaluation layer)
    """
    if not isinstance(config, SideInfoComparisonConfig):
        raise TypeError(f"config must be a SideInfoComparisonConfig, got {type(config).__name__}.")
    runs_root_path = Path(runs_root)
    cmp_id = comparison_id(config)
    run_dir = runs_root_path / cmp_id

    if run_dir.exists() and not force:
        raise FileExistsError(
            f"Comparison directory already exists at {run_dir}; pass force=True to overwrite."
        )

    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    reproducible = validate_data_reproducibility(config, stacklevel=3)
    returns = load_returns_from_source(config.data, frequency=config.frequency)

    baseline_wf = _checkpointed_stage(
        _CHECKPOINT_BASELINE_WF,
        checkpoint_path,
        expected_type=WalkForwardResult,
        factory=lambda: walk_forward(
            returns,
            config.walk_forward,
            cost_bps_per_turnover=config.cost_bps_per_turnover,
            signal_policy=config.signal_policy,
            signal_threshold=config.signal_threshold,
        ),
    )
    baseline_variant = _checkpointed_stage(
        BASELINE_VARIANT,
        checkpoint_path,
        expected_type=SideInfoVariantResult,
        factory=lambda: _build_baseline_variant(baseline_wf, config.cost_bps_per_turnover),
    )
    vol_variant = _checkpointed_stage(
        VOLATILITY_RATIO_VARIANT,
        checkpoint_path,
        expected_type=SideInfoVariantResult,
        factory=lambda: _run_side_info_variant(
            VOLATILITY_RATIO_VARIANT,
            returns=returns,
            config=config,
            baseline_windows=baseline_wf.windows,
        ),
    )
    seasonality_variant = _checkpointed_stage(
        SEASONALITY_VARIANT,
        checkpoint_path,
        expected_type=SideInfoVariantResult,
        factory=lambda: _run_side_info_variant(
            SEASONALITY_VARIANT,
            returns=returns,
            config=config,
            baseline_windows=baseline_wf.windows,
        ),
    )
    vol_hmc_variant = _checkpointed_stage(
        VOLATILITY_RATIO_HMC_CONTINUOUS_VARIANT,
        checkpoint_path,
        expected_type=SideInfoVariantResult,
        factory=lambda: _run_side_info_variant(
            VOLATILITY_RATIO_HMC_CONTINUOUS_VARIANT,
            returns=returns,
            config=config,
            baseline_windows=baseline_wf.windows,
        ),
    )
    seasonality_hmc_variant = _checkpointed_stage(
        SEASONALITY_HMC_CONTINUOUS_VARIANT,
        checkpoint_path,
        expected_type=SideInfoVariantResult,
        factory=lambda: _run_side_info_variant(
            SEASONALITY_HMC_CONTINUOUS_VARIANT,
            returns=returns,
            config=config,
            baseline_windows=baseline_wf.windows,
        ),
    )
    combined_hmc_variant = _checkpointed_stage(
        VOL_RATIO_SEASONALITY_HMC_CONTINUOUS_VARIANT,
        checkpoint_path,
        expected_type=SideInfoVariantResult,
        factory=lambda: _run_side_info_variant(
            VOL_RATIO_SEASONALITY_HMC_CONTINUOUS_VARIANT,
            returns=returns,
            config=config,
            baseline_windows=baseline_wf.windows,
        ),
    )
    dmm_variant = _checkpointed_stage(
        DMM_VARIANT,
        checkpoint_path,
        expected_type=SideInfoVariantResult,
        factory=lambda: _run_dmm_variant(
            returns=returns,
            config=config,
            baseline_windows=baseline_wf.windows,
        ),
    )
    default_hmm_variant = _checkpointed_stage(
        DEFAULT_HMM_VARIANT,
        checkpoint_path,
        expected_type=SideInfoVariantResult,
        factory=lambda: _run_default_hmm_variant(
            returns=returns,
            config=config,
            baseline_windows=baseline_wf.windows,
        ),
    )

    result = SideInfoComparisonResult(
        config=config,
        comparison_id=cmp_id,
        variants={
            BASELINE_VARIANT: baseline_variant,
            VOLATILITY_RATIO_VARIANT: vol_variant,
            SEASONALITY_VARIANT: seasonality_variant,
            VOLATILITY_RATIO_HMC_CONTINUOUS_VARIANT: vol_hmc_variant,
            SEASONALITY_HMC_CONTINUOUS_VARIANT: seasonality_hmc_variant,
            VOL_RATIO_SEASONALITY_HMC_CONTINUOUS_VARIANT: combined_hmc_variant,
            DMM_VARIANT: dmm_variant,
            DEFAULT_HMM_VARIANT: default_hmm_variant,
        },
    )

    runs_root_path.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f"{cmp_id}.tmp-", dir=runs_root_path))
    backup_dir: Path | None = None
    try:
        _write_artifacts(staging_dir, config, cmp_id, result, reproducible=reproducible)
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

    if checkpoint_path is not None and checkpoint_path.exists():
        shutil.rmtree(checkpoint_path, ignore_errors=True)

    return SideInfoComparisonArtifacts(
        comparison_id=cmp_id,
        directory=run_dir,
        config=config,
        result=result,
    )


def _checkpointed_stage(
    name: str,
    checkpoint_dir: Path | None,
    *,
    expected_type: type[_T],
    factory: Callable[[], _T],
) -> _T:
    """Load ``name`` from ``checkpoint_dir`` if present; otherwise compute and save it.

    When ``checkpoint_dir`` is ``None`` this is a transparent passthrough to
    ``factory()`` — the existing behavior of ``run_side_info_comparison`` is
    preserved bit-for-bit.

    The on-disk format is a plain pickle file at
    ``{checkpoint_dir}/{name}.pkl``. Writes are atomic (write to ``.tmp`` then
    ``os.replace``) so a crash mid-write does not leave a partial file. Reads
    that fail (unpicklable, wrong type, truncated) are treated as a missing
    checkpoint: the file is removed and the stage is recomputed.
    """
    if checkpoint_dir is None:
        return factory()
    target = checkpoint_dir / f"{name}.pkl"
    if target.is_file():
        try:
            with target.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            print(
                f"[checkpoint] {name}: failed to load ({exc!r}); recomputing",
                file=sys.stderr,
                flush=True,
            )
            target.unlink(missing_ok=True)
        else:
            if isinstance(payload, expected_type):
                print(
                    f"[checkpoint] {name}: loaded from {target}",
                    file=sys.stderr,
                    flush=True,
                )
                return payload
            print(
                f"[checkpoint] {name}: pickle at {target} is "
                f"{type(payload).__name__}, expected {expected_type.__name__}; "
                "recomputing",
                file=sys.stderr,
                flush=True,
            )
            target.unlink(missing_ok=True)

    print(f"[checkpoint] {name}: computing", file=sys.stderr, flush=True)
    value = factory()
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with tmp.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    print(f"[checkpoint] {name}: saved to {target}", file=sys.stderr, flush=True)
    return value


# ---------------------------------------------------------------------------
# Variant builders
# ---------------------------------------------------------------------------


def _build_baseline_variant(
    baseline_wf: WalkForwardResult,
    cost_bps_per_turnover: float,
) -> SideInfoVariantResult:
    """Wrap a baseline ``WalkForwardResult`` as a ``SideInfoVariantResult``.

    The baseline summary is taken verbatim from ``walk_forward`` so it stays
    bit-for-bit identical to a direct invocation.
    """
    return SideInfoVariantResult(
        variant=BASELINE_VARIANT,
        chosen_k_per_window=tuple(int(w.chosen_k) for w in baseline_wf.windows),
        windows=baseline_wf.windows,
        signal=baseline_wf.signal,
        pre_cost_returns=baseline_wf.pre_cost_returns,
        post_cost_returns=baseline_wf.post_cost_returns,
        summary=baseline_wf.summary,
        cost_bps_per_turnover=float(cost_bps_per_turnover),
    )


def _run_side_info_variant(
    variant: str,
    *,
    returns: pd.Series,
    config: SideInfoComparisonConfig,
    baseline_windows: tuple[WalkForwardWindow, ...],
) -> SideInfoVariantResult:
    """Run one side-information variant on the same windows as the baseline."""
    if variant not in _SIDE_INFO_VARIANTS:
        raise ValueError(f"unsupported variant {variant!r}.")

    cost = config.cost_bps_per_turnover
    wf = config.walk_forward
    bar_dates = returns.index.date

    windows: list[SideInfoVariantWindow | ContinuousSideInfoVariantWindow] = []
    signal_parts: list[pd.Series] = []
    pre_cost_parts: list[pd.Series] = []
    post_cost_parts: list[pd.Series] = []
    chosen_ks: list[int] = []
    initial_position = 0
    total_windows = len(baseline_windows)
    is_hmc_variant = variant in _HMC_CONTINUOUS_VARIANTS

    for window_position, baseline_window in enumerate(baseline_windows, start=1):
        window_start_time = time.monotonic()
        train_mask = (bar_dates >= baseline_window.train_start.date()) & (
            bar_dates <= baseline_window.train_end.date()
        )
        train_slice = returns.loc[train_mask]
        effective_forecast = returns.loc[
            baseline_window.forecast_start : baseline_window.forecast_end
        ]
        if effective_forecast.shape[0] < 2:
            raise ValueError(
                "each effective forecast window must contain at least 2 observations; "
                f"window {baseline_window.index} has {effective_forecast.shape[0]}."
            )

        chosen_k = int(baseline_window.chosen_k)
        wrapper = GaussianHMMWrapper(
            n_states=chosen_k,
            random_state=wf.random_state,
            n_iter=wf.n_iter,
            tol=wf.tol,
            min_variance=wf.min_variance,
            variance_floor_policy=wf.variance_floor_policy,
        )
        fitted = wrapper.fit(train_slice)
        decoded = wrapper.predict(train_slice)

        feature_full = _build_feature(variant, pd.concat([train_slice, effective_forecast]), config)
        feature_train = feature_full.reindex(train_slice.index)
        feature_forecast = feature_full.reindex(effective_forecast.index)

        forecast_has_nan = (
            bool(feature_forecast.isna().any().any())
            if isinstance(feature_forecast, pd.DataFrame)
            else bool(feature_forecast.isna().any())
        )
        if forecast_has_nan:
            raise ValueError(
                f"forecast feature for variant {variant!r} contains NaN in window "
                f"{baseline_window.index}; increase h_days so the rolling window is "
                "fully initialized before the forecast slice."
            )

        feature_train_clean = feature_train.dropna()
        if feature_train_clean.shape[0] < 2:
            raise ValueError(
                f"variant {variant!r} window {baseline_window.index} has fewer than 2 "
                "finite training feature observations after dropping the NaN prefix."
            )
        nonnan_mask = (
            feature_train.notna().all(axis=1).to_numpy()
            if isinstance(feature_train, pd.DataFrame)
            else feature_train.notna().to_numpy()
        )
        decoded_aligned = decoded[nonnan_mask]
        posterior_at_train_end = _terminal_training_posterior(train_slice, fitted)
        x_train_end = np.asarray(feature_train_clean.iloc[-1], dtype=float)
        if x_train_end.ndim == 0:
            x_train_end = x_train_end.reshape(1)

        bucketed: BucketedTransitionResult | None = None
        continuous: ContinuousIOHMMResult | None = None
        boundary_mode: BoundaryMode | None = None
        if variant in _HMC_CONTINUOUS_VARIANTS:
            continuous = fit_continuous_iohmm(
                state_sequence=decoded_aligned,
                side_information=feature_train_clean.to_numpy(),
                n_states=chosen_k,
                config=config.continuous_iohmm,
            )
            train_end_transition = transition_probabilities_at(
                result=continuous,
                x_values=x_train_end.reshape(1, -1),
            )[0]
            expected = _dynamic_forward_expected_returns_continuous(
                forecast_returns=effective_forecast.to_numpy(),
                forecast_features=feature_forecast.to_numpy(),
                fitted=fitted,
                continuous=continuous,
                initial_state_distribution=posterior_at_train_end,
                initial_transition_matrix=train_end_transition,
            )
        else:
            spline_result = fit_spline_predictor(feature_train, train_slice, config=config.spline)
            boundaries, boundary_mode = _bucket_boundaries_for_mode(
                spline_result=spline_result,
                training_values=feature_train_clean.to_numpy(),
                config=config.bucketed_transition,
            )
            bucketed_config = config.bucketed_transition
            if boundary_mode != bucketed_config.boundary_mode:
                bucketed_config = replace(bucketed_config, boundary_mode=boundary_mode)

            bucketed = fit_bucketed_transition_model(
                state_sequence=decoded_aligned,
                side_information=feature_train_clean.to_numpy(),
                n_states=chosen_k,
                baseline_transition_matrix=fitted.transition_matrix,
                bucket_boundaries=boundaries,
                config=bucketed_config,
            )
            train_end_transition = bucketed.transition_matrix_for(float(x_train_end[0]))
            expected = _dynamic_forward_expected_returns(
                forecast_returns=effective_forecast.to_numpy(),
                forecast_features=feature_forecast.to_numpy(),
                fitted=fitted,
                bucketed=bucketed,
                initial_state_distribution=posterior_at_train_end,
                initial_transition_matrix=train_end_transition,
            )
        expected_series = pd.Series(expected, index=effective_forecast.index)

        signal_scale: float | None = None
        if config.signal_policy == "conviction_weighted":
            if continuous is not None:
                train_expected = _dynamic_forward_expected_returns_continuous(
                    forecast_returns=train_slice.loc[feature_train_clean.index].to_numpy(),
                    forecast_features=feature_train_clean.to_numpy(),
                    fitted=fitted,
                    continuous=continuous,
                    initial_state_distribution=fitted.initial_distribution,
                )
            else:
                assert bucketed is not None
                train_expected = _dynamic_forward_expected_returns(
                    forecast_returns=train_slice.loc[feature_train_clean.index].to_numpy(),
                    forecast_features=feature_train_clean.to_numpy(),
                    fitted=fitted,
                    bucketed=bucketed,
                    initial_state_distribution=fitted.initial_distribution,
                )
            signal_scale = conviction_scale_from_train_predictions(train_expected)

        window_signal = build_signal(
            expected_series,
            policy=config.signal_policy,
            threshold=config.signal_threshold,
            initial_position=initial_position,
            scale=signal_scale,
        )
        if config.signal_policy == "thresholded_hold":
            initial_position = int(window_signal.iloc[-1])
        window_pre_cost = align_signal_with_future_return(window_signal, effective_forecast)
        window_post_cost = apply_turnover_cost(
            window_pre_cost,
            signal_turnover(window_signal),
            cost_bps_per_turnover=cost,
        )
        window_summary = _summarize_return_modes(
            window_pre_cost, window_post_cost, cost_bps_per_turnover=cost
        )

        signal_parts.append(window_signal)
        pre_cost_parts.append(window_pre_cost)
        post_cost_parts.append(window_post_cost)
        chosen_ks.append(chosen_k)
        if continuous is not None:
            windows.append(
                ContinuousSideInfoVariantWindow(
                    index=int(baseline_window.index),
                    train_start=train_slice.index.min(),
                    train_end=train_slice.index.max(),
                    forecast_start=effective_forecast.index.min(),
                    forecast_end=effective_forecast.index.max(),
                    chosen_k=chosen_k,
                    n_train_obs=int(train_slice.shape[0]),
                    n_forecast_obs=int(effective_forecast.shape[0]),
                    converged=continuous.converged,
                    rhat_max=_max_parameter_metric(continuous.rhat),
                    ess_bulk_min=_min_parameter_metric(continuous.ess_bulk),
                    posterior_mean_W=continuous.posterior_mean_W,
                    posterior_mean_b=continuous.posterior_mean_b,
                    posterior_samples=continuous.posterior_samples,
                    summary=window_summary,
                )
            )
        else:
            assert bucketed is not None
            assert boundary_mode is not None
            windows.append(
                SideInfoVariantWindow(
                    index=int(baseline_window.index),
                    train_start=train_slice.index.min(),
                    train_end=train_slice.index.max(),
                    forecast_start=effective_forecast.index.min(),
                    forecast_end=effective_forecast.index.max(),
                    chosen_k=chosen_k,
                    n_train_obs=int(train_slice.shape[0]),
                    n_forecast_obs=int(effective_forecast.shape[0]),
                    bucket_observation_counts=tuple(
                        int(c) for c in bucketed.bucket_observation_counts
                    ),
                    boundary_mode=boundary_mode,
                    summary=window_summary,
                )
            )

        elapsed = time.monotonic() - window_start_time
        if is_hmc_variant and continuous is not None:
            print(
                f"[{variant}] window {window_position}/{total_windows} "
                f"index={int(baseline_window.index)} done in {elapsed:.1f}s "
                f"converged={continuous.converged} "
                f"rhat_max={_max_parameter_metric(continuous.rhat):.3f} "
                f"ess_bulk_min={_min_parameter_metric(continuous.ess_bulk)}",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[{variant}] window {window_position}/{total_windows} "
                f"index={int(baseline_window.index)} done in {elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    combined_signal = pd.concat(signal_parts)
    if config.signal_policy in _DISCRETE_SIGNAL_POLICIES:
        combined_signal = combined_signal.astype(np.int8)
    combined_signal.name = "signal"
    pre_cost_returns = pd.concat(pre_cost_parts)
    post_cost_returns = pd.concat(post_cost_parts)
    summary = _summarize_return_modes(
        pre_cost_returns, post_cost_returns, cost_bps_per_turnover=cost
    )

    return SideInfoVariantResult(
        variant=variant,
        chosen_k_per_window=tuple(chosen_ks),
        windows=tuple(windows),
        signal=combined_signal,
        pre_cost_returns=pre_cost_returns,
        post_cost_returns=post_cost_returns,
        summary=summary,
        cost_bps_per_turnover=float(cost),
    )


def _run_default_hmm_variant(
    *,
    returns: pd.Series,
    config: SideInfoComparisonConfig,
    baseline_windows: tuple[WalkForwardWindow, ...],
) -> SideInfoVariantResult:
    """Run the Default HMM variant on the same windows as the baseline.

    Emission means/variances come from PLR segment statistics on the training
    window; A = (1/K)·ones((K,K)) and π = ones(K)/K are uniform throughout.
    The fixed-A forward filter (``hft_hmm.inference.forward_filter``) produces
    per-bar expected returns, which are passed to the shared signal builder.

    References: Figure 8 / §3 Default HMM (paper-faithful)
    """
    cost = config.cost_bps_per_turnover
    wf = config.walk_forward
    bar_dates = returns.index.date

    windows: list[WalkForwardWindow] = []
    signal_parts: list[pd.Series] = []
    pre_cost_parts: list[pd.Series] = []
    post_cost_parts: list[pd.Series] = []
    chosen_ks: list[int] = []
    initial_position = 0
    total_windows = len(baseline_windows)

    for window_position, baseline_window in enumerate(baseline_windows, start=1):
        window_start_time = time.monotonic()
        train_mask = (bar_dates >= baseline_window.train_start.date()) & (
            bar_dates <= baseline_window.train_end.date()
        )
        train_slice = returns.loc[train_mask]
        effective_forecast = returns.loc[
            baseline_window.forecast_start : baseline_window.forecast_end
        ]
        if effective_forecast.shape[0] < 2:
            raise ValueError(
                "each effective forecast window must contain at least 2 observations; "
                f"window {baseline_window.index} has {effective_forecast.shape[0]}."
            )

        chosen_k = int(baseline_window.chosen_k)
        fitted = fit_default_hmm(
            train_slice,
            n_states=chosen_k,
            min_variance=wf.min_variance,
        )

        filter_result = forward_filter(effective_forecast, fitted)
        expected_series = pd.Series(
            filter_result.expected_next_returns, index=effective_forecast.index
        )

        signal_scale: float | None = None
        if config.signal_policy == "conviction_weighted":
            train_filter = forward_filter(train_slice, fitted)
            signal_scale = conviction_scale_from_train_predictions(
                train_filter.expected_next_returns
            )

        window_signal = build_signal(
            expected_series,
            policy=config.signal_policy,
            threshold=config.signal_threshold,
            initial_position=initial_position,
            scale=signal_scale,
        )
        if config.signal_policy == "thresholded_hold":
            initial_position = int(window_signal.iloc[-1])
        window_pre_cost = align_signal_with_future_return(window_signal, effective_forecast)
        window_post_cost = apply_turnover_cost(
            window_pre_cost,
            signal_turnover(window_signal),
            cost_bps_per_turnover=cost,
        )
        window_summary = _summarize_return_modes(
            window_pre_cost, window_post_cost, cost_bps_per_turnover=cost
        )

        signal_parts.append(window_signal)
        pre_cost_parts.append(window_pre_cost)
        post_cost_parts.append(window_post_cost)
        chosen_ks.append(chosen_k)
        windows.append(
            WalkForwardWindow(
                index=int(baseline_window.index),
                train_start=train_slice.index.min(),
                train_end=train_slice.index.max(),
                forecast_start=effective_forecast.index.min(),
                forecast_end=effective_forecast.index.max(),
                chosen_k=chosen_k,
                log_likelihood=fitted.log_likelihood,
                n_train_obs=int(train_slice.shape[0]),
                n_forecast_obs=int(effective_forecast.shape[0]),
                summary=window_summary,
            )
        )

        elapsed = time.monotonic() - window_start_time
        print(
            f"[{DEFAULT_HMM_VARIANT}] window {window_position}/{total_windows} "
            f"index={int(baseline_window.index)} done in {elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    combined_signal = pd.concat(signal_parts)
    if config.signal_policy in _DISCRETE_SIGNAL_POLICIES:
        combined_signal = combined_signal.astype(np.int8)
    combined_signal.name = "signal"
    pre_cost_returns = pd.concat(pre_cost_parts)
    post_cost_returns = pd.concat(post_cost_parts)
    summary = _summarize_return_modes(
        pre_cost_returns, post_cost_returns, cost_bps_per_turnover=cost
    )

    return SideInfoVariantResult(
        variant=DEFAULT_HMM_VARIANT,
        chosen_k_per_window=tuple(chosen_ks),
        windows=tuple(windows),
        signal=combined_signal,
        pre_cost_returns=pre_cost_returns,
        post_cost_returns=post_cost_returns,
        summary=summary,
        cost_bps_per_turnover=float(cost),
    )


def _run_dmm_variant(
    *,
    returns: pd.Series,
    config: SideInfoComparisonConfig,
    baseline_windows: tuple[WalkForwardWindow, ...],
) -> SideInfoVariantResult:
    """Run the DMM benchmark on the same windows as the baseline."""
    if config.dmm.obs_dim != 1:
        raise ValueError(f"config.dmm.obs_dim must be 1, got {config.dmm.obs_dim}.")

    cost = config.cost_bps_per_turnover
    bar_dates = returns.index.date

    windows: list[DmmVariantWindow] = []
    signal_parts: list[pd.Series] = []
    pre_cost_parts: list[pd.Series] = []
    post_cost_parts: list[pd.Series] = []
    chosen_ks: list[int] = []
    initial_position = 0
    total_windows = len(baseline_windows)

    for window_position, baseline_window in enumerate(baseline_windows, start=1):
        window_start_time = time.monotonic()
        train_mask = (bar_dates >= baseline_window.train_start.date()) & (
            bar_dates <= baseline_window.train_end.date()
        )
        train_slice = returns.loc[train_mask]
        effective_forecast = returns.loc[
            baseline_window.forecast_start : baseline_window.forecast_end
        ]
        if effective_forecast.shape[0] < 2:
            raise ValueError(
                "each effective forecast window must contain at least 2 observations; "
                f"window {baseline_window.index} has {effective_forecast.shape[0]}."
            )

        feature_full = _build_feature(
            VOLATILITY_RATIO_VARIANT,
            pd.concat([train_slice, effective_forecast]),
            config,
        )
        feature_train = feature_full.reindex(train_slice.index)
        feature_forecast = feature_full.reindex(effective_forecast.index)

        forecast_has_nan = (
            bool(feature_forecast.isna().any().any())
            if isinstance(feature_forecast, pd.DataFrame)
            else bool(feature_forecast.isna().any())
        )
        if forecast_has_nan:
            raise ValueError(
                f"forecast feature for variant {DMM_VARIANT!r} contains NaN in window "
                f"{baseline_window.index}; increase h_days so the rolling window is "
                "fully initialized before the forecast slice."
            )

        feature_train_clean = feature_train.dropna()
        if feature_train_clean.shape[0] < 2:
            raise ValueError(
                f"variant {DMM_VARIANT!r} window {baseline_window.index} has fewer than 2 "
                "finite training feature observations after dropping the NaN prefix."
            )
        train_returns_clean = train_slice.loc[feature_train_clean.index]
        standardized_train_feature, standardized_forecast_feature = _standardize_feature_train_only(
            feature_train_clean,
            feature_forecast,
        )

        side_info_dim = _feature_side_info_dim(standardized_train_feature)
        dmm_config = replace(config.dmm, side_info_dim=side_info_dim)
        observations = torch.as_tensor(
            train_returns_clean.to_numpy(dtype=np.float32).reshape(1, -1, 1),
            dtype=torch.float32,
        )
        side_info = torch.as_tensor(
            _feature_to_2d_numpy(standardized_train_feature).astype(np.float32, copy=False)[
                None, :, :
            ],
            dtype=torch.float32,
        )
        seq_lengths = torch.tensor([train_returns_clean.shape[0]], dtype=torch.long)
        fitted = fit_dmm(dmm_config, observations, side_info, seq_lengths)
        expected_series = expected_next_returns_from_dmm(
            fitted.model,
            effective_forecast,
            standardized_forecast_feature,
        )

        signal_scale: float | None = None
        if config.signal_policy == "conviction_weighted":
            train_expected = expected_next_returns_from_dmm(
                fitted.model,
                train_returns_clean,
                standardized_train_feature,
            )
            signal_scale = conviction_scale_from_train_predictions(train_expected.to_numpy())

        window_signal = build_signal(
            expected_series,
            policy=config.signal_policy,
            threshold=config.signal_threshold,
            initial_position=initial_position,
            scale=signal_scale,
        )
        if config.signal_policy == "thresholded_hold":
            initial_position = int(window_signal.iloc[-1])
        window_pre_cost = align_signal_with_future_return(window_signal, effective_forecast)
        window_post_cost = apply_turnover_cost(
            window_pre_cost,
            signal_turnover(window_signal),
            cost_bps_per_turnover=cost,
        )
        window_summary = _summarize_return_modes(
            window_pre_cost, window_post_cost, cost_bps_per_turnover=cost
        )

        signal_parts.append(window_signal)
        pre_cost_parts.append(window_pre_cost)
        post_cost_parts.append(window_post_cost)
        chosen_ks.append(int(baseline_window.chosen_k))
        windows.append(
            DmmVariantWindow(
                index=int(baseline_window.index),
                train_start=train_slice.index.min(),
                train_end=train_slice.index.max(),
                forecast_start=effective_forecast.index.min(),
                forecast_end=effective_forecast.index.max(),
                chosen_k=int(baseline_window.chosen_k),
                n_train_obs=int(train_slice.shape[0]),
                n_forecast_obs=int(effective_forecast.shape[0]),
                feature_identity=_DMM_FEATURE_IDENTITY,
                side_info_dim=side_info_dim,
                num_epochs=int(dmm_config.num_epochs),
                mini_batch_size=int(dmm_config.mini_batch_size),
                learning_rate=float(dmm_config.learning_rate),
                beta1=float(dmm_config.beta1),
                beta2=float(dmm_config.beta2),
                clip_norm=float(dmm_config.clip_norm),
                lr_decay=float(dmm_config.lr_decay),
                weight_decay=float(dmm_config.weight_decay),
                min_annealing_factor=float(dmm_config.min_annealing_factor),
                annealing_epochs=int(dmm_config.annealing_epochs),
                seed=int(dmm_config.seed),
                final_loss=float(fitted.loss_history[-1]),
                summary=window_summary,
            )
        )

        elapsed = time.monotonic() - window_start_time
        print(
            f"[{DMM_VARIANT}] window {window_position}/{total_windows} "
            f"index={int(baseline_window.index)} done in {elapsed:.1f}s "
            f"final_loss={float(fitted.loss_history[-1]):.6f}",
            file=sys.stderr,
            flush=True,
        )

    combined_signal = pd.concat(signal_parts)
    if config.signal_policy in _DISCRETE_SIGNAL_POLICIES:
        combined_signal = combined_signal.astype(np.int8)
    combined_signal.name = "signal"
    pre_cost_returns = pd.concat(pre_cost_parts)
    post_cost_returns = pd.concat(post_cost_parts)
    summary = _summarize_return_modes(
        pre_cost_returns, post_cost_returns, cost_bps_per_turnover=cost
    )

    return SideInfoVariantResult(
        variant=DMM_VARIANT,
        chosen_k_per_window=tuple(chosen_ks),
        windows=tuple(windows),
        signal=combined_signal,
        pre_cost_returns=pre_cost_returns,
        post_cost_returns=post_cost_returns,
        summary=summary,
        cost_bps_per_turnover=float(cost),
    )


def _bucket_boundaries_for_mode(
    *,
    spline_result: SplinePredictorResult,
    training_values: np.ndarray,
    config: BucketedTransitionConfig,
) -> tuple[np.ndarray, BoundaryMode]:
    """Resolve bucket boundaries and return the effective boundary mode."""
    if config.boundary_mode == "grid":
        return bucket_boundaries_from_spline_grid(spline_result, config=config), "grid"
    if config.boundary_mode == "quantile":
        try:
            return (
                bucket_boundaries_from_quantiles(
                    np.asarray(training_values, dtype=float), config=config
                ),
                "quantile",
            )
        except ValueError:
            return bucket_boundaries_from_spline_grid(spline_result, config=config), "grid"
    raise ValueError(f"unsupported boundary_mode {config.boundary_mode!r}.")


def _build_feature(
    variant: str,
    series: pd.Series,
    config: SideInfoComparisonConfig,
) -> pd.Series | pd.DataFrame:
    if variant in (VOLATILITY_RATIO_VARIANT, VOLATILITY_RATIO_HMC_CONTINUOUS_VARIANT):
        return volatility_ratio(
            series,
            fast_window=config.vol_ratio.fast_window,
            slow_window=config.vol_ratio.slow_window,
            decay=config.vol_ratio.decay,
        )
    if variant in (SEASONALITY_VARIANT, SEASONALITY_HMC_CONTINUOUS_VARIANT):
        return intraday_seasonality(series, config=config.seasonality)
    if variant == VOL_RATIO_SEASONALITY_HMC_CONTINUOUS_VARIANT:
        vol_ratio_feature = volatility_ratio(
            series,
            fast_window=config.vol_ratio.fast_window,
            slow_window=config.vol_ratio.slow_window,
            decay=config.vol_ratio.decay,
        )
        seasonality_feature = intraday_seasonality(series, config=config.seasonality)
        return pd.concat(
            [vol_ratio_feature, seasonality_feature],
            axis=1,
            keys=["vol_ratio", "seasonality"],
        ).dropna()
    raise ValueError(f"unsupported variant {variant!r}.")


def _standardize_feature_train_only(
    train_feature: pd.Series | pd.DataFrame,
    forecast_feature: pd.Series | pd.DataFrame,
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    """Z-score side information using train-slice moments only."""
    if isinstance(train_feature, pd.DataFrame):
        mean = train_feature.mean(axis=0)
        std = train_feature.std(axis=0, ddof=0)
        std = std.where(std > _MIN_STANDARDIZATION_STD, 1.0)
        return (
            (train_feature - mean) / std,
            (forecast_feature - mean) / std,
        )

    mean = float(train_feature.mean())
    std = float(train_feature.std(ddof=0))
    if std <= _MIN_STANDARDIZATION_STD:
        std = 1.0
    return (
        (train_feature - mean) / std,
        (forecast_feature - mean) / std,
    )


def _feature_to_2d_numpy(feature: pd.Series | pd.DataFrame) -> np.ndarray:
    values = np.asarray(feature.to_numpy(dtype=float), dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"feature must coerce to a 2-D array, got shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("feature values must contain only finite numbers.")
    return values


def _feature_side_info_dim(feature: pd.Series | pd.DataFrame) -> int:
    return int(_feature_to_2d_numpy(feature).shape[1])


def _dynamic_forward_expected_returns(
    *,
    forecast_returns: np.ndarray,
    forecast_features: np.ndarray,
    fitted: GaussianHMMResult,
    bucketed: BucketedTransitionResult,
    initial_state_distribution: np.ndarray,
    initial_transition_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Run the side-info-conditioned forward filter and return E[r_{t+1}|info_t].

    Filtering uses ``A(x_{t-1})`` to advance the predictive distribution from
    ``t-1`` to ``t``; the one-step prediction at ``t`` then uses ``A(x_t)``,
    matching the IOHMM convention where the side-information observed at time
    ``t`` parameterises the next transition. ``initial_state_distribution`` is
    the terminal filtering distribution from the training window, so each
    forecast window carries training history instead of restarting from the
    fitted unconditional initial distribution. When the caller passes that
    terminal distribution it must also supply ``initial_transition_matrix =
    A(x_train_end)`` so the prior used for the first forecast emission
    accounts for the jump out of the training window.
    """
    forecast_features = np.asarray(forecast_features, dtype=float)
    if forecast_features.shape != np.asarray(forecast_returns).shape:
        raise ValueError(
            "forecast_features must have the same shape as forecast_returns; "
            f"got {forecast_features.shape} vs {np.asarray(forecast_returns).shape}."
        )
    if not np.all(np.isfinite(forecast_features)):
        raise ValueError("forecast_features must contain only finite values.")
    transition_matrices = np.stack(
        [bucketed.transition_matrix_for(float(x)) for x in forecast_features],
        axis=0,
    )
    return _dynamic_forward_expected_returns_from_transition_matrices(
        forecast_returns=forecast_returns,
        fitted=fitted,
        transition_matrices=transition_matrices,
        initial_state_distribution=initial_state_distribution,
        initial_transition_matrix=initial_transition_matrix,
    )


def _dynamic_forward_expected_returns_continuous(
    *,
    forecast_returns: np.ndarray,
    forecast_features: np.ndarray,
    fitted: GaussianHMMResult,
    continuous: ContinuousIOHMMResult,
    initial_state_distribution: np.ndarray,
    initial_transition_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Run the continuous-IOHMM forward filter and return E[r_{t+1}|info_t]."""
    transition_matrices = transition_probabilities_at(
        result=continuous,
        x_values=forecast_features,
    )
    return _dynamic_forward_expected_returns_from_transition_matrices(
        forecast_returns=forecast_returns,
        fitted=fitted,
        transition_matrices=transition_matrices,
        initial_state_distribution=initial_state_distribution,
        initial_transition_matrix=initial_transition_matrix,
    )


def _dynamic_forward_expected_returns_from_transition_matrices(
    *,
    forecast_returns: np.ndarray,
    fitted: GaussianHMMResult,
    transition_matrices: np.ndarray,
    initial_state_distribution: np.ndarray,
    initial_transition_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Run a forward filter using one transition matrix per forecast timestamp.

    ``initial_transition_matrix`` advances ``initial_state_distribution`` by one
    step before the first emission is consumed. Callers that pass the terminal
    *training* filtering distribution as ``initial_state_distribution`` must
    supply ``A(x_train_end)`` here so the prior for ``forecast_returns[0]`` is
    ``P(z_train_end) @ A(x_train_end)`` rather than ``P(z_train_end)`` itself.
    """
    means = fitted.means
    variances = fitted.variances
    forecast_returns = np.asarray(forecast_returns, dtype=float)
    if forecast_returns.ndim != 1:
        raise ValueError(f"forecast_returns must be 1-D, got shape {forecast_returns.shape}.")
    if not np.all(np.isfinite(forecast_returns)):
        raise ValueError("forecast_returns must contain only finite values.")

    n_obs = forecast_returns.shape[0]
    k = means.shape[0]
    transition_matrices = np.asarray(transition_matrices, dtype=float)
    if transition_matrices.shape != (n_obs, k, k):
        raise ValueError(
            f"transition_matrices must have shape ({n_obs}, {k}, {k}), "
            f"got {transition_matrices.shape}."
        )
    if not np.all(np.isfinite(transition_matrices)):
        raise ValueError("transition_matrices must contain only finite values.")
    if not np.all(transition_matrices >= 0.0):
        raise ValueError("transition_matrices entries must be non-negative.")
    if not np.allclose(transition_matrices.sum(axis=2), 1.0):
        raise ValueError("transition_matrices rows must sum to 1.")
    initial_distribution = _coerce_state_distribution(
        initial_state_distribution,
        k=k,
        name="initial_state_distribution",
    )
    if initial_transition_matrix is not None:
        initial_transition = np.asarray(initial_transition_matrix, dtype=float)
        if initial_transition.shape != (k, k):
            raise ValueError(
                f"initial_transition_matrix must have shape ({k}, {k}), "
                f"got {initial_transition.shape}."
            )
        if not np.all(np.isfinite(initial_transition)):
            raise ValueError("initial_transition_matrix must contain only finite values.")
        if not np.all(initial_transition >= 0.0):
            raise ValueError("initial_transition_matrix entries must be non-negative.")
        if not np.allclose(initial_transition.sum(axis=1), 1.0):
            raise ValueError("initial_transition_matrix rows must sum to 1.")
        initial_distribution = initial_distribution @ initial_transition
    centered = forecast_returns[:, None] - means[None, :]
    log_emissions = -0.5 * (
        np.log(2.0 * np.pi * variances[None, :]) + (centered * centered) / variances[None, :]
    )

    log_filtering = np.empty((n_obs, k), dtype=float)
    with np.errstate(divide="ignore"):
        log_start = np.log(initial_distribution)
    log_alpha = log_start + log_emissions[0]
    log_filtering[0] = log_alpha - logsumexp(log_alpha)

    for t in range(1, n_obs):
        prev_matrix = transition_matrices[t - 1]
        with np.errstate(divide="ignore"):
            log_transition = np.log(prev_matrix)
        log_predicted = logsumexp(log_filtering[t - 1][:, None] + log_transition, axis=0)
        log_alpha = log_emissions[t] + log_predicted
        log_filtering[t] = log_alpha - logsumexp(log_alpha)

    filtering = np.exp(log_filtering)
    expected = np.empty(n_obs, dtype=float)
    for t in range(n_obs):
        next_matrix = transition_matrices[t]
        expected[t] = filtering[t] @ next_matrix @ means
    return expected


def _terminal_training_posterior(
    train_returns: pd.Series,
    fitted: GaussianHMMResult,
) -> np.ndarray:
    """Return P(state at train_end | training returns) for a fitted HMM."""
    filter_result = filter_from_result(fitted, train_returns)
    k = fitted.means.shape[0]
    return _coerce_state_distribution(
        filter_result.filtering_probabilities[-1],
        k=k,
        name="posterior_at_train_end",
    )


def _coerce_state_distribution(
    values: np.ndarray,
    *,
    k: int,
    name: str,
) -> np.ndarray:
    distribution = np.asarray(values, dtype=float)
    if distribution.shape != (k,):
        raise ValueError(f"{name} must have shape ({k},), got {distribution.shape}.")
    if not np.all(np.isfinite(distribution)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.all(distribution >= 0.0):
        raise ValueError(f"{name} entries must be non-negative.")
    if not np.isclose(distribution.sum(), 1.0):
        raise ValueError(f"{name} must sum to 1.")
    return distribution.copy()


def _max_parameter_metric(metrics: dict[str, np.ndarray]) -> float:
    values = np.concatenate([np.asarray(value, dtype=float).ravel() for value in metrics.values()])
    return float(values.max())


def _min_parameter_metric(metrics: dict[str, np.ndarray]) -> int:
    values = np.concatenate([np.asarray(value, dtype=float).ravel() for value in metrics.values()])
    return int(math.floor(float(values.min())))


# ---------------------------------------------------------------------------
# Summaries and artifacts
# ---------------------------------------------------------------------------


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


def _write_artifacts(
    run_dir: Path,
    config: SideInfoComparisonConfig,
    cmp_id: str,
    result: SideInfoComparisonResult,
    *,
    reproducible: bool,
) -> None:
    (run_dir / "figures").mkdir()
    (run_dir / "config.yaml").write_bytes(config.to_yaml_bytes())
    _write_summary(run_dir / "summary.json", config, cmp_id, result, reproducible=reproducible)
    for variant, variant_result in result.variants.items():
        _write_variant_log(run_dir / f"{variant}.log.jsonl", variant_result)
        _write_posterior_samples(run_dir, variant, variant_result)


def _write_summary(
    path: Path,
    config: SideInfoComparisonConfig,
    cmp_id: str,
    result: SideInfoComparisonResult,
    *,
    reproducible: bool,
) -> None:
    payload: dict[str, Any] = {
        "comparison_id": cmp_id,
        "reproducible": reproducible,
        "frequency": config.frequency,
        "cost_bps_per_turnover": float(config.cost_bps_per_turnover),
        "metric_interpretation": {
            "pre-cost": (
                "primary academic signal-quality metric for paper comparison; "
                "no execution costs applied"
            ),
            "post-cost": (
                "cost-sensitivity diagnostic under the repo's linear turnover "
                "cost convention; not calibrated to the paper's unspecified "
                "execution model"
            ),
        },
        "variants": {
            name: _variant_payload(name, cmp_id, result.variants[name])
            for name in EXPECTED_VARIANTS
        },
    }
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _variant_payload(
    variant: str,
    cmp_id: str,
    result: SideInfoVariantResult,
) -> dict[str, Any]:
    aligned_returns = result.pre_cost_returns
    diagnostics = turnover_diagnostics(result.signal)
    sample_window = {
        "start": aligned_returns.index.min().isoformat(),
        "end": aligned_returns.index.max().isoformat(),
    }
    return {
        "variant": variant,
        "comparison_id": cmp_id,
        "sample_window": sample_window,
        "chosen_k_per_window": [int(k) for k in result.chosen_k_per_window],
        "n_windows": len(result.windows),
        "n_forecast_obs": int(aligned_returns.shape[0]),
        "cost_bps_per_turnover": float(result.cost_bps_per_turnover),
        "summary": _summary_to_payload(result.summary),
        "daily_annualized_sharpe": {
            "trading_days_per_year": 258.0,
            "method": "sum intraday returns by UTC date, then scale daily Sharpe by sqrt(258)",
            "pre-cost": _json_safe(daily_annualized_sharpe_ratio(result.pre_cost_returns)),
            "post-cost": _json_safe(daily_annualized_sharpe_ratio(result.post_cost_returns)),
        },
        "turnover_diagnostics": {
            "total_turnover": _json_safe(diagnostics.total_turnover),
            "mean_turnover_per_period": _json_safe(diagnostics.mean_turnover_per_period),
            "position_change_count": int(diagnostics.position_change_count),
            "mean_holding_periods": _json_safe(diagnostics.mean_holding_periods),
            "cost_drag_cumulative_return": _json_safe(
                cumulative_return(result.pre_cost_returns)
                - cumulative_return(result.post_cost_returns)
            ),
        },
    }


def _write_variant_log(path: Path, result: SideInfoVariantResult) -> None:
    lines = []
    for w in result.windows:
        record: dict[str, Any] = {
            "index": int(w.index),
            "train_start": w.train_start.isoformat(),
            "train_end": w.train_end.isoformat(),
            "forecast_start": w.forecast_start.isoformat(),
            "forecast_end": w.forecast_end.isoformat(),
            "chosen_k": int(w.chosen_k),
            "n_train_obs": int(w.n_train_obs),
            "n_forecast_obs": int(w.n_forecast_obs),
            "summary": _summary_to_payload(w.summary),
        }
        if isinstance(w, SideInfoVariantWindow):
            record["bucket_observation_counts"] = list(w.bucket_observation_counts)
            record["boundary_mode"] = w.boundary_mode
        elif isinstance(w, ContinuousSideInfoVariantWindow):
            record["converged"] = w.converged
            record["rhat_max"] = _json_safe(w.rhat_max)
            record["ess_bulk_min"] = int(w.ess_bulk_min)
            record["posterior_mean_W"] = w.posterior_mean_W.tolist()
            record["posterior_mean_b"] = w.posterior_mean_b.tolist()
        elif isinstance(w, DmmVariantWindow):
            record["feature_identity"] = w.feature_identity
            record["side_info_dim"] = int(w.side_info_dim)
            record["num_epochs"] = int(w.num_epochs)
            record["mini_batch_size"] = int(w.mini_batch_size)
            record["learning_rate"] = _json_safe(w.learning_rate)
            record["beta1"] = _json_safe(w.beta1)
            record["beta2"] = _json_safe(w.beta2)
            record["clip_norm"] = _json_safe(w.clip_norm)
            record["lr_decay"] = _json_safe(w.lr_decay)
            record["weight_decay"] = _json_safe(w.weight_decay)
            record["min_annealing_factor"] = _json_safe(w.min_annealing_factor)
            record["annealing_epochs"] = int(w.annealing_epochs)
            record["seed"] = int(w.seed)
            record["final_loss"] = _json_safe(w.final_loss)
        lines.append(json.dumps(record, sort_keys=True, allow_nan=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_posterior_samples(run_dir: Path, variant: str, result: SideInfoVariantResult) -> None:
    """Persist per-window HMC posterior samples as ``<variant>.posterior/window_<i>.npz``.

    No-op for variants whose windows are not ``ContinuousSideInfoVariantWindow``
    (baseline, bucketed-grid, bucketed-quantile). Each window file contains
    ``W`` with shape ``(chains, samples, K, K, D)`` and ``b`` with shape
    ``(chains, samples, K, K)`` — the full posterior, not just the mean — so
    credible intervals and posterior-predictive checks are reproducible from
    the artifact alone.
    """
    hmc_windows = [w for w in result.windows if isinstance(w, ContinuousSideInfoVariantWindow)]
    if not hmc_windows:
        return
    posterior_dir = run_dir / f"{variant}.posterior"
    posterior_dir.mkdir(exist_ok=True)
    for window in hmc_windows:
        path = posterior_dir / f"window_{window.index:04d}.npz"
        np.savez(
            path,
            W=window.posterior_samples["W"],
            b=window.posterior_samples["b"],
        )


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
