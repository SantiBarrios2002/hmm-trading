"""Gate H side-information comparison experiment.

Runs three walk-forward variants on the same return series with the same
window geometry, K selection, cost model, and reporting units, so the
variants are directly comparable:

1. ``baseline`` — Gaussian HMM walk-forward (delegates to
   :func:`hft_hmm.experiments.walk_forward.walk_forward` so the summary
   matches the existing baseline path exactly).
2. ``volatility_ratio_conditioned`` — same walk-forward geometry, but the
   transition matrix is replaced per-bar by an Issue-16 bucketed
   approximation conditioned on the volatility-ratio side information.
3. ``seasonality_conditioned`` — same as above with the intraday seasonality
   feature.

The IOHMM-style conditioning is the engineering approximation from
``hft_hmm.models.iohmm_approx``: a bucket index is derived from the
side-information value, and a separate K × K transition matrix is fitted per
bucket on the training slice, with the per-window MLE matrix as the smoothing
prior.

References: §4 side-information / IOHMM evaluation (evaluation layer)
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
from typing import Any, Final

import numpy as np
import pandas as pd
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
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    signal_turnover,
)
from hft_hmm.experiments._data_loading import (
    load_returns_from_source,
    validate_data_reproducibility,
)
from hft_hmm.experiments.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
    walk_forward,
)
from hft_hmm.features.seasonality import SeasonalityConfig, intraday_seasonality
from hft_hmm.features.splines import SplinePredictorConfig, fit_spline_predictor
from hft_hmm.features.volatility_ratio import VolatilityRatioConfig, volatility_ratio
from hft_hmm.models.gaussian_hmm import GaussianHMMResult, GaussianHMMWrapper
from hft_hmm.models.iohmm_approx import (
    BucketedTransitionConfig,
    BucketedTransitionResult,
    bucket_boundaries_from_spline_grid,
    fit_bucketed_transition_model,
)
from hft_hmm.strategy import align_signal_with_future_return, sign_signal

__category__: Final[str] = EVALUATION_LAYER
SIDE_INFO_COMPARISON_REFERENCE: Final[PaperReference] = reference(
    "§4", "side-information IOHMM comparison (Gate H)"
)

_VALID_FREQUENCIES: Final[tuple[str, ...]] = ("1min", "5min", "1D")
_SHA256_HEX_LENGTH: Final[int] = 64

BASELINE_VARIANT: Final[str] = "baseline"
VOLATILITY_RATIO_VARIANT: Final[str] = "volatility_ratio_conditioned"
SEASONALITY_VARIANT: Final[str] = "seasonality_conditioned"
EXPECTED_VARIANTS: Final[tuple[str, ...]] = (
    BASELINE_VARIANT,
    VOLATILITY_RATIO_VARIANT,
    SEASONALITY_VARIANT,
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
                "grid_size": int(self.bucketed_transition.grid_size),
                "n_buckets": int(self.bucketed_transition.n_buckets),
                "smoothing": float(self.bucketed_transition.smoothing),
            },
            "cost_bps_per_turnover": float(self.cost_bps_per_turnover),
            "data": self.data.to_dict(),
            "frequency": self.frequency,
            "notes": self.notes,
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
            vol_ratio=vol_ratio_cfg,
            seasonality=seasonality_cfg,
            cost_bps_per_turnover=float(raw.get("cost_bps_per_turnover", 0.0)),
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
        if not isinstance(self.summary, pd.DataFrame):
            raise TypeError("summary must be a pd.DataFrame.")


@dataclass(frozen=True)
class SideInfoVariantResult:
    """Per-variant result of a side-information comparison."""

    variant: str
    chosen_k_per_window: tuple[int, ...]
    windows: tuple[SideInfoVariantWindow, ...] | tuple[WalkForwardWindow, ...]
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


def run_side_info_comparison(
    config: SideInfoComparisonConfig,
    *,
    runs_root: Path | str = Path("runs"),
    force: bool = False,
) -> SideInfoComparisonArtifacts:
    """Execute the comparison described by ``config`` and write artifacts.

    Target directory is ``runs_root / comparison_id(config)``; ``force=True``
    replaces an existing directory atomically (with rollback on failure).

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

    reproducible = validate_data_reproducibility(config, stacklevel=3)
    returns = load_returns_from_source(config.data, frequency=config.frequency)

    baseline_wf = walk_forward(
        returns,
        config.walk_forward,
        cost_bps_per_turnover=config.cost_bps_per_turnover,
    )
    baseline_variant = _build_baseline_variant(baseline_wf, config.cost_bps_per_turnover)
    vol_variant = _run_side_info_variant(
        VOLATILITY_RATIO_VARIANT,
        returns=returns,
        config=config,
        baseline_windows=baseline_wf.windows,
    )
    seasonality_variant = _run_side_info_variant(
        SEASONALITY_VARIANT,
        returns=returns,
        config=config,
        baseline_windows=baseline_wf.windows,
    )

    result = SideInfoComparisonResult(
        config=config,
        comparison_id=cmp_id,
        variants={
            BASELINE_VARIANT: baseline_variant,
            VOLATILITY_RATIO_VARIANT: vol_variant,
            SEASONALITY_VARIANT: seasonality_variant,
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

    return SideInfoComparisonArtifacts(
        comparison_id=cmp_id,
        directory=run_dir,
        config=config,
        result=result,
    )


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
    if variant not in (VOLATILITY_RATIO_VARIANT, SEASONALITY_VARIANT):
        raise ValueError(f"unsupported variant {variant!r}.")

    cost = config.cost_bps_per_turnover
    wf = config.walk_forward
    bar_dates = returns.index.date

    windows: list[SideInfoVariantWindow] = []
    signal_parts: list[pd.Series] = []
    pre_cost_parts: list[pd.Series] = []
    post_cost_parts: list[pd.Series] = []
    chosen_ks: list[int] = []

    for baseline_window in baseline_windows:
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
        feature_train = feature_full.loc[train_slice.index]
        feature_forecast = feature_full.loc[effective_forecast.index]

        if feature_forecast.isna().any():
            raise ValueError(
                f"forecast feature for variant {variant!r} contains NaN in window "
                f"{baseline_window.index}; increase h_days so the rolling window is "
                "fully initialized before the forecast slice."
            )

        spline_result = fit_spline_predictor(feature_train, train_slice, config=config.spline)
        boundaries = bucket_boundaries_from_spline_grid(
            spline_result, config=config.bucketed_transition
        )

        feature_train_clean = feature_train.dropna()
        if feature_train_clean.shape[0] < 2:
            raise ValueError(
                f"variant {variant!r} window {baseline_window.index} has fewer than 2 "
                "finite training feature observations after dropping the NaN prefix."
            )
        nonnan_mask = feature_train.notna().to_numpy()
        decoded_aligned = decoded[nonnan_mask]

        bucketed = fit_bucketed_transition_model(
            state_sequence=decoded_aligned,
            side_information=feature_train_clean.to_numpy(),
            n_states=chosen_k,
            baseline_transition_matrix=fitted.transition_matrix,
            bucket_boundaries=boundaries,
            config=config.bucketed_transition,
        )

        expected = _dynamic_forward_expected_returns(
            forecast_returns=effective_forecast.to_numpy(),
            forecast_features=feature_forecast.to_numpy(),
            fitted=fitted,
            bucketed=bucketed,
        )
        expected_series = pd.Series(expected, index=effective_forecast.index)
        window_signal = sign_signal(expected_series)
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
            SideInfoVariantWindow(
                index=int(baseline_window.index),
                train_start=train_slice.index.min(),
                train_end=train_slice.index.max(),
                forecast_start=effective_forecast.index.min(),
                forecast_end=effective_forecast.index.max(),
                chosen_k=chosen_k,
                n_train_obs=int(train_slice.shape[0]),
                n_forecast_obs=int(effective_forecast.shape[0]),
                bucket_observation_counts=tuple(int(c) for c in bucketed.bucket_observation_counts),
                summary=window_summary,
            )
        )

    combined_signal = pd.concat(signal_parts).astype(np.int8)
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


def _build_feature(
    variant: str,
    series: pd.Series,
    config: SideInfoComparisonConfig,
) -> pd.Series:
    if variant == VOLATILITY_RATIO_VARIANT:
        return volatility_ratio(
            series,
            fast_window=config.vol_ratio.fast_window,
            slow_window=config.vol_ratio.slow_window,
            decay=config.vol_ratio.decay,
        )
    if variant == SEASONALITY_VARIANT:
        return intraday_seasonality(series, config=config.seasonality)
    raise ValueError(f"unsupported variant {variant!r}.")


def _dynamic_forward_expected_returns(
    *,
    forecast_returns: np.ndarray,
    forecast_features: np.ndarray,
    fitted: GaussianHMMResult,
    bucketed: BucketedTransitionResult,
) -> np.ndarray:
    """Run the side-info-conditioned forward filter and return E[r_{t+1}|info_t].

    Filtering uses ``A(x_{t-1})`` to advance the predictive distribution from
    ``t-1`` to ``t``; the one-step prediction at ``t`` then uses ``A(x_t)``,
    matching the IOHMM convention where the side-information observed at time
    ``t`` parameterises the next transition.
    """
    means = fitted.means
    variances = fitted.variances
    if forecast_returns.ndim != 1:
        raise ValueError(f"forecast_returns must be 1-D, got shape {forecast_returns.shape}.")
    if forecast_features.shape != forecast_returns.shape:
        raise ValueError(
            "forecast_features must have the same shape as forecast_returns; "
            f"got {forecast_features.shape} vs {forecast_returns.shape}."
        )
    if not np.all(np.isfinite(forecast_returns)):
        raise ValueError("forecast_returns must contain only finite values.")
    if not np.all(np.isfinite(forecast_features)):
        raise ValueError("forecast_features must contain only finite values.")

    n_obs = forecast_returns.shape[0]
    k = means.shape[0]
    centered = forecast_returns[:, None] - means[None, :]
    log_emissions = -0.5 * (
        np.log(2.0 * np.pi * variances[None, :]) + (centered * centered) / variances[None, :]
    )

    log_filtering = np.empty((n_obs, k), dtype=float)
    with np.errstate(divide="ignore"):
        log_start = np.log(fitted.initial_distribution)
    log_alpha = log_start + log_emissions[0]
    log_filtering[0] = log_alpha - logsumexp(log_alpha)

    for t in range(1, n_obs):
        prev_matrix = bucketed.transition_matrix_for(float(forecast_features[t - 1]))
        with np.errstate(divide="ignore"):
            log_transition = np.log(prev_matrix)
        log_predicted = logsumexp(log_filtering[t - 1][:, None] + log_transition, axis=0)
        log_alpha = log_emissions[t] + log_predicted
        log_filtering[t] = log_alpha - logsumexp(log_alpha)

    filtering = np.exp(log_filtering)
    expected = np.empty(n_obs, dtype=float)
    for t in range(n_obs):
        next_matrix = bucketed.transition_matrix_for(float(forecast_features[t]))
        expected[t] = filtering[t] @ next_matrix @ means
    return expected


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
    sample_window = {
        "start": result.signal.index.min().isoformat(),
        "end": result.signal.index.max().isoformat(),
    }
    return {
        "variant": variant,
        "comparison_id": cmp_id,
        "sample_window": sample_window,
        "chosen_k_per_window": [int(k) for k in result.chosen_k_per_window],
        "n_windows": len(result.windows),
        "n_forecast_obs": int(result.signal.shape[0]),
        "cost_bps_per_turnover": float(result.cost_bps_per_turnover),
        "summary": _summary_to_payload(result.summary),
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
        lines.append(json.dumps(record, sort_keys=True, allow_nan=False))
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
