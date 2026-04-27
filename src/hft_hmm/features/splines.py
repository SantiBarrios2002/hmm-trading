"""Spline-based predictor fitting — engineering approximation of §4.1.

This module fits a deterministic cubic spline f(x_t) ≈ E[r_{t+1} | x_t]
from aligned side-information and future-return observations. It is the
fitting primitive for standalone predictor backtests (Issue 22) and the
IOHMM-style bucketed transition approximation (Issue 16).

This is an engineering approximation: Christensen, Turner & Godsill (§4.1)
describe using a spline-estimated conditional mean to derive regime bucket
boundaries; the exact numerical procedure is not specified. This module
implements least-squares cubic spline regression via
``scipy.interpolate.LSQUnivariateSpline`` as a practical substitute.

Duplicate feature values (common with discrete predictors such as intraday
seasonality buckets) are aggregated by mean future return before fitting, so
the spline estimates E[r_{t+1} | x_t = v] for each unique v. Bucket-boundary
derivation (locating sign changes or crossings on the fitted spline) is out of
scope; callers should use ``SplinePredictorResult.evaluation_grid()`` to obtain
a dense (x, y) grid from which Issue 16 can locate bucket edges.

References: §4.1 spline-based predictor
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy.interpolate import LSQUnivariateSpline

from hft_hmm.core.references import ENGINEERING_APPROXIMATION, PaperReference, reference

__category__: Final[str] = ENGINEERING_APPROXIMATION
SPLINE_PREDICTOR_REFERENCE: Final[PaperReference] = reference("§4.1", "spline-based predictor")

DEFAULT_N_KNOTS: Final[int] = 5
DEFAULT_DEGREE: Final[int] = 3
DEFAULT_MIN_OBS: Final[int] = 20


@dataclass(frozen=True)
class SplinePredictorConfig:
    """Typed parameter bundle for the spline predictor.

    ``n_knots`` is the requested number of interior knots; the actual count
    used may be lower if quantile placement produces duplicate positions
    (recorded in ``SplinePredictorResult.n_knots_effective``).
    ``degree`` is the spline polynomial degree (default 3, cubic).
    ``min_obs`` is the minimum number of valid aligned pairs required after
    NaN-dropping; fewer raises ``ValueError``.
    ``demean`` subtracts the mean prediction evaluated over all unique observed
    x values so the spline is centered on its own support.

    References: §4.1 spline-based predictor
    """

    n_knots: int = DEFAULT_N_KNOTS
    degree: int = DEFAULT_DEGREE
    min_obs: int = DEFAULT_MIN_OBS
    demean: bool = False

    def __post_init__(self) -> None:
        _validate_positive_int(self.n_knots, "n_knots")
        _validate_positive_int(self.degree, "degree")
        _validate_positive_int(self.min_obs, "min_obs")
        if not isinstance(self.demean, bool):
            raise TypeError(f"demean must be a bool; got {type(self.demean).__name__}.")


@dataclass(frozen=True)
class SplinePredictorResult:
    """Fitted spline predictor with evaluation helpers.

    ``spline`` is the underlying ``scipy.interpolate.LSQUnivariateSpline``
    object. ``n_obs`` is the number of raw aligned pairs (before aggregation).
    ``x_min`` and ``x_max`` are the observed feature support bounds.
    ``knots`` are the interior knot positions actually used after deduplication;
    ``n_knots_effective`` is ``len(knots)`` and may be less than
    ``config.n_knots`` when quantile placement collapsed duplicate positions.
    ``prediction_mean`` is non-``None`` when ``config.demean=True``; it is the
    mean fitted value over a uniform grid on the observed support and is
    subtracted from every ``evaluate()`` and ``evaluation_grid()`` call.

    References: §4.1 spline-based predictor
    """

    spline: Any  # scipy.interpolate.LSQUnivariateSpline
    n_obs: int
    x_min: float
    x_max: float
    knots: np.ndarray
    n_knots_effective: int
    prediction_mean: float | None

    def evaluate(
        self, x: float | np.ndarray | pd.Series
    ) -> float | np.ndarray | pd.Series:
        """Return predicted returns for scalar, array, or Series input.

        A ``pd.Series`` input preserves the original index. Values outside
        ``[x_min, x_max]`` are extrapolated by the spline (scipy default).

        References: §4.1 spline-based predictor
        """
        if isinstance(x, pd.Series):
            index = x.index
            arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
            predicted = np.asarray(self.spline(arr), dtype=np.float64)
            if self.prediction_mean is not None:
                predicted = predicted - self.prediction_mean
            return pd.Series(predicted, index=index, name="spline_prediction")

        is_scalar = np.ndim(x) == 0
        arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        predicted = np.asarray(self.spline(arr), dtype=np.float64)

        if self.prediction_mean is not None:
            predicted = predicted - self.prediction_mean

        if is_scalar:
            return float(predicted[0])
        return predicted

    def evaluation_grid(self, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x_grid, y_grid)`` over a uniform grid on ``[x_min, x_max]``.

        Useful for deriving bucket boundaries from evaluated spline values.

        References: §4.1 spline-based predictor
        """
        x_grid = np.linspace(self.x_min, self.x_max, n)
        y_grid = np.asarray(self.spline(x_grid), dtype=np.float64)
        if self.prediction_mean is not None:
            y_grid = y_grid - self.prediction_mean
        return x_grid, y_grid


def fit_spline_predictor(
    feature: pd.Series,
    returns: pd.Series,
    *,
    config: SplinePredictorConfig | None = None,
) -> SplinePredictorResult:
    """Fit a cubic spline f(x_t) ≈ E[r_{t+1} | x_t].

    Aligns ``feature[t]`` with ``returns[t+1]`` via index-based shift to
    avoid look-ahead leakage. NaN rows are silently dropped (e.g. the
    slow-window prefix produced by the volatility-ratio predictor); infinite
    values raise ``ValueError``. Duplicate feature values are aggregated by
    mean future return before fitting so the spline is valid for discrete
    predictors such as intraday seasonality buckets.

    Pass ``config`` to control knot count, polynomial degree, minimum
    observation threshold, and demeaning.

    References: §4.1 spline-based predictor
    """
    if config is None:
        config = SplinePredictorConfig()

    if not isinstance(feature, pd.Series):
        raise TypeError(f"feature must be a pd.Series; got {type(feature).__name__}.")
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a pd.Series; got {type(returns).__name__}.")

    future_returns = returns.shift(-1)
    aligned = pd.concat(
        [feature.rename("x"), future_returns.rename("y")], axis=1
    ).dropna()

    x_arr = aligned["x"].to_numpy(dtype=np.float64)
    y_arr = aligned["y"].to_numpy(dtype=np.float64)

    if not np.isfinite(x_arr).all():
        raise ValueError("feature contains non-finite values; remove inf before calling.")
    if not np.isfinite(y_arr).all():
        raise ValueError("returns contains non-finite values; remove inf before calling.")

    if len(x_arr) < config.min_obs:
        raise ValueError(
            f"Only {len(x_arr)} valid (feature, future_return) pairs after alignment; "
            f"need at least {config.min_obs} (config.min_obs)."
        )

    # Aggregate duplicate feature values: group by unique x, take mean y.
    # This satisfies LSQUnivariateSpline's requirement for no repeated x values
    # and correctly estimates E[r_{t+1} | x_t = v] for discrete predictors.
    x_unique, inverse = np.unique(x_arr, return_inverse=True)
    counts = np.bincount(inverse)
    y_agg = np.bincount(inverse, weights=y_arr) / counts

    n_unique = len(x_unique)
    required_unique = config.n_knots + config.degree + 1
    if n_unique < required_unique:
        raise ValueError(
            f"Only {n_unique} unique feature values after aggregation; need at least "
            f"n_knots + degree + 1 = {required_unique} to fit a degree-{config.degree} "
            f"spline with {config.n_knots} interior knots."
        )

    # Place interior knots at quantiles of unique x, then deduplicate.
    # Quantiles of x_unique (not raw x_arr) ensure spread proportional to
    # unique support; np.unique collapses any coincident positions from
    # heavy-tailed or heavily-discretised features.
    q = np.linspace(0.0, 1.0, config.n_knots + 2)[1:-1]
    knots = np.unique(np.quantile(x_unique, q))

    if len(knots) == 0:
        raise ValueError(
            "No unique interior knots could be placed; the feature support is too narrow."
        )

    fitted = LSQUnivariateSpline(x_unique, y_agg, t=knots, k=config.degree)

    x_min = float(x_unique[0])
    x_max = float(x_unique[-1])
    n_knots_effective = len(knots)

    prediction_mean: float | None = None
    if config.demean:
        support_grid = np.linspace(x_min, x_max, 1_000)
        prediction_mean = float(np.mean(fitted(support_grid)))

    return SplinePredictorResult(
        spline=fitted,
        n_obs=len(x_arr),
        x_min=x_min,
        x_max=x_max,
        knots=knots,
        n_knots_effective=n_knots_effective,
        prediction_mean=prediction_mean,
    )


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer; got {type(value).__name__}.")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value}.")
