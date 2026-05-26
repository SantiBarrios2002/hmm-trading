"""Default HMM trading variant with PLR-derived emissions and uniform transitions.

Paper Figure 8 and p.17 define the Default HMM: emission means/variances come from
PLR segment statistics, and the transition matrix A and initial state distribution π
are held at uniform 1/K. This produces the worst-performing model in the paper's
trading comparison because uniform A conveys no regime-transition information.

Category: paper-faithful — PLR is a documented engineering approximation of the
paper's initialization strategy; uniform A and π are exact to the paper definition.

References: §3 Default HMM / Figure 8 trading-model comparison (paper-faithful)
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from hft_hmm.core import PAPER_FAITHFUL, PaperReference, StateGrid, default_labels, reference
from hft_hmm.inference.forward_filter import forward_filter
from hft_hmm.models.gaussian_hmm import GaussianHMMResult
from hft_hmm.models.plr_baseline import fit_piecewise_linear_regression

__category__: Final[str] = PAPER_FAITHFUL
DEFAULT_HMM_REFERENCE: Final[PaperReference] = reference(
    "§3", "Default HMM — PLR emissions, uniform A/π"
)

_DEFAULT_MIN_VARIANCE: Final[float] = 1e-8


def fit_default_hmm(
    returns: pd.Series | np.ndarray,
    n_states: int,
    *,
    min_variance: float = _DEFAULT_MIN_VARIANCE,
) -> GaussianHMMResult:
    """Fit the Default HMM using PLR-derived emissions and uniform transitions.

    Emission means and variances come from K PLR segments fit on ``returns``.
    The transition matrix ``A = (1/K) · ones((K, K))`` and initial distribution
    ``π = ones(K) / K`` hold all states equally likely, reproducing the Default
    HMM from the paper (p.17): *"A contains no information about the market, as
    all states are equally likely."*

    The function is deterministic for fixed inputs: PLR uses dynamic programming
    with no stochastic components, and A/π are computed analytically.

    Parameters
    ----------
    returns:
        Training return series (1-D, finite, monotonic timestamps if pd.Series).
    n_states:
        Number of hidden states K (must match ``walk_forward.chosen_k``).
    min_variance:
        Variance floor applied to PLR residual variances before constructing the
        result, preventing degenerate Gaussian emission densities.

    References: Figure 8 / §3 Default HMM (paper-faithful)
    """
    values = np.asarray(returns, dtype=float).reshape(-1)
    n_obs = int(values.size)
    plr = fit_piecewise_linear_regression(returns, n_segments=n_states)
    means = np.asarray(plr.state_means, dtype=float)  # sorted ascending by StateGrid
    variances = np.maximum(np.asarray(plr.state_variances, dtype=float), min_variance)
    k = n_states
    uniform_matrix = np.full((k, k), 1.0 / k, dtype=float)
    uniform_dist = np.full(k, 1.0 / k, dtype=float)
    state_grid = StateGrid(k=k, means=means, labels=default_labels(k))

    # Two-step construction: compute log_likelihood via forward filter on the
    # training data before building the final immutable result object.
    _prelim = GaussianHMMResult(
        state_grid=state_grid,
        means=means,
        variances=variances,
        transition_matrix=uniform_matrix,
        initial_distribution=uniform_dist,
        log_likelihood=0.0,
        n_observations=n_obs,
        converged=True,
        n_iter=0,
        random_state=None,
        min_variance=min_variance,
    )
    ll = forward_filter(returns, _prelim).log_likelihood

    return GaussianHMMResult(
        state_grid=state_grid,
        means=means,
        variances=variances,
        transition_matrix=uniform_matrix,
        initial_distribution=uniform_dist,
        log_likelihood=ll,
        n_observations=n_obs,
        converged=True,
        n_iter=0,
        random_state=None,
        min_variance=min_variance,
    )
