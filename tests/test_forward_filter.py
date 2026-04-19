"""Tests for the log-space Gaussian-HMM forward filter."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from hmmlearn import _hmmc
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

from hft_hmm.core import PAPER_FAITHFUL, StateGrid, module_category
from hft_hmm.inference import ForwardFilterResult, filter_from_result, forward_filter
from hft_hmm.models.gaussian_hmm import GaussianHMMResult

forward_filter_module = importlib.import_module("hft_hmm.inference.forward_filter")


def _toy_model() -> GaussianHMMResult:
    means = np.array([-1.0, 1.0], dtype=float)
    variances = np.array([0.1, 0.1], dtype=float)
    transition_matrix = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)
    initial_distribution = np.array([0.5, 0.5], dtype=float)
    return GaussianHMMResult(
        state_grid=StateGrid(k=2, means=means, labels=("down", "up")),
        means=means,
        variances=variances,
        transition_matrix=transition_matrix,
        initial_distribution=initial_distribution,
        log_likelihood=0.0,
        n_observations=4,
        converged=True,
        n_iter=10,
        random_state=0,
    )


def test_forward_filter_module_declares_paper_faithful_category() -> None:
    assert module_category(forward_filter_module) == PAPER_FAITHFUL


def test_forward_filter_returns_normalized_probabilities_and_expected_returns() -> None:
    model = _toy_model()
    returns = np.array([-1.2, -0.8, 0.9, 1.1], dtype=float)

    result = forward_filter(returns, model)

    assert isinstance(result, ForwardFilterResult)
    assert result.filtering_probabilities.shape == (4, 2)
    assert result.predicted_next_state_probabilities.shape == (4, 2)
    assert result.expected_next_returns.shape == (4,)
    assert np.allclose(result.filtering_probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.predicted_next_state_probabilities.sum(axis=1), 1.0)
    assert result.filtering_probabilities[0, 0] > result.filtering_probabilities[0, 1]
    assert result.filtering_probabilities[-1, 1] > result.filtering_probabilities[-1, 0]
    assert result.expected_next_returns[0] < 0.0
    assert result.expected_next_returns[-1] > 0.0
    assert np.isfinite(result.log_likelihood)


def test_forward_filter_matches_manual_first_step_posterior() -> None:
    model = _toy_model()
    returns = np.array([-1.0], dtype=float)

    result = forward_filter(returns, model)

    emission = np.exp(
        -0.5
        * (
            np.log(2.0 * np.pi * model.variances)
            + ((returns[0] - model.means) ** 2) / model.variances
        )
    )
    posterior = model.initial_distribution * emission
    posterior = posterior / posterior.sum()

    np.testing.assert_allclose(result.filtering_probabilities[0], posterior)


def test_filter_from_result_wrapper_matches_forward_filter() -> None:
    fitted = _toy_model()
    returns = np.array([-1.2, -0.8, 0.9, 1.1], dtype=float)

    wrapped = filter_from_result(fitted, returns)
    direct = forward_filter(returns, model=fitted)

    np.testing.assert_allclose(wrapped.filtering_probabilities, direct.filtering_probabilities)
    np.testing.assert_allclose(
        wrapped.predicted_next_state_probabilities,
        direct.predicted_next_state_probabilities,
    )
    np.testing.assert_allclose(wrapped.expected_next_returns, direct.expected_next_returns)
    assert np.isclose(wrapped.log_likelihood, direct.log_likelihood)


def test_forward_filter_matches_hmmlearn_reference() -> None:
    fitted = _toy_model()
    rng = np.random.default_rng(0)
    returns = rng.normal(size=500)
    observations = returns.reshape(-1, 1)

    ref = GaussianHMM(
        n_components=fitted.state_grid.k,
        covariance_type="diag",
        implementation="log",
    )
    ref.startprob_ = fitted.initial_distribution
    ref.transmat_ = fitted.transition_matrix
    ref.means_ = fitted.means.reshape(-1, 1)
    ref.covars_ = fitted.variances.reshape(-1, 1)

    result = forward_filter(returns, model=fitted)
    log_frameprob = ref._compute_log_likelihood(observations)
    _, forward_log_lattice = _hmmc.forward_log(ref.startprob_, ref.transmat_, log_frameprob)
    forward_probabilities = np.exp(
        forward_log_lattice - logsumexp(forward_log_lattice, axis=1, keepdims=True)
    )

    np.testing.assert_allclose(
        result.filtering_probabilities,
        forward_probabilities,
        atol=1e-10,
    )
    assert np.isclose(result.log_likelihood, ref.score(observations), atol=1e-8)


def test_forward_filter_rejects_invalid_returns() -> None:
    model = _toy_model()

    with pytest.raises(ValueError, match="one-dimensional"):
        forward_filter(np.zeros((3, 2)), model)
    with pytest.raises(ValueError, match="at least one observation"):
        forward_filter(np.array([], dtype=float), model)
    with pytest.raises(ValueError, match="finite"):
        forward_filter(np.array([0.0, np.nan]), model)


def test_forward_filter_stays_finite_on_long_sequence() -> None:
    model = _toy_model()
    returns = np.tile(np.array([-0.95, 0.95], dtype=float), 3000)

    result = forward_filter(returns, model)

    assert result.filtering_probabilities.shape == (6000, 2)
    assert np.all(np.isfinite(result.filtering_probabilities))
    assert np.all(np.isfinite(result.predicted_next_state_probabilities))
    assert np.all(np.isfinite(result.expected_next_returns))
    assert np.isfinite(result.log_likelihood)
    per_step_ll = result.log_likelihood / returns.size
    assert np.isfinite(per_step_ll)
    assert -10.0 < per_step_ll < 0.0
    assert np.allclose(result.filtering_probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.predicted_next_state_probabilities.sum(axis=1), 1.0)
