"""Tests for the Gate K continuous-parametric IOHMM transition model."""

from __future__ import annotations

import importlib
import time

import numpy as np

from hft_hmm.core.references import ENGINEERING_APPROXIMATION, module_category
from hft_hmm.models.iohmm_continuous import (
    ContinuousIOHMMConfig,
    fit_continuous_iohmm,
    transition_probabilities_at,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _simulate_continuous_iohmm(
    *,
    W: np.ndarray,
    b: np.ndarray,
    n_obs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_states = W.shape[0]
    x = rng.normal(size=n_obs)
    states = np.zeros(n_obs, dtype=int)
    states[0] = int(rng.integers(0, n_states))
    for t in range(n_obs - 1):
        transition = _softmax(W * x[t] + b)
        states[t + 1] = int(rng.choice(n_states, p=transition[states[t]]))
    return states, x


def test_module_declares_engineering_approximation() -> None:
    mod = importlib.import_module("hft_hmm.models.iohmm_continuous")
    assert module_category(mod) == ENGINEERING_APPROXIMATION


def test_synthetic_recovery_within_two_posterior_sigmas() -> None:
    true_W = np.array([[-0.8, 0.8], [0.6, -0.6]])
    true_b = np.array([[1.2, -1.2], [-0.5, 0.5]])
    states, x = _simulate_continuous_iohmm(W=true_W, b=true_b, n_obs=5000, seed=12)

    result = fit_continuous_iohmm(
        state_sequence=states,
        side_information=x,
        n_states=2,
        config=ContinuousIOHMMConfig(
            num_warmup=100,
            num_samples=200,
            seed=3,
            rhat_threshold=1.2,
            ess_bulk_threshold=50,
        ),
    )

    for name, true_values, posterior_mean in (
        ("W", true_W, result.posterior_mean_W),
        ("b", true_b, result.posterior_mean_b),
    ):
        posterior_std = result.posterior_samples[name].reshape(-1, 2, 2).std(axis=0, ddof=1)
        np.testing.assert_array_less(np.abs(posterior_mean - true_values), 2.0 * posterior_std)


def test_same_seed_produces_identical_posterior_samples() -> None:
    W = np.array([[-0.5, 0.5], [0.4, -0.4]])
    b = np.array([[0.7, -0.7], [-0.3, 0.3]])
    states, x = _simulate_continuous_iohmm(W=W, b=b, n_obs=120, seed=7)
    config = ContinuousIOHMMConfig(
        num_warmup=8,
        num_samples=12,
        seed=11,
        rhat_threshold=10.0,
        ess_bulk_threshold=1,
    )

    first = fit_continuous_iohmm(
        state_sequence=states,
        side_information=x,
        n_states=2,
        config=config,
    )
    second = fit_continuous_iohmm(
        state_sequence=states,
        side_information=x,
        n_states=2,
        config=config,
    )

    np.testing.assert_array_equal(first.posterior_samples["W"], second.posterior_samples["W"])
    np.testing.assert_array_equal(first.posterior_samples["b"], second.posterior_samples["b"])


def test_short_cpu_fit_stays_under_speed_budget() -> None:
    W = np.array([[-0.7, 0.7], [0.5, -0.5]])
    b = np.array([[0.8, -0.8], [-0.4, 0.4]])
    states, x = _simulate_continuous_iohmm(W=W, b=b, n_obs=500, seed=21)
    config = ContinuousIOHMMConfig(
        num_warmup=100,
        num_samples=200,
        seed=9,
        rhat_threshold=1.5,
        ess_bulk_threshold=20,
    )

    start = time.monotonic()
    result = fit_continuous_iohmm(
        state_sequence=states,
        side_information=x,
        n_states=2,
        config=config,
    )
    elapsed = time.monotonic() - start

    assert elapsed < 60.0
    assert result.posterior_samples["W"].shape == (2, 200, 2, 2)


def test_pathological_unreachable_state_marks_nonconvergence() -> None:
    rng = np.random.default_rng(42)
    states = np.zeros(100, dtype=int)
    x = rng.normal(size=100)
    config = ContinuousIOHMMConfig(
        num_warmup=8,
        num_samples=12,
        seed=5,
        rhat_threshold=1.05,
        ess_bulk_threshold=20,
    )

    result = fit_continuous_iohmm(
        state_sequence=states,
        side_information=x,
        n_states=2,
        config=config,
    )

    rhat_max = max(float(values.max()) for values in result.rhat.values())
    assert result.converged is False
    assert rhat_max > config.rhat_threshold


def test_transition_probabilities_are_standardized_and_row_stochastic() -> None:
    W = np.array([[-0.5, 0.5], [0.4, -0.4]])
    b = np.array([[0.7, -0.7], [-0.3, 0.3]])
    states, x = _simulate_continuous_iohmm(W=W, b=b, n_obs=120, seed=10)
    result = fit_continuous_iohmm(
        state_sequence=states,
        side_information=x,
        n_states=2,
        config=ContinuousIOHMMConfig(
            num_warmup=8,
            num_samples=12,
            seed=13,
            rhat_threshold=10.0,
            ess_bulk_threshold=1,
        ),
    )

    matrices = transition_probabilities_at(result=result, x_values=np.array([-1.0, 0.0, 1.0]))

    assert matrices.shape == (3, 2, 2)
    assert np.all(matrices >= 0.0)
    np.testing.assert_allclose(matrices.sum(axis=2), 1.0, atol=1e-12)
    assert not np.allclose(matrices[0], matrices[-1])
