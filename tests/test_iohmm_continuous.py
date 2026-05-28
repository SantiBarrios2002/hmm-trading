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
    n_features = 1 if W.ndim == 2 else W.shape[2]
    x = rng.normal(size=n_obs) if n_features == 1 else rng.normal(size=(n_obs, n_features))
    states = np.zeros(n_obs, dtype=int)
    states[0] = int(rng.integers(0, n_states))
    for t in range(n_obs - 1):
        logits = W * x[t] + b if n_features == 1 else np.einsum("ijd,d->ij", W, x[t]) + b
        transition = _softmax(logits)
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
        ("W", true_W, result.posterior_mean_W[..., 0]),
        ("b", true_b, result.posterior_mean_b),
    ):
        samples = result.posterior_samples[name]
        if name == "W":
            samples = samples[..., 0]
        posterior_std = samples.reshape(-1, 2, 2).std(axis=0, ddof=1)
        np.testing.assert_array_less(np.abs(posterior_mean - true_values), 2.0 * posterior_std)


def test_d1_backward_compatibility_regression() -> None:
    expected_w = np.array(
        [
            [
                [
                    [-0.08780285716056824, 0.08780285716056824],
                    [1.0802294313907623, -1.0802294313907623],
                ],
                [
                    [-0.5058563388884068, 0.5058563388884068],
                    [0.397137850522995, -0.397137850522995],
                ],
                [
                    [-0.9694189727306366, 0.9694189727306366],
                    [0.5519367456436157, -0.5519367456436157],
                ],
                [
                    [-0.7609425336122513, 0.7609425336122513],
                    [1.291548639535904, -1.291548639535904],
                ],
                [
                    [-0.44042253494262695, 0.44042253494262695],
                    [0.31410200893878937, -0.31410200893878937],
                ],
                [
                    [-0.43127211928367615, 0.43127211928367615],
                    [0.8329690098762512, -0.8329690098762512],
                ],
            ],
            [
                [
                    [-0.3371464256197214, 0.3371464256197214],
                    [0.7485049217939377, -0.7485049217939377],
                ],
                [
                    [-0.7067218869924545, 0.7067218869924545],
                    [0.38034521136432886, -0.38034521136432886],
                ],
                [
                    [-0.29504184424877167, 0.29504184424877167],
                    [0.4838552102446556, -0.4838552102446556],
                ],
                [
                    [-0.8364518731832504, 0.8364518731832504],
                    [0.7299068868160248, -0.7299068868160248],
                ],
                [
                    [-0.6796903014183044, 0.6796903014183044],
                    [0.5249980986118317, -0.5249980986118317],
                ],
                [
                    [-0.3747633099555969, 0.3747633099555969],
                    [0.7260288298130035, -0.7260288298130035],
                ],
            ],
        ]
    )
    expected_b = np.array(
        [
            [
                [
                    [1.057192251086235, -1.057192251086235],
                    [-0.315414622426033, 0.315414622426033],
                ],
                [
                    [0.7806781213730574, -0.7806781213730574],
                    [0.4061571955680847, -0.4061571955680847],
                ],
                [
                    [1.1842710971832275, -1.1842710971832275],
                    [-1.28862726688385, 1.28862726688385],
                ],
                [
                    [1.1014857590198517, -1.1014857590198517],
                    [-0.31101539731025696, 0.31101539731025696],
                ],
                [
                    [0.8831271529197693, -0.8831271529197693],
                    [0.20850633829832077, -0.20850633829832077],
                ],
                [
                    [1.2444812655448914, -1.2444812655448914],
                    [-0.7576411068439484, 0.7576411068439484],
                ],
            ],
            [
                [
                    [1.0898419916629791, -1.0898419916629791],
                    [-0.670932948589325, 0.670932948589325],
                ],
                [
                    [1.3751777410507202, -1.3751777410507202],
                    [-0.821102499961853, 0.821102499961853],
                ],
                [
                    [1.2397245466709137, -1.2397245466709137],
                    [-0.8842011094093323, 0.8842011094093323],
                ],
                [
                    [0.9676795303821564, -0.9676795303821564],
                    [0.45578765869140625, -0.45578765869140625],
                ],
                [
                    [0.9317888021469116, -0.9317888021469116],
                    [0.3572673499584198, -0.3572673499584198],
                ],
                [
                    [1.0318421125411987, -1.0318421125411987],
                    [0.2588411867618561, -0.2588411867618561],
                ],
            ],
        ]
    )
    expected_rhat_w = np.array(
        [
            [0.8573229241533895, 0.8573229241533895],
            [0.8878853154009674, 0.8878853154009674],
        ]
    )
    expected_rhat_b = np.array(
        [
            [1.0971086800413397, 1.0971086800413397],
            [1.263539177084734, 1.263539177084734],
        ]
    )

    W = np.array([[-0.5, 0.5], [0.4, -0.4]])
    b = np.array([[0.7, -0.7], [-0.3, 0.3]])
    states, x = _simulate_continuous_iohmm(W=W, b=b, n_obs=60, seed=17)
    result = fit_continuous_iohmm(
        state_sequence=states,
        side_information=x,
        n_states=2,
        config=ContinuousIOHMMConfig(
            num_warmup=5,
            num_samples=6,
            seed=23,
            rhat_threshold=10.0,
            ess_bulk_threshold=1,
        ),
    )

    assert result.posterior_samples["W"].shape == (2, 6, 2, 2, 1)
    np.testing.assert_array_equal(result.posterior_samples["W"][..., 0], expected_w)
    np.testing.assert_array_equal(result.posterior_samples["b"], expected_b)
    np.testing.assert_array_equal(result.rhat["W"][..., 0], expected_rhat_w)
    np.testing.assert_array_equal(result.rhat["b"], expected_rhat_b)


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
    assert result.posterior_samples["W"].shape == (2, 200, 2, 2, 1)


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


def test_d2_transition_probabilities_are_finite_and_row_stochastic() -> None:
    W = np.array([[[-0.5, 0.3], [0.5, -0.3]], [[0.4, -0.2], [-0.4, 0.2]]])
    b = np.array([[0.6, -0.6], [-0.2, 0.2]])
    states, x = _simulate_continuous_iohmm(W=W, b=b, n_obs=160, seed=31)
    result = fit_continuous_iohmm(
        state_sequence=states,
        side_information=x,
        n_states=2,
        config=ContinuousIOHMMConfig(
            num_warmup=8,
            num_samples=12,
            seed=37,
            rhat_threshold=10.0,
            ess_bulk_threshold=1,
        ),
    )

    grid_1 = np.linspace(-2.0, 2.0, 5)
    grid_2 = np.linspace(-1.5, 1.5, 5)
    x_grid = np.array([(a, b_) for a in grid_1 for b_ in grid_2])
    matrices = transition_probabilities_at(result=result, x_values=x_grid)

    assert matrices.shape == (25, 2, 2)
    assert np.all(np.isfinite(matrices))
    np.testing.assert_allclose(matrices.sum(axis=2), 1.0, atol=1e-9)


def test_d2_synthetic_recovery_within_two_posterior_sigmas() -> None:
    true_W = np.array(
        [
            [[-0.9, 0.5], [0.9, -0.5]],
            [[0.6, -0.4], [-0.6, 0.4]],
        ]
    )
    true_b = np.array([[0.8, -0.8], [-0.5, 0.5]])
    states, x = _simulate_continuous_iohmm(W=true_W, b=true_b, n_obs=4000, seed=41)

    result = fit_continuous_iohmm(
        state_sequence=states,
        side_information=x,
        n_states=2,
        config=ContinuousIOHMMConfig(
            num_warmup=500,
            num_samples=1000,
            num_chains=2,
            seed=43,
            rhat_threshold=1.1,
            ess_bulk_threshold=50,
        ),
    )

    mean, std = result.standardization
    expected_w = true_W * std[None, None, :]
    expected_b = true_b + np.einsum("ijd,d->ij", true_W, mean)
    expected_b = expected_b - expected_b.mean(axis=1, keepdims=True)

    w_std = result.posterior_samples["W"].reshape(-1, 2, 2, 2).std(axis=0, ddof=1)
    b_std = result.posterior_samples["b"].reshape(-1, 2, 2).std(axis=0, ddof=1)
    np.testing.assert_array_less(np.abs(result.posterior_mean_W - expected_w), 2.0 * w_std)
    np.testing.assert_array_less(np.abs(result.posterior_mean_b - expected_b), 2.0 * b_std)


def test_standardization_is_per_feature_and_training_slice_only() -> None:
    rng = np.random.default_rng(47)
    train_x = rng.normal(loc=np.array([1.0, -2.0]), scale=np.array([0.5, 2.0]), size=(80, 2))
    heldout_x = rng.normal(
        loc=np.array([100.0, -100.0]), scale=np.array([10.0, 20.0]), size=(20, 2)
    )
    full_x = np.vstack([train_x, heldout_x])
    states = np.arange(train_x.shape[0]) % 2

    result = fit_continuous_iohmm(
        state_sequence=states,
        side_information=full_x[: train_x.shape[0]],
        n_states=2,
        config=ContinuousIOHMMConfig(
            num_warmup=5,
            num_samples=8,
            seed=53,
            rhat_threshold=10.0,
            ess_bulk_threshold=1,
        ),
    )

    mean, std = result.standardization
    assert mean.shape == (2,)
    assert std.shape == (2,)
    np.testing.assert_allclose(mean, train_x.mean(axis=0))
    np.testing.assert_allclose(std, train_x.std(axis=0))
    assert not np.allclose(mean, full_x.mean(axis=0))
    assert not np.allclose(std, full_x.std(axis=0))
