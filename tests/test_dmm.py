"""Tests for the continuous-emission Deep Markov Model architecture."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("pyro")

import pyro  # noqa: E402
import pyro.poutine as poutine  # noqa: E402
import torch  # noqa: E402

from hft_hmm.core import ENGINEERING_APPROXIMATION, module_category  # noqa: E402
from hft_hmm.models.dmm import (  # noqa: E402
    DMM,
    Combiner,
    DMMConfig,
    DMMFitResult,
    Emitter,
    GatedTransition,
    expected_next_returns_from_dmm,
    fit_dmm,
    kl_annealing_factor,
)
from hft_hmm.strategy.signals import build_signal  # noqa: E402


def _synthetic_training_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = 8
    time_steps = 6

    time_grid = torch.linspace(-1.0, 1.0, steps=time_steps, dtype=torch.float32)
    side_template = torch.stack((time_grid, time_grid.square()), dim=1)
    side_info = side_template.unsqueeze(0).repeat(batch_size, 1, 1)

    offsets = torch.linspace(-0.2, 0.2, steps=batch_size, dtype=torch.float32).view(
        batch_size,
        1,
        1,
    )
    slope = 0.35 * side_info[..., :1]
    curvature = -0.15 * side_info[..., 1:].clone()
    observations = slope + curvature + offsets
    seq_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
    return observations, side_info, seq_lengths


@pytest.fixture(scope="module")
def fitted_dmm_for_forecast() -> DMMFitResult:
    observations, side_info, seq_lengths = _synthetic_training_batch()
    config = DMMConfig(
        obs_dim=1,
        z_dim=3,
        emission_dim=6,
        transition_dim=6,
        rnn_dim=7,
        side_info_dim=2,
        num_epochs=4,
        mini_batch_size=4,
        learning_rate=1e-2,
        beta1=0.9,
        beta2=0.999,
        clip_norm=5.0,
        lr_decay=1.0,
        weight_decay=0.0,
        rnn_dropout_rate=0.0,
        min_annealing_factor=0.2,
        annealing_epochs=2,
        seed=31,
    )
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return fit_dmm(config, observations, side_info, seq_lengths)
    finally:
        torch.set_num_threads(previous_threads)


def _forecast_window_inputs(
    length: int = 6,
) -> tuple[pd.Series, pd.DataFrame]:
    index = pd.date_range("2024-01-02 09:30:00", periods=length, freq="min")
    returns = pd.Series(np.linspace(-0.015, 0.018, num=length), index=index, name="returns")
    side_info = pd.DataFrame(
        {
            "level": np.linspace(-1.0, 1.0, num=length),
            "curvature": np.linspace(0.25, 1.25, num=length),
        },
        index=index,
    )
    return returns, side_info


def _constant_forecast_dmm(*, constant: float, side_info_dim: int) -> DMM:
    dmm = DMM(
        obs_dim=1,
        z_dim=2,
        emission_dim=4,
        transition_dim=4,
        rnn_dim=4,
        side_info_dim=side_info_dim,
        rnn_dropout_rate=0.0,
    )
    with torch.no_grad():
        for parameter in dmm.parameters():
            parameter.zero_()
        dmm.emitter.lin_hidden_to_loc.bias.fill_(constant)
    dmm.eval()
    return dmm


def test_module_declares_engineering_approximation() -> None:
    mod = importlib.import_module("hft_hmm.models.dmm")
    assert module_category(mod) == ENGINEERING_APPROXIMATION


def test_component_forward_shapes() -> None:
    batch_size = 4
    z_dim = 5
    obs_dim = 2
    transition_dim = 7
    emission_dim = 6
    rnn_dim = 9
    side_info_dim = 3

    z_t = torch.randn(batch_size, z_dim)
    x_t = torch.randn(batch_size, side_info_dim)
    h_rnn = torch.randn(batch_size, rnn_dim)

    emitter = Emitter(obs_dim=obs_dim, z_dim=z_dim, emission_dim=emission_dim)
    transition = GatedTransition(
        z_dim=z_dim,
        transition_dim=transition_dim,
        side_info_dim=side_info_dim,
    )
    combiner = Combiner(z_dim=z_dim, rnn_dim=rnn_dim)

    emission_loc, emission_scale = emitter(z_t)
    transition_loc, transition_scale = transition(z_t, x_t)
    guide_loc, guide_scale = combiner(z_t, h_rnn)

    assert emission_loc.shape == (batch_size, obs_dim)
    assert emission_scale.shape == (batch_size, obs_dim)
    assert torch.all(emission_scale > 0.0)

    assert transition_loc.shape == (batch_size, z_dim)
    assert transition_scale.shape == (batch_size, z_dim)
    assert torch.all(transition_scale > 0.0)

    assert guide_loc.shape == (batch_size, z_dim)
    assert guide_scale.shape == (batch_size, z_dim)
    assert torch.all(guide_scale > 0.0)


def test_dmm_model_and_guide_trace_on_synthetic_batch() -> None:
    pyro.clear_param_store()
    batch_size = 3
    time_steps = 4
    obs_dim = 1
    side_info_dim = 2
    z_dim = 4
    observations = torch.randn(batch_size, time_steps, obs_dim)
    side_info = torch.randn(batch_size, time_steps, side_info_dim)
    seq_lengths = torch.tensor([4, 3, 2], dtype=torch.long)

    dmm = DMM(obs_dim=obs_dim, z_dim=z_dim, side_info_dim=side_info_dim, rnn_dim=8)

    guide_trace = poutine.trace(dmm.guide).get_trace(
        observations,
        side_info,
        seq_lengths,
        1.0,
    )
    model_trace = poutine.trace(poutine.replay(dmm.model, trace=guide_trace)).get_trace(
        observations,
        side_info,
        seq_lengths,
        1.0,
    )

    for trace in (guide_trace, model_trace):
        trace.compute_log_prob()

    assert guide_trace.nodes["z_1"]["value"].shape == (batch_size, z_dim)
    assert guide_trace.nodes["z_4"]["value"].shape == (batch_size, z_dim)
    assert model_trace.nodes["obs_y_1"]["value"].shape == (batch_size, obs_dim)
    assert model_trace.nodes["obs_y_4"]["value"].shape == (batch_size, obs_dim)
    assert torch.isfinite(torch.as_tensor(model_trace.log_prob_sum()))
    assert torch.isfinite(torch.as_tensor(guide_trace.log_prob_sum()))


def test_gated_side_info_changes_transition_distribution() -> None:
    torch.manual_seed(0)
    transition = GatedTransition(z_dim=4, transition_dim=10, side_info_dim=2)
    z_prev = torch.randn(1, 4)
    x_low = torch.tensor([[0.0, 0.0]])
    x_high = torch.tensor([[2.5, -3.0]])

    loc_low, scale_low = transition(z_prev, x_low)
    loc_high, scale_high = transition(z_prev, x_high)

    assert not torch.allclose(loc_low, loc_high, atol=1e-4, rtol=0.0)
    assert not torch.allclose(scale_low, scale_high, atol=1e-4, rtol=0.0)


def test_masking_ignores_padded_timesteps_in_model_log_prob() -> None:
    pyro.clear_param_store()
    batch_size = 2
    time_steps = 4
    obs_dim = 1
    side_info_dim = 2
    z_dim = 3

    observations = torch.tensor(
        [
            [[0.10], [0.20], [0.30], [0.40]],
            [[1.00], [2.00], [0.00], [0.00]],
        ],
        dtype=torch.float32,
    )
    side_info = torch.tensor(
        [
            [[0.1, 0.0], [0.2, 0.1], [0.3, 0.2], [0.4, 0.3]],
            [[0.9, 0.0], [1.1, 0.2], [0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    seq_lengths = torch.tensor([4, 2], dtype=torch.long)

    observations_alt = observations.clone()
    observations_alt[1, 2:, 0] = torch.tensor([123.0, -456.0])
    side_info_alt = side_info.clone()
    side_info_alt[1, 2:, :] = torch.tensor([[10.0, -10.0], [-20.0, 20.0]])

    dmm = DMM(obs_dim=obs_dim, z_dim=z_dim, side_info_dim=side_info_dim, rnn_dim=7)
    latent_values = {f"z_{step + 1}": torch.randn(batch_size, z_dim) for step in range(time_steps)}

    conditioned_model = poutine.condition(dmm.model, data=latent_values)
    trace_base = poutine.trace(conditioned_model).get_trace(
        observations,
        side_info,
        seq_lengths,
        1.0,
    )
    trace_alt = poutine.trace(conditioned_model).get_trace(
        observations_alt,
        side_info_alt,
        seq_lengths,
        1.0,
    )
    trace_base.compute_log_prob()
    trace_alt.compute_log_prob()

    assert trace_base.log_prob_sum() == trace_alt.log_prob_sum()


def test_dmm_runs_without_side_info() -> None:
    pyro.clear_param_store()
    batch_size = 2
    time_steps = 4
    obs_dim = 1
    z_dim = 3
    observations = torch.randn(batch_size, time_steps, obs_dim)

    dmm = DMM(obs_dim=obs_dim, z_dim=z_dim, rnn_dim=8)  # side_info_dim defaults to 0
    assert dmm.side_info_dim == 0

    guide_trace = poutine.trace(dmm.guide).get_trace(observations, None, None, 1.0)
    model_trace = poutine.trace(poutine.replay(dmm.model, trace=guide_trace)).get_trace(
        observations, None, None, 1.0
    )
    for trace in (guide_trace, model_trace):
        trace.compute_log_prob()

    assert model_trace.nodes["obs_y_4"]["value"].shape == (batch_size, obs_dim)
    assert torch.isfinite(torch.as_tensor(model_trace.log_prob_sum()))
    assert torch.isfinite(torch.as_tensor(guide_trace.log_prob_sum()))


def test_guide_handles_padding_wider_than_longest_sequence() -> None:
    pyro.clear_param_store()
    batch_size = 2
    time_steps = 5
    obs_dim = 1
    side_info_dim = 2
    z_dim = 3
    observations = torch.randn(batch_size, time_steps, obs_dim)
    side_info = torch.randn(batch_size, time_steps, side_info_dim)
    seq_lengths = torch.tensor([3, 2], dtype=torch.long)  # max length 3 < time_steps 5

    dmm = DMM(obs_dim=obs_dim, z_dim=z_dim, side_info_dim=side_info_dim, rnn_dim=8)

    guide_trace = poutine.trace(dmm.guide).get_trace(observations, side_info, seq_lengths, 1.0)
    guide_trace.compute_log_prob()

    assert guide_trace.nodes["z_1"]["value"].shape == (batch_size, z_dim)
    assert guide_trace.nodes[f"z_{time_steps}"]["value"].shape == (batch_size, z_dim)
    assert torch.isfinite(torch.as_tensor(guide_trace.log_prob_sum()))


def test_model_transition_gating_is_predictive() -> None:
    # Predictive (IOHMM-style) gating: x[k] governs the k -> k+1 transition, so it
    # parameterizes the latent that emits y[k+1] (site "z_{k+2}"), not the
    # contemporaneous latent for y[k] (site "z_{k+1}"). Conditioning all latents
    # isolates the transition-site log-probs.
    pyro.clear_param_store()
    torch.manual_seed(0)
    batch_size = 1
    time_steps = 5
    obs_dim = 1
    side_info_dim = 2
    z_dim = 3
    changed_index = 2  # interior bar k

    observations = torch.randn(batch_size, time_steps, obs_dim)
    side_info = torch.randn(batch_size, time_steps, side_info_dim)
    side_info_alt = side_info.clone()
    side_info_alt[:, changed_index, :] += torch.tensor([3.0, -3.0])

    dmm = DMM(obs_dim=obs_dim, z_dim=z_dim, side_info_dim=side_info_dim, rnn_dim=6)
    latent_values = {f"z_{step + 1}": torch.randn(batch_size, z_dim) for step in range(time_steps)}
    conditioned = poutine.condition(dmm.model, data=latent_values)

    def latent_site_log_probs(side: torch.Tensor) -> dict[str, float]:
        trace = poutine.trace(conditioned).get_trace(observations, side, None, 1.0)
        trace.compute_log_prob()
        return {
            name: float(node["log_prob"].sum().item())
            for name, node in trace.nodes.items()
            if name.startswith("z_")
        }

    base = latent_site_log_probs(side_info)
    alt = latent_site_log_probs(side_info_alt)

    predictive_site = f"z_{changed_index + 2}"  # latent emitting y[k+1]
    contemporaneous_site = f"z_{changed_index + 1}"  # latent emitting y[k]
    for name in base:
        if name == predictive_site:
            assert not np.isclose(base[name], alt[name]), f"{name} should change with x[k]."
        else:
            assert np.isclose(base[name], alt[name]), f"{name} should be unchanged by x[k]."
    # The contemporaneous latent in particular must not move (rules out the old lag).
    assert np.isclose(base[contemporaneous_site], alt[contemporaneous_site])


def test_kl_annealing_factor_matches_expected_schedule() -> None:
    kwargs = {
        "num_minibatches": 5,
        "min_annealing_factor": 0.2,
        "annealing_epochs": 20,
    }

    first = kl_annealing_factor(epoch=0, minibatch_index=0, **kwargs)
    midpoint = kl_annealing_factor(epoch=10, minibatch_index=0, **kwargs)
    schedule_end = kl_annealing_factor(epoch=19, minibatch_index=4, **kwargs)
    after_schedule = kl_annealing_factor(epoch=20, minibatch_index=0, **kwargs)
    monotone_prefix = [
        kl_annealing_factor(epoch=0, minibatch_index=index, **kwargs) for index in range(5)
    ]

    assert first == 0.2
    assert np.isclose(midpoint, 0.6)
    assert np.isclose(schedule_end, 0.992)
    assert after_schedule == 1.0
    assert monotone_prefix == sorted(monotone_prefix)


def test_fit_dmm_reduces_elbo_on_synthetic_batch() -> None:
    observations, side_info, seq_lengths = _synthetic_training_batch()
    config = DMMConfig(
        obs_dim=1,
        z_dim=3,
        emission_dim=6,
        transition_dim=6,
        rnn_dim=7,
        side_info_dim=2,
        num_epochs=8,
        mini_batch_size=4,
        learning_rate=1e-2,
        beta1=0.9,
        beta2=0.999,
        clip_norm=5.0,
        lr_decay=1.0,
        weight_decay=0.0,
        rnn_dropout_rate=0.0,
        min_annealing_factor=0.2,
        annealing_epochs=4,
        seed=17,
    )

    result = fit_dmm(config, observations, side_info, seq_lengths)

    assert len(result.loss_history) == config.num_epochs
    assert result.loss_history[-1] < result.loss_history[0] - 0.05


def test_fit_dmm_is_reproducible_within_tolerance_at_one_thread() -> None:
    observations, side_info, seq_lengths = _synthetic_training_batch()
    config = DMMConfig(
        obs_dim=1,
        z_dim=3,
        emission_dim=6,
        transition_dim=6,
        rnn_dim=7,
        side_info_dim=2,
        num_epochs=5,
        mini_batch_size=4,
        learning_rate=8e-3,
        beta1=0.9,
        beta2=0.999,
        clip_norm=5.0,
        lr_decay=1.0,
        weight_decay=0.0,
        rnn_dropout_rate=0.0,
        min_annealing_factor=0.2,
        annealing_epochs=3,
        seed=23,
    )
    atol = 1e-6
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        first = fit_dmm(config, observations, side_info, seq_lengths)
        second = fit_dmm(config, observations, side_info, seq_lengths)
    finally:
        torch.set_num_threads(previous_threads)

    np.testing.assert_allclose(first.loss_history, second.loss_history, atol=atol, rtol=0.0)


def test_expected_next_returns_from_dmm_returns_index_aligned_series(
    fitted_dmm_for_forecast: DMMFitResult,
) -> None:
    forecast_returns, forecast_side_info = _forecast_window_inputs()

    expected = expected_next_returns_from_dmm(
        fitted_dmm_for_forecast,
        forecast_returns,
        forecast_side_info,
    )

    assert isinstance(expected, pd.Series)
    assert len(expected) == len(forecast_returns)
    pd.testing.assert_index_equal(expected.index, forecast_returns.index)
    assert np.all(np.isfinite(expected.to_numpy()))


def test_expected_next_returns_from_dmm_is_causal_on_prefixes(
    fitted_dmm_for_forecast: DMMFitResult,
) -> None:
    forecast_returns, forecast_side_info = _forecast_window_inputs(length=7)
    prefix_length = 4

    full_expected = expected_next_returns_from_dmm(
        fitted_dmm_for_forecast,
        forecast_returns,
        forecast_side_info,
    )
    prefix_expected = expected_next_returns_from_dmm(
        fitted_dmm_for_forecast,
        forecast_returns.iloc[:prefix_length],
        forecast_side_info.iloc[:prefix_length],
    )

    pd.testing.assert_series_equal(full_expected.iloc[:prefix_length], prefix_expected)


def test_expected_next_returns_from_dmm_collapses_on_near_constant_series() -> None:
    constant_level = 0.0125
    forecast_model = _constant_forecast_dmm(constant=constant_level, side_info_dim=1)
    index = pd.date_range("2024-01-03 09:30:00", periods=5, freq="min")
    returns = pd.Series(
        constant_level + np.array([0.0, 1e-4, -1e-4, 5e-5, -5e-5]),
        index=index,
    )
    side_info = pd.Series(np.linspace(-0.5, 0.5, num=len(index)), index=index, name="feature")

    expected = expected_next_returns_from_dmm(forecast_model, returns, side_info)

    assert np.all(np.isfinite(expected.to_numpy()))
    np.testing.assert_allclose(
        expected.to_numpy(),
        np.full(len(expected), constant_level),
        atol=1e-6,
        rtol=0.0,
    )


def test_expected_next_returns_from_dmm_feeds_build_signal(
    fitted_dmm_for_forecast: DMMFitResult,
) -> None:
    forecast_returns, forecast_side_info = _forecast_window_inputs()
    expected = expected_next_returns_from_dmm(
        fitted_dmm_for_forecast,
        forecast_returns,
        forecast_side_info,
    )

    signal = build_signal(expected, policy="sign")

    assert isinstance(signal, pd.Series)
    pd.testing.assert_index_equal(signal.index, forecast_returns.index)
