"""Tests for the DMM latent-trajectory plotting helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("pyro")
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from hft_hmm.models.dmm import (  # noqa: E402
    DMMConfig,
    expected_next_returns_from_dmm,
    filtered_latent_mean_trajectory_from_dmm,
    fit_dmm,
)
from hft_hmm.selection.plots import plot_dmm_filtered_latent_trajectory  # noqa: E402


def _synthetic_training_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = 6
    time_steps = 6
    time_grid = torch.linspace(-1.0, 1.0, steps=time_steps, dtype=torch.float32)
    side_template = torch.stack((time_grid, time_grid.square()), dim=1)
    side_info = side_template.unsqueeze(0).repeat(batch_size, 1, 1)

    offsets = torch.linspace(-0.15, 0.15, steps=batch_size, dtype=torch.float32).view(
        batch_size,
        1,
        1,
    )
    observations = 0.3 * side_info[..., :1] - 0.12 * side_info[..., 1:] + offsets
    seq_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
    return observations, side_info, seq_lengths


def _forecast_window_inputs(length: int = 7) -> tuple[pd.Series, pd.DataFrame]:
    index = pd.date_range("2024-01-04 09:30:00", periods=length, freq="min")
    returns = pd.Series(np.linspace(-0.012, 0.016, num=length), index=index, name="returns")
    side_info = pd.DataFrame(
        {
            "level": np.linspace(-0.75, 0.9, num=length),
            "curvature": np.linspace(0.2, 1.1, num=length),
        },
        index=index,
    )
    return returns, side_info


def test_filtered_latent_trajectory_and_plot_smoke(tmp_path) -> None:
    observations, side_info, seq_lengths = _synthetic_training_batch()
    config = DMMConfig(
        obs_dim=1,
        z_dim=3,
        emission_dim=6,
        transition_dim=6,
        rnn_dim=7,
        side_info_dim=2,
        num_epochs=4,
        mini_batch_size=3,
        learning_rate=1e-2,
        beta1=0.9,
        beta2=0.999,
        clip_norm=5.0,
        lr_decay=1.0,
        weight_decay=0.0,
        rnn_dropout_rate=0.0,
        min_annealing_factor=0.2,
        annealing_epochs=2,
        seed=37,
    )

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        fitted = fit_dmm(config, observations, side_info, seq_lengths)
    finally:
        torch.set_num_threads(previous_threads)

    forecast_returns, forecast_side_info = _forecast_window_inputs()
    latent_trajectory = filtered_latent_mean_trajectory_from_dmm(
        fitted,
        forecast_returns,
        forecast_side_info,
    )
    expected = expected_next_returns_from_dmm(
        fitted,
        forecast_returns,
        forecast_side_info,
    )

    assert latent_trajectory.shape == (len(forecast_returns), config.z_dim)
    pd.testing.assert_index_equal(latent_trajectory.index, forecast_returns.index)
    assert np.isfinite(latent_trajectory.to_numpy()).all()

    fig, ax = plt.subplots()
    output_path = tmp_path / "dmm_latent_trajectory.png"
    try:
        returned_ax = plot_dmm_filtered_latent_trajectory(
            latent_trajectory,
            returns=forecast_returns,
            expected_next_returns=expected,
            ax=ax,
        )
        returned_ax.figure.savefig(output_path, dpi=100)
        assert returned_ax is ax
        assert len(ax.get_lines()) == config.z_dim
        assert len(returned_ax.figure.axes) == 2
        overlay_labels = {line.get_label() for line in returned_ax.figure.axes[1].get_lines()}
        assert {"returns", "expected_next_returns"} <= overlay_labels
    finally:
        plt.close(fig)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
