"""Deep Markov Model with continuous emissions and side-information-gated transitions.

This module implements the architecture layer of the Deep Markov Model (DMM)
described by Krishnan, Shalit, and Sontag (2017, §3-4), adapted for return
sequences with optional exogenous side information. The latent process is a
nonlinear Gaussian state-space model:

``z_t | z_{t-1}, x_{t-1} ~ Normal(mu_trans(z_{t-1}, x_{t-1}), sigma_trans(z_{t-1}, x_{t-1}))``

``y_t | z_t ~ Normal(mu_emit(z_t), sigma_emit(z_t))``

``q(z_t | z_{t-1}, y_{t:T}) = Normal(mu_q(z_{t-1}, h_t), sigma_q(z_{t-1}, h_t))``

where ``h_t`` is produced by a reverse-time recurrent network over the observed
return sequence. Side information gates the transition **predictively**, IOHMM-style:
the exogenous row ``x_{t-1}`` governs the ``(t-1) -> t`` transition (the first
latent has no incoming row and is left ungated). This matches the bucketed/continuous
IOHMM convention where ``A(x_t)`` parameterizes the ``t -> t+1`` step, so the causal
one-step-ahead forecast in :func:`expected_next_returns_from_dmm` (which uses
``x_t`` for the ``t -> t+1`` transition) is consistent with training. When no side
information is provided the model falls back to a zero vector, so the same module can
be used in ablations without exogenous features.

Variable-length mini-batches are represented as padded tensors plus
``seq_lengths``. The model and guide preserve the P0 spike's masking strategy so
padded timesteps do not contribute to latent or observation log-probability.
The ``fit_dmm()`` entry point seeds Python's ``random`` module, NumPy, Torch,
and Pyro from ``DMMConfig.seed``. The reproducibility contract for the SVI path
assumes a single CPU Torch thread: the tests pin ``torch.set_num_threads(1)``
and assert repeated seeded fits match within ``1e-6`` absolute tolerance.
Across other BLAS threading setups the same seed can still drift slightly, so
the DMM is documented as reproducible within tolerance rather than bit-for-bit
identical across all environments.

References: Krishnan et al. (2017) §3-4
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from pyro.distributions import TorchDistribution
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from hft_hmm.core import ENGINEERING_APPROXIMATION, PaperReference, reference

__category__: Final[str] = ENGINEERING_APPROXIMATION
DMM_REFERENCE: Final[PaperReference] = reference(
    "§3-4", "deep Markov model with structured inference"
)

DEFAULT_OBS_DIM: Final[int] = 1
DEFAULT_LATENT_DIM: Final[int] = 16
DEFAULT_EMISSION_DIM: Final[int] = 32
DEFAULT_TRANSITION_DIM: Final[int] = 32
DEFAULT_RNN_DIM: Final[int] = 32
DEFAULT_NUM_LAYERS: Final[int] = 1
DEFAULT_DROPOUT_RATE: Final[float] = 0.0
DEFAULT_MIN_SCALE: Final[float] = 1e-4
DEFAULT_NUM_EPOCHS: Final[int] = 150
DEFAULT_MINI_BATCH_SIZE: Final[int] = 20
DEFAULT_LEARNING_RATE: Final[float] = 3e-4
DEFAULT_BETA1: Final[float] = 0.96
DEFAULT_BETA2: Final[float] = 0.999
DEFAULT_CLIP_NORM: Final[float] = 10.0
DEFAULT_LR_DECAY: Final[float] = 0.99996
DEFAULT_WEIGHT_DECAY: Final[float] = 2.0
DEFAULT_MIN_ANNEALING_FACTOR: Final[float] = 0.2
DEFAULT_ANNEALING_EPOCHS: Final[int] = 1000
DEFAULT_SEED: Final[int] = 54
SINGLE_THREAD_REPRO_TORCH_THREADS: Final[int] = 1
SINGLE_THREAD_REPRO_ABS_TOLERANCE: Final[float] = 1e-6


@dataclass(frozen=True)
class DMMConfig:
    """Hyperparameters for DMM architecture and SVI training.

    The architecture fields map directly to :class:`DMM`. Training defaults are
    lifted from the merged P0 Pyro spike, with ``rnn_dropout_rate=0.1`` as the
    recommended optimization-time setting from that script.

    Reproducibility note: :func:`fit_dmm` seeds Python's ``random`` module,
    NumPy, Torch, and Pyro from ``seed`` before constructing the model and SVI
    objects. The documented guarantee assumes ``torch.set_num_threads(1)``:
    repeated seeded fits should then agree within
    ``SINGLE_THREAD_REPRO_ABS_TOLERANCE`` absolute tolerance. Callers should
    still treat the fit as reproducible within tolerance rather than exactly
    identical across machines or BLAS threading setups.

    References: Krishnan et al. (2017) §3-4
    """

    obs_dim: int = DEFAULT_OBS_DIM
    z_dim: int = DEFAULT_LATENT_DIM
    emission_dim: int = DEFAULT_EMISSION_DIM
    transition_dim: int = DEFAULT_TRANSITION_DIM
    rnn_dim: int = DEFAULT_RNN_DIM
    side_info_dim: int = 0
    num_layers: int = DEFAULT_NUM_LAYERS
    rnn_dropout_rate: float = 0.1
    min_scale: float = DEFAULT_MIN_SCALE
    num_epochs: int = DEFAULT_NUM_EPOCHS
    mini_batch_size: int = DEFAULT_MINI_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    beta1: float = DEFAULT_BETA1
    beta2: float = DEFAULT_BETA2
    clip_norm: float = DEFAULT_CLIP_NORM
    lr_decay: float = DEFAULT_LR_DECAY
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    min_annealing_factor: float = DEFAULT_MIN_ANNEALING_FACTOR
    annealing_epochs: int = DEFAULT_ANNEALING_EPOCHS
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        _validate_positive_int(self.obs_dim, "obs_dim")
        _validate_positive_int(self.z_dim, "z_dim")
        _validate_positive_int(self.emission_dim, "emission_dim")
        _validate_positive_int(self.transition_dim, "transition_dim")
        _validate_positive_int(self.rnn_dim, "rnn_dim")
        _validate_nonnegative_int(self.side_info_dim, "side_info_dim")
        _validate_positive_int(self.num_layers, "num_layers")
        dropout = _validate_nonnegative_float(self.rnn_dropout_rate, "rnn_dropout_rate")
        if dropout >= 1.0:
            raise ValueError(f"rnn_dropout_rate must be < 1.0, got {dropout}.")
        min_scale = _validate_positive_float(self.min_scale, "min_scale")

        _validate_positive_int(self.num_epochs, "num_epochs")
        _validate_positive_int(self.mini_batch_size, "mini_batch_size")
        learning_rate = _validate_positive_float(self.learning_rate, "learning_rate")
        beta1 = _validate_probability_open_interval(self.beta1, "beta1")
        beta2 = _validate_probability_open_interval(self.beta2, "beta2")
        clip_norm = _validate_positive_float(self.clip_norm, "clip_norm")
        lr_decay = _validate_positive_float(self.lr_decay, "lr_decay")
        weight_decay = _validate_nonnegative_float(self.weight_decay, "weight_decay")
        min_annealing_factor = _validate_probability_closed_interval(
            self.min_annealing_factor,
            "min_annealing_factor",
        )
        _validate_nonnegative_int(self.annealing_epochs, "annealing_epochs")
        _validate_nonnegative_int(self.seed, "seed")

        object.__setattr__(self, "rnn_dropout_rate", dropout)
        object.__setattr__(self, "min_scale", min_scale)
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "beta1", beta1)
        object.__setattr__(self, "beta2", beta2)
        object.__setattr__(self, "clip_norm", clip_norm)
        object.__setattr__(self, "lr_decay", lr_decay)
        object.__setattr__(self, "weight_decay", weight_decay)
        object.__setattr__(self, "min_annealing_factor", min_annealing_factor)


class Emitter(nn.Module):
    """Map latent states to a diagonal Normal emission distribution.

    Args:
        obs_dim: Observation dimensionality.
        z_dim: Latent-state dimensionality.
        emission_dim: Hidden width of the emission MLP.
        min_scale: Positive floor added after ``softplus`` for numerical
            stability.

    References: Krishnan et al. (2017) §3-4
    """

    def __init__(
        self,
        obs_dim: int = DEFAULT_OBS_DIM,
        z_dim: int = DEFAULT_LATENT_DIM,
        emission_dim: int = DEFAULT_EMISSION_DIM,
        min_scale: float = DEFAULT_MIN_SCALE,
    ) -> None:
        super().__init__()
        _validate_positive_int(obs_dim, "obs_dim")
        _validate_positive_int(z_dim, "z_dim")
        _validate_positive_int(emission_dim, "emission_dim")
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.emission_dim = emission_dim
        self.min_scale = _validate_positive_float(min_scale, "min_scale")

        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_loc = nn.Linear(emission_dim, obs_dim)
        self.lin_hidden_to_scale = nn.Linear(emission_dim, obs_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loc, scale)`` for ``y_t | z_t``.

        Args:
            z_t: Tensor with trailing shape ``(..., z_dim)``.

        Returns:
            A pair of tensors with trailing shape ``(..., obs_dim)``.

        References: Krishnan et al. (2017) §3-4
        """
        _validate_last_dim(z_t, self.z_dim, "z_t")
        hidden = self.relu(self.lin_z_to_hidden(z_t))
        hidden = self.relu(self.lin_hidden_to_hidden(hidden))
        loc = self.lin_hidden_to_loc(hidden)
        scale = self.softplus(self.lin_hidden_to_scale(hidden)) + self.min_scale
        return loc, scale


class GatedTransition(nn.Module):
    """Map ``(z_{t-1}, x_t)`` to a diagonal Normal transition distribution.

    The network follows the gating pattern from the Pyro DMM example, extended
    so both the gate and proposal can depend on exogenous side information
    ``x_t``. If ``x_t`` is omitted, a zero vector is substituted.

    Args:
        z_dim: Latent-state dimensionality.
        transition_dim: Hidden width of the transition MLPs.
        side_info_dim: Exogenous feature dimensionality. Set to ``0`` to build a
            transition that ignores side information.
        min_scale: Positive floor added after ``softplus`` for numerical
            stability.

    References: Krishnan et al. (2017) §3-4
    """

    def __init__(
        self,
        z_dim: int = DEFAULT_LATENT_DIM,
        transition_dim: int = DEFAULT_TRANSITION_DIM,
        side_info_dim: int = 0,
        min_scale: float = DEFAULT_MIN_SCALE,
    ) -> None:
        super().__init__()
        _validate_positive_int(z_dim, "z_dim")
        _validate_positive_int(transition_dim, "transition_dim")
        _validate_nonnegative_int(side_info_dim, "side_info_dim")
        self.z_dim = z_dim
        self.transition_dim = transition_dim
        self.side_info_dim = side_info_dim
        self.min_scale = _validate_positive_float(min_scale, "min_scale")

        transition_input_dim = z_dim + side_info_dim
        self.lin_gate_input_to_hidden = nn.Linear(transition_input_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_input_to_hidden = nn.Linear(transition_input_dim, transition_dim)
        self.lin_proposed_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_scale_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(
        self, z_t_1: torch.Tensor, x_t: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loc, scale)`` for ``z_t | z_{t-1}, x_t``.

        Args:
            z_t_1: Tensor with trailing shape ``(..., z_dim)``.
            x_t: Optional tensor with trailing shape ``(..., side_info_dim)``.
                When omitted, a zero vector is used.

        Returns:
            A pair of tensors with trailing shape ``(..., z_dim)``.

        References: Krishnan et al. (2017) §3-4
        """
        _validate_last_dim(z_t_1, self.z_dim, "z_t_1")
        side_info = _coerce_side_info_step(
            x_t,
            batch_shape=z_t_1.shape[:-1],
            side_info_dim=self.side_info_dim,
            reference=z_t_1,
        )
        transition_input = torch.cat((z_t_1, side_info), dim=-1)

        gate_hidden = self.relu(self.lin_gate_input_to_hidden(transition_input))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(gate_hidden))
        proposed_hidden = self.relu(self.lin_proposed_input_to_hidden(transition_input))
        proposed_mean = self.lin_proposed_hidden_to_z(proposed_hidden)
        loc = (1.0 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        scale = self.softplus(self.lin_scale_hidden_to_z(proposed_hidden)) + self.min_scale
        return loc, scale


class Combiner(nn.Module):
    """Structured inference network for ``q(z_t | z_{t-1}, y_{t:T})``.

    Args:
        z_dim: Latent-state dimensionality.
        rnn_dim: Hidden width of the reverse-time RNN output.
        min_scale: Positive floor added after ``softplus`` for numerical
            stability.

    References: Krishnan et al. (2017) §3-4
    """

    def __init__(
        self,
        z_dim: int = DEFAULT_LATENT_DIM,
        rnn_dim: int = DEFAULT_RNN_DIM,
        min_scale: float = DEFAULT_MIN_SCALE,
    ) -> None:
        super().__init__()
        _validate_positive_int(z_dim, "z_dim")
        _validate_positive_int(rnn_dim, "rnn_dim")
        self.z_dim = z_dim
        self.rnn_dim = rnn_dim
        self.min_scale = _validate_positive_float(min_scale, "min_scale")

        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(
        self, z_t_1: torch.Tensor, h_rnn: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loc, scale)`` for the guide's latent site at time ``t``.

        Args:
            z_t_1: Tensor with trailing shape ``(..., z_dim)``.
            h_rnn: Tensor with trailing shape ``(..., rnn_dim)``.

        Returns:
            A pair of tensors with trailing shape ``(..., z_dim)``.

        References: Krishnan et al. (2017) §3-4
        """
        _validate_last_dim(z_t_1, self.z_dim, "z_t_1")
        _validate_last_dim(h_rnn, self.rnn_dim, "h_rnn")
        hidden = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        loc = self.lin_hidden_to_loc(hidden)
        scale = self.softplus(self.lin_hidden_to_scale(hidden)) + self.min_scale
        return loc, scale


class DMM(nn.Module):
    """Continuous-emission DMM with optional side-information-gated transitions.

    Args:
        obs_dim: Observation dimensionality. Defaults to univariate returns.
        z_dim: Latent-state dimensionality.
        emission_dim: Hidden width of the emission MLP.
        transition_dim: Hidden width of the transition MLPs.
        rnn_dim: Hidden width of the structured-inference RNN.
        side_info_dim: Exogenous feature dimensionality for ``x_t``.
        num_layers: Number of recurrent layers in the guide RNN.
        rnn_dropout_rate: Dropout between guide-RNN layers. Ignored when
            ``num_layers == 1``.
        min_scale: Positive floor added after ``softplus`` for all Gaussian
            scales.

    References: Krishnan et al. (2017) §3-4
    """

    def __init__(
        self,
        obs_dim: int = DEFAULT_OBS_DIM,
        z_dim: int = DEFAULT_LATENT_DIM,
        emission_dim: int = DEFAULT_EMISSION_DIM,
        transition_dim: int = DEFAULT_TRANSITION_DIM,
        rnn_dim: int = DEFAULT_RNN_DIM,
        side_info_dim: int = 0,
        num_layers: int = DEFAULT_NUM_LAYERS,
        rnn_dropout_rate: float = DEFAULT_DROPOUT_RATE,
        min_scale: float = DEFAULT_MIN_SCALE,
    ) -> None:
        super().__init__()
        _validate_positive_int(obs_dim, "obs_dim")
        _validate_positive_int(z_dim, "z_dim")
        _validate_positive_int(emission_dim, "emission_dim")
        _validate_positive_int(transition_dim, "transition_dim")
        _validate_positive_int(rnn_dim, "rnn_dim")
        _validate_nonnegative_int(side_info_dim, "side_info_dim")
        _validate_positive_int(num_layers, "num_layers")
        dropout = _validate_nonnegative_float(rnn_dropout_rate, "rnn_dropout_rate")
        min_scale_value = _validate_positive_float(min_scale, "min_scale")

        if dropout >= 1.0:
            raise ValueError(f"rnn_dropout_rate must be < 1.0, got {dropout}.")

        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.emission_dim = emission_dim
        self.transition_dim = transition_dim
        self.rnn_dim = rnn_dim
        self.side_info_dim = side_info_dim
        self.num_layers = num_layers
        self.rnn_dropout_rate = dropout
        self.min_scale = min_scale_value

        self.emitter = Emitter(
            obs_dim=obs_dim,
            z_dim=z_dim,
            emission_dim=emission_dim,
            min_scale=min_scale_value,
        )
        self.trans = GatedTransition(
            z_dim=z_dim,
            transition_dim=transition_dim,
            side_info_dim=side_info_dim,
            min_scale=min_scale_value,
        )
        self.combiner = Combiner(z_dim=z_dim, rnn_dim=rnn_dim, min_scale=min_scale_value)
        effective_dropout = 0.0 if num_layers == 1 else dropout
        self.rnn = nn.RNN(
            input_size=obs_dim,
            hidden_size=rnn_dim,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
            num_layers=num_layers,
            dropout=effective_dropout,
        )
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        self.h_0 = nn.Parameter(torch.zeros(num_layers, 1, rnn_dim))

    def model(
        self,
        observations: torch.Tensor,
        side_info: torch.Tensor | None = None,
        seq_lengths: torch.Tensor | None = None,
        annealing_factor: float = 1.0,
    ) -> None:
        """Pyro model for padded return sequences.

        Args:
            observations: Padded observation tensor with shape
                ``(batch, time, obs_dim)``.
            side_info: Optional aligned side-information tensor with shape
                ``(batch, time, side_info_dim)``.
            seq_lengths: Optional valid lengths with shape ``(batch,)``. When
                omitted, all sequences are treated as full length.
            annealing_factor: Multiplicative scale applied to latent KL terms.

        References: Krishnan et al. (2017) §3-4
        """
        (
            observations_prepared,
            side_info_prepared,
            mask,
            seq_lengths_prepared,
        ) = self._prepare_inputs(
            observations=observations,
            side_info=side_info,
            seq_lengths=seq_lengths,
        )
        del seq_lengths_prepared
        time_steps = observations_prepared.size(1)
        pyro.module("dmm", self)
        z_prev = self.z_0.expand(observations_prepared.size(0), self.z_dim)

        with pyro.plate("z_minibatch", observations_prepared.size(0)):
            for t in pyro.markov(range(time_steps)):
                # Predictive (IOHMM-style) gating: the exogenous row x[t-1]
                # governs the (t-1)->t transition, so the causal one-step-ahead
                # forecast (which uses x[t] for the t->t+1 transition) is
                # consistent with training. The first latent has no incoming
                # exogenous row (initial transition) and is left ungated.
                if self.side_info_dim > 0 and t > 0:
                    side_info_step = side_info_prepared[:, t - 1, :]
                else:
                    side_info_step = None
                z_loc, z_scale = self.trans(z_prev, side_info_step)
                with poutine.scale(scale=float(annealing_factor)):
                    z_t = pyro.sample(
                        f"z_{t + 1}",
                        self._masked_diagonal_normal(
                            z_loc,
                            z_scale,
                            mask[:, t],
                        ),
                    )
                emission_loc, emission_scale = self.emitter(z_t)
                pyro.sample(
                    f"obs_y_{t + 1}",
                    self._masked_diagonal_normal(
                        emission_loc,
                        emission_scale,
                        mask[:, t],
                    ),
                    obs=observations_prepared[:, t, :],
                )
                z_prev = z_t

    def guide(
        self,
        observations: torch.Tensor,
        side_info: torch.Tensor | None = None,
        seq_lengths: torch.Tensor | None = None,
        annealing_factor: float = 1.0,
    ) -> None:
        """Pyro guide for padded return sequences.

        The guide currently conditions on observations through the reverse-time
        RNN and accepts ``side_info`` only to keep its call signature aligned
        with :meth:`model`.

        Args:
            observations: Padded observation tensor with shape
                ``(batch, time, obs_dim)``.
            side_info: Optional aligned side-information tensor with shape
                ``(batch, time, side_info_dim)``.
            seq_lengths: Optional valid lengths with shape ``(batch,)``. When
                omitted, all sequences are treated as full length.
            annealing_factor: Multiplicative scale applied to latent KL terms.

        References: Krishnan et al. (2017) §3-4
        """
        (
            observations_prepared,
            side_info_prepared,
            mask,
            seq_lengths_prepared,
        ) = self._prepare_inputs(
            observations=observations,
            side_info=side_info,
            seq_lengths=seq_lengths,
        )
        del side_info_prepared
        time_steps = observations_prepared.size(1)
        pyro.module("dmm", self)
        h_0_contiguous = self.h_0.expand(
            self.num_layers,
            observations_prepared.size(0),
            self.rnn_dim,
        )
        h_0_contiguous = h_0_contiguous.contiguous()
        rnn_output, _ = self.rnn(
            _pack_reversed_sequences(observations_prepared, seq_lengths_prepared), h_0_contiguous
        )
        rnn_output_padded = _pad_and_reverse(
            rnn_output, seq_lengths_prepared, total_length=time_steps
        )
        z_prev = self.z_q_0.expand(observations_prepared.size(0), self.z_dim)

        with pyro.plate("z_minibatch", observations_prepared.size(0)):
            for t in pyro.markov(range(time_steps)):
                z_loc, z_scale = self.combiner(z_prev, rnn_output_padded[:, t, :])
                with poutine.scale(scale=float(annealing_factor)):
                    z_t = pyro.sample(
                        f"z_{t + 1}",
                        self._masked_diagonal_normal(
                            z_loc,
                            z_scale,
                            mask[:, t],
                        ),
                    )
                z_prev = z_t

    def _prepare_inputs(
        self,
        *,
        observations: torch.Tensor,
        side_info: torch.Tensor | None,
        seq_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observations_prepared = _coerce_observations(observations, obs_dim=self.obs_dim)
        batch_size, time_steps, _ = observations_prepared.shape
        seq_lengths_prepared = _coerce_seq_lengths(
            seq_lengths,
            batch_size=batch_size,
            time_steps=time_steps,
            device=observations_prepared.device,
        )
        side_info_prepared = _coerce_side_info_batch(
            side_info,
            batch_size=batch_size,
            time_steps=time_steps,
            side_info_dim=self.side_info_dim,
            device=observations_prepared.device,
            dtype=observations_prepared.dtype,
        )
        mask = _sequence_mask(seq_lengths_prepared, time_steps=time_steps)
        return observations_prepared, side_info_prepared, mask, seq_lengths_prepared

    @staticmethod
    def _masked_diagonal_normal(
        loc: torch.Tensor,
        scale: torch.Tensor,
        mask_t: torch.Tensor,
    ) -> TorchDistribution:
        distribution = dist.Normal(loc, scale)
        return cast(
            TorchDistribution,
            distribution.mask(mask_t.unsqueeze(-1).to(dtype=torch.bool)).to_event(1),
        )


@dataclass(frozen=True)
class DMMFitResult:
    """Fitted DMM and per-epoch normalized loss history.

    ``loss_history`` stores one normalized negative-ELBO value per training
    epoch, scaled by the total number of observed (unpadded) timesteps in the
    training batch. ``model`` is the fitted :class:`DMM` instance returned by
    :func:`fit_dmm`.

    References: Krishnan et al. (2017) §3-4
    """

    config: DMMConfig
    model: DMM
    loss_history: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.config, DMMConfig):
            raise TypeError(f"config must be a DMMConfig, got {type(self.config).__name__}.")
        if not isinstance(self.model, DMM):
            raise TypeError(f"model must be a DMM, got {type(self.model).__name__}.")
        normalized_history = tuple(float(value) for value in self.loss_history)
        if len(normalized_history) != self.config.num_epochs:
            raise ValueError(
                "loss_history must contain one value per epoch; "
                f"expected {self.config.num_epochs}, got {len(normalized_history)}."
            )
        if not all(np.isfinite(normalized_history)):
            raise ValueError("loss_history must contain only finite values.")
        object.__setattr__(self, "loss_history", normalized_history)


def kl_annealing_factor(
    *,
    epoch: int,
    minibatch_index: int,
    num_minibatches: int,
    min_annealing_factor: float,
    annealing_epochs: int,
) -> float:
    """Return the KL weight for one zero-based SVI minibatch step.

    The schedule ramps linearly from ``min_annealing_factor`` on the first
    annealed step toward 1.0 over ``annealing_epochs * num_minibatches``
    zero-based steps, then clamps to exactly 1.0 after the annealing window.

    References: Krishnan et al. (2017) §3-4
    """

    _validate_nonnegative_int(epoch, "epoch")
    _validate_nonnegative_int(minibatch_index, "minibatch_index")
    _validate_positive_int(num_minibatches, "num_minibatches")
    if minibatch_index >= num_minibatches:
        raise ValueError(
            "minibatch_index must be strictly less than num_minibatches; "
            f"got {minibatch_index} and {num_minibatches}."
        )
    min_factor = _validate_probability_closed_interval(
        min_annealing_factor,
        "min_annealing_factor",
    )
    _validate_nonnegative_int(annealing_epochs, "annealing_epochs")

    total_annealing_steps = annealing_epochs * num_minibatches
    if total_annealing_steps == 0:
        return 1.0

    global_step = epoch * num_minibatches + minibatch_index
    if global_step >= total_annealing_steps:
        return 1.0

    factor = min_factor + (1.0 - min_factor) * (float(global_step) / float(total_annealing_steps))
    return min(1.0, factor)


def fit_dmm(
    config: DMMConfig,
    observations: torch.Tensor,
    side_info: torch.Tensor | None = None,
    seq_lengths: torch.Tensor | None = None,
) -> DMMFitResult:
    """Fit a DMM with SVI and KL annealing on padded sequences.

    Args:
        config: Architecture and training hyperparameters.
        observations: Padded observation tensor with shape
            ``(batch, time, obs_dim)``.
        side_info: Optional aligned side-information tensor with shape
            ``(batch, time, side_info_dim)``.
        seq_lengths: Optional valid lengths with shape ``(batch,)``. When
            omitted, every sequence is treated as full length.

    Returns:
        A :class:`DMMFitResult` containing the fitted model and one normalized
        negative-ELBO value per epoch.

    References: Krishnan et al. (2017) §3-4
    """

    if not isinstance(config, DMMConfig):
        raise TypeError(f"config must be a DMMConfig, got {type(config).__name__}.")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    pyro.set_rng_seed(config.seed)
    pyro.clear_param_store()

    observations_prepared = _coerce_observations(observations, obs_dim=config.obs_dim)
    batch_size, time_steps, _ = observations_prepared.shape
    _validate_positive_int(batch_size, "batch_size")
    _validate_positive_int(time_steps, "time_steps")

    seq_lengths_prepared = _coerce_seq_lengths(
        seq_lengths,
        batch_size=batch_size,
        time_steps=time_steps,
        device=observations_prepared.device,
    )
    if torch.any(seq_lengths_prepared <= 0):
        raise ValueError("fit_dmm requires every seq_length to be at least 1.")

    side_info_prepared: torch.Tensor | None
    if config.side_info_dim > 0 or side_info is not None:
        side_info_prepared = _coerce_side_info_batch(
            side_info,
            batch_size=batch_size,
            time_steps=time_steps,
            side_info_dim=config.side_info_dim,
            device=observations_prepared.device,
            dtype=observations_prepared.dtype,
        )
    else:
        side_info_prepared = None

    dmm = DMM(
        obs_dim=config.obs_dim,
        z_dim=config.z_dim,
        emission_dim=config.emission_dim,
        transition_dim=config.transition_dim,
        rnn_dim=config.rnn_dim,
        side_info_dim=config.side_info_dim,
        num_layers=config.num_layers,
        rnn_dropout_rate=config.rnn_dropout_rate,
        min_scale=config.min_scale,
    )
    dmm = cast(DMM, dmm.to(device=observations_prepared.device, dtype=observations_prepared.dtype))
    dmm.train()

    optimizer = ClippedAdam(
        {
            "lr": config.learning_rate,
            "betas": (config.beta1, config.beta2),
            "clip_norm": config.clip_norm,
            "lrd": config.lr_decay,
            "weight_decay": config.weight_decay,
        }
    )
    svi = SVI(dmm.model, dmm.guide, optimizer, loss=Trace_ELBO())

    num_sequences = batch_size
    num_minibatches = (num_sequences + config.mini_batch_size - 1) // config.mini_batch_size
    train_time_slices = float(seq_lengths_prepared.sum().item())
    loss_history: list[float] = []

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        shuffled_indices = np.arange(num_sequences, dtype=np.int64)
        np.random.shuffle(shuffled_indices)

        for minibatch_index in range(num_minibatches):
            start = minibatch_index * config.mini_batch_size
            stop = min((minibatch_index + 1) * config.mini_batch_size, num_sequences)
            batch_indices = torch.as_tensor(
                shuffled_indices[start:stop],
                dtype=torch.long,
                device=observations_prepared.device,
            )
            observation_batch = observations_prepared.index_select(0, batch_indices)
            seq_length_batch = seq_lengths_prepared.index_select(0, batch_indices)
            side_info_batch = (
                side_info_prepared.index_select(0, batch_indices)
                if side_info_prepared is not None
                else None
            )
            annealing_factor = kl_annealing_factor(
                epoch=epoch,
                minibatch_index=minibatch_index,
                num_minibatches=num_minibatches,
                min_annealing_factor=config.min_annealing_factor,
                annealing_epochs=config.annealing_epochs,
            )
            epoch_loss += float(
                svi.step(
                    observation_batch,
                    side_info_batch,
                    seq_length_batch,
                    annealing_factor,
                )
            )

        loss_history.append(epoch_loss / train_time_slices)

    dmm.eval()
    return DMMFitResult(config=config, model=dmm, loss_history=tuple(loss_history))


def expected_next_returns_from_dmm(
    fitted: DMM | DMMFitResult,
    returns: pd.Series | np.ndarray,
    side_info: pd.Series | pd.DataFrame | np.ndarray | None = None,
) -> pd.Series:
    """Return the causal DMM forecast surface ``E[Δy_{t+1} | Δy_{1:t}]``.

    The guide is a reverse-time smoother, so this helper avoids look-ahead by
    running the guide on each expanding prefix ``Δy_{1:t}`` separately,
    propagating the deterministic guide mean at ``z_t`` one step ahead through
    the transition network, and mapping that next-latent mean through the
    emission network. The one-step-ahead transition uses ``side_info[t]``: under
    the model's predictive (IOHMM-style) gating, ``x_t`` is exactly the exogenous
    row that governs the ``t -> t+1`` transition, so this forecast is consistent
    with how the DMM was trained (no one-step input lag). Only causally available
    rows are read; the forecast for bar ``t + 1`` uses ``x_t``, never a future row.

    Returns a finite ``pd.Series`` aligned to the input return index (or a
    positional ``RangeIndex`` when ``returns`` is a raw numpy array).

    References: Krishnan et al. (2017) §3-4
    """

    dmm = _coerce_dmm_forecast_model(fitted)
    if dmm.obs_dim != 1:
        raise ValueError(
            "expected_next_returns_from_dmm currently supports only obs_dim=1 "
            f"models, got obs_dim={dmm.obs_dim}."
        )

    return_values, return_index = _coerce_return_forecast_series(returns)
    n_obs = return_values.shape[0]
    parameter = next(dmm.parameters())
    device = parameter.device
    dtype = parameter.dtype

    observations = torch.as_tensor(
        return_values.reshape(1, n_obs, dmm.obs_dim),
        device=device,
        dtype=dtype,
    )
    side_info_tensor = _coerce_forecast_side_info(
        side_info,
        return_index=return_index,
        n_obs=n_obs,
        side_info_dim=dmm.side_info_dim,
        device=device,
        dtype=dtype,
    )

    was_training = dmm.training
    dmm.eval()
    expected = np.empty(n_obs, dtype=float)

    try:
        with torch.no_grad():
            filtered_latent_means = _filtered_latent_mean_trajectory_tensor(
                dmm,
                observations=observations,
                side_info=side_info_tensor,
            )
            for t in range(n_obs):
                transition_side_info = (
                    side_info_tensor[:, t, :] if side_info_tensor is not None else None
                )
                next_latent_loc, _ = dmm.trans(
                    filtered_latent_means[t, :].unsqueeze(0),
                    transition_side_info,
                )
                emission_loc, _ = dmm.emitter(next_latent_loc)
                expected[t] = float(emission_loc.squeeze().item())
    finally:
        dmm.train(was_training)

    if not np.all(np.isfinite(expected)):
        raise ValueError("DMM expected_next_returns must contain only finite values.")
    return pd.Series(expected, index=return_index, name="expected_next_returns")


def filtered_latent_mean_trajectory_from_dmm(
    fitted: DMM | DMMFitResult,
    returns: pd.Series | np.ndarray,
    side_info: pd.Series | pd.DataFrame | np.ndarray | None = None,
) -> pd.DataFrame:
    """Return the causal filtered latent-mean trajectory for a forecast window.

    The DMM guide is a reverse-time smoother, so recovering a filtered
    ``E[z_t | Δy_{1:t}]`` trajectory without look-ahead requires evaluating the
    guide on each expanding prefix ``Δy_{1:t}`` separately. This helper reuses
    the same prefix filter as :func:`expected_next_returns_from_dmm` and returns
    a finite, index-aligned dataframe with one row per observed bar and one
    column per latent dimension.

    References: Krishnan et al. (2017) §3-4
    """

    dmm = _coerce_dmm_forecast_model(fitted)
    if dmm.obs_dim != 1:
        raise ValueError(
            "filtered_latent_mean_trajectory_from_dmm currently supports only obs_dim=1 "
            f"models, got obs_dim={dmm.obs_dim}."
        )
    return_values, return_index = _coerce_return_forecast_series(returns)
    n_obs = return_values.shape[0]
    parameter = next(dmm.parameters())
    device = parameter.device
    dtype = parameter.dtype

    observations = torch.as_tensor(
        return_values.reshape(1, n_obs, dmm.obs_dim),
        device=device,
        dtype=dtype,
    )
    side_info_tensor = _coerce_forecast_side_info(
        side_info,
        return_index=return_index,
        n_obs=n_obs,
        side_info_dim=dmm.side_info_dim,
        device=device,
        dtype=dtype,
    )

    was_training = dmm.training
    dmm.eval()
    try:
        with torch.no_grad():
            filtered_latent_means = _filtered_latent_mean_trajectory_tensor(
                dmm,
                observations=observations,
                side_info=side_info_tensor,
            )
    finally:
        dmm.train(was_training)

    latent_values = filtered_latent_means.detach().cpu().numpy()
    if not np.all(np.isfinite(latent_values)):
        raise ValueError("DMM filtered latent trajectory must contain only finite values.")
    columns = [f"latent_{latent_dim + 1}" for latent_dim in range(dmm.z_dim)]
    return pd.DataFrame(latent_values, index=return_index, columns=columns)


def _coerce_observations(observations: torch.Tensor, *, obs_dim: int) -> torch.Tensor:
    if observations.ndim != 3:
        raise ValueError(
            "observations must have shape (batch, time, obs_dim), "
            f"got tensor with shape {tuple(observations.shape)}."
        )
    if observations.shape[2] != obs_dim:
        raise ValueError(
            f"observations last dimension must equal obs_dim={obs_dim}, "
            f"got {observations.shape[2]}."
        )
    if not torch.is_floating_point(observations):
        raise TypeError("observations must be a floating-point tensor.")
    return observations


def _coerce_seq_lengths(
    seq_lengths: torch.Tensor | None,
    *,
    batch_size: int,
    time_steps: int,
    device: torch.device,
) -> torch.Tensor:
    if seq_lengths is None:
        return torch.full((batch_size,), time_steps, dtype=torch.long, device=device)
    if seq_lengths.ndim != 1 or seq_lengths.shape[0] != batch_size:
        raise ValueError(
            f"seq_lengths must have shape ({batch_size},), got {tuple(seq_lengths.shape)}."
        )
    lengths = seq_lengths.to(device=device, dtype=torch.long)
    if torch.any(lengths < 0):
        raise ValueError("seq_lengths must be nonnegative.")
    if torch.any(lengths > time_steps):
        raise ValueError(f"seq_lengths entries must be <= time_steps={time_steps}.")
    return lengths


def _coerce_side_info_batch(
    side_info: torch.Tensor | None,
    *,
    batch_size: int,
    time_steps: int,
    side_info_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if side_info is None:
        return torch.zeros(batch_size, time_steps, side_info_dim, device=device, dtype=dtype)
    if side_info_dim == 0:
        raise ValueError("side_info was provided but side_info_dim=0 for this DMM.")
    if side_info.ndim != 3:
        raise ValueError(
            "side_info must have shape (batch, time, side_info_dim), "
            f"got tensor with shape {tuple(side_info.shape)}."
        )
    expected_shape = (batch_size, time_steps, side_info_dim)
    if tuple(side_info.shape) != expected_shape:
        raise ValueError(
            f"side_info must have shape {expected_shape}, got {tuple(side_info.shape)}."
        )
    if not torch.is_floating_point(side_info):
        raise TypeError("side_info must be a floating-point tensor.")
    return side_info.to(device=device, dtype=dtype)


def _coerce_side_info_step(
    x_t: torch.Tensor | None,
    *,
    batch_shape: torch.Size,
    side_info_dim: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if x_t is None:
        return torch.zeros(
            *batch_shape,
            side_info_dim,
            device=reference.device,
            dtype=reference.dtype,
        )
    if side_info_dim == 0:
        raise ValueError("x_t was provided but side_info_dim=0 for this transition.")
    if x_t.shape[:-1] != batch_shape or x_t.shape[-1] != side_info_dim:
        raise ValueError(
            f"x_t must have shape {tuple(batch_shape) + (side_info_dim,)}, "
            f"got {tuple(x_t.shape)}."
        )
    if not torch.is_floating_point(x_t):
        raise TypeError("x_t must be a floating-point tensor.")
    return x_t


def _sequence_mask(seq_lengths: torch.Tensor, *, time_steps: int) -> torch.Tensor:
    time_index = torch.arange(time_steps, device=seq_lengths.device)
    return time_index.unsqueeze(0) < seq_lengths.unsqueeze(1)


def _reverse_sequences(sequences: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
    reversed_sequences = torch.zeros_like(sequences)
    for batch_idx in range(sequences.size(0)):
        seq_length = int(seq_lengths[batch_idx].item())
        if seq_length == 0:
            continue
        time_slice = torch.arange(seq_length - 1, -1, -1, device=sequences.device)
        reversed_sequence = torch.index_select(sequences[batch_idx, :, :], 0, time_slice)
        reversed_sequences[batch_idx, 0:seq_length, :] = reversed_sequence
    return reversed_sequences


def _pack_reversed_sequences(sequences: torch.Tensor, seq_lengths: torch.Tensor) -> PackedSequence:
    reversed_sequences = _reverse_sequences(sequences, seq_lengths)
    lengths_cpu = seq_lengths.to(device="cpu", dtype=torch.long)
    return pack_padded_sequence(
        reversed_sequences,
        lengths_cpu,
        batch_first=True,
        enforce_sorted=False,
    )


def _pad_and_reverse(
    rnn_output: PackedSequence, seq_lengths: torch.Tensor, *, total_length: int
) -> torch.Tensor:
    unpacked, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=total_length)
    return _reverse_sequences(unpacked, seq_lengths)


def _coerce_dmm_forecast_model(fitted: DMM | DMMFitResult) -> DMM:
    if isinstance(fitted, DMMFitResult):
        return fitted.model
    if isinstance(fitted, DMM):
        return fitted
    raise TypeError(f"fitted must be a DMM or DMMFitResult, got {type(fitted).__name__}.")


def _coerce_return_forecast_series(
    returns: pd.Series | np.ndarray,
) -> tuple[np.ndarray, pd.Index]:
    if isinstance(returns, pd.Series):
        values = np.asarray(returns, dtype=float)
        index = returns.index
    else:
        values = np.asarray(returns, dtype=float)
        index = pd.RangeIndex(start=0, stop=values.shape[0])
    if values.ndim != 1:
        raise ValueError(f"returns must be one-dimensional, got shape {values.shape}.")
    if values.size < 1:
        raise ValueError("returns must contain at least one observation.")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite values; drop NaN/inf first.")
    return values.copy(), index


def _coerce_forecast_side_info(
    side_info: pd.Series | pd.DataFrame | np.ndarray | None,
    *,
    return_index: pd.Index,
    n_obs: int,
    side_info_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if side_info_dim == 0:
        if side_info is not None:
            raise ValueError("side_info was provided but side_info_dim=0 for this DMM.")
        return None
    if side_info is None:
        raise ValueError(
            "side_info must be provided when forecasting with a DMM that expects exogenous inputs."
        )

    if isinstance(side_info, pd.DataFrame):
        side_values = side_info.to_numpy(dtype=float, copy=True)
        _validate_forecast_side_info_index(side_info.index, return_index)
    elif isinstance(side_info, pd.Series):
        side_values = side_info.to_numpy(dtype=float, copy=True).reshape(-1, 1)
        _validate_forecast_side_info_index(side_info.index, return_index)
    else:
        side_values = np.asarray(side_info, dtype=float)
        if side_values.ndim == 1:
            side_values = side_values.reshape(-1, 1)

    if side_values.ndim != 2:
        raise ValueError(
            "side_info must be one-dimensional or two-dimensional, "
            f"got shape {side_values.shape}."
        )
    expected_shape = (n_obs, side_info_dim)
    if side_values.shape != expected_shape:
        raise ValueError(f"side_info must have shape {expected_shape}, got {side_values.shape}.")
    if not np.all(np.isfinite(side_values)):
        raise ValueError("side_info must contain only finite values; drop NaN/inf first.")
    return torch.as_tensor(side_values.reshape(1, n_obs, side_info_dim), device=device, dtype=dtype)


def _validate_forecast_side_info_index(side_info_index: pd.Index, return_index: pd.Index) -> None:
    if not side_info_index.equals(return_index):
        raise ValueError("side_info index must exactly match the return-series index.")


def _filtered_latent_mean_for_prefix(
    dmm: DMM,
    *,
    prefix_observations: torch.Tensor,
    prefix_side_info: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, time_steps, _ = prefix_observations.shape
    seq_lengths = torch.full(
        (batch_size,),
        time_steps,
        dtype=torch.long,
        device=prefix_observations.device,
    )
    observations_prepared, _, _, seq_lengths_prepared = dmm._prepare_inputs(
        observations=prefix_observations,
        side_info=prefix_side_info,
        seq_lengths=seq_lengths,
    )
    h_0_contiguous = dmm.h_0.expand(dmm.num_layers, batch_size, dmm.rnn_dim).contiguous()
    rnn_output, _ = dmm.rnn(
        _pack_reversed_sequences(observations_prepared, seq_lengths_prepared),
        h_0_contiguous,
    )
    rnn_output_padded = _pad_and_reverse(
        rnn_output,
        seq_lengths_prepared,
        total_length=time_steps,
    )
    z_prev = dmm.z_q_0.expand(batch_size, dmm.z_dim)
    for step in range(time_steps):
        z_loc, _ = dmm.combiner(z_prev, rnn_output_padded[:, step, :])
        z_prev = z_loc
    return z_prev


def _filtered_latent_mean_trajectory_tensor(
    dmm: DMM,
    *,
    observations: torch.Tensor,
    side_info: torch.Tensor | None,
) -> torch.Tensor:
    _, time_steps, _ = observations.shape
    latent_trajectory = torch.empty(
        time_steps,
        dmm.z_dim,
        device=observations.device,
        dtype=observations.dtype,
    )
    for step in range(time_steps):
        prefix_length = step + 1
        prefix_observations = observations[:, :prefix_length, :]
        prefix_side_info = side_info[:, :prefix_length, :] if side_info is not None else None
        latent_trajectory[step, :] = _filtered_latent_mean_for_prefix(
            dmm,
            prefix_observations=prefix_observations,
            prefix_side_info=prefix_side_info,
        ).squeeze(0)
    return latent_trajectory


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_nonnegative_int(value: int, name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}.")


def _validate_positive_float(value: float, name: str) -> float:
    validated = float(value)
    if not torch.isfinite(torch.tensor(validated)):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if validated <= 0.0:
        raise ValueError(f"{name} must be positive, got {validated}.")
    return validated


def _validate_nonnegative_float(value: float, name: str) -> float:
    validated = float(value)
    if not torch.isfinite(torch.tensor(validated)):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if validated < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {validated}.")
    return validated


def _validate_probability_open_interval(value: float, name: str) -> float:
    validated = float(value)
    if not torch.isfinite(torch.tensor(validated)):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if not 0.0 < validated < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {validated}.")
    return validated


def _validate_probability_closed_interval(value: float, name: str) -> float:
    validated = float(value)
    if not torch.isfinite(torch.tensor(validated)):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if not 0.0 <= validated <= 1.0:
        raise ValueError(f"{name} must lie between 0 and 1 inclusive, got {validated}.")
    return validated


def _validate_last_dim(tensor: torch.Tensor, expected_last_dim: int, name: str) -> None:
    if tensor.shape[-1] != expected_last_dim:
        raise ValueError(
            f"{name} last dimension must equal {expected_last_dim}, got {tensor.shape[-1]}."
        )
