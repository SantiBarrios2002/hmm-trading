"""Deep Markov Model with continuous emissions and side-information-gated transitions.

This module implements the architecture layer of the Deep Markov Model (DMM)
described by Krishnan, Shalit, and Sontag (2017, §3-4), adapted for return
sequences with optional exogenous side information. The latent process is a
nonlinear Gaussian state-space model:

``z_t | z_{t-1}, x_t ~ Normal(mu_trans(z_{t-1}, x_t), sigma_trans(z_{t-1}, x_t))``

``y_t | z_t ~ Normal(mu_emit(z_t), sigma_emit(z_t))``

``q(z_t | z_{t-1}, y_{t:T}) = Normal(mu_q(z_{t-1}, h_t), sigma_q(z_{t-1}, h_t))``

where ``h_t`` is produced by a reverse-time recurrent network over the observed
return sequence. The transition network accepts optional side information
``x_t``; when no side information is provided, the model falls back to a zero
vector so the same module can be used in ablations without exogenous features.

Variable-length mini-batches are represented as padded tensors plus
``seq_lengths``. The model and guide preserve the P0 spike's masking strategy so
padded timesteps do not contribute to latent or observation log-probability.

References: Krishnan et al. (2017) §3-4
"""

from __future__ import annotations

from typing import Final, cast

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from pyro.distributions import TorchDistribution
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
                side_info_step = side_info_prepared[:, t, :] if self.side_info_dim > 0 else None
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


def _validate_last_dim(tensor: torch.Tensor, expected_last_dim: int, name: str) -> None:
    if tensor.shape[-1] != expected_last_dim:
        raise ValueError(
            f"{name} last dimension must equal {expected_last_dim}, got {tensor.shape[-1]}."
        )
