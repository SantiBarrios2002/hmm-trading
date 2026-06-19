#!/usr/bin/env python3
"""Throwaway P0 spike: run Pyro's DMM example on CPU and record timing/NLL."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from collections import namedtuple
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from torch.nn.utils.rnn import pad_sequence

PUBLISHED_PYRO_TEST_NLL = 6.87
PAPER_TEST_NLL = 6.93

dset = namedtuple("dset", ["name", "url", "filename"])

JSB_CHORALES = dset(
    "jsb_chorales",
    "https://github.com/pyro-ppl/datasets/blob/master/polyphonic/jsb_chorales.pickle?raw=true",
    "jsb_chorales.pkl",
)


def get_data_directory(path: str) -> str:
    return os.path.abspath(path)


def process_data(base_path: str, dataset: dset, min_note: int = 21, note_range: int = 88) -> None:
    output = os.path.join(base_path, dataset.filename)
    if os.path.exists(output):
        return

    print(f"downloading and processing raw data: {dataset.name}")
    request = Request(dataset.url, headers={"User-Agent": "hft-hmm-dmm-spike/0"})
    with urlopen(request) as response:
        data = pickle.load(response)

    processed_dataset: dict[str, dict[str, Any]] = {}
    for split, data_split in data.items():
        processed_dataset[split] = {}
        n_seqs = len(data_split)
        processed_dataset[split]["sequence_lengths"] = torch.zeros(n_seqs, dtype=torch.long)
        processed_dataset[split]["sequences"] = []
        for seq in range(n_seqs):
            seq_length = len(data_split[seq])
            processed_dataset[split]["sequence_lengths"][seq] = seq_length
            processed_sequence = torch.zeros((seq_length, note_range))
            for t in range(seq_length):
                note_slice = torch.tensor(list(data_split[seq][t])) - min_note
                if len(note_slice) > 0:
                    processed_sequence[t, note_slice] = 1.0
            processed_dataset[split]["sequences"].append(processed_sequence)

    with open(output, "wb") as handle:
        pickle.dump(processed_dataset, handle, pickle.HIGHEST_PROTOCOL)


def load_data(dataset: dset, data_dir: str) -> dict[str, dict[str, Any]]:
    os.makedirs(data_dir, exist_ok=True)
    process_data(data_dir, dataset)
    file_loc = os.path.join(data_dir, dataset.filename)
    with open(file_loc, "rb") as handle:
        loaded = pickle.load(handle)

    for split in loaded.values():
        sequences = split["sequences"]
        split["sequences"] = pad_sequence(sequences, batch_first=True).type(torch.Tensor)
        split["sequence_lengths"] = split["sequence_lengths"].to(device=torch.Tensor().device)
    return loaded


def reverse_sequences(mini_batch: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
    reversed_mini_batch = torch.zeros_like(mini_batch)
    for batch_idx in range(mini_batch.size(0)):
        seq_length = seq_lengths[batch_idx]
        time_slice = torch.arange(seq_length - 1, -1, -1, device=mini_batch.device)
        reversed_sequence = torch.index_select(mini_batch[batch_idx, :, :], 0, time_slice)
        reversed_mini_batch[batch_idx, 0:seq_length, :] = reversed_sequence
    return reversed_mini_batch


def pad_and_reverse(
    rnn_output: torch.nn.utils.rnn.PackedSequence, seq_lengths: torch.Tensor
) -> torch.Tensor:
    unpacked, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    return reverse_sequences(unpacked, seq_lengths)


def get_mini_batch_mask(mini_batch: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(mini_batch.shape[0:2])
    for batch_idx in range(mini_batch.shape[0]):
        mask[batch_idx, 0 : seq_lengths[batch_idx]] = 1.0
    return mask


def get_mini_batch(
    mini_batch_indices: torch.Tensor, sequences: torch.Tensor, seq_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.nn.utils.rnn.PackedSequence, torch.Tensor, torch.Tensor]:
    batch_seq_lengths = seq_lengths[mini_batch_indices]
    _, sorted_seq_length_indices = torch.sort(batch_seq_lengths)
    sorted_seq_length_indices = sorted_seq_length_indices.flip(0)
    sorted_seq_lengths = batch_seq_lengths[sorted_seq_length_indices]
    sorted_mini_batch_indices = mini_batch_indices[sorted_seq_length_indices]

    max_length = torch.max(batch_seq_lengths)
    mini_batch = sequences[sorted_mini_batch_indices, 0:max_length, :]
    mini_batch_reversed = reverse_sequences(mini_batch, sorted_seq_lengths)
    mini_batch_mask = get_mini_batch_mask(mini_batch, sorted_seq_lengths)
    mini_batch_reversed = nn.utils.rnn.pack_padded_sequence(
        mini_batch_reversed, sorted_seq_lengths, batch_first=True
    )
    return mini_batch, mini_batch_reversed, mini_batch_mask, sorted_seq_lengths


class Emitter(nn.Module):
    def __init__(self, input_dim: int, z_dim: int, emission_dim: int) -> None:
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        return torch.sigmoid(self.lin_hidden_to_input(h2))


class GatedTransition(nn.Module):
    def __init__(self, z_dim: int, transition_dim: int) -> None:
        super().__init__()
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_hidden = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(gate_hidden))
        proposed_hidden = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(proposed_hidden)
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        return loc, scale


class Combiner(nn.Module):
    def __init__(self, z_dim: int, rnn_dim: int) -> None:
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(
        self, z_t_1: torch.Tensor, h_rnn: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        loc = self.lin_hidden_to_loc(h_combined)
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        return loc, scale


class DMM(nn.Module):
    def __init__(
        self,
        input_dim: int = 88,
        z_dim: int = 100,
        emission_dim: int = 100,
        transition_dim: int = 200,
        rnn_dim: int = 600,
        num_layers: int = 1,
        rnn_dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim)
        effective_dropout = 0.0 if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=rnn_dim,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
            num_layers=num_layers,
            dropout=effective_dropout,
        )
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

    def model(
        self,
        mini_batch: torch.Tensor,
        mini_batch_reversed: torch.nn.utils.rnn.PackedSequence,
        mini_batch_mask: torch.Tensor,
        mini_batch_seq_lengths: torch.Tensor,
        annealing_factor: float = 1.0,
    ) -> None:
        del mini_batch_reversed, mini_batch_seq_lengths
        time_steps = mini_batch.size(1)
        pyro.module("dmm", self)
        z_prev = self.z_0.expand(mini_batch.size(0), self.z_0.size(0))

        with pyro.plate("z_minibatch", len(mini_batch)):
            for t in pyro.markov(range(1, time_steps + 1)):
                z_loc, z_scale = self.trans(z_prev)
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(
                        f"z_{t}",
                        dist.Normal(z_loc, z_scale).mask(mini_batch_mask[:, t - 1 : t]).to_event(1),
                    )
                emission_probs_t = self.emitter(z_t)
                pyro.sample(
                    f"obs_x_{t}",
                    dist.Bernoulli(emission_probs_t)
                    .mask(mini_batch_mask[:, t - 1 : t])
                    .to_event(1),
                    obs=mini_batch[:, t - 1, :],
                )
                z_prev = z_t

    def guide(
        self,
        mini_batch: torch.Tensor,
        mini_batch_reversed: torch.nn.utils.rnn.PackedSequence,
        mini_batch_mask: torch.Tensor,
        mini_batch_seq_lengths: torch.Tensor,
        annealing_factor: float = 1.0,
    ) -> None:
        time_steps = mini_batch.size(1)
        pyro.module("dmm", self)
        h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        rnn_output = pad_and_reverse(rnn_output, mini_batch_seq_lengths)
        z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))

        with pyro.plate("z_minibatch", len(mini_batch)):
            for t in pyro.markov(range(1, time_steps + 1)):
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(
                        f"z_{t}",
                        dist.Normal(z_loc, z_scale).mask(mini_batch_mask[:, t - 1 : t]).to_event(1),
                    )
                z_prev = z_t


def vectorized_eval_batch(
    sequences: torch.Tensor, seq_lengths: torch.Tensor, n_eval_samples: int
) -> tuple[torch.Tensor, torch.nn.utils.rnn.PackedSequence, torch.Tensor, torch.Tensor]:
    def rep(tensor: torch.Tensor) -> torch.Tensor:
        rep_shape = torch.Size([tensor.size(0) * n_eval_samples]) + tensor.size()[1:]
        repeat_dims = [1] * len(tensor.size())
        repeat_dims[0] = n_eval_samples
        return (
            tensor.repeat(repeat_dims)
            .reshape(n_eval_samples, -1)
            .transpose(1, 0)
            .reshape(rep_shape)
        )

    expanded_seq_lengths = rep(seq_lengths)
    expanded_sequences = rep(sequences)
    indices = torch.arange(n_eval_samples * sequences.shape[0])
    return get_mini_batch(indices, expanded_sequences, expanded_seq_lengths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-epochs", type=int, default=150)
    parser.add_argument("--mini-batch-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.96)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--lr-decay", type=float, default=0.99996)
    parser.add_argument("--weight-decay", type=float, default=2.0)
    parser.add_argument("--annealing-epochs", type=int, default=1000)
    parser.add_argument("--minimum-annealing-factor", type=float, default=0.2)
    parser.add_argument("--rnn-dropout-rate", type=float, default=0.1)
    parser.add_argument("--eval-frequency", type=int, default=25)
    parser.add_argument("--eval-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/tmp/hft_hmm_dmm_spike_data"),
        help="Directory for the JSB Chorales download/cache.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("/tmp/hft_hmm_dmm_spike_summary.json"),
        help="Optional JSON summary output path.",
    )
    parser.add_argument("--torch-threads", type=int, default=0)
    return parser


def main() -> None:
    assert pyro.__version__.startswith("1.9.1")
    args = build_parser().parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()

    data_dir = get_data_directory(str(args.data_dir))
    data = load_data(JSB_CHORALES, data_dir)

    training_seq_lengths = data["train"]["sequence_lengths"]
    training_data_sequences = data["train"]["sequences"]
    test_seq_lengths = data["test"]["sequence_lengths"]
    test_data_sequences = data["test"]["sequences"]
    val_seq_lengths = data["valid"]["sequence_lengths"]
    val_data_sequences = data["valid"]["sequences"]

    num_train_data = len(training_seq_lengths)
    train_time_slices = float(torch.sum(training_seq_lengths))
    num_mini_batches = int(
        num_train_data / args.mini_batch_size + int(num_train_data % args.mini_batch_size > 0)
    )

    print(
        json.dumps(
            {
                "event": "config",
                "seed": args.seed,
                "num_epochs": args.num_epochs,
                "mini_batch_size": args.mini_batch_size,
                "eval_frequency": args.eval_frequency,
                "eval_samples": args.eval_samples,
                "torch_threads": torch.get_num_threads(),
                "num_train_data": num_train_data,
                "num_mini_batches": num_mini_batches,
                "avg_train_seq_len": float(training_seq_lengths.float().mean()),
            }
        ),
        flush=True,
    )

    val_batch, val_batch_reversed, val_batch_mask, val_batch_lengths = vectorized_eval_batch(
        val_data_sequences, val_seq_lengths, args.eval_samples
    )
    test_batch, test_batch_reversed, test_batch_mask, test_batch_lengths = vectorized_eval_batch(
        test_data_sequences, test_seq_lengths, args.eval_samples
    )

    dmm = DMM(rnn_dropout_rate=args.rnn_dropout_rate)
    adam = ClippedAdam(
        {
            "lr": args.learning_rate,
            "betas": (args.beta1, args.beta2),
            "clip_norm": args.clip_norm,
            "lrd": args.lr_decay,
            "weight_decay": args.weight_decay,
        }
    )
    svi = SVI(dmm.model, dmm.guide, adam, loss=Trace_ELBO())

    def do_evaluation() -> tuple[float, float]:
        dmm.rnn.eval()
        val_nll = (
            svi.evaluate_loss(val_batch, val_batch_reversed, val_batch_mask, val_batch_lengths)
            / float(torch.sum(val_batch_lengths))
        )
        test_nll = (
            svi.evaluate_loss(test_batch, test_batch_reversed, test_batch_mask, test_batch_lengths)
            / float(torch.sum(test_batch_lengths))
        )
        dmm.rnn.train()
        return val_nll, test_nll

    epoch_times: list[float] = []
    eval_history: list[dict[str, float | int]] = []
    best_test_nll = math.inf
    best_test_epoch = -1
    last_train_nll = math.nan
    start_time = time.perf_counter()

    for epoch in range(args.num_epochs):
        epoch_start = time.perf_counter()
        epoch_nll = 0.0
        shuffled_indices = np.arange(num_train_data)
        np.random.shuffle(shuffled_indices)

        for which_mini_batch in range(num_mini_batches):
            if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
                min_af = args.minimum_annealing_factor
                annealing_factor = min_af + (1.0 - min_af) * (
                    float(which_mini_batch + epoch * num_mini_batches + 1)
                    / float(args.annealing_epochs * num_mini_batches)
                )
            else:
                annealing_factor = 1.0

            batch_start = which_mini_batch * args.mini_batch_size
            batch_end = min((which_mini_batch + 1) * args.mini_batch_size, num_train_data)
            mini_batch_indices = torch.tensor(
                shuffled_indices[batch_start:batch_end], dtype=torch.long
            )
            batch = get_mini_batch(
                mini_batch_indices, training_data_sequences, training_seq_lengths
            )
            epoch_nll += svi.step(*batch, annealing_factor)

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        last_train_nll = epoch_nll / train_time_slices
        print(
            json.dumps(
                {
                    "event": "train_epoch",
                    "epoch": epoch + 1,
                    "normalized_negative_elbo": last_train_nll,
                    "epoch_seconds": epoch_time,
                }
            ),
            flush=True,
        )

        should_eval = (epoch + 1) % args.eval_frequency == 0 or (epoch + 1) == args.num_epochs
        if should_eval:
            val_nll, test_nll = do_evaluation()
            eval_item = {
                "epoch": epoch + 1,
                "val_nll": val_nll,
                "test_nll": test_nll,
            }
            eval_history.append(eval_item)
            if test_nll < best_test_nll:
                best_test_nll = test_nll
                best_test_epoch = epoch + 1
            print(json.dumps({"event": "eval", **eval_item}), flush=True)

    total_seconds = time.perf_counter() - start_time
    summary = {
        "pyro_version": pyro.__version__,
        "torch_version": torch.__version__,
        "published_pyro_test_nll": PUBLISHED_PYRO_TEST_NLL,
        "paper_test_nll": PAPER_TEST_NLL,
        "seed": args.seed,
        "num_epochs": args.num_epochs,
        "mini_batch_size": args.mini_batch_size,
        "eval_frequency": args.eval_frequency,
        "eval_samples": args.eval_samples,
        "torch_threads": torch.get_num_threads(),
        "last_train_normalized_negative_elbo": last_train_nll,
        "best_test_nll": best_test_nll,
        "best_test_epoch": best_test_epoch,
        "final_val_nll": eval_history[-1]["val_nll"] if eval_history else None,
        "final_test_nll": eval_history[-1]["test_nll"] if eval_history else None,
        "mean_epoch_seconds": float(np.mean(epoch_times)),
        "median_epoch_seconds": float(np.median(epoch_times)),
        "total_seconds": total_seconds,
        "eval_history": eval_history,
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(json.dumps({"event": "summary", **summary}), flush=True)


if __name__ == "__main__":
    main()
