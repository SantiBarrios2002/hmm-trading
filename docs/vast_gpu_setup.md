# Running HMC variants on a Vast.ai GPU box

The HMC continuous-parametric IOHMM variants (NumPyro NUTS) take **~12 min/window
on 16 CPU threads**, so a full 92-window × 2-HMC-variant run is ~36 h locally.
On a single rented NVIDIA GPU the same run finishes in **~75 min for under $1**.

This doc covers how to do that with [Vast.ai](https://vast.ai). The two scripts
in `bin/` automate everything except creating the instance.

## 1. Pick a GPU

Recommendation: **RTX 3090** from a *verified* Vast.ai host.

- This is a *small* model (K=2, n_knots=5, warmup=500, samples=1000, 2 chains).
  VRAM is not the bottleneck — even 8 GB is plenty.
- What matters is CUDA core count, memory bandwidth, and driver maturity.
- RTX 3090 is the value sweet spot: 24 GB, ~$0.20–0.30/hr verified, abundant
  supply. RTX 4090 (~$0.40/hr) is a fine fallback.
- **Skip** T4 (older Turing, slower for modern NumPyro) and A100/H100
  (10× the cost, will not be 10× faster on this tiny model).

### Vast.ai search filters

- **GPU:** RTX 3090 (or RTX 4090)
- **Verified:** on (mandatory — unverified hosts are routinely broken)
- **CUDA:** ≥ 12.1
- **Disk:** ≥ 50 GB
- **Internet:** ≥ 30 Mbps up
- **Template:** *PyTorch (cuDNN Runtime)* — ships CUDA 12 + Python ready.
  Do **not** pick a bare Ubuntu image unless you enjoy installing CUDA.

## 2. Launch the instance

1. Click **Rent** on Vast.ai. Wait ~30 s for it to boot.
2. Open the instance card → **Connect**. Copy the SSH command, e.g.
   `ssh -p 12345 root@ssh5.vast.ai`. Note the host and port — you'll pass them
   to the launcher.

## 3. Push code + data and start the job

From the repo root on your laptop:

```bash
bin/vast_launch.sh ssh5.vast.ai 12345
```

What this does:

1. `rsync`s the repo to `/workspace/hmm-trading/` (excludes `.venv`, `.git`,
   `runs/`, `data/`).
2. `rsync`s the ES 1-minute parquet (~600 MB) into
   `data/databento/databento/`.
3. SSHes in and runs `bin/vast_remote_setup.sh`, which:
   - Builds a venv, installs `requirements.txt`.
   - Replaces the CPU `jax` with `jax[cuda12]` (~2 GB of `nvidia-*` wheels,
     ~5 min).
   - Sanity-checks that JAX sees the GPU.
   - Starts `scripts/repro.py … --force` inside a tmux session named `hmc`,
     teeing to `hmc_run.log`.

The launcher exits as soon as the job is running. The job survives SSH drops.

### Different config

```bash
bin/vast_launch.sh ssh5.vast.ai 12345 configs/your_other_hmc_config.yaml
```

## 4. Monitor the run

Recent output (cheap, non-interactive):

```bash
ssh -p 12345 root@ssh5.vast.ai 'tmux capture-pane -pt hmc -S -200'
```

Or tail the log file directly:

```bash
ssh -p 12345 root@ssh5.vast.ai 'tail -f /workspace/hmm-trading/hmc_run.log'
```

Live attach (Ctrl-b then d to detach without killing the job):

```bash
ssh -p 12345 root@ssh5.vast.ai -t 'tmux attach -t hmc'
```

You'll see per-window progress lines like:

```
[volatility_ratio_hmc_continuous] window 12/92 index=11 done in 24.3s converged=True rhat_max=1.012 ess_bulk_min=412
```

Expected per-window time on RTX 3090: **15–40 s**.

## 5. Pull artifacts back

When `tmux capture-pane` shows the job finished:

```bash
rsync -avz -e "ssh -p 12345" \
  root@ssh5.vast.ai:/workspace/hmm-trading/runs/ ./runs/
```

## 6. Shut it down

**Important:** Vast.ai bills as long as the instance exists, even when idle.
Destroy it from the web UI as soon as you've rsync'd `runs/` back. Don't just
"stop" it — stopped instances still incur storage cost.

## Gotchas

- **First `import jax` is slow** (~30 s on a fresh box) — CUDA libs warming up,
  not a hang.
- **Preemption = restart from scratch.** Vast.ai's cheap "interruptible" tiers
  can reclaim the box. The side-information runner accumulates all variants
  **in memory** and only writes `runs/<cmp_id>/` atomically at the end — there
  are no per-window checkpoints on disk. If the instance is killed mid-run,
  you lose everything and start over. Mitigations: pick an on-demand
  (non-interruptible) instance, or accept the ~$1 worst case and re-run.
  `--force` in `bin/vast_remote_setup.sh` is correct: it overwrites the run
  dir on a second attempt (without it, the runner refuses to start if the
  directory exists).
- **Host SSD vs network storage.** Pick "on-demand" or non-shared storage if
  available — network-backed instances have slow disk that bottlenecks parquet
  reads.
- **VPN / corporate proxy.** Vast.ai SSH ports are random high ports
  (10000–60000). If you're behind a restrictive firewall, the `rsync` upload
  step will hang.
- **Cost sanity check.** At $0.25/hr a full run is ~$0.30. Budget $1 to cover
  setup overhead and an occasional retry. If you see hours of compute racking
  up, something is wrong — kill and inspect rather than letting it ride.
