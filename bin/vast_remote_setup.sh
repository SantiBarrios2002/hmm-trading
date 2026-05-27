#!/usr/bin/env bash
# Remote-side bootstrap, executed on the Vast.ai box by vast_launch.sh.
# Builds a venv, installs project deps + CUDA jax, sanity-checks the GPU, then
# starts the HMC run inside a detached tmux session named 'hmc'.

set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/example_es_databento_side_info_comparison_hmc.yaml}"
PROJECT_DIR="$(pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on remote — pick a PyTorch/CUDA Vast template" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "[remote] creating .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[remote] installing project deps"
pip install --upgrade pip wheel >/dev/null
pip install -r requirements.txt

echo "[remote] swapping CPU jax for CUDA 12 build"
pip uninstall -y jax jaxlib >/dev/null 2>&1 || true
pip install --upgrade "jax[cuda12]"

echo "[remote] sanity-check: jax devices"
python - <<'PY'
import jax
devs = jax.devices()
print("jax devices:", devs)
if not any(d.platform == "gpu" for d in devs):
    raise SystemExit("No GPU visible to JAX — check CUDA install / drivers.")
PY

if ! command -v tmux >/dev/null 2>&1; then
  echo "[remote] installing tmux"
  apt-get update -qq && apt-get install -y -qq tmux
fi

echo "[remote] (re)starting tmux session 'hmc'"
tmux kill-session -t hmc 2>/dev/null || true
LOG="${PROJECT_DIR}/hmc_run.log"
tmux new-session -d -s hmc \
  "cd ${PROJECT_DIR} && source .venv/bin/activate && \
   .venv/bin/python scripts/repro.py ${CONFIG_PATH} --force 2>&1 | tee ${LOG}"

echo "[remote] job running. log: ${LOG}"
