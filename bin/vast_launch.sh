#!/usr/bin/env bash
# Local-side launcher: rsync repo + ES parquet to a Vast.ai box, start the HMC run
# in tmux, then exit. Pull artifacts back manually when the job is done (see
# docs/vast_gpu_setup.md).
#
# Usage:
#   bin/vast_launch.sh <ssh_host> <ssh_port> [config_relpath]
# Defaults config to configs/example_es_databento_side_info_comparison_hmc.yaml.

set -euo pipefail

HOST="${1:?ssh host required (e.g. ssh5.vast.ai)}"
PORT="${2:?ssh port required (e.g. 12345)}"
CONFIG="${3:-configs/example_es_databento_side_info_comparison_hmc.yaml}"

REMOTE="root@${HOST}"
REMOTE_DIR="/workspace/hmm-trading"
SSH_OPTS=(-p "${PORT}" -o StrictHostKeyChecking=accept-new)
SSH=(ssh "${SSH_OPTS[@]}" "${REMOTE}")
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

PARQUET="data/databento/databento/ES_c_0_ohlcv-1m_2019-01-01_2024-12-31.parquet"
if [[ ! -f "${PARQUET}" ]]; then
  echo "Missing local parquet: ${PARQUET}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing local config: ${CONFIG}" >&2
  exit 1
fi

echo "[1/4] pushing repo to ${REMOTE}:${REMOTE_DIR}"
"${SSH[@]}" "mkdir -p ${REMOTE_DIR}"
rsync -avz --delete \
  --exclude '.venv' --exclude '.git' --exclude 'runs' \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '*.pyo' \
  --exclude 'data' --exclude '.pytest_cache' --exclude '.mypy_cache' \
  --exclude '.ruff_cache' --exclude '.ipynb_checkpoints' \
  -e "${RSYNC_SSH}" ./ "${REMOTE}:${REMOTE_DIR}/"

echo "[2/4] pushing ES parquet (~600MB, one-shot)"
"${SSH[@]}" "mkdir -p ${REMOTE_DIR}/data/databento/databento"
rsync -avz --progress \
  -e "${RSYNC_SSH}" "${PARQUET}" \
  "${REMOTE}:${REMOTE_DIR}/data/databento/databento/"

echo "[3/4] running remote setup + starting tmux session 'hmc'"
"${SSH[@]}" "bash -lc 'cd ${REMOTE_DIR} && CONFIG_PATH=${CONFIG} bash bin/vast_remote_setup.sh'"

echo "[4/4] done. job is detached in tmux session 'hmc'."
cat <<EOF

Useful commands:
  Tail recent output:
    ssh -p ${PORT} ${REMOTE} 'tmux capture-pane -pt hmc -S -200'
  Live attach (Ctrl-b then d to detach):
    ssh -p ${PORT} ${REMOTE} -t 'tmux attach -t hmc'
  Pull artifacts when done:
    rsync -avz -e "ssh -p ${PORT}" \\
      ${REMOTE}:${REMOTE_DIR}/runs/ ./runs/
EOF
