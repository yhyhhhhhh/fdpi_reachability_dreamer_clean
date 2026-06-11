#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${EXPERIMENT_DIR}/../.." && pwd)"

mkdir -p "${EXPERIMENT_DIR}/日志" "${EXPERIMENT_DIR}/图片" "${EXPERIMENT_DIR}/检查点" "${EXPERIMENT_DIR}/配置快照"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-isaaclab}"
export ISAACLAB_ROOT="${ISAACLAB_ROOT:-/root/IsaacLab}"
export SURGICAL_ROBOT5_EXT="${SURGICAL_ROBOT5_EXT:-/root/gpufree-data/surgical_robot5/exts/surgical_robot5}"

cd "${PROJECT_ROOT}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

python scripts/profiling/benchmark_training_speed.py \
  --config "${EXPERIMENT_DIR}/配置.yaml" \
  2>&1 | tee "${EXPERIMENT_DIR}/日志/运行日志.txt"
