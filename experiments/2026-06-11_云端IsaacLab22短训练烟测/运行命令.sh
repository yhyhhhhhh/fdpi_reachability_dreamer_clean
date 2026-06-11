#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${EXPERIMENT_DIR}/../.." && pwd)"

export CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-isaaclab}"
export ISAACLAB_ROOT="${ISAACLAB_ROOT:-/root/IsaacLab}"
export SURGICAL_ROBOT5_EXT="${SURGICAL_ROBOT5_EXT:-/root/gpufree-data/surgical_robot5/exts/surgical_robot5}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CONFIG_PATH="${CONFIG_PATH:-${EXPERIMENT_DIR}/配置快照/reachability_gp_isaaclab22_短训练烟测.yaml}"
export RUN_NAME="${RUN_NAME:-fdpi-cloud-isaaclab22-smoke}"
export RUN_ROOT="${RUN_ROOT:-${EXPERIMENT_DIR}/检查点}"
export RUN_ID="${RUN_ID:-cloud_smoke}"
export NOTE="${NOTE:-云端 IsaacLab22 SurgicalRobot5 短训练烟测}"
export TAGS="${TAGS:-fdpi,isaaclab22,surgical_robot5,cloud_smoke}"
export NUM_ENVS="${NUM_ENVS:-2}"
export SAMPLE_MAX_STEPS="${SAMPLE_MAX_STEPS:-1024}"
export BUFFER_WARMUP_STEPS="${BUFFER_WARMUP_STEPS:-64}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-512}"

mkdir -p "${EXPERIMENT_DIR}/日志" "${EXPERIMENT_DIR}/检查点"

cd "${PROJECT_ROOT}"
bash scripts/train_isaaclab22.sh 2>&1 | tee "${EXPERIMENT_DIR}/日志/运行日志.txt"
