#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${EXPERIMENT_DIR}/../.." && pwd)"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-isaaclab}"
export ISAACLAB_ROOT="${ISAACLAB_ROOT:-/root/IsaacLab}"
export SURGICAL_ROBOT5_EXT="${SURGICAL_ROBOT5_EXT:-/root/gpufree-data/surgical_robot5/exts/surgical_robot5}"
export TERM="${TERM:-xterm}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm
fi

cd "${PROJECT_ROOT}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

for name in baseline balanced fast; do
  case "${name}" in
    baseline) config="${EXPERIMENT_DIR}/配置_baseline.yaml" ;;
    balanced) config="${EXPERIMENT_DIR}/配置_balanced.yaml" ;;
    fast) config="${EXPERIMENT_DIR}/配置_fast.yaml" ;;
  esac
  run_dir="${EXPERIMENT_DIR}/日志/${name}"
  mkdir -p "${run_dir}"
  python scripts/profiling/benchmark_training_speed.py \
    --config "${config}" \
    2>&1 | tee "${run_dir}/运行日志.txt"
done
