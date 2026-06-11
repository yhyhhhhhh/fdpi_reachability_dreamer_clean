#!/usr/bin/env bash
set -e

EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${EXPERIMENT_DIR}/../.." && pwd)"

export CONFIG_PATH="${PROJECT_ROOT}/configs/reachability_gp_isaaclab22_4090_fast_wm_safety50.yaml"
export RUN_NAME="${RUN_NAME:-fdpi-4090fast-wm-safety50}"
export RUN_ID="${RUN_ID:-wm_safety50_fast_$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT="${RUN_ROOT:-${EXPERIMENT_DIR}/检查点}"
export TAGS="${TAGS:-fdpi,reachability,wm-safety50,4090-fast,dual-data,isaaclab22,surgical_robot5}"
export NOTE="${NOTE:-4090 fast 参数 + world model 50% uniform / 50% safety-critical replay 采样，验证危险区域建模误差是否下降。}"

bash "${PROJECT_ROOT}/scripts/train.sh"
