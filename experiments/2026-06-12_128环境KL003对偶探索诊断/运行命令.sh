#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${EXPERIMENT_DIR}/../.." && pwd)"

mkdir -p \
  "${EXPERIMENT_DIR}/日志" \
  "${EXPERIMENT_DIR}/检查点" \
  "${EXPERIMENT_DIR}/图片" \
  "${EXPERIMENT_DIR}/配置快照"

export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/reachability_gp_isaaclab22_baseline_engineering_dual_kl003_128env.yaml}"
export RUN_NAME="${RUN_NAME:-fdpi-baseline-engineering-dual-kl003-128env}"
export RUN_ROOT="${RUN_ROOT:-${EXPERIMENT_DIR}/检查点}"
export TAGS="${TAGS:-fdpi,reachability,baseline-engineering,dual-kl003,128env,dual-diagnostic,isaaclab22,surgical_robot5}"
export NOTE="${NOTE:-128-env baseline-engineering diagnostic run with DualUpdate.KLCoeff=0.03, retaining imagined_risk_return and all kl01_128env training density settings.}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

cd "${PROJECT_ROOT}"
bash scripts/train.sh "$@"
