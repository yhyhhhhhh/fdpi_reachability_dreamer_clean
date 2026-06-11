#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${EXPERIMENT_DIR}/../.." && pwd)"

mkdir -p \
  "${EXPERIMENT_DIR}/日志" \
  "${EXPERIMENT_DIR}/检查点" \
  "${EXPERIMENT_DIR}/图片" \
  "${EXPERIMENT_DIR}/配置快照"

export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/reachability_gp_isaaclab22_4090_policy_recovery.yaml}"
export RUN_NAME="${RUN_NAME:-fdpi-4090-policy-recovery}"
export RUN_ROOT="${RUN_ROOT:-${EXPERIMENT_DIR}/检查点}"
export TAGS="${TAGS:-fdpi,reachability,4090-policy-recovery,isaaclab22,surgical_robot5}"
export NOTE="${NOTE:-4090 policy recovery: keep 128 env + batch 256 + AMP/TF32, restore stronger agent/Gp/Gd/dual update density than fast, and validate grasp learning recovery.}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

_report_exit() {
  local status=$?
  printf '\n[FDPI_EXPERIMENT_EXIT] status=%s time=%s run_id=%s\n' \
    "${status}" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${RUN_ID:-}" >&2
}

_handle_signal() {
  local signal_name="$1"
  local status="$2"
  printf '\n[FDPI_EXPERIMENT_SIGNAL] signal=%s status=%s time=%s run_id=%s\n' \
    "${signal_name}" "${status}" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${RUN_ID:-}" >&2
  exit "${status}"
}

trap _report_exit EXIT
trap '_handle_signal INT 130' INT
trap '_handle_signal TERM 143' TERM
trap '_handle_signal HUP 129' HUP

cd "${PROJECT_ROOT}"
set +e
bash scripts/train.sh "$@"
status=$?
set -e
exit "${status}"
