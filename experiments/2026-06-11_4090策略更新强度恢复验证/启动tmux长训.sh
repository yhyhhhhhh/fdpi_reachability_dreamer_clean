#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${EXPERIMENT_DIR}/../.." && pwd)"
TMUX_SESSION="${TMUX_SESSION:-policy_recovery_long}"
RUN_ID="${RUN_ID:-policy_recovery_diag_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-${EXPERIMENT_DIR}/日志/tmux训练_${RUN_ID}.log}"

mkdir -p "${EXPERIMENT_DIR}/日志"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  printf 'ERROR: tmux session already exists: %s\n' "${TMUX_SESSION}" >&2
  exit 1
fi

export PROJECT_ROOT
export EXPERIMENT_DIR
export RUN_ID
export LOG_PATH

tmux new-session -d -s "${TMUX_SESSION}" bash -lc '
set -o pipefail
cd "${PROJECT_ROOT}"
export RUN_ID
bash "${EXPERIMENT_DIR}/运行命令.sh" 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
printf "\n[TMUX_WRAPPER_EXIT] status=%s time=%s run_id=%s\n" "${status}" "$(date "+%Y-%m-%d %H:%M:%S %Z")" "${RUN_ID}" | tee -a "${LOG_PATH}"
exit "${status}"
'

printf 'TMUX_SESSION=%s\nRUN_ID=%s\nLOG_PATH=%s\n' "${TMUX_SESSION}" "${RUN_ID}" "${LOG_PATH}"
