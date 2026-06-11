#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FINAL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-isaaclab}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/root/IsaacLab}"
ISAACLAB_SH="${ISAACLAB_SH:-${ISAACLAB_ROOT}/isaaclab.sh}"
SURGICAL_ROBOT5_EXT="${SURGICAL_ROBOT5_EXT:-/root/gpufree-data/surgical_robot5/exts/surgical_robot5}"

RUN_NAME="${RUN_NAME:-fdpi-reachability-dreamer-isaaclab22}"
SEED="${SEED:-0}"
CONFIG_PATH="${CONFIG_PATH:-${FINAL_ROOT}/configs/reachability_gp_isaaclab22.yaml}"
ENV_NAME="${ENV_NAME:-SurgicalRobot5-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1}"
DEVICE="${DEVICE:-cuda:0}"
RUN_ROOT="${RUN_ROOT:-${FINAL_ROOT}/ckpt_isaaclab22}"
RUN_ID="${RUN_ID:-}"
NOTE="${NOTE:-}"
TAGS="${TAGS:-fdpi,reachability,nstep-gp,isaaclab22,surgical_robot5}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
FULL_CHECKPOINT_PATH="${FULL_CHECKPOINT_PATH:-}"
COMPONENT_CHECKPOINT_DIR="${COMPONENT_CHECKPOINT_DIR:-}"
COMPONENT_CHECKPOINT_STEP="${COMPONENT_CHECKPOINT_STEP:-}"
RESUME_ENV_STEPS="${RESUME_ENV_STEPS:-}"
SAMPLE_MAX_STEPS="${SAMPLE_MAX_STEPS:-}"
NUM_ENVS="${NUM_ENVS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
BATCH_LENGTH="${BATCH_LENGTH:-}"
IMAGINE_BATCH_SIZE="${IMAGINE_BATCH_SIZE:-}"
IMAGINE_CONTEXT="${IMAGINE_CONTEXT:-}"
IMAGINE_HORIZON="${IMAGINE_HORIZON:-}"
TRAIN_MODEL_EVERY_STEPS="${TRAIN_MODEL_EVERY_STEPS:-}"
TRAIN_AGENT_EVERY_STEPS="${TRAIN_AGENT_EVERY_STEPS:-}"
MODEL_UPDATE="${MODEL_UPDATE:-}"
AGENT_UPDATE="${AGENT_UPDATE:-}"
MAIN_FDPI_START_STEP="${MAIN_FDPI_START_STEP:-}"
BUFFER_WARMUP_STEPS="${BUFFER_WARMUP_STEPS:-}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-}"
NO_LOAD_REPLAY_BUFFER="${NO_LOAD_REPLAY_BUFFER:-}"
NO_LOAD_OPTIMIZER="${NO_LOAD_OPTIMIZER:-}"
NO_LOAD_RNG="${NO_LOAD_RNG:-}"

if [[ ! -f "${CONDA_SH}" ]]; then
  printf 'ERROR: conda activation script does not exist: %q\n' "${CONDA_SH}" >&2
  exit 1
fi

if [[ ! -x "${ISAACLAB_SH}" ]]; then
  printf 'ERROR: ISAACLAB_SH is not executable: %q\n' "${ISAACLAB_SH}" >&2
  exit 1
fi

if [[ ! -d "${SURGICAL_ROBOT5_EXT}" ]]; then
  printf 'ERROR: SURGICAL_ROBOT5_EXT does not exist: %q\n' "${SURGICAL_ROBOT5_EXT}" >&2
  exit 1
fi

set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"
set -u

python - <<'PY'
try:
    import yacs  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "ERROR: yacs is missing in conda env `isaaclab`. "
        "Install it with: conda activate isaaclab && python -m pip install yacs"
    ) from exc
PY

export TERM=xterm
export PYTHONPATH="${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${ISAACLAB_ROOT}/source/isaaclab_assets:${ISAACLAB_ROOT}/source/isaaclab_rl:${SURGICAL_ROBOT5_EXT}:${FINAL_ROOT}:${PYTHONPATH:-}"

cd "${FINAL_ROOT}"

args=(
  -n "${RUN_NAME}"
  -seed "${SEED}"
  -config_path "${CONFIG_PATH}"
  -env_name "${ENV_NAME}"
  -device "${DEVICE}"
  --run_root "${RUN_ROOT}"
  --no_run_info_prompt
)

if [[ -n "${RUN_ID}" ]]; then
  args+=(--run_id "${RUN_ID}")
fi

if [[ -n "${NOTE}" ]]; then
  args+=(--note "${NOTE}")
fi

if [[ -n "${TAGS}" ]]; then
  args+=(--tags "${TAGS}")
fi

if [[ -n "${FULL_CHECKPOINT_PATH}" ]]; then
  if [[ ! -f "${FULL_CHECKPOINT_PATH}" ]]; then
    printf 'ERROR: FULL_CHECKPOINT_PATH does not exist: %q\n' "${FULL_CHECKPOINT_PATH}" >&2
    exit 1
  fi
  args+=(--full_checkpoint_path "${FULL_CHECKPOINT_PATH}")
fi

if [[ -n "${COMPONENT_CHECKPOINT_DIR}" ]]; then
  if [[ ! -d "${COMPONENT_CHECKPOINT_DIR}" ]]; then
    printf 'ERROR: COMPONENT_CHECKPOINT_DIR does not exist: %q\n' "${COMPONENT_CHECKPOINT_DIR}" >&2
    exit 1
  fi
  args+=(--component_checkpoint_dir "${COMPONENT_CHECKPOINT_DIR}")
fi

if [[ -n "${COMPONENT_CHECKPOINT_STEP}" ]]; then
  args+=(--component_checkpoint_step "${COMPONENT_CHECKPOINT_STEP}")
fi

if [[ -n "${RESUME_ENV_STEPS}" ]]; then
  args+=(--resume_env_steps "${RESUME_ENV_STEPS}")
fi

if [[ -n "${SAMPLE_MAX_STEPS}" ]]; then
  args+=(--max_steps "${SAMPLE_MAX_STEPS}")
fi

if [[ -n "${NUM_ENVS}" ]]; then
  args+=(--num_envs "${NUM_ENVS}")
fi

if [[ -n "${BATCH_SIZE}" ]]; then
  args+=(--batch_size "${BATCH_SIZE}")
fi

if [[ -n "${BATCH_LENGTH}" ]]; then
  args+=(--batch_length "${BATCH_LENGTH}")
fi

if [[ -n "${IMAGINE_BATCH_SIZE}" ]]; then
  args+=(--imagine_batch_size "${IMAGINE_BATCH_SIZE}")
fi

if [[ -n "${IMAGINE_CONTEXT}" ]]; then
  args+=(--imagine_context "${IMAGINE_CONTEXT}")
fi

if [[ -n "${IMAGINE_HORIZON}" ]]; then
  args+=(--imagine_horizon "${IMAGINE_HORIZON}")
fi

if [[ -n "${TRAIN_MODEL_EVERY_STEPS}" ]]; then
  args+=(--train_model_every_steps "${TRAIN_MODEL_EVERY_STEPS}")
fi

if [[ -n "${TRAIN_AGENT_EVERY_STEPS}" ]]; then
  args+=(--train_agent_every_steps "${TRAIN_AGENT_EVERY_STEPS}")
fi

if [[ -n "${MODEL_UPDATE}" ]]; then
  args+=(--model_update "${MODEL_UPDATE}")
fi

if [[ -n "${AGENT_UPDATE}" ]]; then
  args+=(--agent_update "${AGENT_UPDATE}")
fi

if [[ -n "${MAIN_FDPI_START_STEP}" ]]; then
  args+=(--main_fdpi_start_step "${MAIN_FDPI_START_STEP}")
fi

if [[ -n "${BUFFER_WARMUP_STEPS}" ]]; then
  args+=(--buffer_warmup_steps "${BUFFER_WARMUP_STEPS}")
fi

if [[ -n "${SAVE_EVERY_STEPS}" ]]; then
  args+=(--save_every_steps "${SAVE_EVERY_STEPS}")
fi

if [[ -n "${NO_LOAD_REPLAY_BUFFER}" ]]; then
  args+=(--no_load_replay_buffer)
fi

if [[ -n "${NO_LOAD_OPTIMIZER}" ]]; then
  args+=(--no_load_optimizer)
fi

if [[ -n "${NO_LOAD_RNG}" ]]; then
  args+=(--no_load_rng)
fi

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    printf 'ERROR: CHECKPOINT_PATH does not exist: %q\n' "${CHECKPOINT_PATH}" >&2
    exit 1
  fi
  args+=(-checkpoint_path "${CHECKPOINT_PATH}")
fi

exec "${ISAACLAB_SH}" -p "${FINAL_ROOT}/fdpi_reachability_dreamer_isaaclab22/train.py" "${args[@]}" "$@"
