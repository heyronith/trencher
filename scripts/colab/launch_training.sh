#!/usr/bin/env bash
# Launch Stage-A (or other phase) world-model training from staged frozen artifacts.
# Does NOT embed credentials. Does NOT auto-run Stage B/C.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

STAGE="${TRENCHER_LOCAL_STAGE:-/content/trencher_stage}"
WORLD_ROOT="${TRENCHER_WORLD_ROOT:-${STAGE}/world_model}"
SPLIT_ROOT="${TRENCHER_SPLIT_ROOT:-${WORLD_ROOT}/splits/split_v1}"
# Config comes from the checked-out Git SHA unless explicitly overridden.
CONFIG="${TRENCHER_WM_CONFIG:-${REPO_ROOT}/configs/world_model/world_model_candidate_v2.yaml}"
PHASE="${TRENCHER_WM_PHASE:-ssl}"
DEVICE="${TRENCHER_WM_DEVICE:-cuda}"
CKPT_DIR="${TRENCHER_WM_CKPT_DIR:-${STAGE}/checkpoints}"
MODE="${TRENCHER_WM_MODE:-final_train}"
POPULATION="${TRENCHER_WM_POPULATION:-broad}"
RESUME="${TRENCHER_WM_RESUME:-}"
MAX_STEPS="${TRENCHER_WM_MAX_STEPS:-}"
GIT_SHA="${TRENCHER_GIT_SHA:-}"
NUM_WORKERS="${TRENCHER_WM_NUM_WORKERS:-2}"
RUN_ID="${TRENCHER_WM_RUN_ID:-stage_a_ssl}"

echo "[launch_training] REPO_ROOT=${REPO_ROOT}"
echo "[launch_training] STAGE=${STAGE}"
echo "[launch_training] WORLD_ROOT=${WORLD_ROOT}"
echo "[launch_training] CONFIG=${CONFIG}"
echo "[launch_training] PHASE=${PHASE} MODE=${MODE} POPULATION=${POPULATION}"
echo "[launch_training] DEVICE=${DEVICE} NUM_WORKERS=${NUM_WORKERS} RUN_ID=${RUN_ID}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${WORLD_ROOT}" ]]; then
  echo "Missing world root: ${WORLD_ROOT} — run stage_from_gcs.py first" >&2
  exit 1
fi

mkdir -p "${CKPT_DIR}"

export TRENCHER_WM_CONFIG="${CONFIG}"
export TRENCHER_LOCAL_STAGE="${STAGE}"
export TRENCHER_WORLD_ROOT="${WORLD_ROOT}"
export TRENCHER_SPLIT_ROOT="${SPLIT_ROOT}"
export TRENCHER_WM_PHASE="${PHASE}"
export TRENCHER_WM_DEVICE="${DEVICE}"
export TRENCHER_WM_CKPT_DIR="${CKPT_DIR}"
export TRENCHER_WM_MODE="${MODE}"
export TRENCHER_WM_POPULATION="${POPULATION}"
export TRENCHER_WM_RESUME="${RESUME}"
export TRENCHER_WM_MAX_STEPS="${MAX_STEPS}"
export TRENCHER_GIT_SHA="${GIT_SHA}"
export TRENCHER_WM_NUM_WORKERS="${NUM_WORKERS}"
export TRENCHER_WM_RUN_ID="${RUN_ID}"

python -m trencher.world_model.launch_hardened
