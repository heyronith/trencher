#!/usr/bin/env bash
# Wait for the in-flight broad feature cache, verify with a resume pass on the
# newest commit, then launch full-population trains + rich cache.
#
# Completion gate (BOTH required before verification):
#   1) progress artifact message contains "feature cache complete"
#      (NOT merely progress=100 or partitions_built=256)
#   2) original broad Modal app is stopped (final Volume commit done)
#
# If the app stops without the explicit completion message → abort.
# If verification exits non-zero → abort (no training on a partial cache).
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/v2_pipeline_after_broad.log
BROAD_APP_ID="ap-jdn18q2koPc353fAVW1S4i"
echo "watching broad feature job ${BROAD_APP_ID}…" | tee -a "$LOG"

_has_explicit_complete() {
  # Match the terminal ProgressEmitter message only — never bare progress=100.
  echo "$1" | rg -q 'feature cache complete'
}

_app_stopped() {
  local row
  row=$(uv run modal app list 2>/dev/null | rg "${BROAD_APP_ID}" || true)
  if [[ -z "${row}" ]]; then
    # App no longer listed → treat as finished/committed.
    return 0
  fi
  echo "${row}" | rg -q 'stopped'
}

_app_actively_running() {
  local row
  row=$(uv run modal app list 2>/dev/null | rg "${BROAD_APP_ID}" || true)
  [[ -n "${row}" ]] && echo "${row}" | rg -q 'ephemeral|running'
}

seen_complete=0
while true; do
  prog=$(uv run python -u -c 'import modal,json; print(json.dumps(json.loads(b"".join(modal.Volume.from_name("trencher-baseline-artifacts-v1").read_file("feature_cache_broad_progress.json")))))' 2>/dev/null || echo "{}")
  echo "$(date -u +%H:%M:%S) $prog" | tee -a "$LOG"

  if _has_explicit_complete "$prog"; then
    seen_complete=1
    echo "saw explicit completion message; waiting for app ${BROAD_APP_ID} to stop…" | tee -a "$LOG"
    if _app_stopped; then
      echo "BROAD COMPLETE (message + app stopped)" | tee -a "$LOG"
      break
    fi
  fi

  if ! _app_actively_running; then
    # Re-read progress after stop so we don't race the final artifact commit.
    sleep 15
    prog=$(uv run python -u -c 'import modal,json; print(json.dumps(json.loads(b"".join(modal.Volume.from_name("trencher-baseline-artifacts-v1").read_file("feature_cache_broad_progress.json")))))' 2>/dev/null || echo "{}")
    echo "$(date -u +%H:%M:%S) post-stop $prog" | tee -a "$LOG"
    if _has_explicit_complete "$prog"; then
      echo "BROAD COMPLETE (app stopped + message)" | tee -a "$LOG"
      break
    fi
    echo "STOPPED WITHOUT explicit 'feature cache complete' — abort pipeline" | tee -a "$LOG"
    exit 2
  fi

  sleep 180
done

if [[ "${seen_complete}" -ne 1 ]] && ! _has_explicit_complete "$prog"; then
  echo "INTERNAL: exited wait loop without completion message — abort" | tee -a "$LOG"
  exit 2
fi

echo "Verification resume pass (population=broad, resume=True)…" | tee -a "$LOG"
# Blocking; must exit 0 before any training.
set +e
uv run modal run modal_apps/baselines_app.py::build_features --population broad --resume 2>&1 | tee -a "$LOG"
verify_rc=${PIPESTATUS[0]}
set -e
if [[ "${verify_rc}" -ne 0 ]]; then
  echo "VERIFICATION FAILED (exit=${verify_rc}) — abort; no training" | tee -a "$LOG"
  exit "${verify_rc}"
fi
echo "Verification OK" | tee -a "$LOG"

echo "Launching full-population trains (detached)…" | tee -a "$LOG"
uv run modal run --detach modal_apps/baselines_app.py::train_lightgbm --no-smoke 2>&1 | tee -a "$LOG"
uv run modal run --detach modal_apps/baselines_app.py::train_xgboost --no-smoke 2>&1 | tee -a "$LOG"
uv run modal run --detach modal_apps/baselines_app.py::train_catboost --no-smoke 2>&1 | tee -a "$LOG"
uv run modal run --detach modal_apps/baselines_app.py::train_survival --no-smoke 2>&1 | tee -a "$LOG"

echo "Launching rich feature cache…" | tee -a "$LOG"
uv run modal run --detach modal_apps/baselines_app.py::build_features --population rich --resume 2>&1 | tee -a "$LOG"

echo "PIPELINE STAGE1 LAUNCHED (verify+trains+rich cache)." | tee -a "$LOG"
