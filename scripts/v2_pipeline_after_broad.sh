#!/usr/bin/env bash
# Wait for the in-flight broad feature cache, verify with a resume pass on the
# newest commit, then launch full-population trains + rich cache.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/v2_pipeline_after_broad.log
echo "watching broad feature job…" | tee "$LOG"

while true; do
  prog=$(uv run python -u -c 'import modal,json; print(json.dumps(json.loads(b"".join(modal.Volume.from_name("trencher-baseline-artifacts-v1").read_file("feature_cache_broad_progress.json")))))' 2>/dev/null || echo "{}")
  echo "$(date -u +%H:%M:%S) $prog" | tee -a "$LOG"
  if echo "$prog" | rg -q 'feature cache complete|"progress": 100|"partitions_built": 256'; then
    echo "BROAD COMPLETE" | tee -a "$LOG"
    break
  fi
  state=$(uv run modal app list 2>/dev/null | rg "jdn18q2" || true)
  if echo "$state" | rg -q "stopped"; then
    echo "app stopped; checking final progress" | tee -a "$LOG"
    echo "$prog" | tee -a "$LOG"
    if echo "$prog" | rg -q 'complete|"progress": 100|"partitions_built": 256'; then
      break
    fi
    echo "STOPPED WITHOUT COMPLETE — abort pipeline" | tee -a "$LOG"
    exit 2
  fi
  sleep 180
done

echo "Verification resume pass (population=broad, resume=True)…" | tee -a "$LOG"
# Blocks until the verify job finishes; must succeed before any training.
uv run modal run modal_apps/baselines_app.py::build_features --population broad --resume 2>&1 | tee -a "$LOG"

echo "Launching full-population trains (detached)…" | tee -a "$LOG"
uv run modal run --detach modal_apps/baselines_app.py::train_lightgbm --no-smoke 2>&1 | tee -a "$LOG"
uv run modal run --detach modal_apps/baselines_app.py::train_xgboost --no-smoke 2>&1 | tee -a "$LOG"
uv run modal run --detach modal_apps/baselines_app.py::train_catboost --no-smoke 2>&1 | tee -a "$LOG"
uv run modal run --detach modal_apps/baselines_app.py::train_survival --no-smoke 2>&1 | tee -a "$LOG"

echo "Launching rich feature cache…" | tee -a "$LOG"
uv run modal run --detach modal_apps/baselines_app.py::build_features --population rich --resume 2>&1 | tee -a "$LOG"

echo "PIPELINE STAGE1 LAUNCHED (verify+trains+rich cache)." | tee -a "$LOG"
