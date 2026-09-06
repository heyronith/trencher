#!/usr/bin/env bash
# Resume V2 after CatBoost GPU-PRAUC fix.
# Does NOT rerun: broad feature cache, LightGBM, XGBoost.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/v2_resume_after_catboost_fix.log
mkdir -p logs
SHA=$(git rev-parse HEAD)
echo "V2 resume from SHA=${SHA}" | tee -a "$LOG"

echo "Launching corrected full-pop CatBoost…" | tee -a "$LOG"
uv run modal run --detach modal_apps/baselines_app.py::train_catboost --no-smoke 2>&1 | tee -a "$LOG"

echo "Launching survival…" | tee -a "$LOG"
uv run modal run --detach modal_apps/baselines_app.py::train_survival --no-smoke 2>&1 | tee -a "$LOG"

echo "Launching rich 146-feature cache…" | tee -a "$LOG"
uv run modal run --detach modal_apps/baselines_app.py::build_features --population rich --resume 2>&1 | tee -a "$LOG"

echo "STAGE A LAUNCHED (catboost + survival + rich cache). SHA=${SHA}" | tee -a "$LOG"
echo "After those finish: run_rich_ablation then publish_results_v2 --git-sha ${SHA}" | tee -a "$LOG"
