#!/usr/bin/env bash
# Record CatBoost incomplete status on Modal artifacts volume, then run
# discrete-hazard survival (economical). Does not touch LGBM/XGB/rich cache.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/v2_baseline_closeout.log
mkdir -p logs
SHA=$(git rev-parse HEAD)
echo "baseline closeout SHA=${SHA}" | tee -a "$LOG"

uv run python -u <<'PY' 2>&1 | tee -a logs/v2_baseline_closeout.log
import json
import modal
from trencher.data.world_model.io import utc_now

vol = modal.Volume.from_name("trencher-baseline-artifacts-v1")
# Inventory progress files already on the volume.
heads = {}
with vol.batch_upload(force=True) as batch:
    # Build status from a lightweight inventory via read API first.
    pass

inv = {}
try:
    for e in vol.listdir("catboost"):
        p = getattr(e, "path", str(e))
        if p.endswith(".json"):
            continue
        name = p.rsplit("/", 1)[-1]
        parts = {}
        try:
            for e2 in vol.listdir(p):
                pp = getattr(e2, "path", str(e2))
                leaf = pp.rsplit("/", 1)[-1]
                try:
                    raw = b"".join(vol.read_file(f"{pp}/progress.json"))
                    prog = json.loads(raw)
                    parts[leaf] = {
                        "progress": prog.get("progress"),
                        "message": prog.get("message"),
                        "elapsed_sec": prog.get("elapsed_sec"),
                        "best_iteration": prog.get("best_iteration"),
                    }
                except Exception:
                    parts[leaf] = {"present": True}
        except Exception as exc:
            parts = {"list_error": str(exc)[:120]}
        inv[name] = parts
except Exception as exc:
    inv = {"inventory_error": str(exc)}

status = {
    "family": "catboost",
    "status": "incomplete_compute_cost_stop",
    "reason": "unacceptable_compute_efficiency",
    "operator_requested_compute_stop": True,
    "operator_reported_approx_cost_usd": 20,
    "operator_reported_cost_note": (
        "Approximate operator-reported spend on this CatBoost run; "
        "not a machine-measured Modal invoice line item."
    ),
    "partial_artifacts_retained": True,
    "app_id": "ap-o4Ntw6HOYbp2iYCnOsFoon",
    "git_sha_at_stop": "7917b7ef392e3ff7671ea2af27ef1ff64739406a",
    "generated_at": utc_now(),
    "do_not_treat_as_completed_family": True,
    "scientifically_failed": False,
    "fold_inventory": inv,
    "note": (
        "CatBoost was stopped to avoid further GPU burn. "
        "LightGBM and XGBoost remain the completed tree controls. "
        "Partial fold artifacts/progress are preserved for audit only."
    ),
}
# Write via a tiny ephemeral mount using Volume API
import tempfile, pathlib
td = pathlib.Path(tempfile.mkdtemp())
(td / "FAMILY_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
# Also annotate results.json if present without deleting heads
try:
    raw = b"".join(vol.read_file("catboost/results.json"))
    results = json.loads(raw)
except Exception:
    results = {"family": "catboost", "heads": {}}
results["family_status"] = "incomplete_compute_cost_stop"
results["metrics_scope"] = "partial_or_legacy_not_full_v2_complete"
results["do_not_treat_as_full_population_complete"] = True
results["status_meta"] = {
    "status": "incomplete_compute_cost_stop",
    "operator_reported_approx_cost_usd": 20,
}
(td / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

with vol.batch_upload(force=True) as batch:
    batch.put_file(str(td / "FAMILY_STATUS.json"), "catboost/FAMILY_STATUS.json")
    batch.put_file(str(td / "results.json"), "catboost/results.json")
print("wrote catboost FAMILY_STATUS + annotated results.json")
print(json.dumps({k: status[k] for k in ("status", "app_id", "operator_reported_approx_cost_usd")}, indent=2))
PY

echo "Launching discrete-hazard survival…" | tee -a "$LOG"
uv run modal run --detach modal_apps/baselines_app.py::train_survival --no-smoke 2>&1 | tee -a "$LOG"
echo "survival launch attempted" | tee -a "$LOG"
