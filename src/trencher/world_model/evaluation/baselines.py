"""Compare world-model metrics against frozen Baseline V2 controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_BASELINE_CORE = Path(
    "manifests/world_model_data/baseline_v2/BASELINE_CORE_METRICS_V2.json"
)

# Overlapping heads where target definition + horizon + permanent-val protocol match.
COMPARABLE_HEADS: dict[str, dict[str, Any]] = {
    "explosion_60s": {
        "wm_family": "explosion",
        "wm_key": "60",
        "wm_metric": "auroc",
        "baseline_metric": "roc_auc",
        "direction": "higher_better",
    },
    "explosion_300s": {
        "wm_family": "explosion",
        "wm_key": "300",
        "wm_metric": "auroc",
        "baseline_metric": "roc_auc",
        "direction": "higher_better",
    },
}


def load_baseline_core(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else REPO_BASELINE_CORE
    return json.loads(p.read_text(encoding="utf-8"))


def extract_baseline_head(
    core: dict[str, Any],
    *,
    family: str,
    head_id: str,
    metric: str,
) -> dict[str, Any]:
    fam = core.get(family) or {}
    status = fam.get("family_status")
    if status and status != "complete":
        return {
            "comparable": False,
            "reason": f"baseline family {family} status={status}",
            "value": None,
        }
    heads = fam.get("heads") or {}
    head = heads.get(head_id)
    if not head:
        return {"comparable": False, "reason": f"missing head {head_id}", "value": None}
    # Prefer calibrated block when present; else raw.
    block = head.get("calibrated") or head.get("raw") or head
    if metric not in block:
        return {
            "comparable": False,
            "reason": f"metric {metric} absent for {head_id}",
            "value": None,
        }
    return {
        "comparable": True,
        "reason": None,
        "value": block[metric],
        "support": (head.get("counts_val") or {}).get("eligible"),
        "family_status": status,
        "eval_only_proof": fam.get("eval_only_proof"),
    }


def compare_world_model_to_baselines(
    wm_metrics: dict[str, Any],
    *,
    baseline_core: dict[str, Any] | None = None,
    families: tuple[str, ...] = ("lightgbm", "xgboost"),
) -> dict[str, Any]:
    """Flag non-comparable cells; never invent rankings across mismatched protocols."""
    core = baseline_core or load_baseline_core()
    rows: list[dict[str, Any]] = []
    for head_id, spec in COMPARABLE_HEADS.items():
        fam = spec["wm_family"]
        key = spec["wm_key"]
        wm_block = ((wm_metrics.get(fam) or {}).get(key)) or {}
        wm_val = wm_block.get(spec["wm_metric"])
        if wm_val is None and spec["wm_metric"] == "auroc":
            wm_val = wm_block.get("auroc")
        for family in families:
            base = extract_baseline_head(
                core, family=family, head_id=head_id, metric=spec["baseline_metric"]
            )
            comparable = bool(base["comparable"] and wm_val is not None)
            delta = None
            if comparable:
                delta = float(wm_val) - float(base["value"])
            rows.append(
                {
                    "head_id": head_id,
                    "baseline_family": family,
                    "wm_value": wm_val,
                    "baseline_value": base.get("value"),
                    "delta_wm_minus_baseline": delta,
                    "comparable": comparable,
                    "reason": None
                    if comparable
                    else (base.get("reason") or "missing_wm_metric"),
                    "direction": spec["direction"],
                    "wm_support": wm_block.get("support"),
                    "baseline_support": base.get("support"),
                }
            )
    # CatBoost intentionally incomplete
    cat = (core.get("catboost") or {}).get("family_status")
    return {
        "rows": rows,
        "catboost_status": cat or "unknown",
        "catboost_note": "Intentionally incomplete due to compute cost; excluded from ranking.",
        "non_comparable_policy": (
            "Only report deltas where target definition, horizon, and permanent-val "
            "protocol match. Do not manufacture cross-model rankings from incompatible cells."
        ),
    }
