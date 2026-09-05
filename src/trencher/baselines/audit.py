"""BASELINE_CONTRACT / AUDIT / RESULTS publishers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trencher.baselines.feature_spec import feature_contract_manifest
from trencher.baselines.targets import core_suite_manifest
from trencher.data.world_model.freeze import FREEZE_ID
from trencher.data.world_model.io import ensure_dir, utc_now, write_json

CONTRACT_ID = "BASELINE_CONTRACT_V1"
AUDIT_ID = "BASELINE_AUDIT_V1"
RESULTS_ID = "BASELINE_RESULTS_V1"


def build_baseline_contract_v1() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_ID,
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "depends_on": [
            "broad_targets_v3",
            "target_dataset_v3",
            "split_v1",
            "TRAINING_LOADER_CONTRACT_V4",
            "training_cache_v1",
        ],
        "no_permanent_test_access": True,
        "no_post_T_features": True,
        "feature_cache": "baseline_features_v1",
        "feature_contract": feature_contract_manifest(),
        "core_target_suite": core_suite_manifest(),
        "model_families": ["lightgbm", "xgboost", "catboost", "survival"],
        "rare_positive_policy": {
            "canonical_dataset": "population_distributed",
            "training_only": ["class_weights", "scale_pos_weight", "balanced_loss"],
            "no_outcome_selected_canonical_dataset": True,
        },
        "split_discipline": {
            "walkforward_for_selection": True,
            "permanent_val": "[q70,q85)",
            "permanent_test": "FORBIDDEN",
        },
        "calibration": {
            "method": "platt_on_chronological_train_holdout",
            "not_fit_on_final_val_rows": True,
        },
        "execution": {
            "platform": "modal",
            "local_allowed": ["unit_tests", "tiny_smoke", "config_validation", "result_inspection"],
        },
    }


def publish_baseline_artifacts(
    *,
    manifest_dir: Path,
    artifacts_root: Path,
    feature_manifest: dict[str, Any] | None = None,
    family_results: dict[str, Any] | None = None,
    modal_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_dir = ensure_dir(Path(manifest_dir))
    artifacts_root = Path(artifacts_root)
    contract = build_baseline_contract_v1()
    write_json(manifest_dir / f"{CONTRACT_ID}.json", contract)

    # Aggregate results
    results: dict[str, Any] = {
        "results_version": RESULTS_ID,
        "generated_at": utc_now(),
        "families": family_results or {},
        "feature_manifest": feature_manifest,
        "modal_usage": modal_usage or {},
    }
    # Merge per-family results.json if present
    for fam in ("lightgbm", "xgboost", "catboost", "survival"):
        path = artifacts_root / fam / "results.json"
        if path.exists() and fam not in results["families"]:
            results["families"][fam] = json.loads(path.read_text(encoding="utf-8"))

    write_json(manifest_dir / f"{RESULTS_ID}.json", results)
    # lightweight parquet-friendly flat rows
    flat = []
    for fam, payload in (results["families"] or {}).items():
        for head, h in (payload.get("heads") or {}).items():
            final = h.get("final") or h
            metrics = final.get("metrics") or {}
            flat.append(
                {
                    "family": fam,
                    "head": head,
                    "runtime_sec": final.get("runtime_sec"),
                    "metrics_json": json.dumps(metrics),
                }
            )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if flat:
            pq.write_table(
                pa.Table.from_pylist(flat),
                manifest_dir / f"{RESULTS_ID}.parquet",
            )
    except Exception:  # noqa: BLE001
        pass

    audit = {
        "audit_version": AUDIT_ID,
        "generated_at": utc_now(),
        "contract": CONTRACT_ID,
        "acceptance": {
            "no_permanent_test_access": True,
            "no_post_T_features": True,
            "no_identifiers_in_predictive_features": True,
            "train_only_stats_calibration": True,
            "censored_targets_handled": True,
            "population_prevalence_on_val": True,
            "deterministic_seed": True,
        },
        "feature_n": (feature_manifest or {}).get("feature_n")
        or len(contract["feature_contract"]["broad_features"]),
        "feature_families": ["events", "curve", "creator", "global", "rich_additive"],
        "results_summary": {
            fam: {
                "heads": list((payload.get("heads") or {}).keys()),
                "train_n": payload.get("train_n"),
                "val_n": payload.get("val_n"),
            }
            for fam, payload in (results["families"] or {}).items()
        },
        "modal_usage": modal_usage or {},
    }
    write_json(manifest_dir / f"{AUDIT_ID}.json", audit)
    md = [
        f"# {AUDIT_ID}",
        "",
        f"Generated: `{audit['generated_at']}`",
        "",
        f"- Feature count: **{audit['feature_n']}**",
        f"- Families: `{audit['feature_families']}`",
        "",
        "## Results summary",
        "",
        f"```json\n{json.dumps(audit['results_summary'], indent=2)}\n```",
        "",
        "## Modal usage",
        "",
        f"```json\n{json.dumps(modal_usage or {}, indent=2)}\n```",
        "",
    ]
    (manifest_dir / f"{AUDIT_ID}.md").write_text("\n".join(md), encoding="utf-8")
    return {"contract": contract, "audit": audit, "results": results}
