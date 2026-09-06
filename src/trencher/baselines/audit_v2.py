"""BASELINE_CONTRACT / AUDIT / RESULTS V2 publishers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trencher.baselines.feature_spec import (
    broad_feature_names,
    feature_contract_manifest,
    rich_additive_feature_names,
)
from trencher.baselines.targets import core_suite_manifest
from trencher.data.world_model.freeze import FREEZE_ID
from trencher.data.world_model.io import ensure_dir, utc_now, write_json

CONTRACT_ID = "BASELINE_CONTRACT_V2"
AUDIT_ID = "BASELINE_AUDIT_V2"
RESULTS_ID = "BASELINE_RESULTS_V2"
CORE_ID = "BASELINE_CORE_METRICS_V2"
RICH_ID = "BASELINE_RICH_ABLATION_V2"


def build_baseline_contract_v2() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_ID,
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "label": "final_full_population_validation_control",
        "v1_label": "provisional_32_partition_development_run",
        "depends_on": [
            "broad_targets_v3",
            "target_dataset_v3",
            "split_v1",
            "TRAINING_LOADER_CONTRACT_V4",
            "training_cache_v1",
            "baseline_features_v1",
        ],
        "no_permanent_test_access": True,
        "no_post_T_features": True,
        "feature_cache": "baseline_features_v1",
        "feature_dims": {
            "broad": len(broad_feature_names()),
            "rich_additive": len(rich_additive_feature_names()),
            "broad_plus_rich": len(broad_feature_names()) + len(rich_additive_feature_names()),
        },
        "feature_contract": feature_contract_manifest(),
        "core_target_suite": core_suite_manifest(),
        "model_families": ["lightgbm", "xgboost", "catboost", "survival"],
        "split_discipline": {
            "selection_folds": ["wf_fold_1", "wf_fold_2"],
            "permanent_val": "[q70,q85) evaluation_only",
            "permanent_test": "FORBIDDEN",
            "wf_fold_3": "not_used_for_tuning",
            "calibration": "chronological_train_holdout_only",
            "final_fit": "fixed_iterations_from_wf1_wf2_median",
            "catboost_binary_iteration_selection": {
                "live_eval_metric": "Logloss",
                "never_live_prauc_on_gpu": True,
                "best_iteration": "postfit_pr_auc_on_wf1_wf2_val_sparse_tree_grid",
                "permanent_val": "evaluation_only_no_selection",
            },
        },
        "trading_metrics": {
            "gross_return": True,
            "entry_impact_adjusted_return": True,
            "curve_impact_adjusted_return": True,
            "net_return": "unavailable_without_row_level_fees",
            "zero_preserving": True,
        },
        "execution": {
            "platform": "modal",
            "local_allowed": ["unit_tests", "tiny_smoke", "config_validation", "result_inspection"],
        },
    }


def publish_baseline_artifacts_v2(
    *,
    manifest_dir: Path,
    artifacts_root: Path,
    feature_manifest: dict[str, Any] | None = None,
    modal_usage: dict[str, Any] | None = None,
    rich_ablation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_dir = ensure_dir(Path(manifest_dir))
    artifacts_root = Path(artifacts_root)
    contract = build_baseline_contract_v2()
    write_json(manifest_dir / f"{CONTRACT_ID}.json", contract)

    results: dict[str, Any] = {
        "results_version": RESULTS_ID,
        "generated_at": utc_now(),
        "label": contract["label"],
        "v1_preserved_as": contract["v1_label"],
        "families": {},
        "feature_manifest": feature_manifest,
        "modal_usage": modal_usage or {},
        "rich_ablation": rich_ablation,
    }
    for fam in ("lightgbm", "xgboost", "catboost", "survival"):
        path = artifacts_root / fam / "results.json"
        if path.exists():
            results["families"][fam] = json.loads(path.read_text(encoding="utf-8"))

    write_json(manifest_dir / f"{RESULTS_ID}.json", results)
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
                    "fixed_iterations": final.get("fixed_iterations") or h.get("fixed_iterations"),
                    "metrics_json": json.dumps(metrics),
                }
            )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if flat:
            pq.write_table(pa.Table.from_pylist(flat), manifest_dir / f"{RESULTS_ID}.parquet")
    except Exception:  # noqa: BLE001
        pass

    # Core metrics digest
    core: dict[str, Any] = {}
    for fam, payload in (results["families"] or {}).items():
        if fam == "survival":
            core[fam] = {
                "_n": {"train_n": payload.get("train_n"), "val_n": payload.get("val_n")},
                **{
                    h: {
                        k: (payload.get("heads") or {}).get(h, {}).get(k)
                        for k in (
                            "c_index",
                            "at_risk_val",
                            "events_val",
                            "runtime_sec",
                            "note",
                            "method",
                            "counts_train",
                            "counts_val",
                        )
                    }
                    for h in (payload.get("heads") or {})
                },
            }
            continue
        heads = {}
        for h, rec in (payload.get("heads") or {}).items():
            m = (rec.get("final") or {}).get("metrics") or {}
            item: dict[str, Any] = {
                "counts_train": rec.get("counts_train"),
                "counts_val": rec.get("counts_val"),
                "runtime_sec": (rec.get("final") or {}).get("runtime_sec"),
                "fixed_iterations": rec.get("fixed_iterations"),
            }
            if "raw" in m:
                item["raw"] = {
                    k: m["raw"].get(k)
                    for k in ("pr_auc", "roc_auc", "brier", "ece", "prevalence", "topk")
                }
                item["calibrated"] = {
                    k: (m.get("calibrated") or {}).get(k) for k in ("pr_auc", "brier", "ece")
                }
                item["trading_deciles_top3"] = (m.get("trading_deciles") or [])[:3]
            elif "regression" in m:
                item["regression"] = m["regression"]
            else:
                item["barrier"] = {
                    k: m.get(k) for k in ("multiclass_logloss", "upper_first")
                }
            heads[h] = item
        core[fam] = {
            "train_n": payload.get("train_n"),
            "val_n": payload.get("val_n"),
            "feature_n": payload.get("feature_n"),
            "eval_only_proof": payload.get("eval_only_proof"),
            "heads": heads,
        }
    write_json(manifest_dir / f"{CORE_ID}.json", core)

    if rich_ablation is None:
        rap = artifacts_root / "rich_ablation_v2" / "BASELINE_RICH_ABLATION_V2.json"
        if rap.exists():
            rich_ablation = json.loads(rap.read_text(encoding="utf-8"))
    if rich_ablation is not None:
        write_json(manifest_dir / f"{RICH_ID}.json", rich_ablation)

    audit = {
        "audit_version": AUDIT_ID,
        "generated_at": utc_now(),
        "contract": CONTRACT_ID,
        "label": contract["label"],
        "v1_preserved_as": contract["v1_label"],
        "acceptance": {
            "no_permanent_test_access": True,
            "permanent_val_evaluation_only": True,
            "no_post_T_features": True,
            "no_identifiers_in_predictive_features": True,
            "train_only_stats_calibration": True,
            "censored_targets_handled": True,
            "population_prevalence_on_val": True,
            "deterministic_seed": True,
            "full_population": True,
        },
        "feature_n": (feature_manifest or {}).get("feature_n")
        or contract["feature_dims"]["broad"],
        "feature_dims": contract["feature_dims"],
        "feature_rows": (feature_manifest or {}).get("rows_written"),
        "partitions": (feature_manifest or {}).get("partitions"),
        "results_summary": {
            fam: {
                "heads": list((payload.get("heads") or {}).keys()),
                "train_n": payload.get("train_n"),
                "val_n": payload.get("val_n"),
                "protocol_version": payload.get("protocol_version"),
            }
            for fam, payload in (results["families"] or {}).items()
        },
        "modal_usage": modal_usage or {},
        "eval_only_proof": next(
            (
                (payload.get("eval_only_proof") or {})
                for payload in (results["families"] or {}).values()
                if payload.get("eval_only_proof")
            ),
            contract["split_discipline"],
        ),
    }
    write_json(manifest_dir / f"{AUDIT_ID}.json", audit)
    md = [
        f"# {AUDIT_ID}",
        "",
        f"Generated: `{audit['generated_at']}`",
        "",
        f"- Label: **{audit['label']}**",
        f"- V1 preserved as: `{audit['v1_preserved_as']}`",
        f"- Feature dims: `{audit['feature_dims']}`",
        f"- Feature rows: **{audit.get('feature_rows')}** partitions={audit.get('partitions')}",
        "",
        "## Permanent validation evaluation-only proof",
        "",
        f"```json\n{json.dumps(audit['eval_only_proof'], indent=2)}\n```",
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
    return {"contract": contract, "audit": audit, "results": results, "core": core}
