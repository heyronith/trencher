"""SPLIT_CONTRACT_V1 / SPLIT_AUDIT_V1 publishers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trencher.data.world_model.freeze import FREEZE_ID
from trencher.data.world_model.io import utc_now, write_json
from trencher.data.world_model.splits.purge import all_eligibility_columns, purge_reason_summary

CONTRACT_ID = "SPLIT_CONTRACT_V1"
AUDIT_ID = "SPLIT_AUDIT_V1"


def build_split_contract_v1(
    *, cutoffs: dict[str, Any], walkforward: dict[str, Any]
) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_ID,
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "scope": "shared_broad_and_rich_chronological_splits",
        "frozen_inputs_not_modified": [
            "broad_targets_v3",
            "BROAD_SUPERVISION_CONTRACT_V3",
            "target_dataset_v3",
            "sequence_dataset_v2",
            "observation_dataset_v1",
            "pit_v4",
        ],
        "split_key": "token_detected_at_unix",
        "observation_rule": "all_observations_of_a_token_share_one_base_split",
        "no_random_observation_split": True,
        "cutoff_method": cutoffs.get("method"),
        "cutoffs_unix": cutoffs.get("quantiles_unix"),
        "cutoffs_iso": cutoffs.get("quantiles_iso"),
        "tie_policy": cutoffs.get("tie_policy"),
        "purge_rules": purge_reason_summary(),
        "eligibility_columns": all_eligibility_columns(),
        "walkforward": {
            "folds": [f["fold_id"] for f in walkforward.get("folds", [])],
            "final_test_isolation": walkforward.get("final_test_isolation"),
            "method": walkforward.get("method"),
        },
        "no_training": True,
        "eval_controls_are_not_training_features": True,
    }


def publish_split_audit_v1(
    *,
    meta: dict[str, Any],
    contract: dict[str, Any],
    leakage_audit: dict[str, Any],
    manifest_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    cutoffs = meta["cutoffs"]
    actual = meta["actual_split_percentages"]
    broad_obs = meta["broad_observation_stats"]
    rich_obs = meta["rich_observation_stats"]

    # Headline purge counts by horizon for returns + survival
    def _horizon_purge(stats: dict[str, Any], head: str, horizons: list[int]) -> dict[str, Any]:
        elig = stats.get("eligible_counts") or {}
        purg = stats.get("purged_or_ineligible_counts") or {}
        out = {}
        for h in horizons:
            key = f"eligible_{head}_h{h}"
            out[str(h)] = {
                "eligible": elig.get(key),
                "purged_or_ineligible": purg.get(key),
            }
        return out

    audit = {
        "audit_id": AUDIT_ID,
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "contract_version": CONTRACT_ID,
        "cutoffs": {
            "quantiles_unix": cutoffs.get("quantiles_unix"),
            "quantiles_iso": cutoffs.get("quantiles_iso"),
            "launch_min_iso": cutoffs.get("launch_min_iso"),
            "launch_max_iso": cutoffs.get("launch_max_iso"),
            "method": cutoffs.get("method"),
            "tie_policy": cutoffs.get("tie_policy"),
        },
        "actual_split_percentages": actual,
        "token_counts": {
            "broad_event_curve": (actual.get("broad_event_curve") or {}).get("counts"),
            "rich_subset": (actual.get("rich_subset") or {}).get("counts"),
        },
        "observation_counts": {
            "broad": broad_obs.get("observations_by_base_split"),
            "broad_total": broad_obs.get("observations_assigned"),
            "broad_late_state_purged": broad_obs.get("observations_purged_late_state"),
            "rich": rich_obs.get("observations_by_base_split"),
            "rich_total": rich_obs.get("observations_assigned"),
            "rich_late_state_purged": rich_obs.get("observations_purged_late_state"),
        },
        "purge_counts_by_horizon": {
            "broad_returns": _horizon_purge(
                broad_obs, "returns", [10, 30, 60, 300, 3600, 86400]
            ),
            "broad_survival": _horizon_purge(
                broad_obs, "survival", [30, 60, 300, 3600, 86400]
            ),
            "broad_explosion_imminence": _horizon_purge(
                broad_obs, "explosion_imminence", [10, 30, 60, 120, 300]
            ),
            "rich_returns": _horizon_purge(
                rich_obs, "returns", [10, 30, 60, 300, 3600, 86400]
            ),
            "rich_survival": _horizon_purge(
                rich_obs, "survival", [30, 60, 300, 3600, 86400]
            ),
        },
        "walkforward_folds": meta.get("walkforward"),
        "eval_controls": {
            "broad": meta.get("eval_controls_broad"),
            "rich": meta.get("eval_controls_rich"),
        },
        "leakage_invariants": leakage_audit.get("invariants"),
        "artifacts": meta.get("artifacts"),
    }
    write_json(manifest_dir / f"{AUDIT_ID}.json", audit)
    write_json(out_dir / f"{AUDIT_ID}.json", audit)
    md = render_split_audit_md(audit)
    (manifest_dir / f"{AUDIT_ID}.md").write_text(md, encoding="utf-8")
    (out_dir / f"{AUDIT_ID}.md").write_text(md, encoding="utf-8")
    return audit


def render_split_audit_md(audit: dict[str, Any]) -> str:
    c = audit.get("cutoffs") or {}
    inv = audit.get("leakage_invariants") or {}
    lines = [
        f"# Split Audit (`{audit['audit_id']}`)",
        "",
        f"Generated at: `{audit['generated_at']}`",
        f"Freeze: `{audit['freeze_id']}`",
        "",
        "## Chronological cutoffs",
        "",
        f"- method: `{c.get('method')}`",
        f"- quantiles_iso: `{c.get('quantiles_iso')}`",
        f"- launch range: `{c.get('launch_min_iso')}` → `{c.get('launch_max_iso')}`",
        f"- tie_policy: {c.get('tie_policy')}",
        "",
        "## Actual split percentages (after ties)",
        "",
        f"- `{audit.get('actual_split_percentages')}`",
        "",
        "## Token / observation counts",
        "",
        f"- tokens: `{audit.get('token_counts')}`",
        f"- observations: `{audit.get('observation_counts')}`",
        "",
        "## Leakage invariants",
        "",
        f"- all_passed: **{inv.get('all_passed')}**",
        f"- details: `{inv}`",
        "",
        "## Walk-forward",
        "",
        f"- `{(audit.get('walkforward_folds') or {}).get('folds')}`",
        "",
    ]
    return "\n".join(lines)
