"""PIT v2 audit: ownership completeness, arbitrary-T coverage, leakage results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.ownership import (
    build_fusion_coverage_truth,
    holder_state_completeness_spec,
)
from trencher.data.world_model.pit import (
    assert_no_post_t_leakage,
    ensure_atomic_trades_index,
    load_launch_times,
    load_pit_indexes,
    materialize_pit_states_for_subset,
)


def build_pit_v2_audit(
    *,
    world_root: Path,
    manifest_dir: Path,
    pump_root: Path | None = None,
    max_mints: int = 50,
    freeze_id: str = "world_model_freeze_v1",
) -> dict[str, Any]:
    """Recompute derived ownership/coverage + PIT sample; do not mutate freeze artifacts."""
    ensure_atomic_trades_index(world_root=world_root, pump_root=pump_root, force=False)
    coverage = build_fusion_coverage_truth(world_root, manifest_dir=manifest_dir)
    pit_manifest = materialize_pit_states_for_subset(
        world_root=world_root,
        max_mints=max_mints,
        pump_root=pump_root,
    )

    launches = load_launch_times(world_root)
    sample_mints = sorted(launches)[: min(5, len(launches))]
    indexes = load_pit_indexes(
        world_root, mint_filter=set(sample_mints), pump_root=pump_root
    )
    leakage_results = []
    for mint in sample_mints:
        # Observe at launch+60s so pre-T wallet edges exist in synthetic tests.
        t = launches[mint] + 60
        leakage_results.append(
            assert_no_post_t_leakage(
                mint=mint, observation_time_t=t, indexes=indexes
            )
        )

    truth = coverage["truth"]
    counts = truth.get("counts") or {}
    own_counts = truth.get("ownership_evidence_counts") or {}
    audit = {
        "audit_id": "pit_v2",
        "generated_at": utc_now(),
        "acquisition_freeze_id": freeze_id,
        "acquisition_freeze_policy": "unchanged_append_only_no_new_external_data",
        "ownership_evidence_counts": own_counts,
        "truly_reconstructable_holder_state_count": own_counts.get(
            "holder_state_reconstructable", 0
        ),
        "core_plus_any_genuine_ownership": counts.get(
            "core_plus_any_genuine_ownership"
        ),
        "core_plus_reconstructable_holder_state": counts.get(
            "core_plus_holder_state_reconstructable"
        ),
        "core_event_graph_micro_exec_market": counts.get(
            "core_event_graph_micro_exec_market"
        ),
        "fusion_coverage": counts,
        "completeness_spec": holder_state_completeness_spec(),
        "arbitrary_t_reconstruction": {
            "engine_version": pit_manifest.get("engine_version"),
            "mints_materialized": pit_manifest.get("mints_materialized"),
            "rows_materialized": pit_manifest.get("rows_materialized"),
            "age_offsets_sec": pit_manifest.get("age_offsets_sec"),
            "coverage_mints_with_any_modality": pit_manifest.get(
                "arbitrary_t_coverage_mints_with_any_modality"
            ),
            "path": pit_manifest.get("path"),
        },
        "leakage_tests": {
            "n_mints": len(leakage_results),
            "all_passed": all(r.get("passed") for r in leakage_results),
            "results": leakage_results,
        },
        "fusion_readiness_policy": truth.get("fusion_readiness_policy"),
        "known_remaining_limitations": [
            "holder_state_reconstructable is rare under documented completeness; "
            "most BQ ledgers remain transfer_ledger_observed_partial.",
            "RPC token-balance ownership is sampled-tx evidence only.",
            "Lifetime SOL funding covers a wallet subset only.",
            "Global same-day context uses rolling subset launches + mint trades; "
            "full-universe same-day aggregates are not used.",
            "Frozen lifetime edges_wallet_token.parquet is lineage/QA only — "
            "never a PIT model input.",
            "No supervised labels, training splits, or models in this audit.",
        ],
        "labels": "not_generated",
        "targets": "not_generated",
        "training": "not_started",
    }
    ensure_dir(manifest_dir)
    write_json(manifest_dir / "PIT_V2_AUDIT.json", audit)
    write_json(world_root / "pit" / "PIT_V2_AUDIT.json", audit)
    (manifest_dir / "PIT_V2_AUDIT.md").write_text(
        render_pit_v2_audit_md(audit), encoding="utf-8"
    )
    return audit


def render_pit_v2_audit_md(audit: dict[str, Any]) -> str:
    lines = [
        f"# PIT v2 Audit (`{audit['audit_id']}`)",
        "",
        f"Generated at: `{audit['generated_at']}`",
        f"Acquisition freeze: `{audit['acquisition_freeze_id']}` (unchanged)",
        "",
        "## Ownership evidence counts",
        "",
    ]
    for k, v in (audit.get("ownership_evidence_counts") or {}).items():
        lines.append(f"- `{k}`: **{v}**")
    lines.extend(
        [
            "",
            f"- Truly reconstructable holder-state: "
            f"**{audit.get('truly_reconstructable_holder_state_count')}**",
            f"- Core + any genuine ownership: "
            f"**{audit.get('core_plus_any_genuine_ownership')}**",
            f"- Core + reconstructable holder-state: "
            f"**{audit.get('core_plus_reconstructable_holder_state')}**",
            "",
            "## Arbitrary-T reconstruction coverage",
            "",
        ]
    )
    arb = audit.get("arbitrary_t_reconstruction") or {}
    lines.append(f"- Mints materialized: {arb.get('mints_materialized')}")
    lines.append(f"- Rows materialized: {arb.get('rows_materialized')}")
    cov = arb.get("coverage_mints_with_any_modality") or {}
    for age, n in cov.items():
        lines.append(f"- Age `{age}`: {n} mints with any modality")
    leak = audit.get("leakage_tests") or {}
    lines.extend(
        [
            "",
            "## Leakage tests",
            "",
            f"- All passed: **{leak.get('all_passed')}**",
            f"- Mints tested: {leak.get('n_mints')}",
            "",
            "## Known remaining limitations",
            "",
        ]
    )
    lines.extend(f"- {x}" for x in audit.get("known_remaining_limitations") or [])
    lines.append("")
    return "\n".join(lines)
