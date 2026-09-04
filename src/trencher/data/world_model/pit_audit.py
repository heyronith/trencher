"""PIT v3 audit: as-of-T ownership, global context, full-payload leakage."""

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
    ensure_global_context_index,
    find_forbidden_future_fields,
    load_launch_times,
    load_pit_indexes,
    materialize_pit_states_for_subset,
    reconstruct_token_state_at_t,
)


def build_pit_v3_audit(
    *,
    world_root: Path,
    manifest_dir: Path,
    pump_root: Path | None = None,
    max_mints: int = 50,
    freeze_id: str = "world_model_freeze_v1",
) -> dict[str, Any]:
    """Derived PIT v3 audit; does not mutate acquisition freeze artifacts."""
    ensure_atomic_trades_index(world_root=world_root, pump_root=pump_root, force=False)
    ensure_global_context_index(world_root=world_root, pump_root=pump_root, force=False)
    coverage = build_fusion_coverage_truth(world_root, manifest_dir=manifest_dir)

    launches = load_launch_times(world_root)
    recon_mints: list[str] = []
    own_path = world_root / "universe" / "ownership_evidence.parquet"
    if own_path.exists():
        import pyarrow.parquet as pq

        ot = pq.read_table(own_path, columns=["mint", "ownership_evidence_level"])
        for m, level in zip(
            ot.column("mint").to_pylist(),
            ot.column("ownership_evidence_level").to_pylist(),
            strict=True,
        ):
            if level == "holder_state_reconstructable" and m in launches:
                recon_mints.append(m)

    sample_for_materialize = list(dict.fromkeys(recon_mints + sorted(launches)))[
        : max(max_mints, len(recon_mints))
    ]
    pit_manifest = materialize_pit_states_for_subset(
        world_root=world_root,
        mint_list=sample_for_materialize,
        pump_root=pump_root,
    )
    pit_manifest["full_history_reconstructable_mints_included"] = len(recon_mints)

    sample_mints = list(dict.fromkeys(recon_mints[:2] + sorted(launches)))[:5]
    indexes = load_pit_indexes(
        world_root, mint_filter=set(sample_mints), pump_root=pump_root
    )
    leakage_results = []
    ownership_at_t_vs_full = []
    for mint in sample_mints:
        for age_name, offset in (("launch", 0), ("1m", 60), ("5m", 300)):
            t = launches[mint] + offset
            leak = assert_no_post_t_leakage(
                mint=mint, observation_time_t=t, indexes=indexes
            )
            leakage_results.append({"age": age_name, **leak})
        # Full-history reconstructable must not appear at launch unless as-of-T passes.
        full = indexes.get("own_level_full_history_audit", {}).get(mint)
        state0 = reconstruct_token_state_at_t(
            mint=mint, observation_time_t=launches[mint], indexes=indexes
        )["model_state"]
        ownership_at_t_vs_full.append(
            {
                "mint": mint,
                "full_history_level": full,
                "ownership_level_at_launch": state0["ownership_level_at_t"],
                "holder_state_reconstructable_at_launch": state0[
                    "holder_state_reconstructable_at_t"
                ],
                "ok_no_retroactive_upgrade": not (
                    full == "holder_state_reconstructable"
                    and state0["holder_state_reconstructable_at_t"]
                    and state0["ownership_rows_observed_at_t"] < 20
                ),
            }
        )

    # Structural absence of future fields on a sample state.
    sample_state = reconstruct_token_state_at_t(
        mint=sample_mints[0],
        observation_time_t=launches[sample_mints[0]] + 60,
        indexes=indexes,
    )
    forbidden = find_forbidden_future_fields(sample_state["model_state"])

    truth = coverage["truth"]
    counts = truth.get("counts") or {}
    own_counts = truth.get("ownership_evidence_counts") or {}
    audit = {
        "audit_id": "pit_v3",
        "generated_at": utc_now(),
        "acquisition_freeze_id": freeze_id,
        "acquisition_freeze_policy": "unchanged_append_only_no_new_external_data",
        "fusion_status": "MASKED",
        "ownership_evidence_counts_full_history_qa": own_counts,
        "as_of_t_ownership_level_distribution_by_age": pit_manifest.get(
            "ownership_level_at_t_by_age"
        ),
        "holder_state_reconstructable_at_t_by_age": pit_manifest.get(
            "holder_state_reconstructable_at_t_by_age"
        ),
        "global_context_coverage_by_age": pit_manifest.get(
            "global_context_coverage_by_age"
        ),
        "core_plus_any_genuine_ownership": counts.get(
            "core_plus_any_genuine_ownership"
        ),
        "core_plus_holder_state_reconstructable_full_history_qa": counts.get(
            "core_plus_holder_state_reconstructable"
        ),
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
            "n_cases": len(leakage_results),
            "all_passed": all(r.get("passed") for r in leakage_results),
            "results": leakage_results,
        },
        "full_history_vs_as_of_t_ownership": ownership_at_t_vs_full,
        "future_labels_structurally_absent_from_model_state": not forbidden,
        "forbidden_field_hits_in_sample_model_state": forbidden,
        "model_state_excludes_provenance_generated_at": (
            "generated_at" not in sample_state["model_state"]
            and "generated_at" in sample_state.get("provenance", {})
        ),
        "known_remaining_limitations": [
            "Full-history holder_state_reconstructable remains rare; as-of-T is "
            "usually lower.",
            "Window unique mint/wallet counts are not materialized in v3 "
            "second buckets (volume/flow only).",
            "RPC ownership remains sampled-tx evidence.",
            "Native SOL funding covers a wallet subset only.",
            "Fusion remains MASKED; do not claim full ownership coverage.",
            "No supervised labels, splits, or models in this audit.",
        ],
        "labels": "not_generated",
        "targets": "not_generated",
        "training": "not_started",
    }
    ensure_dir(manifest_dir)
    write_json(manifest_dir / "PIT_V3_AUDIT.json", audit)
    write_json(world_root / "pit" / "PIT_V3_AUDIT.json", audit)
    (manifest_dir / "PIT_V3_AUDIT.md").write_text(
        render_pit_v3_audit_md(audit), encoding="utf-8"
    )
    return audit


def render_pit_v3_audit_md(audit: dict[str, Any]) -> str:
    lines = [
        f"# PIT v3 Audit (`{audit['audit_id']}`)",
        "",
        f"Generated at: `{audit['generated_at']}`",
        f"Acquisition freeze: `{audit['acquisition_freeze_id']}` (unchanged)",
        f"Fusion status: `{audit.get('fusion_status')}`",
        "",
        "## As-of-T ownership by age",
        "",
    ]
    for age, dist in (audit.get("as_of_t_ownership_level_distribution_by_age") or {}).items():
        lines.append(f"- `{age}`: {dist}")
    lines.extend(["", "## Holder-state reconstructable at T", ""])
    for age, n in (audit.get("holder_state_reconstructable_at_t_by_age") or {}).items():
        lines.append(f"- `{age}`: **{n}**")
    lines.extend(["", "## Global context coverage", ""])
    for age, n in (audit.get("global_context_coverage_by_age") or {}).items():
        lines.append(f"- `{age}`: {n}")
    leak = audit.get("leakage_tests") or {}
    lines.extend(
        [
            "",
            "## Leakage / structural checks",
            "",
            f"- Leakage all passed: **{leak.get('all_passed')}**",
            f"- Future labels absent from model_state: "
            f"**{audit.get('future_labels_structurally_absent_from_model_state')}**",
            f"- generated_at only in provenance: "
            f"**{audit.get('model_state_excludes_provenance_generated_at')}**",
            "",
            "## Known remaining limitations",
            "",
        ]
    )
    lines.extend(f"- {x}" for x in audit.get("known_remaining_limitations") or [])
    lines.append("")
    return "\n".join(lines)


# Keep v2 name as alias for pipeline compatibility during transition.
def build_pit_v2_audit(**kwargs: Any) -> dict[str, Any]:
    return build_pit_v3_audit(**kwargs)
