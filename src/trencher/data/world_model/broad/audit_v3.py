"""Broad supervision contract/audit v3 (missing-primary stub fix)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trencher.data.world_model.broad.audit import (
    _dir_bytes,
    _read_json,
    _recorded_or_not,
    _recover_build_timings,
    rich_vs_broad_capability_matrix,
)
from trencher.data.world_model.broad.audit_v2 import (
    _count_primary_exists,
    _rich_explosion_imminence_report,
)
from trencher.data.world_model.broad.graduation_censor import graduation_censor_decision
from trencher.data.world_model.broad.targets_v2 import (
    TARGETS_V1_STORE,
    TARGETS_V2_STORE,
    load_explosion_defs_by_mint,
)
from trencher.data.world_model.broad.targets_v3 import TARGETS_V3_STORE
from trencher.data.world_model.freeze import FREEZE_ID
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.targets.status import (
    EXPLOSION_IMMINENCE_HORIZONS_SEC,
    STATUS_VALID_NEGATIVE,
)
from trencher.data.world_model.targets.sufficiency import aggregate_sufficiency_streaming

logger = logging.getLogger(__name__)

CONTRACT_ID = "BROAD_SUPERVISION_CONTRACT_V3"
AUDIT_ID = "BROAD_SUPERVISION_AUDIT_V3"


def build_broad_supervision_contract_v3() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_ID,
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "supersedes": "BROAD_SUPERVISION_CONTRACT_V2",
        "scope": "broad_event_curve_supervision",
        "explosion_primary_semantics": (
            "Keep every stored primary_event row whether explosion_exists is "
            "true or false. If no stored primary detector row exists for a mint, "
            "explosion imminence is modality_unavailable — never fabricate a "
            "synthetic explosion_exists=False stub."
        ),
        "invariant": "no_detector_row_never_valid_explosion_negative",
        "targets": {
            "v1_preserved": TARGETS_V1_STORE,
            "v2_preserved": TARGETS_V2_STORE,
            "v3_canonical": TARGETS_V3_STORE,
            "v3_repair": (
                "explosion_columns_only; missing_primary_no_stub_from_v2"
            ),
        },
        "frozen_semantics_not_modified": [
            "world_model_freeze_v1",
            "pit_v4",
            "observation_dataset_v1",
            "sequence_dataset_v2",
            "model_input_contract_v2",
            "target_dataset_v3_target_values",
            "target_contract_v3",
            "target_source_coverage_v2",
            "explosion_detector_v1",
        ],
        "no_splits": True,
        "no_training": True,
        "no_external_acquisition": True,
    }


def _population_mint_sets(pop_path: Path) -> dict[str, Any]:
    if not pop_path.exists():
        return {}
    rows = pq.read_table(pop_path).to_pylist()
    event = {str(r["mint"]) for r in rows if r.get("event_curve_population")}
    price = {str(r["mint"]) for r in rows if r.get("price_target_population")}
    return {
        "launch_population": sum(1 for r in rows if r.get("launch_population")),
        "event_curve_population": len(event),
        "price_target_population": len(price),
        "graduation_event_population": sum(
            1 for r in rows if r.get("graduation_event_population")
        ),
        "has_migration_evidence_metadata": sum(
            1 for r in rows if r.get("has_migration_evidence")
        ),
        "_event_mints": event,
        "_price_mints": price,
    }


def _verify_no_detector_never_valid_negative(
    targets_dir: Path,
    primary_mints: set[str],
    *,
    sample_horizon: int = 60,
) -> dict[str, Any]:
    """Scan observation parts: missing-primary rows must not be valid_negative."""
    status_col = f"explosion_within_{sample_horizon}s_status"
    parts = sorted((targets_dir / "observation_targets").glob("part-*.parquet"))
    scanned = 0
    missing_primary_obs = 0
    violations = 0
    for path in parts:
        table = pq.read_table(path, columns=["mint", status_col])
        mints = table.column("mint").to_pylist()
        statuses = table.column(status_col).to_pylist()
        for mint, st in zip(mints, statuses, strict=True):
            scanned += 1
            if str(mint) in primary_mints:
                continue
            missing_primary_obs += 1
            if st == STATUS_VALID_NEGATIVE:
                violations += 1
    return {
        "horizon_sec": sample_horizon,
        "observations_scanned": scanned,
        "observations_without_primary_row": missing_primary_obs,
        "valid_negative_violations": violations,
        "passed": violations == 0,
    }


def build_broad_supervision_audit_v3(
    *,
    world_root: Path,
    manifest_dir: Path,
    pump_root: Path,
    broad_root: Path | None = None,
    phase_timings: dict[str, float] | None = None,
    phase_metas: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write BROAD_SUPERVISION_CONTRACT/AUDIT V3 after targets_v3 repair."""
    broad_root = broad_root or world_root / "broad"
    manifest_dir = ensure_dir(manifest_dir)
    phase_metas = phase_metas or {}
    phase_timings = phase_timings or {}

    contract = build_broad_supervision_contract_v3()
    write_json(manifest_dir / f"{CONTRACT_ID}.json", contract)
    write_json(broad_root / f"{CONTRACT_ID}.json", contract)

    censor = graduation_censor_decision(pump_root)
    pop_path = broad_root / "broad_population_v1.parquet"
    pop_raw = _population_mint_sets(pop_path)
    event_mints: set[str] = pop_raw.pop("_event_mints", set())
    price_mints: set[str] = pop_raw.pop("_price_mints", set())
    pop_counts = {k: v for k, v in pop_raw.items()}

    expl_path = broad_root / TARGETS_V1_STORE / "explosion_events_v1.parquet"
    primary_by_mint, _, primary_exists_n = load_explosion_defs_by_mint(expl_path)
    primary_mints = set(primary_by_mint)
    without_detector = event_mints - primary_mints

    detector_breakdown = {
        "event_curve_mints": len(event_mints),
        "price_capable_mints": len(price_mints),
        "mints_with_stored_primary_detector_row": len(primary_mints),
        "mints_without_detector_row": len(without_detector),
        "primary_exists_true": primary_exists_n,
        "primary_exists_false": len(primary_mints) - primary_exists_n,
        "price_capable_without_primary": len(price_mints - primary_mints),
        "event_curve_without_primary": len(without_detector),
    }

    broad_primary, tokens_with_expl = _count_primary_exists(expl_path)
    rich_primary, _ = _count_primary_exists(
        world_root / "targets" / "target_dataset_v3" / "explosion_events_v1.parquet"
    )

    logger.info("Streaming sufficiency for broad_targets_v2 (stub baseline)")
    suff_v2, extras_v2 = aggregate_sufficiency_streaming(
        broad_root / TARGETS_V2_STORE,
        primary_explosion_events=broad_primary,
        tokens_with_primary_explosion=tokens_with_expl,
        population="broad_targets_v2_stub_missing_primary",
    )
    logger.info("Streaming sufficiency for broad_targets_v3 (no stub)")
    suff_v3, extras_v3 = aggregate_sufficiency_streaming(
        broad_root / TARGETS_V3_STORE,
        primary_explosion_events=broad_primary,
        tokens_with_primary_explosion=tokens_with_expl,
        population="broad_targets_v3_no_stub",
    )

    logger.info("Verifying invariant: no detector row → never valid negative")
    invariant_scan = _verify_no_detector_never_valid_negative(
        broad_root / TARGETS_V3_STORE, primary_mints
    )
    if not invariant_scan["passed"]:
        raise RuntimeError(
            "AUDIT_V3 invariant failed: "
            f"{invariant_scan['valid_negative_violations']} valid_negative "
            "statuses on observations without a primary detector row"
        )

    rich_imminence = _rich_explosion_imminence_report(
        world_root=world_root, independent_events=rich_primary
    )

    obs_meta = phase_metas.get("observations") or _read_json(
        broad_root / "broad_observation_index_v1" / "manifest.json"
    )
    tgt_meta = phase_metas.get("targets_v3") or _read_json(
        broad_root / TARGETS_V3_STORE / "broad_targets_manifest.json"
    )
    events_manifest = _read_json(
        broad_root / "broad_sequence_events_v1" / "manifest.json"
    )
    creator_manifest = _read_json(
        broad_root / "broad_sequence_creator_v1" / "manifest.json"
    )

    recovered = _recover_build_timings(broad_root)
    build_timings: dict[str, Any] = {}
    build_timings.update(recovered)
    build_timings.update(phase_timings)
    if not build_timings:
        build_timings = {"status": "not_recorded"}

    disk = {
        "population_bytes": _dir_bytes(pop_path),
        "sequences_bytes": (
            _dir_bytes(broad_root / "broad_sequence_events_v1")
            + _dir_bytes(broad_root / "broad_sequence_curve_v1")
            + _dir_bytes(broad_root / "broad_sequence_creator_v1")
        ),
        "observations_bytes": (
            _dir_bytes(broad_root / "broad_observation_index_v1")
            + _dir_bytes(broad_root / "broad_observation_sequence_refs_v1")
        ),
        "targets_v1_bytes": _dir_bytes(broad_root / TARGETS_V1_STORE),
        "targets_v2_bytes": _dir_bytes(broad_root / TARGETS_V2_STORE),
        "targets_v3_bytes": _dir_bytes(broad_root / TARGETS_V3_STORE),
    }
    disk["total_bytes"] = sum(disk.values())
    disk["total_mb"] = round(disk["total_bytes"] / (1024 * 1024), 3)

    v3_horizons = extras_v3.get("explosion_imminence_by_horizon") or {}
    explosion_comparison = {
        "independent_primary_explosion_events_broad": broad_primary,
        "independent_primary_explosion_events_rich": rich_primary,
        "rich_4k": rich_imminence,
        "broad_targets_v2_stub_missing_primary": extras_v2.get(
            "explosion_imminence_by_horizon"
        ),
        "broad_targets_v3_no_stub": v3_horizons,
        "detector_row_breakdown": detector_breakdown,
        "invariant_scan": invariant_scan,
        "invariants": {
            "positive_subseteq_valid_v3": all(
                (v3_horizons.get(str(h), {}).get("distinct_positive_tokens", 0))
                <= (
                    v3_horizons.get(str(h), {}).get(
                        "unique_tokens_with_valid_supervision", 0
                    )
                )
                for h in EXPLOSION_IMMINENCE_HORIZONS_SEC
            ),
            "primary_event_count_unchanged": broad_primary == 49836,
            "no_detector_row_never_valid_explosion_negative": invariant_scan["passed"],
        },
    }

    audit = {
        "audit_id": AUDIT_ID,
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "contract_version": CONTRACT_ID,
        "supersedes": "BROAD_SUPERVISION_AUDIT_V2",
        "population_counts": pop_counts,
        "detector_row_breakdown": detector_breakdown,
        "sequence_counts": {
            "event_rows": events_manifest.get("deduped_event_rows"),
            "curve_rows": events_manifest.get("curve_rows"),
            "creator_rows": creator_manifest.get("rows"),
            "excluded_system_program_rows": _recorded_or_not(
                events_manifest.get("excluded_system_program_rows")
            ),
            "exact_duplicate_groups_removed": _recorded_or_not(
                events_manifest.get("exact_duplicate_groups_removed")
            ),
            "signatures_with_heterogeneous_event_fingerprints": _recorded_or_not(
                events_manifest.get("signatures_with_heterogeneous_event_fingerprints")
            ),
            "invalid_nonfinite_rows_excluded": _recorded_or_not(
                events_manifest.get("invalid_nonfinite_rows_excluded")
            ),
        },
        "observation_counts": {
            "tokens_n": obs_meta.get("tokens_n"),
            "observations_n": obs_meta.get("observations_n"),
            "projected": obs_meta.get("projected_row_counts"),
            "actual": obs_meta.get("actual_row_counts"),
        },
        "target_counts": {
            "v1_preserved": True,
            "v2_preserved": True,
            "v3_observations_processed": tgt_meta.get("observations_processed"),
            "primary_explosion_events": tgt_meta.get("primary_explosion_events")
            or broad_primary,
            "mints_with_primary_definition_row": tgt_meta.get(
                "mints_with_primary_definition_row"
            )
            or len(primary_mints),
            "mints_without_primary_definition_row": tgt_meta.get(
                "mints_without_primary_definition_row"
            )
            or len(without_detector),
            "explosion_repair": tgt_meta.get("explosion_repair"),
        },
        "disk": disk,
        "build_timings_sec": build_timings,
        "graduation_decision": censor,
        "rich_vs_broad_capability_matrix": rich_vs_broad_capability_matrix(),
        "supervision_sufficiency_v3": suff_v3,
        "supervision_sufficiency_v2_stub_baseline": suff_v2,
        "explosion_imminence_comparison": explosion_comparison,
        "explosion_imminence_by_horizon_v3": v3_horizons,
        "sufficiency_field_contract_version": "supervision_sufficiency_v3",
        "phase_metas": {
            k: {
                kk: vv
                for kk, vv in (v or {}).items()
                if kk != "price_capable_mint_set"
            }
            for k, v in phase_metas.items()
        },
    }
    write_json(manifest_dir / f"{AUDIT_ID}.json", audit)
    write_json(broad_root / f"{AUDIT_ID}.json", audit)
    md = render_broad_audit_v3_md(audit)
    (manifest_dir / f"{AUDIT_ID}.md").write_text(md, encoding="utf-8")
    (broad_root / f"{AUDIT_ID}.md").write_text(md, encoding="utf-8")
    return audit


def render_broad_audit_v3_md(audit: dict[str, Any]) -> str:
    lines = [
        f"# Broad Supervision Audit (`{audit['audit_id']}`)",
        "",
        f"Generated at: `{audit['generated_at']}`",
        f"Freeze: `{audit['freeze_id']}`",
        f"Supersedes: `{audit.get('supersedes')}`",
        "",
        "## Detector row breakdown",
        "",
        f"- `{audit.get('detector_row_breakdown')}`",
        "",
        "## Explosion imminence by horizon (v3)",
        "",
    ]
    for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
        h60 = (audit.get("explosion_imminence_by_horizon_v3") or {}).get(str(h)) or {}
        if not h60:
            continue
        lines.append(
            f"- **{h}s**: valid={h60.get('valid_observations')}, "
            f"neg={h60.get('valid_negative_observations')}, "
            f"pos={h60.get('valid_positive_observations')}, "
            f"censored={h60.get('censored_observations')}, "
            f"unavailable={h60.get('unavailable_observations')}, "
            f"prev_obs={h60.get('prevalence_among_valid_observations')}, "
            f"prev_tok={h60.get('prevalence_among_valid_tokens')}"
        )
    comp = audit.get("explosion_imminence_comparison") or {}
    lines.extend(
        [
            "",
            f"- invariants: `{comp.get('invariants')}`",
            f"- invariant_scan: `{comp.get('invariant_scan')}`",
            "",
            "## Population / counts",
            "",
            f"- population: `{audit.get('population_counts')}`",
            f"- sequences: `{audit.get('sequence_counts')}`",
            f"- observations: `{audit.get('observation_counts')}`",
            f"- targets: `{audit.get('target_counts')}`",
            f"- timings: `{audit.get('build_timings_sec')}`",
            f"- disk MB: **{(audit.get('disk') or {}).get('total_mb')}**",
            "",
        ]
    )
    return "\n".join(lines)
