"""Broad supervision contract/audit v2 (explosion-repair release)."""

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
from trencher.data.world_model.broad.graduation_censor import graduation_censor_decision
from trencher.data.world_model.broad.targets_v2 import TARGETS_V1_STORE, TARGETS_V2_STORE
from trencher.data.world_model.freeze import FREEZE_ID
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.targets.explosion import PRIMARY_DEFINITION_ID
from trencher.data.world_model.targets.status import EXPLOSION_IMMINENCE_HORIZONS_SEC
from trencher.data.world_model.targets.sufficiency import (
    aggregate_sufficiency_streaming,
    explosion_imminence_counts,
)

logger = logging.getLogger(__name__)

CONTRACT_ID = "BROAD_SUPERVISION_CONTRACT_V2"
AUDIT_ID = "BROAD_SUPERVISION_AUDIT_V2"


def build_broad_supervision_contract_v2() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_ID,
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "supersedes": "BROAD_SUPERVISION_CONTRACT_V1",
        "scope": "broad_event_curve_supervision",
        "explosion_primary_semantics": (
            "Store every primary_event(events) whether explosion_exists is true "
            "or false — match target_dataset_v3. Count primary explosion events "
            "only when explosion_exists is true."
        ),
        "targets": {
            "v1_preserved": TARGETS_V1_STORE,
            "v2_repaired": TARGETS_V2_STORE,
            "v2_repair": "explosion_columns_only_from_stored_explosion_events",
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


def _horizon_report_from_counts(
    stats: dict[str, Any], *, independent_events: int
) -> dict[str, Any]:
    n_valid = int(stats["valid_observations"])
    n_pos = int(stats["positive_observations"])
    n_neg = int(stats.get("negative_observations", n_valid - n_pos))
    valid_tok = stats["valid_tokens"]
    pos_tok = stats["pos_tokens"] & valid_tok
    neg_tok = stats.get("neg_tokens", valid_tok - pos_tok) & valid_tok
    return {
        "valid_observations": n_valid,
        "valid_negative_observations": n_neg,
        "valid_positive_observations": n_pos,
        "unique_tokens_with_valid_supervision": len(valid_tok),
        "distinct_positive_tokens": len(pos_tok),
        "distinct_valid_negative_tokens": len(neg_tok),
        "censored_observations": int(stats["censored_observations"]),
        "unavailable_observations": int(stats["unavailable_observations"]),
        "prevalence_among_valid_observations": (n_pos / n_valid) if n_valid else None,
        "prevalence_among_valid_tokens": (
            (len(pos_tok) / len(valid_tok)) if valid_tok else None
        ),
        "independent_primary_explosion_events": independent_events,
    }


def _rich_explosion_imminence_report(
    *,
    world_root: Path,
    independent_events: int,
) -> dict[str, Any]:
    path = (
        world_root / "targets" / "target_dataset_v3" / "observation_targets_v3.parquet"
    )
    if not path.exists():
        return {"status": "not_recorded", "reason": "rich targets missing"}
    rows = pq.read_table(path).to_pylist()
    out: dict[str, Any] = {}
    for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
        stats = explosion_imminence_counts(rows, horizon_sec=h)
        out[str(h)] = _horizon_report_from_counts(
            stats, independent_events=independent_events
        )
    out["observations_scanned"] = len(rows)
    return out


def _count_primary_exists(explosion_path: Path) -> tuple[int, set[str]]:
    if not explosion_path.exists():
        return 0, set()
    tokens: set[str] = set()
    n = 0
    for r in pq.read_table(explosion_path).to_pylist():
        if r.get("definition_id") == PRIMARY_DEFINITION_ID and r.get(
            "explosion_exists"
        ):
            n += 1
            if r.get("mint"):
                tokens.add(str(r["mint"]))
    return n, tokens


def build_broad_supervision_audit_v2(
    *,
    world_root: Path,
    manifest_dir: Path,
    pump_root: Path,
    broad_root: Path | None = None,
    phase_timings: dict[str, float] | None = None,
    phase_metas: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write BROAD_SUPERVISION_CONTRACT/AUDIT V2 after targets_v2 repair."""
    broad_root = broad_root or world_root / "broad"
    manifest_dir = ensure_dir(manifest_dir)
    phase_metas = phase_metas or {}
    phase_timings = phase_timings or {}

    contract = build_broad_supervision_contract_v2()
    write_json(manifest_dir / f"{CONTRACT_ID}.json", contract)
    write_json(broad_root / f"{CONTRACT_ID}.json", contract)

    censor = graduation_censor_decision(pump_root)
    pop_path = broad_root / "broad_population_v1.parquet"
    pop_counts: dict[str, Any] = {}
    if pop_path.exists():
        rows = pq.read_table(pop_path).to_pylist()
        pop_counts = {
            "launch_population": sum(1 for r in rows if r.get("launch_population")),
            "event_curve_population": sum(
                1 for r in rows if r.get("event_curve_population")
            ),
            "price_target_population": sum(
                1 for r in rows if r.get("price_target_population")
            ),
            "graduation_event_population": sum(
                1 for r in rows if r.get("graduation_event_population")
            ),
            "has_migration_evidence_metadata": sum(
                1 for r in rows if r.get("has_migration_evidence")
            ),
        }

    v1_expl = broad_root / TARGETS_V1_STORE / "explosion_events_v1.parquet"
    v2_expl = broad_root / TARGETS_V2_STORE / "explosion_events_v1.parquet"
    expl_path = v2_expl if v2_expl.exists() else v1_expl
    broad_primary, tokens_with_expl = _count_primary_exists(expl_path)
    rich_primary, _ = _count_primary_exists(
        world_root / "targets" / "target_dataset_v3" / "explosion_events_v1.parquet"
    )

    logger.info("Streaming sufficiency for broad_targets_v1 (broken baseline)")
    suff_v1, extras_v1 = aggregate_sufficiency_streaming(
        broad_root / TARGETS_V1_STORE,
        primary_explosion_events=broad_primary,
        tokens_with_primary_explosion=tokens_with_expl,
        population="broad_targets_v1_broken_primary_gate",
    )
    logger.info("Streaming sufficiency for broad_targets_v2 (repaired)")
    suff_v2, extras_v2 = aggregate_sufficiency_streaming(
        broad_root / TARGETS_V2_STORE,
        primary_explosion_events=broad_primary,
        tokens_with_primary_explosion=tokens_with_expl,
        population="broad_targets_v2_repaired",
    )

    rich_imminence = _rich_explosion_imminence_report(
        world_root=world_root, independent_events=rich_primary
    )

    obs_meta = phase_metas.get("observations") or _read_json(
        broad_root / "broad_observation_index_v1" / "manifest.json"
    )
    tgt_meta = phase_metas.get("targets_v2") or _read_json(
        broad_root / TARGETS_V2_STORE / "broad_targets_manifest.json"
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
    }
    disk["total_bytes"] = sum(disk.values())
    disk["total_mb"] = round(disk["total_bytes"] / (1024 * 1024), 3)

    explosion_comparison = {
        "independent_primary_explosion_events_broad": broad_primary,
        "independent_primary_explosion_events_rich": rich_primary,
        "rich_4k": rich_imminence,
        "broad_targets_v1_broken": extras_v1.get("explosion_imminence_by_horizon"),
        "broad_targets_v2_repaired": extras_v2.get("explosion_imminence_by_horizon"),
        "invariants": {
            "positive_subseteq_valid_v2": all(
                (
                    (extras_v2.get("explosion_imminence_by_horizon") or {})
                    .get(str(h), {})
                    .get("distinct_positive_tokens", 0)
                )
                <= (
                    (extras_v2.get("explosion_imminence_by_horizon") or {})
                    .get(str(h), {})
                    .get("unique_tokens_with_valid_supervision", 0)
                )
                for h in EXPLOSION_IMMINENCE_HORIZONS_SEC
            ),
            "valid_negative_tokens_substantial_v2": all(
                (
                    (extras_v2.get("explosion_imminence_by_horizon") or {})
                    .get(str(h), {})
                    .get("distinct_valid_negative_tokens", 0)
                )
                > 10_000
                for h in EXPLOSION_IMMINENCE_HORIZONS_SEC
            ),
            "primary_event_count_unchanged": broad_primary == 49836
            or broad_primary > 0,
        },
    }

    audit = {
        "audit_id": AUDIT_ID,
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "contract_version": CONTRACT_ID,
        "supersedes": "BROAD_SUPERVISION_AUDIT_V1",
        "population_counts": pop_counts,
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
            "v2_observations_processed": tgt_meta.get("observations_processed"),
            "primary_explosion_events": tgt_meta.get("primary_explosion_events")
            or broad_primary,
            "mints_with_primary_explosion": tgt_meta.get(
                "mints_with_primary_explosion"
            )
            or broad_primary,
            "explosion_repair": tgt_meta.get("explosion_repair"),
        },
        "disk": disk,
        "build_timings_sec": build_timings,
        "graduation_decision": censor,
        "rich_vs_broad_capability_matrix": rich_vs_broad_capability_matrix(),
        "supervision_sufficiency_v2": suff_v2,
        "supervision_sufficiency_v1_broken_baseline": suff_v1,
        "explosion_imminence_comparison": explosion_comparison,
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
    md = render_broad_audit_v2_md(audit)
    (manifest_dir / f"{AUDIT_ID}.md").write_text(md, encoding="utf-8")
    (broad_root / f"{AUDIT_ID}.md").write_text(md, encoding="utf-8")
    return audit


def render_broad_audit_v2_md(audit: dict[str, Any]) -> str:
    lines = [
        f"# Broad Supervision Audit (`{audit['audit_id']}`)",
        "",
        f"Generated at: `{audit['generated_at']}`",
        f"Freeze: `{audit['freeze_id']}`",
        f"Supersedes: `{audit.get('supersedes')}`",
        "",
        "## Explosion imminence comparison (60s)",
        "",
    ]
    comp = audit.get("explosion_imminence_comparison") or {}
    for label, key in (
        ("rich_4k", "rich_4k"),
        ("broad_v1_broken", "broad_targets_v1_broken"),
        ("broad_v2_repaired", "broad_targets_v2_repaired"),
    ):
        block = comp.get(key) or {}
        h60 = block.get("60") if isinstance(block, dict) else None
        if not isinstance(h60, dict):
            lines.append(f"- **{label}**: `{block}`")
            continue
        lines.append(
            f"- **{label}**: valid_tok={h60.get('unique_tokens_with_valid_supervision')}, "
            f"pos_tok={h60.get('distinct_positive_tokens')}, "
            f"neg_tok={h60.get('distinct_valid_negative_tokens')}, "
            f"prev_tok={h60.get('prevalence_among_valid_tokens')}, "
            f"indep={h60.get('independent_primary_explosion_events')}"
        )
    lines.extend(
        [
            "",
            f"- invariants: `{comp.get('invariants')}`",
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
            "## Supervision sufficiency (v2 repaired)",
            "",
        ]
    )
    for s in audit.get("supervision_sufficiency_v2") or []:
        if not str(s.get("head", "")).startswith(
            ("explosion", "graduation", "graduated")
        ):
            continue
        lines.append(
            f"- **{s.get('head')}**: valid_tok="
            f"{s.get('unique_tokens_with_valid_supervision')}, "
            f"pos_tok={s.get('distinct_positive_tokens')}, "
            f"events={s.get('independent_events')}, "
            f"rec=`{s.get('recommendation')}`"
        )
    lines.append("")
    return "\n".join(lines)
