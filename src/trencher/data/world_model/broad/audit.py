"""Broad supervision contract + audit artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from trencher.data.world_model.broad.graduation_censor import graduation_censor_decision
from trencher.data.world_model.freeze import FREEZE_ID
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.targets.explosion import PRIMARY_DEFINITION_ID
from trencher.data.world_model.targets.sufficiency import (
    aggregate_sufficiency_streaming,
    sufficiency_row,
)

logger = logging.getLogger(__name__)

CONTRACT_ID = "BROAD_SUPERVISION_CONTRACT_V1"
AUDIT_ID = "BROAD_SUPERVISION_AUDIT_V1"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _recorded_or_not(value: Any) -> Any:
    if value is None:
        return "not_recorded"
    return value


def _recover_build_timings(broad_root: Path) -> dict[str, Any]:
    """Recover phase timings from build_progress.jsonl when available."""
    path = broad_root / "build_progress.jsonl"
    if not path.exists():
        return {}
    recovered: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") != "done":
            continue
        phase = rec.get("phase")
        seconds = rec.get("seconds")
        if isinstance(phase, str) and isinstance(seconds, (int, float)):
            # Last wins — progress file may contain multiple phase runs.
            recovered[phase] = float(seconds)
    return recovered


def build_broad_supervision_contract_v1() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_ID,
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "scope": "broad_event_curve_supervision",
        "frozen_semantics_not_modified": [
            "world_model_freeze_v1",
            "pit_v4",
            "observation_dataset_v1",
            "sequence_dataset_v2",
            "model_input_contract_v2",
            "target_dataset_v3_target_values",
            "target_contract_v3",
            "target_source_coverage_v2",
        ],
        "artifacts": {
            "population": "broad_population_v1.parquet",
            "sequences": [
                "broad_sequence_events_v1/",
                "broad_sequence_curve_v1/",
                "broad_sequence_creator_v1/",
            ],
            "observations": [
                "broad_observation_index_v1/",
                "broad_observation_sequence_refs_v1/",
            ],
            "targets": "broad_targets_v1/",
        },
        "population_policy": {
            "outcome_independent": True,
            "no_preferential_graduate_sampling": True,
            "preserve_all_event_curve_tokens": True,
        },
        "graduation_policy": {
            "retain_conservative_v3": True,
            "tracking_expires_at_as_coverage_end": False,
        },
        "no_splits": True,
        "no_training": True,
        "no_external_acquisition": True,
    }


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _load_target_sample(targets_root: Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    path = targets_root / "observation_targets"
    if not path.exists():
        return []
    dataset = ds.dataset(str(path), format="parquet")
    rows: list[dict[str, Any]] = []
    for batch in dataset.scanner(batch_size=20_000).to_batches():
        names = batch.schema.names
        cols = [batch.column(c).to_pylist() for c in names]
        for vals in zip(*cols, strict=True):
            rows.append(dict(zip(names, vals, strict=True)))
            if max_rows is not None and len(rows) >= max_rows:
                return rows
    return rows


def rich_vs_broad_capability_matrix() -> list[dict[str, Any]]:
    return [
        {
            "capability": "event_history_sequences",
            "rich_4k": True,
            "broad": True,
            "notes": "Broad uses full Pump trades; rich uses atomic subset",
        },
        {
            "capability": "curve_reserve_trajectory",
            "rich_4k": True,
            "broad": True,
            "notes": "Broad from trade reserves; rich also has execution_curve_states",
        },
        {
            "capability": "creator_prior_launches",
            "rich_4k": "graph_edges_partial",
            "broad": True,
            "notes": "Broad reconstructs from tokens timeline (no past_rugs leakage)",
        },
        {
            "capability": "microstructure_rpc",
            "rich_4k": True,
            "broad": False,
            "notes": "RPC sampled on rich subset only",
        },
        {
            "capability": "ownership_ledger_bq",
            "rich_4k": "partial",
            "broad": False,
            "notes": "BQ transfers limited; broad ownership targets sparse/unavailable",
        },
        {
            "capability": "wallet_prior_graph",
            "rich_4k": True,
            "broad": False,
            "notes": "Not in broad event/curve package",
        },
        {
            "capability": "pit_model_state_json",
            "rich_4k": True,
            "broad": False,
            "notes": "Broad stores index + sequence refs only",
        },
        {
            "capability": "price_return_barrier_targets",
            "rich_4k": True,
            "broad": True,
            "notes": "Same compute_* semantics",
        },
        {
            "capability": "graduation_targets",
            "rich_4k": True,
            "broad": True,
            "notes": "Conservative v3 absence censoring retained",
        },
        {
            "capability": "explosion_hazard",
            "rich_4k": "marginal",
            "broad": "scale_audit",
            "notes": "Broad increases independent explosion events",
        },
    ]


def build_broad_supervision_audit_v1(
    *,
    world_root: Path,
    manifest_dir: Path,
    pump_root: Path,
    broad_root: Path | None = None,
    phase_timings: dict[str, float] | None = None,
    phase_metas: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write BROAD_SUPERVISION_CONTRACT/AUDIT JSON + markdown."""
    broad_root = broad_root or world_root / "broad"
    manifest_dir = ensure_dir(manifest_dir)
    phase_metas = phase_metas or {}
    phase_timings = phase_timings or {}

    contract = build_broad_supervision_contract_v1()
    write_json(manifest_dir / f"{CONTRACT_ID}.json", contract)
    write_json(broad_root / f"{CONTRACT_ID}.json", contract)

    censor = graduation_censor_decision(pump_root)
    pop_path = broad_root / "broad_population_v1.parquet"
    pop_counts: dict[str, Any] = {}
    if pop_path.exists():
        pop = pq.read_table(pop_path)
        rows = pop.to_pylist()
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

    # Explosion scale vs rich
    broad_expl_path = broad_root / "broad_targets_v1" / "explosion_events_v1.parquet"
    rich_expl_path = (
        world_root / "targets" / "target_dataset_v3" / "explosion_events_v1.parquet"
    )
    broad_primary = 0
    rich_primary = 0
    tokens_with_expl: set[str] = set()
    if broad_expl_path.exists():
        for r in pq.read_table(broad_expl_path).to_pylist():
            if r.get("definition_id") == PRIMARY_DEFINITION_ID and r.get(
                "explosion_exists"
            ):
                broad_primary += 1
                if r.get("mint"):
                    tokens_with_expl.add(str(r["mint"]))
    if rich_expl_path.exists():
        for r in pq.read_table(rich_expl_path).to_pylist():
            if r.get("definition_id") == PRIMARY_DEFINITION_ID and r.get(
                "explosion_exists"
            ):
                rich_primary += 1

    target_extras: dict[str, Any] = {}
    try:
        sufficiency, target_extras = aggregate_sufficiency_streaming(
            broad_root / "broad_targets_v1",
            primary_explosion_events=broad_primary,
            tokens_with_primary_explosion=tokens_with_expl,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("sufficiency compute failed: %s", exc)
        sufficiency = [
            sufficiency_row(
                head="audit_error",
                valid_observations=0,
                unique_tokens_with_valid_supervision=0,
                positive_observations=0,
                distinct_positive_tokens=0,
                independent_events=0,
                censored_observations=0,
                unavailable_observations=0,
                recommendation="insufficient_data",
                notes=str(exc),
            )
        ]
        target_extras = {}

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
        "targets_bytes": _dir_bytes(broad_root / "broad_targets_v1"),
    }
    disk["total_bytes"] = sum(disk.values())
    disk["total_mb"] = round(disk["total_bytes"] / (1024 * 1024), 3)

    seq_meta = phase_metas.get("sequences") or {}
    obs_meta = phase_metas.get("observations") or {}
    tgt_meta = phase_metas.get("targets") or {}
    # When audit is re-run alone, recover counts from on-disk manifests.
    if not obs_meta:
        obs_meta = _read_json(broad_root / "broad_observation_index_v1" / "manifest.json")
    if not tgt_meta:
        tgt_meta = _read_json(broad_root / "broad_targets_v1" / "broad_targets_manifest.json")
    events_manifest = _read_json(broad_root / "broad_sequence_events_v1" / "manifest.json")
    creator_manifest = _read_json(
        broad_root / "broad_sequence_creator_v1" / "manifest.json"
    )
    if not seq_meta:
        seq_meta = {
            "events": events_manifest,
            "creator": creator_manifest,
        }

    recovered_timings = _recover_build_timings(broad_root)
    build_timings: dict[str, Any] = {}
    # Prefer recovered full-run timings; overlay any timings from this process.
    build_timings.update(recovered_timings)
    build_timings.update(phase_timings)
    if not build_timings:
        build_timings = {"status": "not_recorded"}

    sequence_quality = {
        "excluded_system_program_rows": _recorded_or_not(
            (seq_meta.get("events") or seq_meta).get("excluded_system_program_rows")
            if isinstance(seq_meta.get("events") or seq_meta, dict)
            else None
        ),
        "exact_duplicate_groups_removed": _recorded_or_not(
            events_manifest.get("exact_duplicate_groups_removed")
        ),
        "signatures_with_heterogeneous_event_fingerprints": _recorded_or_not(
            events_manifest.get("signatures_with_heterogeneous_event_fingerprints")
        ),
        "heterogeneous_same_signature_events_preserved": _recorded_or_not(
            events_manifest.get("heterogeneous_same_signature_events_preserved")
        ),
        "invalid_nonfinite_rows_excluded": _recorded_or_not(
            events_manifest.get("invalid_nonfinite_rows_excluded")
        ),
        "source": "broad_sequence_events_v1/manifest.json",
        "note": (
            "Counts taken only from existing manifests; missing fields are "
            "marked not_recorded (resume finalize may omit exclusion tallies)."
        ),
    }
    # Prefer explicit events_manifest values when nested lookup is nullish.
    if sequence_quality["excluded_system_program_rows"] == "not_recorded":
        sequence_quality["excluded_system_program_rows"] = _recorded_or_not(
            events_manifest.get("excluded_system_program_rows")
        )

    audit = {
        "audit_id": AUDIT_ID,
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "contract_version": CONTRACT_ID,
        "population_counts": pop_counts,
        "sequence_counts": {
            "event_rows": (seq_meta.get("events") or seq_meta).get("deduped_event_rows")
            or events_manifest.get("deduped_event_rows"),
            "curve_rows": (seq_meta.get("events") or seq_meta).get("curve_rows")
            or events_manifest.get("curve_rows"),
            "creator_rows": (seq_meta.get("creator") or {}).get("rows")
            or creator_manifest.get("rows"),
            "excluded_system_program_rows": sequence_quality[
                "excluded_system_program_rows"
            ],
            "exact_duplicate_groups_removed": sequence_quality[
                "exact_duplicate_groups_removed"
            ],
            "signatures_with_heterogeneous_event_fingerprints": sequence_quality[
                "signatures_with_heterogeneous_event_fingerprints"
            ],
            "heterogeneous_same_signature_events_preserved": sequence_quality[
                "heterogeneous_same_signature_events_preserved"
            ],
            "invalid_nonfinite_rows_excluded": sequence_quality[
                "invalid_nonfinite_rows_excluded"
            ],
        },
        "sequence_quality": sequence_quality,
        "observation_counts": {
            "tokens_n": obs_meta.get("tokens_n"),
            "observations_n": obs_meta.get("observations_n"),
            "projected": obs_meta.get("projected_row_counts"),
            "actual": obs_meta.get("actual_row_counts"),
        },
        "target_counts": {
            "observations_processed": tgt_meta.get("observations_processed"),
            "primary_explosion_events": tgt_meta.get("primary_explosion_events"),
            "mints_with_primary_explosion": tgt_meta.get(
                "mints_with_primary_explosion"
            ),
            "explosion_definition_rows": tgt_meta.get("explosion_definition_rows"),
        },
        "disk": disk,
        "build_timings_sec": build_timings,
        "explosion_scale_audit": {
            "rich_4k_primary_explosion_events": rich_primary,
            "broad_primary_explosion_events": broad_primary,
            "broad_distinct_tokens_with_primary": len(tokens_with_expl),
            "scale_factor_vs_rich": (
                (broad_primary / rich_primary) if rich_primary else None
            ),
            "imminence_positives_by_horizon": target_extras.get(
                "explosion_imminence_positives_by_horizon"
            ),
            "imminence_valid_tokens_by_horizon": target_extras.get(
                "explosion_imminence_valid_tokens_by_horizon"
            ),
            "pre_explosion_observations": target_extras.get(
                "pre_explosion_observations"
            ),
            "observations_scanned_for_sufficiency": target_extras.get(
                "observations_scanned"
            ),
            "price_target_available_observations": target_extras.get(
                "price_target_available_observations"
            ),
        },
        "graduation_decision": censor,
        "rich_vs_broad_capability_matrix": rich_vs_broad_capability_matrix(),
        "supervision_sufficiency": sufficiency,
        "sufficiency_field_contract_version": "supervision_sufficiency_v3",
        "phase_metas": {
            k: {kk: vv for kk, vv in (v or {}).items() if kk != "price_capable_mint_set"}
            for k, v in phase_metas.items()
        },
    }
    write_json(manifest_dir / f"{AUDIT_ID}.json", audit)
    write_json(broad_root / f"{AUDIT_ID}.json", audit)
    md = render_broad_audit_md(audit)
    (manifest_dir / f"{AUDIT_ID}.md").write_text(md, encoding="utf-8")
    (broad_root / f"{AUDIT_ID}.md").write_text(md, encoding="utf-8")
    return audit


def render_broad_audit_md(audit: dict[str, Any]) -> str:
    lines = [
        f"# Broad Supervision Audit (`{audit['audit_id']}`)",
        "",
        f"Generated at: `{audit['generated_at']}`",
        f"Freeze: `{audit['freeze_id']}`",
        "",
        "## Population",
        "",
        f"```json\n{audit.get('population_counts')}\n```",
        "",
        "## Counts",
        "",
        f"- sequences: `{audit.get('sequence_counts')}`",
        f"- sequence_quality: `{audit.get('sequence_quality')}`",
        f"- observations: `{audit.get('observation_counts')}`",
        f"- targets: `{audit.get('target_counts')}`",
        f"- disk MB: **{(audit.get('disk') or {}).get('total_mb')}**",
        f"- timings_sec: `{audit.get('build_timings_sec')}`",
        "",
        "## Explosion scale vs rich 4k",
        "",
        f"```json\n{audit.get('explosion_scale_audit')}\n```",
        "",
        "## Graduation decision",
        "",
    ]
    gd = audit.get("graduation_decision") or {}
    lines.extend(
        [
            (
                "- retain_conservative_v3: **"
                f"{gd.get('retain_conservative_v3_graduation_treatment')}**"
            ),
            (
                "- tracking_expires proven end: **"
                f"{gd.get('tracking_expires_at_is_proven_graduation_tracking_end')}**"
            ),
            "",
            "## Rich vs broad capability matrix",
            "",
        ]
    )
    for row in audit.get("rich_vs_broad_capability_matrix") or []:
        lines.append(
            f"- `{row['capability']}`: rich=`{row['rich_4k']}` "
            f"broad=`{row['broad']}` — {row.get('notes')}"
        )
    lines.extend(["", "## Supervision sufficiency", ""])
    for s in audit.get("supervision_sufficiency") or []:
        lines.append(
            f"- **{s.get('head')}**: valid_tokens="
            f"{s.get('unique_tokens_with_valid_supervision')}, "
            f"pos_tokens={s.get('distinct_positive_tokens')}, "
            f"events={s.get('independent_events')}, "
            f"rec=`{s.get('recommendation')}`"
        )
    lines.append("")
    return "\n".join(lines)
