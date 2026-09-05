"""Repair broad_targets_v2 explosion columns without rebuilding upstream data.

Copies non-explosion fields from ``broad_targets_v1`` and recomputes only
``compute_explosion_obs`` outputs using stored ``explosion_events`` with rich
semantics: every primary definition row is retained whether or not
``explosion_exists``.
"""

from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.broad.targets import (
    FUTURE_PATH_PARTS,
    OBS_INDEX_STORE,
    _normalize_row,
    _schema_from_rows,
)
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.targets.compute import compute_explosion_obs
from trencher.data.world_model.targets.coverage import build_source_coverage_registry
from trencher.data.world_model.targets.explosion import (
    PRIMARY_DEFINITION_ID,
    primary_event,
)
from trencher.data.world_model.targets.status import EXPLOSION_IMMINENCE_HORIZONS_SEC

logger = logging.getLogger(__name__)

TARGETS_V1_STORE = "broad_targets_v1"
TARGETS_V2_STORE = "broad_targets_v2"

# Columns owned exclusively by compute_explosion_obs (plus related aliases).
_EXPLOSION_PREFIXES = (
    "explosion_",
    "time_to_explosion",
    "time_since_explosion",
    "time_from_t_to_explosion",
    "remaining_upside_to_explosion",
    "peak_to_post_peak",
    "gross_opportunity_if_enter_at_t",
)


def _is_explosion_column(name: str) -> bool:
    return name.startswith(_EXPLOSION_PREFIXES) or name in {
        "time_to_explosion_peak_sec",
        "explosion_duration_onset_to_peak_sec",
    }


def _stub_primary_absent() -> dict[str, Any]:
    return {
        "definition_id": PRIMARY_DEFINITION_ID,
        "definition_label": "primary",
        "explosion_exists": False,
        "onset_unix": None,
        "peak_unix": None,
        "end_unix": None,
        "censoring_status": None,
        "peak_price": None,
        "explosion_magnitude": None,
        "onset_to_peak_sec": None,
        "max_drawdown_after_peak": None,
    }


def load_explosion_defs_by_mint(
    explosion_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], int]:
    """Load stored detector rows → primary_by_mint + all_defs; count exists=True."""
    if not explosion_path.exists():
        return {}, {}, 0
    table = pq.read_table(explosion_path)
    by_mint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table.to_pylist():
        mint = row.get("mint")
        if not mint:
            continue
        by_mint[str(mint)].append(row)

    primary_by_mint: dict[str, dict[str, Any]] = {}
    primary_exists_n = 0
    for mint, events in by_mint.items():
        prim = primary_event(events)
        if prim:
            primary_by_mint[mint] = prim
            if prim.get("explosion_exists"):
                primary_exists_n += 1
    return primary_by_mint, dict(by_mint), primary_exists_n


def _ref_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_price": row.get("reference_price"),
        "reference_price_time": row.get("reference_price_time"),
        "reference_price_source": row.get("reference_price_source"),
        "reference_price_age_sec": row.get("reference_price_age_sec"),
        "price_target_available": bool(row.get("price_target_available")),
        "reference_regime": row.get("reference_regime"),
        "reference_v_sol": row.get("reference_v_sol"),
        "reference_v_tok": row.get("reference_v_tok"),
    }


def repair_explosion_columns_for_row(
    row: dict[str, Any],
    *,
    primary: dict[str, Any] | None,
    all_defs: list[dict[str, Any]],
    coverage: Any,
) -> dict[str, Any]:
    """Return a copy of ``row`` with explosion fields recomputed."""
    out = dict(row)
    T = float(out["observation_time_t"])
    expl = compute_explosion_obs(
        T=T,
        primary=primary if primary is not None else _stub_primary_absent(),
        all_defs=all_defs,
        coverage=coverage,
        ref=_ref_from_row(out),
    )
    # Clear prior explosion columns then apply recomputed values.
    for key in list(out):
        if _is_explosion_column(key):
            out[key] = None
    out.update(expl)
    out["target_dataset_version"] = TARGETS_V2_STORE
    out["label_semantics"] = "target_dataset_v3_compute_reuse_explosion_repaired"
    return out


def build_broad_targets_v2_from_v1(
    *,
    world_root: Path,
    pump_root: Path,
    broad_root: Path | None = None,
    batch_flush: int = 5_000,
) -> dict[str, Any]:
    """Materialize broad_targets_v2 by repairing explosion columns on v1 rows."""
    broad_root = ensure_dir(broad_root or world_root / "broad")
    v1_dir = broad_root / TARGETS_V1_STORE
    v2_dir = ensure_dir(broad_root / TARGETS_V2_STORE)
    v1_targets = v1_dir / "observation_targets"
    if not v1_targets.exists():
        raise FileNotFoundError(f"missing {v1_targets}")

    # Fresh v2 outputs (preserve v1 untouched).
    for name in ("observation_targets", "future_path_refs", "explosion_events"):
        p = v2_dir / name
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    for name in ("explosion_events_v1.parquet", "broad_targets_manifest.json"):
        (v2_dir / name).unlink(missing_ok=True)

    expl_src = v1_dir / "explosion_events_v1.parquet"
    primary_by_mint, all_defs_by_mint, primary_exists_n = load_explosion_defs_by_mint(
        expl_src
    )
    logger.info(
        "Loaded explosion defs for %d mints (%d primary exists=True)",
        len(primary_by_mint),
        primary_exists_n,
    )

    coverage = build_source_coverage_registry(
        world_root=world_root, pump_root=pump_root
    )
    # Reuse coverage + detector config artifacts.
    shutil.copy2(
        v1_dir / "target_source_coverage_v2.json",
        v2_dir / "target_source_coverage_v2.json",
    )
    shutil.copy2(
        v1_dir / "explosion_detector_config_v1.json",
        v2_dir / "explosion_detector_config_v1.json",
    )

    def _link_or_copy(src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    # Unchanged artifacts: symlink to v1 (or copy if symlink unsupported).
    _link_or_copy(expl_src, v2_dir / "explosion_events_v1.parquet")
    if (v1_dir / "explosion_events").exists():
        _link_or_copy(v1_dir / "explosion_events", v2_dir / "explosion_events")
    if (v1_dir / "future_path_refs").exists():
        _link_or_copy(v1_dir / "future_path_refs", v2_dir / "future_path_refs")
    if (v1_dir / FUTURE_PATH_PARTS).exists():
        _link_or_copy(v1_dir / FUTURE_PATH_PARTS, v2_dir / FUTURE_PATH_PARTS)

    out_parts = ensure_dir(v2_dir / "observation_targets")
    part_files = sorted(v1_targets.glob("part-*.parquet"))
    if not part_files:
        raise FileNotFoundError(f"no part files under {v1_targets}")

    schema: pa.Schema | None = None
    observations_processed = 0
    modality_before = 0
    modality_after = 0
    valid_neg_after = 0
    valid_pos_after = 0

    for fi, path in enumerate(part_files):
        logger.info(
            "targets_v2 repair file %s (%d/%d) obs_so_far=%d",
            path.name,
            fi + 1,
            len(part_files),
            observations_processed,
        )
        out_path = out_parts / path.name
        writer: pq.ParquetWriter | None = None
        pending: list[dict[str, Any]] = []

        def _write_pending(dest: Path = out_path) -> None:
            nonlocal writer, pending, schema
            if not pending:
                return
            if schema is None:
                schema = _schema_from_rows(pending)
                v1_schema = pq.read_schema(part_files[0])
                fields = []
                seen: set[str] = set()
                by_name = {f.name: f for f in schema}
                for f in v1_schema:
                    if f.name in by_name:
                        fields.append(
                            pa.field(f.name, by_name[f.name].type, nullable=True)
                        )
                        seen.add(f.name)
                for f in schema:
                    if f.name not in seen:
                        fields.append(f)
                schema = pa.schema(fields)
            assert schema is not None
            table = pa.Table.from_pylist(
                [_normalize_row(r, schema) for r in pending], schema=schema
            )
            if writer is None:
                writer = pq.ParquetWriter(dest, schema)
            writer.write_table(table)
            pending = []

        pf = pq.ParquetFile(path)
        for rb in pf.iter_batches(batch_size=batch_flush):
            for row in rb.to_pylist():
                if row.get("explosion_within_60s_status") == "modality_unavailable":
                    modality_before += 1
                mint = str(row["mint"])
                primary = primary_by_mint.get(mint)
                defs = all_defs_by_mint.get(mint, [])
                if primary is None:
                    primary = _stub_primary_absent()
                    if not defs:
                        defs = [primary]
                repaired = repair_explosion_columns_for_row(
                    row, primary=primary, all_defs=defs, coverage=coverage
                )
                st = repaired.get("explosion_within_60s_status")
                if st == "modality_unavailable":
                    modality_after += 1
                elif st == "valid_negative":
                    valid_neg_after += 1
                elif st == "valid_positive":
                    valid_pos_after += 1
                pending.append(repaired)
                observations_processed += 1
                if len(pending) >= batch_flush:
                    _write_pending()
        _write_pending()
        if writer is not None:
            writer.close()

    meta = {
        "generated_at": utc_now(),
        "version": TARGETS_V2_STORE,
        "repaired_from": TARGETS_V1_STORE,
        "observations_processed": observations_processed,
        "primary_explosion_events": primary_exists_n,
        "mints_with_primary_explosion": primary_exists_n,
        "mints_with_primary_definition_row": len(primary_by_mint),
        "explosion_repair": {
            "method": "recompute_compute_explosion_obs_from_stored_explosion_events",
            "primary_semantics": (
                "store every primary_event(events) whether explosion_exists "
                "true or false (match target_dataset_v3)"
            ),
            "non_explosion_fields": "copied_unchanged_from_v1",
            "modality_unavailable_60s_before": modality_before,
            "modality_unavailable_60s_after": modality_after,
            "valid_negative_60s_after": valid_neg_after,
            "valid_positive_60s_after": valid_pos_after,
            "horizons_sec": list(EXPLOSION_IMMINENCE_HORIZONS_SEC),
        },
        "artifacts": {
            "observation_targets": str(out_parts),
            "future_path_refs": str(v2_dir / "future_path_refs"),
            "explosion_events": str(v2_dir / "explosion_events_v1.parquet"),
            "source_v1": str(v1_dir),
            "observation_index": str(broad_root / OBS_INDEX_STORE),
        },
    }
    write_json(v2_dir / "broad_targets_manifest.json", meta)
    return meta
