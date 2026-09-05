"""Build broad_targets_v3 from repaired v2 with correct missing-primary semantics.

Distinguishes:

1. Stored primary detector row with ``explosion_exists=False`` → coverage-resolved
   valid negatives (same as rich / v2 for those mints).
2. No stored primary detector row → ``modality_unavailable``; never fabricate a
   synthetic ``explosion_exists=False`` stub.
"""

from __future__ import annotations

import logging
import shutil
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
from trencher.data.world_model.broad.targets_v2 import (
    TARGETS_V2_STORE,
    _is_explosion_column,
    _ref_from_row,
    load_explosion_defs_by_mint,
)
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.targets.compute import compute_explosion_obs
from trencher.data.world_model.targets.coverage import build_source_coverage_registry
from trencher.data.world_model.targets.status import (
    EXPLOSION_IMMINENCE_HORIZONS_SEC,
    STATUS_MODALITY_UNAVAILABLE,
    STATUS_VALID_NEGATIVE,
    STATUS_VALID_POSITIVE,
)

logger = logging.getLogger(__name__)

TARGETS_V3_STORE = "broad_targets_v3"
TARGETS_V1_STORE = "broad_targets_v1"


def repair_explosion_columns_for_row_v3(
    row: dict[str, Any],
    *,
    primary: dict[str, Any] | None,
    all_defs: list[dict[str, Any]],
    coverage: Any,
) -> dict[str, Any]:
    """Recompute explosion fields; pass ``primary=None`` when no detector row."""
    out = dict(row)
    T = float(out["observation_time_t"])
    expl = compute_explosion_obs(
        T=T,
        primary=primary,  # None stays modality_unavailable — no stub
        all_defs=all_defs,
        coverage=coverage,
        ref=_ref_from_row(out),
    )
    for key in list(out):
        if _is_explosion_column(key):
            out[key] = None
    out.update(expl)
    out["target_dataset_version"] = TARGETS_V3_STORE
    out["label_semantics"] = (
        "target_dataset_v3_compute_reuse_explosion_repaired_no_stub"
    )
    return out


def _bump_version_only(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["target_dataset_version"] = TARGETS_V3_STORE
    out["label_semantics"] = (
        "target_dataset_v3_compute_reuse_explosion_repaired_no_stub"
    )
    return out


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


def build_broad_targets_v3_from_v2(
    *,
    world_root: Path,
    pump_root: Path,
    broad_root: Path | None = None,
    batch_flush: int = 5_000,
) -> dict[str, Any]:
    """Materialize broad_targets_v3 from v2; fix missing-primary stub leakage."""
    broad_root = ensure_dir(broad_root or world_root / "broad")
    v1_dir = broad_root / TARGETS_V1_STORE
    v2_dir = broad_root / TARGETS_V2_STORE
    v3_dir = ensure_dir(broad_root / TARGETS_V3_STORE)
    v2_targets = v2_dir / "observation_targets"
    if not v2_targets.exists():
        raise FileNotFoundError(f"missing {v2_targets}")

    for name in ("observation_targets", "future_path_refs", "explosion_events"):
        p = v3_dir / name
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    for name in ("explosion_events_v1.parquet", "broad_targets_manifest.json"):
        (v3_dir / name).unlink(missing_ok=True)

    expl_src = v1_dir / "explosion_events_v1.parquet"
    if not expl_src.exists():
        expl_src = v2_dir / "explosion_events_v1.parquet"
    primary_by_mint, all_defs_by_mint, primary_exists_n = load_explosion_defs_by_mint(
        expl_src
    )
    primary_mints = set(primary_by_mint)
    logger.info(
        "Loaded explosion defs for %d mints (%d primary exists=True)",
        len(primary_by_mint),
        primary_exists_n,
    )

    coverage = build_source_coverage_registry(
        world_root=world_root, pump_root=pump_root
    )
    shutil.copy2(
        v2_dir / "target_source_coverage_v2.json"
        if (v2_dir / "target_source_coverage_v2.json").exists()
        else v1_dir / "target_source_coverage_v2.json",
        v3_dir / "target_source_coverage_v2.json",
    )
    shutil.copy2(
        v2_dir / "explosion_detector_config_v1.json"
        if (v2_dir / "explosion_detector_config_v1.json").exists()
        else v1_dir / "explosion_detector_config_v1.json",
        v3_dir / "explosion_detector_config_v1.json",
    )

    _link_or_copy(expl_src, v3_dir / "explosion_events_v1.parquet")
    if (v1_dir / "explosion_events").exists():
        _link_or_copy(v1_dir / "explosion_events", v3_dir / "explosion_events")
    if (v1_dir / "future_path_refs").exists():
        _link_or_copy(v1_dir / "future_path_refs", v3_dir / "future_path_refs")
    if (v1_dir / FUTURE_PATH_PARTS).exists():
        _link_or_copy(v1_dir / FUTURE_PATH_PARTS, v3_dir / FUTURE_PATH_PARTS)

    out_parts = ensure_dir(v3_dir / "observation_targets")
    part_files = sorted(v2_targets.glob("part-*.parquet"))
    if not part_files:
        raise FileNotFoundError(f"no part files under {v2_targets}")

    schema: pa.Schema | None = None
    observations_processed = 0
    copied_with_primary = 0
    recomputed_missing_primary = 0
    modality_after = 0
    valid_neg_after = 0
    valid_pos_after = 0
    invalid_missing_primary_neg = 0
    mints_seen: set[str] = set()
    mints_without_primary_seen: set[str] = set()

    for fi, path in enumerate(part_files):
        logger.info(
            "targets_v3 repair file %s (%d/%d) obs_so_far=%d",
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
                v2_schema = pq.read_schema(part_files[0])
                fields = []
                seen: set[str] = set()
                by_name = {f.name: f for f in schema}
                for f in v2_schema:
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
                mint = str(row["mint"])
                mints_seen.add(mint)
                if mint in primary_mints:
                    out = _bump_version_only(row)
                    copied_with_primary += 1
                else:
                    mints_without_primary_seen.add(mint)
                    out = repair_explosion_columns_for_row_v3(
                        row,
                        primary=None,
                        all_defs=all_defs_by_mint.get(mint, []),
                        coverage=coverage,
                    )
                    recomputed_missing_primary += 1
                    # Invariant: no detector row → never valid explosion negative
                    for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
                        if (
                            out.get(f"explosion_within_{h}s_status")
                            == STATUS_VALID_NEGATIVE
                        ):
                            invalid_missing_primary_neg += 1
                st = out.get("explosion_within_60s_status")
                if st == STATUS_MODALITY_UNAVAILABLE:
                    modality_after += 1
                elif st == STATUS_VALID_NEGATIVE:
                    valid_neg_after += 1
                elif st == STATUS_VALID_POSITIVE:
                    valid_pos_after += 1
                pending.append(out)
                observations_processed += 1
                if len(pending) >= batch_flush:
                    _write_pending()
        _write_pending()
        if writer is not None:
            writer.close()

    if invalid_missing_primary_neg:
        raise RuntimeError(
            "invariant violated: missing primary produced "
            f"{invalid_missing_primary_neg} valid_negative explosion statuses"
        )

    meta = {
        "generated_at": utc_now(),
        "version": TARGETS_V3_STORE,
        "repaired_from": TARGETS_V2_STORE,
        "observations_processed": observations_processed,
        "primary_explosion_events": primary_exists_n,
        "mints_with_primary_explosion": primary_exists_n,
        "mints_with_primary_definition_row": len(primary_by_mint),
        "mints_seen_in_observations": len(mints_seen),
        "mints_without_primary_definition_row": len(mints_without_primary_seen),
        "observations_copied_with_primary": copied_with_primary,
        "observations_recomputed_missing_primary": recomputed_missing_primary,
        "explosion_repair": {
            "method": (
                "copy_v2_explosion_cols_when_primary_row_exists;"
                "recompute_with_primary_None_when_absent"
            ),
            "primary_semantics": (
                "stored primary row (exists true/false) kept; "
                "missing primary → modality_unavailable (no _stub_primary_absent)"
            ),
            "non_explosion_fields": "copied_unchanged_from_v2",
            "modality_unavailable_60s_after": modality_after,
            "valid_negative_60s_after": valid_neg_after,
            "valid_positive_60s_after": valid_pos_after,
            "horizons_sec": list(EXPLOSION_IMMINENCE_HORIZONS_SEC),
            "invariant_no_detector_row_never_valid_negative": True,
        },
        "artifacts": {
            "observation_targets": str(out_parts),
            "future_path_refs": str(v3_dir / "future_path_refs"),
            "explosion_events": str(v3_dir / "explosion_events_v1.parquet"),
            "source_v2": str(v2_dir),
            "observation_index": str(broad_root / OBS_INDEX_STORE),
        },
    }
    write_json(v3_dir / "broad_targets_manifest.json", meta)
    return meta
