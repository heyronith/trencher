"""Derived training cache (slim targets + TTE by mint_partition).

Does not modify frozen stores. Avoids full 8GB target duplication by keeping a
training-head column subset, re-sharded to match assignment mint_partition.
"""

from __future__ import annotations

import logging
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.splits.progress import Progress100
from trencher.data.world_model.targets.status import (
    BARRIER_PAIRS,
    EXPLOSION_IMMINENCE_HORIZONS_SEC,
    HORIZONS_SEC,
    POSITION_SIZES_SOL,
    SURVIVAL_HORIZONS_SEC,
)
from trencher.data.world_model.training.transforms import (
    fit_sequence_transforms_from_unique_rows,
    persist_transforms,
)

logger = logging.getLogger(__name__)

CACHE_VERSION = "training_cache_v1"


def mint_partition(mint: str) -> str:
    return sha256(str(mint).encode()).hexdigest()[:2]


def cache_root(split_root: Path) -> Path:
    return Path(split_root) / CACHE_VERSION


def slim_target_columns(schema_names: set[str]) -> list[str]:
    """Training-head subset — enough for all major families, not full 700+ cols."""
    cols = [
        "observation_id",
        "mint",
        "time_to_graduation_sec",
        "time_to_graduation_status",
        "graduated_at_or_before_t",
        "time_to_explosion_sec",
        "explosion_event_observed",
        "explosion_already_started_at_t",
        "explosion_right_censored",
        "remaining_upside_to_explosion_peak",
        "explosion_peak_future_return",
        "time_to_inactivity_sec",
        "time_to_inactivity_status",
        "ownership_target_available",
        "ownership_target_status",
        "holder_count_change",
        "churn_persistence",
    ]
    for h in HORIZONS_SEC:
        cols.extend(
            [
                f"h{h}_status",
                f"h{h}_simple_return",
                f"h{h}_log_return",
                f"h{h}_mfe",
                f"h{h}_mae",
                f"grad_h{h}_status",
                f"grad_h{h}_graduated",
                f"act_h{h}_status",
                f"act_h{h}_trade_count",
                f"act_h{h}_activity_rate",
                f"act_h{h}_activity_acceleration",
                f"h{h}_future_mark",
                f"h{h}_future_mark_time",
                f"h{h}_future_path_start_idx",
                f"h{h}_future_path_end_idx",
            ]
        )
        for _, _, bid in BARRIER_PAIRS:
            cols.append(f"h{h}_barrier_{bid}_outcome")
    for h in SURVIVAL_HORIZONS_SEC:
        cols.extend([f"surv_h{h}_status", f"surv_h{h}_active"])
    for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
        cols.extend([f"explosion_within_{h}s", f"explosion_within_{h}s_status"])
    for sol in POSITION_SIZES_SOL:
        tag = str(sol).replace(".", "p")
        cols.extend(
            [
                f"exec_{tag}sol_available",
                f"exec_{tag}sol_status",
                f"exec_{tag}sol_gross_return",
                f"exec_{tag}sol_net_return",
                f"exec_{tag}sol_execution_target_available",
                f"exec_{tag}sol_slippage",
            ]
        )
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c in schema_names and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _tte_keep_columns(names: list[str]) -> list[str]:
    keep = ["observation_id", "mint"]
    for c in names:
        is_tte = c.startswith("graduation_tte_") or c.startswith("explosion_tte_")
        is_meta = c in ("admin_boundary_unix", "base_split", "observation_time_t")
        if is_tte or is_meta:
            keep.append(c)
    seen: set[str] = set()
    out: list[str] = []
    for c in keep:
        if c in names and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _add_mint_partition_column(table: pa.Table) -> pa.Table:
    mints = table.column("mint").to_pylist()
    mp = pa.array(
        [mint_partition(str(m) if m is not None else "") for m in mints],
        type=pa.string(),
    )
    if "mint_partition" in table.column_names:
        table = table.drop(["mint_partition"])
    return table.append_column("mint_partition", mp)


def _cast_null_columns(table: pa.Table) -> pa.Table:
    arrays = []
    names = []
    for field in table.schema:
        col = table.column(field.name)
        if pa.types.is_null(field.type):
            col = pc.cast(col, pa.float64())
        arrays.append(col)
        names.append(field.name)
    return pa.Table.from_arrays(arrays, names=names)


def _compact_partitions(stage: Path, out_dir: Path, filename: str) -> int:
    written = 0
    part_dirs = [p for p in stage.iterdir() if p.is_dir()] if stage.exists() else []
    for part_dir in sorted(part_dirs):
        mp = part_dir.name.split("=", 1)[-1]
        files = sorted(part_dir.glob("*.parquet"))
        if not files:
            continue
        merged = pa.concat_tables([pq.read_table(f) for f in files])
        dest = ensure_dir(out_dir / mp)
        pq.write_table(merged, dest / filename, compression="zstd")
        written += 1
    shutil.rmtree(stage, ignore_errors=True)
    return written


def build_training_cache_v1(
    *,
    world_root: Path,
    split_root: Path | None = None,
    progress: Progress100 | None = None,
    refit_transforms: bool = True,
) -> dict[str, Any]:
    progress = progress or Progress100()
    world_root = Path(world_root)
    split_root = Path(split_root or world_root / "splits" / "split_v1")
    root = ensure_dir(cache_root(split_root))
    broad_out = ensure_dir(root / "broad" / "by_mint_partition")
    rich_out = ensure_dir(root / "rich")

    # --- Broad TTE ---
    progress(2, "cache: broad TTE by mint_partition")
    stage_tte = root / "_stage_tte"
    if stage_tte.exists():
        shutil.rmtree(stage_tte, ignore_errors=True)
    stage_tte = ensure_dir(stage_tte)
    tte_parts = sorted((split_root / "tte_overlays_broad").glob("part-*.parquet"))
    tte_n = 0
    for i, path in enumerate(tte_parts):
        t = pq.read_table(path)
        t = _add_mint_partition_column(t.select(_tte_keep_columns(t.column_names)))
        tte_n += t.num_rows
        pads.write_dataset(
            t,
            stage_tte,
            format="parquet",
            partitioning=["mint_partition"],
            existing_data_behavior="overwrite_or_ignore",
            basename_template=f"tte-{i:05d}-{{i}}.parquet",
            max_partitions=512,
            max_open_files=512,
        )
        progress(
            2 + int(18 * (i + 1) / max(len(tte_parts), 1)),
            f"cache: tte {i + 1}/{len(tte_parts)}",
        )
    tte_parts_n = _compact_partitions(stage_tte, broad_out, "tte.parquet")

    # --- Broad slim targets ---
    progress(22, "cache: broad slim targets by mint_partition")
    stage_tgt = root / "_stage_tgt"
    if stage_tgt.exists():
        shutil.rmtree(stage_tgt, ignore_errors=True)
    stage_tgt = ensure_dir(stage_tgt)
    target_parts = sorted(
        (world_root / "broad" / "broad_targets_v3" / "observation_targets").glob(
            "part-*.parquet"
        )
    )
    cols: list[str] = []
    tgt_n = 0
    for i, path in enumerate(target_parts):
        if not cols:
            cols = slim_target_columns(set(pq.read_schema(path).names))
        t = _cast_null_columns(
            _add_mint_partition_column(pq.read_table(path, columns=cols))
        )
        tgt_n += t.num_rows
        pads.write_dataset(
            t,
            stage_tgt,
            format="parquet",
            partitioning=["mint_partition"],
            existing_data_behavior="overwrite_or_ignore",
            basename_template=f"tgt-{i:05d}-{{i}}.parquet",
            max_partitions=512,
            max_open_files=512,
        )
        progress(
            22 + int(40 * (i + 1) / max(len(target_parts), 1)),
            f"cache: slim targets {i+1}/{len(target_parts)}",
        )
    tgt_parts_n = _compact_partitions(stage_tgt, broad_out, "targets_slim.parquet")

    # --- Join TTE + slim targets per partition (in place) ---
    progress(65, "cache: join slim targets + TTE per partition")
    joined_n = 0
    for mp_dir in sorted(broad_out.iterdir()):
        if not mp_dir.is_dir():
            continue
        tp = mp_dir / "targets_slim.parquet"
        ttep = mp_dir / "tte.parquet"
        if not tp.exists():
            continue
        targets = pq.read_table(tp)
        if ttep.exists():
            tte = pq.read_table(ttep)
            # drop duplicate mint from tte
            drop = [c for c in ("mint", "mint_partition") if c in tte.column_names]
            if drop:
                tte = tte.drop(drop)
            joined = targets.join(
                tte, keys="observation_id", join_type="left outer", use_threads=True
            )
        else:
            joined = targets
        pq.write_table(joined, mp_dir / "joined.parquet", compression="zstd")
        joined_n += joined.num_rows
        # remove intermediates to save disk
        tp.unlink(missing_ok=True)
        ttep.unlink(missing_ok=True)

    # --- Rich ---
    progress(78, "cache: rich joined slim+TTE")
    rich_path = (
        world_root / "targets" / "target_dataset_v3" / "observation_targets_v3.parquet"
    )
    rich_cols = slim_target_columns(set(pq.read_schema(rich_path).names))
    rich_t = _cast_null_columns(pq.read_table(rich_path, columns=rich_cols))
    rich_tte = pq.read_table(split_root / "tte_overlays_rich.parquet")
    rich_tte = rich_tte.select(_tte_keep_columns(rich_tte.column_names))
    drop = [c for c in ("mint",) if c in rich_tte.column_names]
    if drop:
        rich_tte = rich_tte.drop(drop)
    rich_joined = rich_t.join(
        _cast_null_columns(rich_tte), keys="observation_id", join_type="left outer"
    )
    pq.write_table(rich_joined, rich_out / "joined.parquet", compression="zstd")

    transform_info: dict[str, Any] = {}
    if refit_transforms:
        progress(85, "cache: fit train transforms")
        tokens = pq.read_table(split_root / "token_assignments.parquet").to_pylist()
        train_mints = {
            r["mint"]
            for r in tokens
            if r["base_split"] == "train" and r.get("in_broad_event_curve")
        }
        events_root = world_root / "broad" / "broad_sequence_events_v1"
        parts = sorted(events_root.rglob("part.parquet"))
        fit_parts = parts[:: max(1, len(parts) // 64)][:64] if parts else []
        sch = set(pq.read_schema(fit_parts[0]).names) if fit_parts else set()
        numeric_cols = [
            c
            for c in (
                "sol_amount",
                "token_amount",
                "price_sol",
                "v_sol",
                "v_tok",
                "curve_pct_depleted",
            )
            if c in sch
        ]
        transforms = fit_sequence_transforms_from_unique_rows(
            sequence_parts=fit_parts,
            train_mints=train_mints,
            numeric_columns=numeric_cols,
            max_rows_per_part=250_000,
        )
        tdir = ensure_dir(root / "transforms_train_v1")
        persist_transforms(tdir / "sequence_numeric_stats.json", transforms)
        persist_transforms(
            split_root / "transforms_train_v1" / "sequence_numeric_stats.json",
            transforms,
        )
        transform_info = {
            "rows_seen": transforms.get("rows_seen"),
            "columns": list((transforms.get("columns") or {}).keys()),
            "order": transforms.get("transform_order"),
        }

    manifest = {
        "cache_version": CACHE_VERSION,
        "generated_at": utc_now(),
        "design": "slim_targets_plus_tte_by_mint_partition",
        "broad": {
            "tte_rows": tte_n,
            "tte_partitions": tte_parts_n,
            "target_rows": tgt_n,
            "target_partitions": tgt_parts_n,
            "joined_rows": joined_n,
            "slim_columns_n": len(cols),
        },
        "rich": {"joined_rows": rich_joined.num_rows, "slim_columns_n": len(rich_cols)},
        "transforms": transform_info,
        "notes": [
            "Frozen stores untouched",
            "Slim training-head columns only (not full frozen target width)",
            "Partition-aligned with observation_assignments_broad",
        ],
    }
    write_json(root / "manifest.json", manifest)
    progress(95, "cache: manifest written")
    return manifest


class JoinedTargetCache:
    """Partition-aligned joined slim-targets + TTE lookup."""

    def __init__(self, cache_dir: Path, *, population: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.population = population
        self._part_tables: dict[str, pa.Table] = {}
        self._part_idx: dict[str, dict[str, int]] = {}
        self._rich_table: pa.Table | None = None
        self._rich_idx: dict[str, int] | None = None

    def _load_broad_part(self, mp: str) -> tuple[pa.Table, dict[str, int]]:
        if mp in self._part_tables:
            return self._part_tables[mp], self._part_idx[mp]
        path = self.cache_dir / "broad" / "by_mint_partition" / mp / "joined.parquet"
        if not path.exists():
            empty = pa.table({})
            self._part_tables[mp] = empty
            self._part_idx[mp] = {}
            return empty, {}
        table = pq.read_table(path)
        idx = {
            str(oid): i
            for i, oid in enumerate(table.column("observation_id").to_pylist())
            if oid is not None
        }
        self._part_tables[mp] = table
        self._part_idx[mp] = idx
        if len(self._part_tables) > 24:
            oldest = next(iter(self._part_tables))
            if oldest != mp:
                del self._part_tables[oldest]
                del self._part_idx[oldest]
        return table, idx

    def _load_rich(self) -> tuple[pa.Table, dict[str, int]]:
        if self._rich_table is None:
            path = self.cache_dir / "rich" / "joined.parquet"
            self._rich_table = pq.read_table(path) if path.exists() else pa.table({})
            self._rich_idx = {
                str(oid): i
                for i, oid in enumerate(
                    self._rich_table.column("observation_id").to_pylist()
                )
                if oid is not None
            }
        return self._rich_table, self._rich_idx or {}

    def prefetch(self, assignments: list[dict[str, Any]]) -> None:
        if self.population == "rich":
            self._load_rich()
            return
        parts = {
            str(a.get("mint_partition") or mint_partition(str(a["mint"])))
            for a in assignments
        }
        for mp in parts:
            self._load_broad_part(mp)

    def get(self, assignment: dict[str, Any]) -> dict[str, Any] | None:
        oid = str(assignment["observation_id"])
        if self.population == "rich":
            table, idx = self._load_rich()
        else:
            mp = str(
                assignment.get("mint_partition")
                or mint_partition(str(assignment["mint"]))
            )
            table, idx = self._load_broad_part(mp)
        i = idx.get(oid)
        if i is None or table.num_rows == 0:
            return None
        return table.slice(i, 1).to_pylist()[0]


# Alias
PartitionTTECache = JoinedTargetCache
