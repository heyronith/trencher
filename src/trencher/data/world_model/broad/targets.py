"""Broad label-only targets reusing target_dataset_v3 compute_* unchanged.

Graduation: conservative v3 (absence censored) per graduation_censor decision.
Coverage: reuse ``target_source_coverage_v2`` registry (do not rewrite policy).
Explosion detector unchanged.

Scalability: process one mint-hash partition at a time. Future-market path
arrays are derived from ``broad_sequence_events_v1`` (+ post-grad snapshots
for graduated mints) so we never materialize all ~623k mint paths in RAM.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from trencher.data.world_model.broad.graduation_censor import graduation_censor_decision
from trencher.data.world_model.broad.sequences import (
    EVENTS_STORE,
    load_entity_index,
    mint_partition,
)
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.targets.compute import (
    compute_activity_survival,
    compute_execution,
    compute_explosion_obs,
    compute_graduation,
    compute_ownership_targets,
    compute_persistence,
    compute_return_barrier_mfe,
    future_path_refs_for_obs,
    reference_price_at_t,
)
from trencher.data.world_model.targets.coverage import (
    build_source_coverage_registry,
    write_coverage_manifest,
)
from trencher.data.world_model.targets.explosion import (
    detect_explosions_for_mint,
    detector_config_manifest,
    primary_event,
)
from trencher.data.world_model.targets.future_path import load_migration_unix
from trencher.data.world_model.targets.timeutil import is_finite_positive, to_unix

logger = logging.getLogger(__name__)

TARGETS_STORE = "broad_targets_v1"
OBS_INDEX_STORE = "broad_observation_index_v1"
FUTURE_PATH_PARTS = "future_market_path_parts"


def _empty_path_arr() -> dict[str, np.ndarray]:
    return {
        "t": np.asarray([], dtype=np.float64),
        "price": np.asarray([], dtype=np.float64),
        "regime": np.asarray([], dtype=np.int8),
        "is_buy": np.asarray([], dtype=np.int8),
        "sol": np.asarray([], dtype=np.float64),
        "token": np.asarray([], dtype=np.float64),
        "wallet": np.asarray([], dtype=object),
        "v_sol": np.asarray([], dtype=np.float64),
        "v_tok": np.asarray([], dtype=np.float64),
        "curve": np.asarray([], dtype=np.float64),
        "price_source": np.asarray([], dtype=object),
        "path_idx": np.asarray([], dtype=np.int64),
        "usable_price": np.asarray([], dtype=bool),
    }


def _rows_to_path_arrays(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
    by_mint_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_mint_rows[str(r["mint"])].append(r)
    by_mint: dict[str, dict[str, np.ndarray]] = {}
    entity: dict[str, dict[str, Any]] = {}
    for mint, mint_rows in by_mint_rows.items():
        mint_rows.sort(
            key=lambda r: (
                float(r["event_time_unix"]),
                str(r.get("order_key") or ""),
            )
        )
        n = len(mint_rows)
        t = np.asarray([float(r["event_time_unix"]) for r in mint_rows], dtype=np.float64)
        price = np.asarray([float(r["price"]) for r in mint_rows], dtype=np.float64)
        regime = np.asarray([int(r["regime"]) for r in mint_rows], dtype=np.int8)
        is_buy = np.asarray(
            [
                1 if r.get("is_buy") is True else (0 if r.get("is_buy") is False else -1)
                for r in mint_rows
            ],
            dtype=np.int8,
        )
        sol = np.asarray(
            [
                float(r["sol_amount"])
                if r.get("sol_amount") is not None
                and math.isfinite(float(r["sol_amount"]))
                else np.nan
                for r in mint_rows
            ],
            dtype=np.float64,
        )
        token = np.asarray(
            [
                float(r["token_amount"])
                if r.get("token_amount") is not None
                and math.isfinite(float(r["token_amount"]))
                else np.nan
                for r in mint_rows
            ],
            dtype=np.float64,
        )
        wallet = np.asarray([r.get("wallet") for r in mint_rows], dtype=object)
        v_sol = np.asarray(
            [
                float(r["v_sol"])
                if r.get("v_sol") is not None and math.isfinite(float(r["v_sol"]))
                else np.nan
                for r in mint_rows
            ],
            dtype=np.float64,
        )
        v_tok = np.asarray(
            [
                float(r["v_tok"])
                if r.get("v_tok") is not None and math.isfinite(float(r["v_tok"]))
                else np.nan
                for r in mint_rows
            ],
            dtype=np.float64,
        )
        curve = np.asarray(
            [
                float(r["curve_pct_depleted"])
                if r.get("curve_pct_depleted") is not None
                and math.isfinite(float(r["curve_pct_depleted"]))
                else np.nan
                for r in mint_rows
            ],
            dtype=np.float64,
        )
        price_source = np.asarray(
            [r.get("price_source") or "trade" for r in mint_rows], dtype=object
        )
        usable = np.asarray(
            [math.isfinite(float(p)) and float(p) > 0 for p in price], dtype=bool
        )
        by_mint[mint] = {
            "t": t,
            "price": price,
            "regime": regime,
            "is_buy": is_buy,
            "sol": sol,
            "token": token,
            "wallet": wallet,
            "v_sol": v_sol,
            "v_tok": v_tok,
            "curve": curve,
            "price_source": price_source,
            "path_idx": np.arange(n, dtype=np.int64),
            "usable_price": usable,
        }
        entity[mint] = {
            "n_events": n,
            "first_event_unix": float(t[0]) if n else None,
            "last_event_unix": float(t[-1]) if n else None,
        }
    return by_mint, entity


def _load_postgrad_by_mint(
    pump_root: Path, migration_unix: dict[str, float]
) -> dict[str, list[dict[str, Any]]]:
    path = pump_root / "postgard_snapshots.parquet"
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists() or not migration_unix:
        return out
    cols = [
        "mint",
        "snapshot_time",
        "price_native",
        "price_usd",
        "incomplete_data",
        "liquidity_usd",
    ]
    for batch in ds.dataset(str(path), format="parquet").scanner(
        columns=cols, batch_size=200_000
    ).to_batches():
        names = batch.schema.names
        cols_l = [batch.column(c).to_pylist() for c in names]
        for vals in zip(*cols_l, strict=True):
            row = dict(zip(names, vals, strict=True))
            mint = row.get("mint")
            if mint not in migration_unix:
                continue
            et = to_unix(row.get("snapshot_time"))
            if et is None or et <= migration_unix[mint]:
                continue
            price = None
            price_src = "postgrad_unusable"
            if is_finite_positive(row.get("price_native")):
                price = float(row["price_native"])
                price_src = "postgard_price_native"
            elif is_finite_positive(row.get("price_usd")):
                price = float(row["price_usd"])
                price_src = "postgard_price_usd_not_sol_continuous"
            if price is None:
                continue
            out[str(mint)].append(
                {
                    "mint": mint,
                    "event_time_unix": float(et),
                    "price": float(price),
                    "price_source": price_src,
                    "side": None,
                    "is_buy": None,
                    "sol_amount": None,
                    "token_amount": None,
                    "wallet": None,
                    "order_key": f"postgrad|{et:.6f}",
                    "v_sol": None,
                    "v_tok": None,
                    "curve_pct_depleted": None,
                    "regime": 1,
                }
            )
    return out


def _path_rows_from_event_partition(
    events_part: Path,
    migration_unix: dict[str, float],
    postgrad_by_mint: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    path = events_part / "part.parquet"
    if not path.exists() or pq.ParquetFile(path).metadata.num_rows == 0:
        return []
    out: list[dict[str, Any]] = []
    for r in pq.read_table(path).to_pylist():
        mint = str(r["mint"])
        et = float(r["event_time_unix"])
        mig = migration_unix.get(mint)
        if mig is not None and et > mig:
            continue
        px = r.get("price_sol")
        if not is_finite_positive(px):
            # Prefer reserve-implied when trade price missing.
            v_sol, v_tok = r.get("v_sol"), r.get("v_tok")
            if is_finite_positive(v_sol) and is_finite_positive(v_tok):
                px = float(v_sol) / float(v_tok)
                src = "reserve_implied"
            else:
                continue
        else:
            px = float(px)
            src = "trade_price_sol"
        out.append(
            {
                "mint": mint,
                "event_time_unix": et,
                "price": px,
                "price_source": src,
                "side": r.get("side"),
                "is_buy": r.get("is_buy"),
                "sol_amount": r.get("sol_amount"),
                "token_amount": r.get("token_amount"),
                "wallet": r.get("wallet"),
                "order_key": r.get("order_key"),
                "v_sol": r.get("v_sol"),
                "v_tok": r.get("v_tok"),
                "curve_pct_depleted": r.get("curve_pct_depleted"),
                "regime": 0,
            }
        )
    # Attach post-grad rows for mints in this partition.
    mints_in_part = {str(r["mint"]) for r in out}
    # Also include graduated mints whose pregrad events were empty but postgrad exists
    for mint, rows in postgrad_by_mint.items():
        if mint_partition(mint) != events_part.name:
            continue
        if mint in mints_in_part or mint in migration_unix:
            out.extend(rows)
    return out


def _iter_obs_for_partition(index_root: Path, part: str):
    part_dir = index_root / part
    if not part_dir.exists():
        # legacy hive-style layout
        part_dir = index_root / f"mint_partition={part}"
    if not part_dir.exists():
        return
    for path in sorted(part_dir.glob("*.parquet")):
        yield from pq.read_table(path).to_pylist()


def _py_to_pa_type(value: Any) -> pa.DataType | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return pa.bool_()
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return pa.int64()
    if isinstance(value, (float, np.floating)):
        return pa.float64()
    if isinstance(value, str):
        return pa.string()
    return pa.string()


def _heuristic_pa_type(name: str) -> pa.DataType:
    """Stable Arrow type when a column is all-null in prototypes."""
    if name.endswith("_execution_target_available") or name.endswith(
        ("_start_idx", "_end_idx", "_count")
    ):
        return pa.int64()
    if name.endswith(
        (
            "_status",
            "_outcome",
            "_source",
            "_note",
            "_type",
            "_regime",
            "_trigger",
            "_version",
            "_semantics",
            "_pseudolabel",
        )
    ) or name in {
        "observation_id",
        "mint",
        "trigger_value",
        "explosion_definition_primary",
        "ownership_evidence_level",
        "future_path_store",
        "survival_definition",
    }:
        return pa.string()
    if (
        name.endswith(
            (
                "_available",
                "_exists",
                "_censored",
                "_graduated",
                "_observed",
                "_reconstructable",
            )
        )
        or name.startswith("explosion_reaches_")
        or name
        in {
            "label_only",
            "explosion_already_started_at_t",
            "explosion_event_observed",
            "explosion_right_censored",
        }
    ):
        return pa.bool_()
    return pa.float64()


def _schema_from_rows(rows: list[dict[str, Any]]) -> pa.Schema:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    fields: list[pa.Field] = []
    for key in keys:
        inferred: pa.DataType | None = None
        for row in rows:
            inferred = _py_to_pa_type(row.get(key))
            if inferred is not None:
                break
        fields.append(pa.field(key, inferred or _heuristic_pa_type(key), nullable=True))
    return pa.schema(fields)


def _normalize_row(row: dict[str, Any], schema: pa.Schema) -> dict[str, Any]:
    return {field.name: row.get(field.name) for field in schema}


def _synthetic_path_arr(*, n: int = 8, t0: float = 1_800_000_000.0) -> dict[str, np.ndarray]:
    t = t0 + np.arange(n, dtype=np.float64) * 30.0
    price = np.linspace(1.0, 2.5, n, dtype=np.float64)
    return {
        "t": t,
        "price": price,
        "regime": np.zeros(n, dtype=np.int8),
        "is_buy": np.ones(n, dtype=np.int8),
        "sol": np.full(n, 0.5, dtype=np.float64),
        "token": np.full(n, 1e6, dtype=np.float64),
        "wallet": np.asarray([f"w{i}" for i in range(n)], dtype=object),
        "v_sol": np.full(n, 30.0, dtype=np.float64),
        "v_tok": np.full(n, 1e9, dtype=np.float64),
        "curve": np.linspace(0.0, 0.4, n, dtype=np.float64),
        "price_source": np.asarray(["trade_price_sol"] * n, dtype=object),
        "path_idx": np.arange(n, dtype=np.int64),
        "usable_price": np.ones(n, dtype=bool),
    }


def _prototype_target_and_ref_rows(
    coverage: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rows covering sparse and dense compute_* branches for a fixed Parquet schema."""
    target_rows: list[dict[str, Any]] = []
    ref_rows: list[dict[str, Any]] = []

    def _one(
        *,
        mint: str,
        arr: dict[str, np.ndarray],
        T: float,
        migration_unix: float | None,
        primary: dict[str, Any] | None,
        all_defs: list[dict[str, Any]],
        ownership_meta: dict[str, Any] | None,
    ) -> None:
        if arr["t"].size == 0:
            ref = {
                "reference_price": None,
                "reference_price_time": None,
                "reference_price_source": None,
                "reference_price_age_sec": None,
                "price_target_available": False,
            }
            regime = None
        else:
            ref = reference_price_at_t(arr, T)
            regime = ref.get("reference_regime")
        row: dict[str, Any] = {
            "observation_id": f"{mint}:proto",
            "mint": mint,
            "observation_time_t": T,
            "token_age_seconds": 0.0,
            "observation_trigger": "fixed_age",
            "trigger_value": "launch",
            "target_dataset_version": "broad_targets_v1",
            "label_semantics": "target_dataset_v3_compute_reuse",
            "label_only": True,
            "reference_regime": regime,
        }
        row.update(
            {
                k: ref.get(k)
                for k in (
                    "reference_price",
                    "reference_price_time",
                    "reference_price_source",
                    "reference_price_age_sec",
                    "price_target_available",
                )
            }
        )
        last_path_t = float(arr["t"][-1]) if arr["t"].size else None
        row.update(compute_return_barrier_mfe(arr, T=T, ref=ref, last_path_t=last_path_t))
        row.update(
            compute_activity_survival(
                arr, T=T, coverage=coverage, migration_unix=migration_unix
            )
        )
        row.update(
            compute_graduation(T=T, migration_unix=migration_unix, coverage=coverage)
        )
        row.update(compute_persistence(arr, T=T, coverage=coverage))
        row.update(
            compute_explosion_obs(
                T=T, primary=primary, all_defs=all_defs, coverage=coverage, ref=ref
            )
        )
        row.update(compute_execution(arr, T=T, ref=ref))
        row.update(
            compute_ownership_targets(mint=mint, T=T, ownership_meta=ownership_meta)
        )
        target_rows.append(row)
        ref_rows.append(
            {
                "observation_id": row["observation_id"],
                "mint": mint,
                "observation_time_t": T,
                "future_path_store": "broad_partitioned_path_v1",
                **future_path_refs_for_obs(arr, T=T),
            }
        )

    empty = _empty_path_arr()
    rich = _synthetic_path_arr()
    t0 = float(rich["t"][0])
    defs = [
        {
            "definition_label": label,
            "explosion_exists": True,
            "onset_unix": t0 + 90.0,
            "peak_unix": t0 + 150.0,
            "end_unix": t0 + 200.0,
            "peak_price": 3.0,
            "explosion_magnitude": 1.2,
            "onset_to_peak_sec": 60.0,
            "max_drawdown_after_peak": 0.3,
            "censoring_status": None,
        }
        for label in ("conservative", "primary", "aggressive")
    ]
    primary = {**defs[1], "definition_id": "explosion_primary_v1"}
    _one(
        mint="proto_empty",
        arr=empty,
        T=t0,
        migration_unix=None,
        primary=None,
        all_defs=[],
        ownership_meta=None,
    )
    _one(
        mint="proto_rich",
        arr=rich,
        T=t0 + 10.0,
        migration_unix=t0 + 10_000.0,
        primary=primary,
        all_defs=defs,
        ownership_meta={
            "holder_state_reconstructable": True,
            "ownership_evidence_level": "edges_only",
        },
    )
    # Force definition flags even when no event detected (matches detect_explosions output).
    _one(
        mint="proto_defs_absent",
        arr=rich,
        T=t0 + 10.0,
        migration_unix=None,
        primary={
            "explosion_exists": False,
            "onset_unix": None,
            "censoring_status": "right_censored_path_end",
        },
        all_defs=[
            {
                "definition_label": label,
                "explosion_exists": False,
                "onset_unix": None,
                "explosion_magnitude": None,
            }
            for label in ("conservative", "primary", "aggressive")
        ],
        ownership_meta=None,
    )
    return target_rows, ref_rows


def build_broad_targets_v1(
    *,
    world_root: Path,
    pump_root: Path,
    broad_root: Path | None = None,
    population_path: Path | None = None,
    batch_flush: int = 5_000,
) -> dict[str, Any]:
    """Generate label-only targets for all broad observations (partitioned)."""
    broad_root = ensure_dir(broad_root or world_root / "broad")
    population_path = population_path or broad_root / "broad_population_v1.parquet"
    out_dir = ensure_dir(broad_root / TARGETS_STORE)
    index_root = broad_root / OBS_INDEX_STORE
    events_root = broad_root / EVENTS_STORE
    # Drop partial outputs from prior failed runs (keep config copies rewritten below).
    for name in (
        "observation_targets",
        "future_path_refs",
        "explosion_events",
        FUTURE_PATH_PARTS,
    ):
        p = out_dir / name
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    for name in ("explosion_events_v1.parquet", "broad_targets_manifest.json"):
        (out_dir / name).unlink(missing_ok=True)
    path_parts_dir = ensure_dir(out_dir / FUTURE_PATH_PARTS)

    censor = graduation_censor_decision(pump_root)
    if not censor.get("retain_conservative_v3_graduation_treatment"):
        raise RuntimeError("unexpected graduation censor decision")

    pop = pq.read_table(
        population_path,
        columns=["mint", "event_curve_population"],
    )
    event_mints = {
        str(r["mint"])
        for r in pop.to_pylist()
        if r.get("event_curve_population") and r.get("mint")
    }
    logger.info(
        "Broad targets for %d event_curve mints (partitioned path build)",
        len(event_mints),
    )

    migration_unix = load_migration_unix(
        world_root=world_root, pump_root=pump_root, mints=event_mints
    )
    logger.info("Loading post-grad snapshots for %d migrated mints", len(migration_unix))
    postgrad_by_mint = _load_postgrad_by_mint(pump_root, migration_unix)

    coverage = build_source_coverage_registry(
        world_root=world_root, pump_root=pump_root
    )
    cov_manifest = write_coverage_manifest(
        coverage, out_dir / "target_source_coverage_v2.json"
    )
    write_json(out_dir / "explosion_detector_config_v1.json", detector_config_manifest())

    proto_targets, proto_refs = _prototype_target_and_ref_rows(coverage)
    target_schema = _schema_from_rows(proto_targets)
    ref_schema = _schema_from_rows(proto_refs)
    logger.info(
        "Locked broad target schemas: targets_cols=%d refs_cols=%d",
        len(target_schema),
        len(ref_schema),
    )

    own_path = world_root / "universe" / "ownership_evidence.parquet"
    own_by_mint: dict[str, dict[str, Any]] = {}
    if own_path.exists():
        for row in pq.read_table(own_path).to_pylist():
            if row.get("mint") in event_mints:
                own_by_mint[row["mint"]] = row

    events_index = load_entity_index(events_root)
    partitions = sorted(
        {
            str(meta.get("partition"))
            for meta in events_index.values()
            if meta.get("partition")
        }
    )
    # Also include obs partitions that might exist without events (launch-only)
    if index_root.exists():
        for p in index_root.iterdir():
            if not p.is_dir():
                continue
            part = (
                p.name.split("=", 1)[-1]
                if p.name.startswith("mint_partition=")
                else p.name
            )
            if len(part) == 2 and part not in partitions:
                partitions.append(part)
        partitions = sorted(set(partitions))

    targets_parts = ensure_dir(out_dir / "observation_targets")
    refs_parts = ensure_dir(out_dir / "future_path_refs")
    expl_parts = ensure_dir(out_dir / "explosion_events")

    observations_processed = 0
    explosion_event_rows = 0
    primary_event_count = 0
    mints_with_primary: set[str] = set()
    part_i = 0
    target_writer: pq.ParquetWriter | None = None
    ref_writer: pq.ParquetWriter | None = None
    targets_path = targets_parts / "part-00000.parquet"
    refs_path = refs_parts / "part-00000.parquet"
    target_batch: list[dict[str, Any]] = []
    ref_batch: list[dict[str, Any]] = []

    def _flush(*, force_rotate: bool = False) -> None:
        nonlocal target_writer, ref_writer, target_batch, ref_batch, part_i
        nonlocal targets_path, refs_path
        if target_batch:
            table = pa.Table.from_pylist(target_batch, schema=target_schema)
            if target_writer is None:
                target_writer = pq.ParquetWriter(targets_path, target_schema)
            target_writer.write_table(table)
            target_batch = []
        if ref_batch:
            table = pa.Table.from_pylist(ref_batch, schema=ref_schema)
            if ref_writer is None:
                ref_writer = pq.ParquetWriter(refs_path, ref_schema)
            ref_writer.write_table(table)
            ref_batch = []
        if force_rotate or (
            observations_processed and observations_processed % 500_000 == 0
        ):
            if target_writer is not None:
                target_writer.close()
                target_writer = None
            if ref_writer is not None:
                ref_writer.close()
                ref_writer = None
            if observations_processed % 500_000 == 0 and observations_processed:
                part_i += 1
                targets_path = targets_parts / f"part-{part_i:05d}.parquet"
                refs_path = refs_parts / f"part-{part_i:05d}.parquet"

    for pi, part in enumerate(partitions):
        pct = int(100 * pi / max(len(partitions), 1))
        logger.info(
            "targets progress=%d%% partition=%s (%d/%d) obs_so_far=%d",
            pct,
            part,
            pi + 1,
            len(partitions),
            observations_processed,
        )
        events_part = events_root / part
        path_rows = _path_rows_from_event_partition(
            events_part, migration_unix, postgrad_by_mint
        )
        by_mint, entity = _rows_to_path_arrays(path_rows)
        # Persist compact path partition for audit/reuse
        if path_rows:
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "mint": r["mint"],
                            "event_time_unix": r["event_time_unix"],
                            "price": r["price"],
                            "price_source": r.get("price_source"),
                            "regime": r["regime"],
                        }
                        for r in path_rows
                    ]
                ),
                path_parts_dir / f"{part}.parquet",
            )

        explosion_rows: list[dict[str, Any]] = []
        primary_by_mint: dict[str, dict[str, Any]] = {}
        all_defs_by_mint: dict[str, list[dict[str, Any]]] = {}
        for mint, arr in by_mint.items():
            events = detect_explosions_for_mint(mint, arr)
            all_defs_by_mint[mint] = events
            for ev in events:
                flat = {k: v for k, v in ev.items() if k != "precursor"}
                prec = ev.get("precursor") or {}
                flat["precursor_json"] = json.dumps(prec, default=str)
                for pk, pv in prec.items():
                    if isinstance(pv, dict):
                        continue
                    flat[f"precursor_{pk}"] = pv
                explosion_rows.append(flat)
            prim = primary_event(events)
            if prim:
                primary_by_mint[mint] = prim
                if prim.get("explosion_exists"):
                    primary_event_count += 1
                    mints_with_primary.add(mint)
        if explosion_rows:
            pq.write_table(
                pa.Table.from_pylist(explosion_rows),
                expl_parts / f"{part}.parquet",
            )
            explosion_event_rows += len(explosion_rows)

        for obs in _iter_obs_for_partition(index_root, part):
            mint = obs["mint"]
            T = float(obs["observation_time_t"])
            oid = obs["observation_id"]
            arr = by_mint.get(mint)
            last_path_t = None
            if mint in entity:
                last_path_t = entity[mint].get("last_event_unix")
            mig = migration_unix.get(mint)

            row: dict[str, Any] = {
                "observation_id": oid,
                "mint": mint,
                "observation_time_t": T,
                "token_age_seconds": obs.get("token_age_seconds"),
                "observation_trigger": obs.get("observation_trigger"),
                "trigger_value": obs.get("trigger_value"),
                "target_dataset_version": "broad_targets_v1",
                "label_semantics": "target_dataset_v3_compute_reuse",
                "label_only": True,
            }

            if arr is None:
                ref = {
                    "reference_price": None,
                    "reference_price_time": None,
                    "reference_price_source": None,
                    "reference_price_age_sec": None,
                    "price_target_available": False,
                }
                row.update(ref)
                arr = _empty_path_arr()
            else:
                ref = reference_price_at_t(arr, T)
                row.update(
                    {
                        k: ref[k]
                        for k in (
                            "reference_price",
                            "reference_price_time",
                            "reference_price_source",
                            "reference_price_age_sec",
                            "price_target_available",
                        )
                    }
                )
                row["reference_regime"] = ref.get("reference_regime")

            row.update(
                compute_return_barrier_mfe(
                    arr, T=T, ref=ref, last_path_t=last_path_t
                )
            )
            row.update(
                compute_activity_survival(
                    arr, T=T, coverage=coverage, migration_unix=mig
                )
            )
            row.update(compute_graduation(T=T, migration_unix=mig, coverage=coverage))
            row.update(compute_persistence(arr, T=T, coverage=coverage))
            row.update(
                compute_explosion_obs(
                    T=T,
                    primary=primary_by_mint.get(mint),
                    all_defs=all_defs_by_mint.get(mint, []),
                    coverage=coverage,
                    ref=ref,
                )
            )
            row.update(compute_execution(arr, T=T, ref=ref))
            row.update(
                compute_ownership_targets(
                    mint=mint, T=T, ownership_meta=own_by_mint.get(mint)
                )
            )
            target_batch.append(_normalize_row(row, target_schema))
            ref_batch.append(
                _normalize_row(
                    {
                        "observation_id": oid,
                        "mint": mint,
                        "observation_time_t": T,
                        "future_path_store": "broad_partitioned_path_v1",
                        **future_path_refs_for_obs(arr, T=T),
                    },
                    ref_schema,
                )
            )
            observations_processed += 1
            if len(target_batch) >= batch_flush:
                _flush()

        # Free partition memory
        del by_mint, entity, path_rows, primary_by_mint, all_defs_by_mint

    _flush(force_rotate=True)
    if target_writer is not None:
        target_writer.close()
    if ref_writer is not None:
        ref_writer.close()

    # Concatenate explosion parts for a single audit-friendly file when small enough
    expl_files = sorted(expl_parts.glob("*.parquet"))
    expl_path = out_dir / "explosion_events_v1.parquet"
    if expl_files:
        tables = [pq.read_table(p) for p in expl_files]
        pq.write_table(pa.concat_tables(tables, promote_options="default"), expl_path)
    else:
        pq.write_table(pa.table({"mint": pa.array([], pa.string())}), expl_path)

    meta = {
        "generated_at": utc_now(),
        "version": TARGETS_STORE,
        "observations_processed": observations_processed,
        "event_curve_mints": len(event_mints),
        "partitions_processed": len(partitions),
        "explosion_definition_rows": explosion_event_rows,
        "primary_explosion_events": primary_event_count,
        "mints_with_primary_explosion": len(mints_with_primary),
        "graduation_censor_decision": censor,
        "coverage_manifest_version": cov_manifest.get("manifest_version"),
        "path_construction": (
            "derived_from_broad_sequence_events_v1_plus_postgard_snapshots"
        ),
        "artifacts": {
            "observation_targets": str(targets_parts),
            "future_path_refs": str(refs_parts),
            "explosion_events": str(expl_path),
            "future_path_parts": str(path_parts_dir),
        },
    }
    write_json(out_dir / "broad_targets_manifest.json", meta)
    return meta
