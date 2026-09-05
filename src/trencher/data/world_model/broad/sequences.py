"""Broad causal event / curve / creator sequence stores from FULL Pump trades.

Partitioning
------------
Outputs are partitioned by ``mint_partition`` = first 2 hex chars of
``sha256(mint)`` (256 partitions). Within each partition, ``seq_idx`` is
**local** (0-based contiguous per partition file). Entity indexes map
``mint -> (partition, start_idx, end_idx)`` with half-open ``[start, end)``.

Streaming
---------
Trade shards are scanned with pyarrow dataset batches. Rows are appended to
per-partition staging parquet files (not held as one giant Python list), then
each partition is loaded, fingerprint-deduped, sorted, and rewritten.
"""

from __future__ import annotations

import logging
import math
import shutil
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from trencher.data.feasibility.concepts import SYSTEM_PROGRAM_WALLET
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.pit import _to_unix
from trencher.data.world_model.sequences import (
    _event_fingerprint,
    _order_key,
)

logger = logging.getLogger(__name__)

EVENTS_STORE = "broad_sequence_events_v1"
CURVE_STORE = "broad_sequence_curve_v1"
CREATOR_STORE = "broad_sequence_creator_v1"
ENTITY_INDEX_NAME = "entity_index.parquet"

_EVENT_SCHEMA_COLS = (
    "mint",
    "event_time_unix",
    "tx_signature",
    "event_fingerprint",
    "order_key",
    "wallet",
    "side",
    "is_buy",
    "sol_amount",
    "token_amount",
    "price_sol",
    "v_sol",
    "v_tok",
    "curve_pct_depleted",
    "source_quality_flag",
    "has_reserves",
    "has_finite_price",
    "mint_partition",
)


def mint_partition(mint: str) -> str:
    return sha256(mint.encode()).hexdigest()[:2]


def _is_finite_positive(x: Any) -> bool:
    if x is None:
        return False
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0.0


def _prefer_price(price_sol: Any, v_sol: Any, v_tok: Any) -> float | None:
    if _is_finite_positive(v_sol) and _is_finite_positive(v_tok):
        px = float(v_sol) / float(v_tok)
        if math.isfinite(px) and px > 0:
            return px
    if _is_finite_positive(price_sol):
        return float(price_sol)
    return None


def _flush_staging(
    staging_dir: Path,
    part: str,
    rows: list[dict[str, Any]],
    writers: dict[str, pq.ParquetWriter],
) -> None:
    if not rows:
        return
    ensure_dir(staging_dir / part)
    table = pa.Table.from_pylist(rows)
    path = staging_dir / part / "staging.parquet"
    if part not in writers:
        writers[part] = pq.ParquetWriter(path, table.schema)
    writers[part].write_table(table)


def _finalize_partition_events(
    staging_part: Path,
    out_part: Path,
) -> tuple[list[dict[str, Any]], int, int, set[str]]:
    """Dedupe + sort staging into final part.parquet; return entity rows."""
    ensure_dir(out_part)
    staging_file = staging_part / "staging.parquet"
    if not staging_file.exists():
        return [], 0, 0, set()

    best: dict[str, dict[str, Any]] = {}
    exact_dup = 0
    price_capable: set[str] = set()
    table = pq.read_table(staging_file)
    for row in table.to_pylist():
        fp = row["event_fingerprint"]
        prev = best.get(fp)
        if prev is None:
            best[fp] = row
        else:
            exact_dup += 1
            fields = ("price_sol", "v_sol", "v_tok", "tx_signature")
            score = sum(1 for k in fields if row.get(k) is not None)
            prev_score = sum(1 for k in fields if prev.get(k) is not None)
            if score >= prev_score:
                best[fp] = row
        if row.get("has_finite_price"):
            price_capable.add(str(row["mint"]))

    rows = list(best.values())
    rows.sort(
        key=lambda r: (
            r["mint"],
            float(r["event_time_unix"]),
            r.get("order_key") or "",
        )
    )
    for i, row in enumerate(rows):
        row["seq_idx"] = i
    out_path = out_part / "part.parquet"
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), out_path)
    else:
        pq.write_table(pa.table({"seq_idx": pa.array([], pa.int64())}), out_path)

    entity: list[dict[str, Any]] = []
    i = 0
    n = len(rows)
    while i < n:
        mint = rows[i]["mint"]
        j = i + 1
        while j < n and rows[j]["mint"] == mint:
            j += 1
        entity.append(
            {
                "mint": mint,
                "partition": out_part.name,
                "start_idx": i,
                "end_idx": j,
                "n_rows": j - i,
            }
        )
        i = j
    return entity, len(rows), exact_dup, price_capable


def _write_curve_from_events(
    events_part: Path, curve_part: Path
) -> tuple[list[dict[str, Any]], int]:
    path = events_part / "part.parquet"
    if not path.exists() or pq.ParquetFile(path).metadata.num_rows == 0:
        ensure_dir(curve_part)
        pq.write_table(
            pa.table({"seq_idx": pa.array([], pa.int64())}),
            curve_part / "part.parquet",
        )
        return [], 0
    rows = []
    for r in pq.read_table(path).to_pylist():
        if (
            r.get("v_sol") is None
            and r.get("v_tok") is None
            and r.get("curve_pct_depleted") is None
            and not r.get("has_finite_price")
        ):
            continue
        rows.append(
            {
                "mint": r["mint"],
                "event_time_unix": r["event_time_unix"],
                "order_key": r["order_key"],
                "side": r.get("side"),
                "is_buy": r.get("is_buy"),
                "sol_amount": r.get("sol_amount"),
                "token_amount": r.get("token_amount"),
                "price_sol": r.get("price_sol"),
                "v_sol": r.get("v_sol"),
                "v_tok": r.get("v_tok"),
                "curve_pct_depleted": r.get("curve_pct_depleted"),
                "has_reserves": r.get("has_reserves"),
                "source_quality_flag": r.get("source_quality_flag"),
                "source": "pump_trade_reserve_trajectory",
                "mint_partition": r.get("mint_partition"),
                "seq_idx": r["seq_idx"],  # temporary; reassigned below
            }
        )
    rows.sort(
        key=lambda r: (
            r["mint"],
            float(r["event_time_unix"]),
            r.get("order_key") or "",
        )
    )
    for i, row in enumerate(rows):
        row["seq_idx"] = i
    ensure_dir(curve_part)
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), curve_part / "part.parquet")
    else:
        pq.write_table(
            pa.table({"seq_idx": pa.array([], pa.int64())}),
            curve_part / "part.parquet",
        )
    entity: list[dict[str, Any]] = []
    i = 0
    n = len(rows)
    while i < n:
        mint = rows[i]["mint"]
        j = i + 1
        while j < n and rows[j]["mint"] == mint:
            j += 1
        entity.append(
            {
                "mint": mint,
                "partition": curve_part.name,
                "start_idx": i,
                "end_idx": j,
                "n_rows": j - i,
            }
        )
        i = j
    return entity, len(rows)


def build_broad_sequence_events_v1(
    *,
    broad_root: Path,
    pump_root: Path,
    event_curve_mints: set[str],
    batch_size: int = 256_000,
    staging_flush_rows: int = 50_000,
) -> dict[str, Any]:
    """Stream full Pump trades into partitioned event + curve stores."""
    events_root = ensure_dir(broad_root / EVENTS_STORE)
    curve_root = ensure_dir(broad_root / CURVE_STORE)
    staging_root = ensure_dir(broad_root / "_staging_events")
    trades_dir = pump_root / "trades"
    if not trades_dir.exists():
        raise FileNotFoundError(f"missing trades dir {trades_dir}")

    excluded_system = 0
    raw_n = 0
    hetero_sig_groups = 0
    sig_fps: dict[str, set[str]] = defaultdict(set)
    pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
    writers: dict[str, pq.ParquetWriter] = {}
    pending_total = 0

    cols = [
        "mint",
        "user_wallet",
        "event_time",
        "is_buy",
        "sol_amount",
        "token_amount",
        "tx_signature",
        "price_sol",
        "v_sol_bonding_curve",
        "v_tokens_bonding_curve",
        "curve_pct_depleted",
    ]
    dataset = ds.dataset(str(trades_dir), format="parquet")
    logger.info(
        "Scanning Pump trades for %d event_curve mints", len(event_curve_mints)
    )
    try:
        for batch in dataset.scanner(columns=cols, batch_size=batch_size).to_batches():
            names = batch.schema.names
            cols_py = [batch.column(c).to_pylist() for c in names]
            for vals in zip(*cols_py, strict=True):
                row = dict(zip(names, vals, strict=True))
                mint = row.get("mint")
                if not mint or mint not in event_curve_mints:
                    continue
                wallet = row.get("user_wallet")
                if wallet == SYSTEM_PROGRAM_WALLET:
                    excluded_system += 1
                    continue
                et = _to_unix(row.get("event_time"))
                if et is None:
                    continue
                raw_n += 1
                mint = str(mint)
                is_buy = row.get("is_buy")
                side = None if is_buy is None else ("buy" if is_buy else "sell")
                v_sol = row.get("v_sol_bonding_curve")
                v_tok = row.get("v_tokens_bonding_curve")
                fp = _event_fingerprint(
                    mint=mint,
                    tx_signature=row.get("tx_signature"),
                    event_time_unix=et,
                    wallet=wallet,
                    is_buy=is_buy if isinstance(is_buy, bool) else None,
                    sol_amount=row.get("sol_amount"),
                    token_amount=row.get("token_amount"),
                    v_sol_bonding_curve=v_sol,
                    v_tokens_bonding_curve=v_tok,
                )
                price = row.get("price_sol")
                part = mint_partition(mint)
                cand = {
                    "mint": mint,
                    "event_time_unix": float(et),
                    "tx_signature": row.get("tx_signature"),
                    "event_fingerprint": fp,
                    "order_key": _order_key(
                        event_time_unix=et,
                        tx_signature=row.get("tx_signature"),
                        wallet=wallet,
                        side=side,
                    ),
                    "wallet": wallet,
                    "side": side,
                    "is_buy": (
                        bool(is_buy) if isinstance(is_buy, bool) else None
                    ),
                    "sol_amount": row.get("sol_amount"),
                    "token_amount": row.get("token_amount"),
                    "price_sol": price,
                    "v_sol": v_sol,
                    "v_tok": v_tok,
                    "curve_pct_depleted": row.get("curve_pct_depleted"),
                    "source_quality_flag": "observed",
                    "has_reserves": bool(
                        v_sol is not None or v_tok is not None
                    ),
                    "has_finite_price": _prefer_price(price, v_sol, v_tok)
                    is not None,
                    "mint_partition": part,
                }
                pending[part].append(cand)
                pending_total += 1
                sig = row.get("tx_signature")
                if sig:
                    sig_fps[f"{mint}:{sig}"].add(fp)
                if pending_total >= staging_flush_rows:
                    for p, rows in list(pending.items()):
                        _flush_staging(staging_root, p, rows, writers)
                        pending[p] = []
                    pending_total = 0
        for p, rows in list(pending.items()):
            _flush_staging(staging_root, p, rows, writers)
            pending[p] = []
    finally:
        for w in writers.values():
            w.close()

    hetero_sig_groups = sum(1 for fps in sig_fps.values() if len(fps) > 1)

    entity_rows: list[dict[str, Any]] = []
    curve_entity: list[dict[str, Any]] = []
    event_rows_total = 0
    curve_rows_total = 0
    exact_dup = 0
    price_capable: set[str] = set()
    parts = sorted(
        d.name for d in staging_root.iterdir() if d.is_dir()
    ) if staging_root.exists() else []
    for part in parts:
        ent, n, dups, caps = _finalize_partition_events(
            staging_root / part, events_root / part
        )
        entity_rows.extend(ent)
        event_rows_total += n
        exact_dup += dups
        price_capable |= caps
        cent, cn = _write_curve_from_events(events_root / part, curve_root / part)
        curve_entity.extend(cent)
        curve_rows_total += cn

    # Cleanup staging
    if staging_root.exists():
        shutil.rmtree(staging_root, ignore_errors=True)

    events_idx = events_root / ENTITY_INDEX_NAME
    curve_idx = curve_root / ENTITY_INDEX_NAME
    if entity_rows:
        pq.write_table(pa.Table.from_pylist(entity_rows), events_idx)
    else:
        pq.write_table(pa.table({"mint": pa.array([], pa.string())}), events_idx)
    if curve_entity:
        pq.write_table(pa.Table.from_pylist(curve_entity), curve_idx)
    else:
        pq.write_table(pa.table({"mint": pa.array([], pa.string())}), curve_idx)

    snap_path = pump_root / "snapshots.parquet"
    snap_note = (
        "snapshots available but not merged (trade trajectory sufficient)"
        if snap_path.exists()
        else "snapshots absent"
    )

    meta = {
        "generated_at": utc_now(),
        "events_store": EVENTS_STORE,
        "curve_store": CURVE_STORE,
        "entity_index_semantics": (
            "per-partition local seq_idx; entity_index columns: "
            "mint, partition, start_idx, end_idx, n_rows "
            "(half-open [start_idx, end_idx) within partition part.parquet)"
        ),
        "partitioning": "sha256(mint).hexdigest()[:2]",
        "raw_input_rows": raw_n,
        "deduped_event_rows": event_rows_total,
        "curve_rows": curve_rows_total,
        "exact_duplicate_groups_removed": exact_dup,
        "signatures_with_heterogeneous_event_fingerprints": hetero_sig_groups,
        "excluded_system_program_rows": excluded_system,
        "dedupe_policy": "conservative_event_fingerprint",
        "price_capable_mints": len(price_capable),
        "entities": len(entity_rows),
        "snapshots_note": snap_note,
        "snap_rows_added": 0,
        "events_root": str(events_root),
        "curve_root": str(curve_root),
    }
    write_json(events_root / "manifest.json", meta)
    write_json(curve_root / "manifest.json", {**meta, "store": CURVE_STORE})
    meta["price_capable_mint_set"] = price_capable
    return meta


def _write_sorted_partition(
    part_dir: Path, rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    ensure_dir(part_dir)
    if not rows:
        pq.write_table(
            pa.table({"seq_idx": pa.array([], pa.int64())}),
            part_dir / "part.parquet",
        )
        return [], 0
    rows.sort(
        key=lambda r: (
            r["mint"],
            float(r["event_time_unix"]),
            r.get("order_key") or "",
        )
    )
    for i, row in enumerate(rows):
        row["seq_idx"] = i
    pq.write_table(pa.Table.from_pylist(rows), part_dir / "part.parquet")
    entity: list[dict[str, Any]] = []
    i = 0
    n = len(rows)
    while i < n:
        mint = rows[i]["mint"]
        j = i + 1
        while j < n and rows[j]["mint"] == mint:
            j += 1
        entity.append(
            {
                "mint": mint,
                "partition": part_dir.name,
                "start_idx": i,
                "end_idx": j,
                "n_rows": j - i,
            }
        )
        i = j
    return entity, n


def build_broad_sequence_creator_v1(
    *,
    broad_root: Path,
    pump_root: Path,
    population_path: Path,
) -> dict[str, Any]:
    """Causal prior launches per creator from tokens (no creator_past_* leakage).

    Only launches with ``detected_at < current mint detected_at`` are included.
    Reconstructs from the tokens table sorted by creator + detected_at rather
    than using ``creator_past_tokens`` / ``creator_past_rugs`` aggregates.
    """
    out_root = ensure_dir(broad_root / CREATOR_STORE)
    pop = pq.read_table(
        population_path,
        columns=["mint", "detected_at_unix", "creator", "event_curve_population"],
    )
    tokens_path = pump_root / "tokens.parquet"
    table = pq.read_table(
        tokens_path, columns=["mint", "detected_at", "creator", "is_mayhem_mode"]
    )
    by_creator: dict[str, list[tuple[float, str, bool | None]]] = defaultdict(list)
    for row in table.to_pylist():
        mint = row.get("mint")
        creator = row.get("creator")
        det = _to_unix(row.get("detected_at"))
        if not mint or not creator or det is None:
            continue
        by_creator[str(creator)].append(
            (float(det), str(mint), row.get("is_mayhem_mode"))
        )
    for creator in by_creator:
        by_creator[creator].sort(key=lambda x: (x[0], x[1]))

    event_mints = {
        str(m)
        for m, flag in zip(
            pop.column("mint").to_pylist(),
            pop.column("event_curve_population").to_pylist(),
            strict=True,
        )
        if flag and m
    }
    det_by_mint = {
        str(m): float(d)
        for m, d in zip(
            pop.column("mint").to_pylist(),
            pop.column("detected_at_unix").to_pylist(),
            strict=True,
        )
        if m and d is not None
    }
    creator_by_mint = {
        str(m): c
        for m, c in zip(
            pop.column("mint").to_pylist(),
            pop.column("creator").to_pylist(),
            strict=True,
        )
        if m and c
    }

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mint in event_mints:
        creator = creator_by_mint.get(mint)
        det = det_by_mint.get(mint)
        if creator is None or det is None:
            continue
        priors = by_creator.get(creator) or []
        prior_rank = 0
        for p_det, p_mint, p_mayhem in priors:
            if p_det >= det:
                break
            if p_mint == mint:
                continue
            prior_rank += 1
            part = mint_partition(mint)
            buckets[part].append(
                {
                    "mint": mint,
                    "event_time_unix": p_det,
                    "order_key": f"{p_det:.6f}|{p_mint}",
                    "creator": creator,
                    "prior_mint": p_mint,
                    "prior_detected_at_unix": p_det,
                    "prior_rank": prior_rank,
                    "prior_is_mayhem_mode": p_mayhem,
                    "source": "tokens_table_creator_timeline",
                    "source_quality_flag": "launch_time_fields_only",
                    "mint_partition": part,
                }
            )

    entity_rows: list[dict[str, Any]] = []
    total = 0
    for part, rows in sorted(buckets.items()):
        ent, n = _write_sorted_partition(out_root / part, rows)
        entity_rows.extend(ent)
        total += n
    idx_path = out_root / ENTITY_INDEX_NAME
    if entity_rows:
        pq.write_table(pa.Table.from_pylist(entity_rows), idx_path)
    else:
        pq.write_table(pa.table({"mint": pa.array([], pa.string())}), idx_path)

    meta = {
        "generated_at": utc_now(),
        "store": CREATOR_STORE,
        "rows": total,
        "entities": len(entity_rows),
        "policy": (
            "Causal prior launches only (detected_at < current). "
            "Does NOT use creator_past_rugs / lifetime outcome aggregates."
        ),
        "entity_index_semantics": (
            "per-partition local seq_idx; see broad_sequence_events_v1"
        ),
        "path": str(out_root),
    }
    write_json(out_root / "manifest.json", meta)
    return meta


def load_entity_index(store_root: Path) -> dict[str, dict[str, Any]]:
    path = store_root / ENTITY_INDEX_NAME
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in pq.read_table(path).to_pylist():
        mint = row.get("mint")
        if mint:
            out[str(mint)] = row
    return out


def load_partition_times(store_root: Path, partition: str) -> list[float]:
    path = store_root / partition / "part.parquet"
    if not path.exists() or pq.ParquetFile(path).metadata.num_rows == 0:
        return []
    return [
        float(x)
        for x in pq.read_table(path, columns=["event_time_unix"])
        .column(0)
        .to_pylist()
    ]


def build_broad_sequences_v1(
    *,
    world_root: Path,
    pump_root: Path,
    broad_root: Path | None = None,
    population_path: Path | None = None,
) -> dict[str, Any]:
    """Build event, curve, and creator sequence stores for the broad population."""
    broad_root = ensure_dir(broad_root or world_root / "broad")
    population_path = population_path or (
        broad_root / "broad_population_v1.parquet"
    )
    from trencher.data.world_model.broad.population import (
        event_curve_mints,
        update_price_target_flags,
    )

    mints = set(event_curve_mints(population_path))
    logger.info("Building broad sequences for %d event_curve mints", len(mints))
    events_meta = build_broad_sequence_events_v1(
        broad_root=broad_root,
        pump_root=pump_root,
        event_curve_mints=mints,
    )
    price_set = events_meta.pop("price_capable_mint_set", set())
    price_meta = update_price_target_flags(population_path, price_set)
    creator_meta = build_broad_sequence_creator_v1(
        broad_root=broad_root,
        pump_root=pump_root,
        population_path=population_path,
    )
    return {
        "generated_at": utc_now(),
        "events": events_meta,
        "creator": creator_meta,
        "price_target_update": price_meta,
        "event_curve_mints": len(mints),
    }
