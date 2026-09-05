"""Broad observation index + causal sequence refs for ALL event_curve tokens.

No token subsample. Observation times reuse ``select_observation_points``
(fixed ages from PIT_AGE_OFFSETS_SEC + power-of-two event counts 1..512).
Coincident times are deduped inside that helper.

Outputs (partitioned by mint hash prefix):
- ``broad_observation_index_v1/``
- ``broad_observation_sequence_refs_v1/`` with causal end offsets into the
  shared event/curve/creator stores (sequences stored once).
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.broad.sequences import (
    CREATOR_STORE,
    CURVE_STORE,
    EVENTS_STORE,
    load_entity_index,
    load_partition_times,
    mint_partition,
)
from trencher.data.world_model.freeze import FREEZE_ID
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.observations import (
    EVENT_TRADE_COUNT_CAP,
    EVENT_TRADE_COUNTS,
    select_observation_points,
)
from trencher.data.world_model.pit import PIT_AGE_OFFSETS_SEC, PIT_ENGINE_VERSION
from trencher.data.world_model.sequences import _end_idx_for_t

logger = logging.getLogger(__name__)

OBS_INDEX_STORE = "broad_observation_index_v1"
OBS_REFS_STORE = "broad_observation_sequence_refs_v1"
BROAD_OBS_VERSION = "broad_observation_v1"


def _projected_obs_upper_bound(n_tokens: int) -> dict[str, Any]:
    """Projected row counts before coincident-time dedupe."""
    n_ages = len(PIT_AGE_OFFSETS_SEC)
    n_event_slots = len(EVENT_TRADE_COUNTS)
    return {
        "tokens": n_tokens,
        "fixed_age_obs_if_all_emitted": n_tokens * n_ages,
        "event_count_obs_upper_bound": n_tokens * n_event_slots,
        "total_upper_bound_before_dedupe": n_tokens * (n_ages + n_event_slots),
        "note": (
            "Actual count is lower after coincident-time dedupe and when "
            "tokens have fewer than 512 trades."
        ),
    }


def _flush_partition_rows(
    root: Path,
    part: str,
    rows: list[dict[str, Any]],
    writers: dict[str, pq.ParquetWriter],
) -> None:
    if not rows:
        return
    ensure_dir(root / part)
    table = pa.Table.from_pylist(rows)
    path = root / part / "part.parquet"
    if part not in writers:
        writers[part] = pq.ParquetWriter(path, table.schema)
    writers[part].write_table(table)


def build_broad_observations_v1(
    *,
    world_root: Path,
    pump_root: Path,
    broad_root: Path | None = None,
    population_path: Path | None = None,
) -> dict[str, Any]:
    """Materialize observation index + causal sequence refs for event_curve tokens."""
    del pump_root  # sequences already built from pump; launch times from population
    broad_root = ensure_dir(broad_root or world_root / "broad")
    population_path = population_path or broad_root / "broad_population_v1.parquet"
    index_root = ensure_dir(broad_root / OBS_INDEX_STORE)
    refs_root = ensure_dir(broad_root / OBS_REFS_STORE)

    pop = pq.read_table(population_path)
    event_rows = [
        r
        for r in pop.to_pylist()
        if r.get("event_curve_population") and r.get("mint") and r.get("detected_at_unix")
    ]
    projected = _projected_obs_upper_bound(len(event_rows))
    logger.info(
        "Building broad observations for %d event_curve tokens "
        "(projected upper bound ~%s)",
        len(event_rows),
        projected["total_upper_bound_before_dedupe"],
    )

    events_idx = load_entity_index(broad_root / EVENTS_STORE)
    curve_idx = load_entity_index(broad_root / CURVE_STORE)
    creator_idx = load_entity_index(broad_root / CREATOR_STORE)

    # Cache partition times as needed.
    events_times_cache: dict[str, list[float]] = {}
    curve_times_cache: dict[str, list[float]] = {}
    creator_times_cache: dict[str, list[float]] = {}

    def _times(
        cache: dict[str, list[float]], store: str, partition: str
    ) -> list[float]:
        if partition not in cache:
            cache[partition] = load_partition_times(broad_root / store, partition)
        return cache[partition]

    index_writers: dict[str, pq.ParquetWriter] = {}
    refs_writers: dict[str, pq.ParquetWriter] = {}
    index_batches: dict[str, list[dict[str, Any]]] = {}
    refs_batches: dict[str, list[dict[str, Any]]] = {}

    observations_n = 0
    tokens_n = 0
    duplicate_collisions = 0
    trigger_counts: Counter[str] = Counter()
    obs_per_token: list[int] = []
    FLUSH = 2_000

    try:
        for prow in event_rows:
            mint = str(prow["mint"])
            launch_t = float(prow["detected_at_unix"])
            part = mint_partition(mint)
            tokens_n += 1

            e_meta = events_idx.get(mint)
            trade_times: list[float] = []
            if e_meta:
                times = _times(events_times_cache, EVENTS_STORE, e_meta["partition"])
                lo, hi = int(e_meta["start_idx"]), int(e_meta["end_idx"])
                trade_times = times[lo:hi]

            before = len(PIT_AGE_OFFSETS_SEC) + sum(
                1 for c in EVENT_TRADE_COUNTS if c <= len(trade_times)
            )
            points = select_observation_points(
                mint=mint, launch_t=launch_t, trade_event_times=trade_times
            )
            if before > len(points):
                duplicate_collisions += before - len(points)
            obs_per_token.append(len(points))

            c_meta = curve_idx.get(mint)
            cr_meta = creator_idx.get(mint)
            if c_meta:
                _times(curve_times_cache, CURVE_STORE, c_meta["partition"])
            if cr_meta:
                _times(creator_times_cache, CREATOR_STORE, cr_meta["partition"])

            for point in points:
                t = float(point["observation_time_t"])
                oid = point["observation_id"]
                trigger_counts[point["observation_trigger"]] += 1
                index_row = {
                    "mint": mint,
                    "observation_id": oid,
                    "observation_time_t": t,
                    "token_age_seconds": point["token_age_seconds"],
                    "observation_trigger": point["observation_trigger"],
                    "trigger_value": str(point["trigger_value"]),
                    "trigger_numeric": point["trigger_numeric"],
                    "source_freeze_version": FREEZE_ID,
                    "pit_engine_version": PIT_ENGINE_VERSION,
                    "observation_dataset_version": BROAD_OBS_VERSION,
                    "mint_partition": part,
                    "detected_at_unix": launch_t,
                }
                index_batches.setdefault(part, []).append(index_row)

                def _ref_slice(
                    meta: dict[str, Any] | None,
                    store: str,
                    cache: dict[str, list[float]],
                    obs_t: float,
                ) -> tuple[str | None, int, int, int]:
                    if not meta:
                        return None, 0, 0, 0
                    p = str(meta["partition"])
                    lo, hi = int(meta["start_idx"]), int(meta["end_idx"])
                    times = _times(cache, store, p)
                    end = _end_idx_for_t(times, lo, hi, obs_t) if hi > lo else lo
                    return p, lo, end, max(0, end - lo)

                e_part, e_lo, e_hi, e_n = _ref_slice(
                    e_meta, EVENTS_STORE, events_times_cache, t
                )
                c_part, c_lo, c_hi, c_n = _ref_slice(
                    c_meta, CURVE_STORE, curve_times_cache, t
                )
                cr_part, cr_lo, cr_hi, cr_n = _ref_slice(
                    cr_meta, CREATOR_STORE, creator_times_cache, t
                )
                refs_batches.setdefault(part, []).append(
                    {
                        "mint": mint,
                        "observation_id": oid,
                        "observation_time_t": t,
                        "mint_partition": part,
                        "events_store": EVENTS_STORE,
                        "events_partition": e_part,
                        "events_start_idx": e_lo,
                        "events_end_idx": e_hi,
                        "events_count": e_n,
                        "curve_store": CURVE_STORE,
                        "curve_partition": c_part,
                        "curve_start_idx": c_lo,
                        "curve_end_idx": c_hi,
                        "curve_count": c_n,
                        "creator_store": CREATOR_STORE,
                        "creator_partition": cr_part,
                        "creator_start_idx": cr_lo,
                        "creator_end_idx": cr_hi,
                        "creator_count": cr_n,
                        "shared_sequences_once": True,
                    }
                )
                observations_n += 1

            if sum(len(v) for v in index_batches.values()) >= FLUSH:
                for p, rows in list(index_batches.items()):
                    _flush_partition_rows(index_root, p, rows, index_writers)
                    index_batches[p] = []
                for p, rows in list(refs_batches.items()):
                    _flush_partition_rows(refs_root, p, rows, refs_writers)
                    refs_batches[p] = []

        for p, rows in list(index_batches.items()):
            _flush_partition_rows(index_root, p, rows, index_writers)
        for p, rows in list(refs_batches.items()):
            _flush_partition_rows(refs_root, p, rows, refs_writers)
    finally:
        for w in index_writers.values():
            w.close()
        for w in refs_writers.values():
            w.close()

    obs_sorted = sorted(obs_per_token)

    def _pct(p: float) -> float | None:
        if not obs_sorted:
            return None
        idx = min(len(obs_sorted) - 1, max(0, int(round(p * (len(obs_sorted) - 1)))))
        return float(obs_sorted[idx])

    meta = {
        "generated_at": utc_now(),
        "version": BROAD_OBS_VERSION,
        "tokens_n": tokens_n,
        "observations_n": observations_n,
        "projected_row_counts": projected,
        "actual_row_counts": {
            "tokens": tokens_n,
            "observations": observations_n,
            "duplicate_time_collisions_removed": duplicate_collisions,
        },
        "observations_by_trigger": dict(trigger_counts),
        "observations_per_token": {
            "min": obs_sorted[0] if obs_sorted else 0,
            "p50": _pct(0.5),
            "p90": _pct(0.9),
            "p99": _pct(0.99),
            "max": obs_sorted[-1] if obs_sorted else 0,
            "mean": (sum(obs_per_token) / len(obs_per_token) if obs_per_token else 0.0),
        },
        "policy": {
            "population": "all event_curve_population tokens (no subsample)",
            "selection": "select_observation_points (fixed ages + event counts)",
            "fixed_ages_sec": {k: v for k, v in PIT_AGE_OFFSETS_SEC},
            "event_trade_counts": list(EVENT_TRADE_COUNTS),
            "event_trade_count_cap": EVENT_TRADE_COUNT_CAP,
            "dedupe": "coincident microsecond keys; prefer fixed_age",
            "sequence_refs": (
                "causal exclusive end_idx via _end_idx_for_t; "
                "shared sequence stores referenced by offsets"
            ),
            "labels": "not_generated_here",
        },
        "artifacts": {
            "observation_index": str(index_root),
            "observation_sequence_refs": str(refs_root),
        },
    }
    write_json(index_root / "manifest.json", meta)
    write_json(refs_root / "manifest.json", meta)
    return meta
