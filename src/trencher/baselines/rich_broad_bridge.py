"""Causal broad-sequence refs for frozen rich observations (mint + T).

Rich and broad observation IDs share the same hash formula but are produced
from different launch / trade streams, so OID identity across populations is
invalid. This module derives exclusive causal event/curve/creator slices from
the shared broad entity indexes at the frozen rich ``(mint, T)`` without
mutating frozen sequence datasets or rich observation IDs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.sequences import _end_idx_for_t

BRIDGE_STORE = "rich_broad_sequence_refs_v1"
BRIDGE_VERSION = "rich_broad_sequence_refs_v1"
DIAGNOSTIC_ID = "RICH_BROAD_REF_DIAGNOSTIC_V1"


class RichBroadBridgeError(RuntimeError):
    """Invalid or incomplete rich→broad causal bridge."""


def _ref_slice_for_t(
    *,
    meta: dict[str, Any] | None,
    store: str,
    times_cache: dict[str, list[float]],
    store_root: Path,
    observation_time_t: float,
) -> tuple[str | None, int, int, int, bool]:
    """Return (partition, start, end_excl, count, modality_available).

    Unavailable modality (no entity row / missing store) → zero-length indices
    with ``available=False``. Present entity with no rows <= T → available but
    zero-length causal window.
    """
    if not meta:
        return None, 0, 0, 0, False
    part = str(meta["partition"])
    lo, hi = int(meta["start_idx"]), int(meta["end_idx"])
    if hi < lo or lo < 0:
        raise RichBroadBridgeError(
            f"corrupt entity index for store={store} partition={part}: "
            f"start={lo} end={hi}"
        )
    if hi == lo:
        return part, lo, lo, 0, True
    key = f"{store}:{part}"
    if key not in times_cache:
        times_cache[key] = load_partition_times(store_root, part)
    times = times_cache[key]
    if hi > len(times):
        raise RichBroadBridgeError(
            f"entity end_idx out of range store={store} partition={part}: "
            f"end={hi} n_times={len(times)}"
        )
    end = _end_idx_for_t(times, lo, hi, float(observation_time_t))
    return part, lo, end, max(0, end - lo), True


def build_rich_broad_sequence_refs_v1(
    *,
    world_root: Path,
    assignment_rows: list[dict[str, Any]],
    out_root: Path | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Build derived bridge keyed by frozen rich observation_id.

    Does not modify broad refs, sequence_dataset_v2, or rich assignments.
    """
    world_root = Path(world_root)
    broad_root = world_root / "broad"
    out_root = ensure_dir(out_root or broad_root / BRIDGE_STORE)

    events_idx = load_entity_index(broad_root / EVENTS_STORE)
    curve_idx = load_entity_index(broad_root / CURVE_STORE)
    creator_idx = load_entity_index(broad_root / CREATOR_STORE)
    events_root = broad_root / EVENTS_STORE
    curve_root = broad_root / CURVE_STORE
    creator_root = broad_root / CREATOR_STORE

    times_cache: dict[str, list[float]] = {}
    by_mp: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    unavailable = {"events": 0, "curve": 0, "creator": 0}

    for i, row in enumerate(assignment_rows):
        oid = str(row["observation_id"])
        if oid in seen:
            raise RichBroadBridgeError(f"duplicate rich observation_id={oid}")
        seen.add(oid)
        mint = str(row["mint"])
        T = float(row["observation_time_t"])
        mp = str(row.get("mint_partition") or mint_partition(mint))

        e_part, e_lo, e_hi, e_n, e_ok = _ref_slice_for_t(
            meta=events_idx.get(mint),
            store=EVENTS_STORE,
            times_cache=times_cache,
            store_root=events_root,
            observation_time_t=T,
        )
        c_part, c_lo, c_hi, c_n, c_ok = _ref_slice_for_t(
            meta=curve_idx.get(mint),
            store=CURVE_STORE,
            times_cache=times_cache,
            store_root=curve_root,
            observation_time_t=T,
        )
        cr_part, cr_lo, cr_hi, cr_n, cr_ok = _ref_slice_for_t(
            meta=creator_idx.get(mint),
            store=CREATOR_STORE,
            times_cache=times_cache,
            store_root=creator_root,
            observation_time_t=T,
        )
        if not e_ok:
            unavailable["events"] += 1
        if not c_ok:
            unavailable["curve"] += 1
        if not cr_ok:
            unavailable["creator"] += 1

        by_mp.setdefault(mp, []).append(
            {
                "observation_id": oid,
                "mint": mint,
                "observation_time_t": T,
                "mint_partition": mp,
                "detected_at_unix": row.get("detected_at_unix"),
                "events_store": EVENTS_STORE,
                "events_partition": e_part,
                "events_start_idx": e_lo,
                "events_end_idx": e_hi,
                "events_count": e_n,
                "events_available": e_ok,
                "curve_store": CURVE_STORE,
                "curve_partition": c_part,
                "curve_start_idx": c_lo,
                "curve_end_idx": c_hi,
                "curve_count": c_n,
                "curve_available": c_ok,
                "creator_store": CREATOR_STORE,
                "creator_partition": cr_part,
                "creator_start_idx": cr_lo,
                "creator_end_idx": cr_hi,
                "creator_count": cr_n,
                "creator_available": cr_ok,
                "bridge_version": BRIDGE_VERSION,
                "shared_sequences_once": True,
                "derived_from": "mint_plus_T_broad_entity_indexes",
            }
        )
        if progress is not None and (i + 1) % 2000 == 0:
            progress.emit(
                min(40, 10 + int(30 * (i + 1) / max(len(assignment_rows), 1))),
                message=f"bridge refs {i + 1}/{len(assignment_rows)}",
            )

    # Validate causal bounds on a sample of partitions with times loaded.
    _validate_bridge_causal(by_mp, times_cache=times_cache, broad_root=broad_root)

    for mp, rows in by_mp.items():
        part_dir = ensure_dir(out_root / mp)
        pq.write_table(
            pa.Table.from_pylist(rows),
            part_dir / "part.parquet",
            compression="zstd",
        )

    manifest = {
        "bridge_version": BRIDGE_VERSION,
        "generated_at": utc_now(),
        "rows": len(seen),
        "unique_observation_ids": len(seen),
        "partitions": len(by_mp),
        "modality_unavailable_counts": unavailable,
        "source": {
            "assignments": "observation_assignments_rich train+val",
            "events": EVENTS_STORE,
            "curve": CURVE_STORE,
            "creator": CREATOR_STORE,
        },
        "frozen_sources_unmodified": True,
        "note": (
            "Derived local cache keyed by frozen rich observation_id; "
            "causal end_idx via _end_idx_for_t on broad entity indexes at exact T."
        ),
    }
    write_json(out_root / "manifest.json", manifest)
    return {"manifest": manifest, "by_mp": by_mp, "out_root": str(out_root)}


def _validate_bridge_causal(
    by_mp: dict[str, list[dict[str, Any]]],
    *,
    times_cache: dict[str, list[float]],
    broad_root: Path,
) -> None:
    """Fail closed on negative / out-of-range / future-leaking indexes."""
    for _mp, rows in by_mp.items():
        for ref in rows:
            T = float(ref["observation_time_t"])
            for prefix, store in (
                ("events", EVENTS_STORE),
                ("curve", CURVE_STORE),
                ("creator", CREATOR_STORE),
            ):
                lo = int(ref[f"{prefix}_start_idx"])
                hi = int(ref[f"{prefix}_end_idx"])
                n = int(ref[f"{prefix}_count"])
                avail = bool(ref[f"{prefix}_available"])
                if lo < 0 or hi < 0 or hi < lo:
                    raise RichBroadBridgeError(
                        f"bad bridge indexes oid={ref['observation_id']} "
                        f"{prefix}=[{lo},{hi})"
                    )
                if n != max(0, hi - lo):
                    raise RichBroadBridgeError(
                        f"bridge count mismatch oid={ref['observation_id']} "
                        f"{prefix} count={n} hi-lo={hi - lo}"
                    )
                if not avail and n != 0:
                    raise RichBroadBridgeError(
                        f"unavailable modality with nonzero count "
                        f"oid={ref['observation_id']} {prefix}"
                    )
                if not avail:
                    continue
                part = ref.get(f"{prefix}_partition")
                if not part or hi == lo:
                    continue
                key = f"{store}:{part}"
                if key not in times_cache:
                    times_cache[key] = load_partition_times(broad_root / store, str(part))
                times = times_cache[key]
                if hi > len(times):
                    raise RichBroadBridgeError(
                        f"bridge end past store oid={ref['observation_id']} "
                        f"{prefix} end={hi} n={len(times)}"
                    )
                # All referenced rows must be <= T.
                window = times[lo:hi]
                if window and float(max(window)) > T + 1e-9:
                    raise RichBroadBridgeError(
                        f"future leak oid={ref['observation_id']} {prefix}: "
                        f"max_t={max(window)} T={T}"
                    )


def load_bridge_refs_for_partition(bridge_root: Path, mp: str) -> dict[str, dict[str, Any]]:
    path = Path(bridge_root) / mp / "part.parquet"
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for r in pq.read_table(path).to_pylist():
        oid = str(r.get("observation_id") or "")
        if oid:
            out[oid] = r
    return out


def assert_bridge_matches_assignments(
    *,
    assignment_rows: list[dict[str, Any]],
    bridge_by_mp: dict[str, list[dict[str, Any]]],
) -> None:
    """One bridge row per assignment; mint/T must match exactly."""
    bridge_rows = [r for rows in bridge_by_mp.values() for r in rows]
    if len(bridge_rows) != len(assignment_rows):
        raise RichBroadBridgeError(
            f"bridge rows={len(bridge_rows)} != assignment rows={len(assignment_rows)}"
        )
    by_oid = {str(r["observation_id"]): r for r in bridge_rows}
    if len(by_oid) != len(bridge_rows):
        raise RichBroadBridgeError("bridge observation_ids not unique")
    for row in assignment_rows:
        oid = str(row["observation_id"])
        ref = by_oid.get(oid)
        if ref is None:
            raise RichBroadBridgeError(f"missing bridge row for oid={oid}")
        if str(ref["mint"]) != str(row["mint"]):
            raise RichBroadBridgeError(f"bridge mint mismatch oid={oid}")
        if abs(float(ref["observation_time_t"]) - float(row["observation_time_t"])) > 1e-9:
            raise RichBroadBridgeError(f"bridge T mismatch oid={oid}")


def verify_no_future_mutation_affects_features(
    *,
    feat_before: dict[str, float],
    feat_after: dict[str, float],
    feature_names: list[str],
) -> None:
    """Bit-identical broad features after appending post-T modality rows."""
    for name in feature_names:
        a = float(feat_before[name])
        b = float(feat_after[name])
        if not (np.isnan(a) and np.isnan(b)) and a != b:
            raise AssertionError(f"feature mutated after post-T append: {name} {a} vs {b}")
