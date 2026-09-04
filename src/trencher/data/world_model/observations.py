"""Canonical causal observation index for the unbiased multimodal population."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.freeze import FREEZE_ID
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.pit import (
    PIT_AGE_OFFSETS_SEC,
    PIT_ENGINE_VERSION,
    _to_unix,
    assert_no_post_t_leakage,
    ensure_atomic_trades_index,
    ensure_global_context_index,
    find_forbidden_future_fields,
    load_launch_times,
    load_pit_indexes,
    model_state_digest,
    reconstruct_token_state_at_t,
)

OBSERVATION_DATASET_VERSION = "observation_dataset_v1"
EVENT_TRADE_COUNT_CAP = 512
EVENT_TRADE_COUNTS = tuple(
    2**i for i in range(0, 10) if 2**i <= EVENT_TRADE_COUNT_CAP
)  # 1..512


def _observation_id(
    *,
    mint: str,
    observation_time_t: float,
    trigger: str,
    trigger_value: str | int,
) -> str:
    raw = f"{mint}|{observation_time_t:.6f}|{trigger}|{trigger_value}"
    return sha256(raw.encode()).hexdigest()[:24]


def select_observation_points(
    *,
    mint: str,
    launch_t: float,
    trade_event_times: list[float],
    fixed_ages: tuple[tuple[str, int], ...] = PIT_AGE_OFFSETS_SEC,
    event_counts: tuple[int, ...] = EVENT_TRADE_COUNTS,
) -> list[dict[str, Any]]:
    """Causal observation times: fixed ages + power-of-two cumulative trade counts.

    Selection uses only information known by each candidate event (trade times
    at/before the observation). No future peak/graduation/outcome features.
    """
    candidates: list[dict[str, Any]] = []
    for age_name, offset in fixed_ages:
        t = launch_t + float(offset)
        candidates.append(
            {
                "mint": mint,
                "observation_time_t": t,
                "token_age_seconds": float(offset),
                "observation_trigger": "fixed_age",
                "trigger_value": age_name,
                "trigger_numeric": offset,
            }
        )

    sorted_times = sorted(t for t in trade_event_times if t is not None)
    next_targets = list(event_counts)
    ti = 0
    for n, et in enumerate(sorted_times, start=1):
        while ti < len(next_targets) and n >= next_targets[ti]:
            target = next_targets[ti]
            candidates.append(
                {
                    "mint": mint,
                    "observation_time_t": float(et),
                    "token_age_seconds": float(et) - launch_t,
                    "observation_trigger": "event_count",
                    "trigger_value": f"trades_{target}",
                    "trigger_numeric": target,
                }
            )
            ti += 1
        if ti >= len(next_targets):
            break

    # Deduplicate coincident times; prefer fixed_age over event_count.
    by_time: dict[int, dict[str, Any]] = {}
    preferred_rank = {"fixed_age": 0, "event_count": 1}
    for cand in candidates:
        key = int(round(cand["observation_time_t"] * 1_000_000))
        prev = by_time.get(key)
        if prev is None or preferred_rank.get(
            cand["observation_trigger"], 9
        ) < preferred_rank.get(prev["observation_trigger"], 9):
            by_time[key] = cand

    rows = sorted(
        by_time.values(), key=lambda r: (r["observation_time_t"], r["trigger_value"])
    )
    for row in rows:
        row["observation_id"] = _observation_id(
            mint=mint,
            observation_time_t=row["observation_time_t"],
            trigger=row["observation_trigger"],
            trigger_value=row["trigger_value"],
        )
        row["source_freeze_version"] = FREEZE_ID
        row["pit_engine_version"] = PIT_ENGINE_VERSION
        row["observation_dataset_version"] = OBSERVATION_DATASET_VERSION
    return rows


def build_observation_dataset_v1(
    *,
    world_root: Path,
    manifest_dir: Path,
    pump_root: Path | None = None,
    max_mints: int | None = None,
    leakage_sample_n: int = 8,
) -> dict[str, Any]:
    """Materialize observation index + PIT-safe model states for the subset."""
    ensure_atomic_trades_index(world_root=world_root, pump_root=pump_root, force=False)
    ensure_global_context_index(world_root=world_root, pump_root=pump_root, force=False)

    launches = load_launch_times(world_root)
    subset_path = world_root / "universe" / "rich_subset_mints.parquet"
    subset = pq.read_table(subset_path)
    mints = [m for m in subset.column("mint").to_pylist() if m in launches]
    if max_mints is not None:
        mints = mints[:max_mints]

    launch_day_by_mint: dict[str, str | None] = {}
    regime_by_mint: dict[str, str | None] = {}
    if "launch_day" in subset.column_names:
        for m, d in zip(
            subset.column("mint").to_pylist(),
            subset.column("launch_day").to_pylist(),
            strict=True,
        ):
            launch_day_by_mint[m] = d
    if "launch_regime" in subset.column_names:
        for m, r in zip(
            subset.column("mint").to_pylist(),
            subset.column("launch_regime").to_pylist(),
            strict=True,
        ):
            regime_by_mint[m] = r

    indexes = load_pit_indexes(
        world_root, mint_filter=set(mints), pump_root=pump_root
    )

    out_dir = ensure_dir(world_root / "observations" / OBSERVATION_DATASET_VERSION)
    index_path = out_dir / "observation_index_v1.parquet"
    states_path = out_dir / "observation_model_states_v1.parquet"

    index_batch: list[dict[str, Any]] = []
    state_batch: list[dict[str, Any]] = []
    index_writer: pq.ParquetWriter | None = None
    state_writer: pq.ParquetWriter | None = None
    leakage_candidates: list[dict[str, Any]] = []

    trigger_counts: Counter[str] = Counter()
    age_counts: Counter[str] = Counter()
    obs_per_token: list[int] = []
    modality_by_age: dict[str, Counter[str]] = defaultdict(Counter)
    ownership_by_age: dict[str, Counter[str]] = defaultdict(Counter)
    day_counts: Counter[str] = Counter()
    regime_counts: Counter[str] = Counter()
    duplicate_time_collisions = 0
    observations_n = 0

    def _flush() -> None:
        nonlocal index_writer, state_writer, index_batch, state_batch
        if index_batch:
            table = pa.Table.from_pylist(index_batch)
            if index_writer is None:
                index_writer = pq.ParquetWriter(index_path, table.schema)
            index_writer.write_table(table)
            index_batch = []
        if state_batch:
            table = pa.Table.from_pylist(state_batch)
            if state_writer is None:
                state_writer = pq.ParquetWriter(states_path, table.schema)
            state_writer.write_table(table)
            state_batch = []

    try:
        for mint in mints:
            launch_t = launches[mint]
            trade_times: list[float] = []
            for row in indexes["trades_by_mint"].get(mint, []):
                u = _to_unix(row.get("event_time"))
                if u is not None:
                    trade_times.append(u)
            if not trade_times:
                for row in indexes["curve_by_mint"].get(mint, []):
                    u = _to_unix(row.get("event_time"))
                    if u is not None:
                        trade_times.append(u)

            before_dedupe_n = len(PIT_AGE_OFFSETS_SEC) + sum(
                1 for c in EVENT_TRADE_COUNTS if c <= len(trade_times)
            )
            points = select_observation_points(
                mint=mint, launch_t=launch_t, trade_event_times=trade_times
            )
            if before_dedupe_n > len(points):
                duplicate_time_collisions += before_dedupe_n - len(points)
            obs_per_token.append(len(points))

            day = launch_day_by_mint.get(mint)
            regime = regime_by_mint.get(mint)
            if day:
                day_counts[str(day)] += 1
            if regime:
                regime_counts[str(regime)] += 1

            for point in points:
                t = float(point["observation_time_t"])
                result = reconstruct_token_state_at_t(
                    mint=mint, observation_time_t=t, indexes=indexes
                )
                state = result["model_state"]
                forbidden = find_forbidden_future_fields(state)
                if forbidden:
                    raise RuntimeError(
                        f"forbidden fields in observation model_state: {forbidden}"
                    )
                digest = model_state_digest(state)
                oid = point["observation_id"]
                trigger = point["observation_trigger"]
                trigger_counts[trigger] += 1
                age_key = (
                    str(point["trigger_value"])
                    if trigger == "fixed_age"
                    else f"event:{point['trigger_value']}"
                )
                age_counts[age_key] += 1
                for mk, mv in (state.get("modality_masks") or {}).items():
                    if mv:
                        modality_by_age[age_key][mk] += 1
                ownership_by_age[age_key][state["ownership_level_at_t"]] += 1

                index_row = {
                    "mint": mint,
                    "observation_id": oid,
                    "observation_time_t": t,
                    "token_age_seconds": point["token_age_seconds"],
                    "observation_trigger": trigger,
                    "trigger_value": str(point["trigger_value"]),
                    "trigger_numeric": point["trigger_numeric"],
                    "source_freeze_version": FREEZE_ID,
                    "pit_engine_version": PIT_ENGINE_VERSION,
                    "observation_dataset_version": OBSERVATION_DATASET_VERSION,
                    "launch_day": day,
                    "launch_regime": regime,
                    "model_state_digest": digest,
                    "ownership_level_at_t": state["ownership_level_at_t"],
                    "holder_state_reconstructable_at_t": state[
                        "holder_state_reconstructable_at_t"
                    ],
                    "mask_event_history": state["modality_masks"]["event_history"],
                    "mask_wallet_token_dynamic": state["modality_masks"][
                        "wallet_token_dynamic"
                    ],
                    "mask_microstructure": state["modality_masks"]["microstructure"],
                    "mask_ownership_any": state["modality_masks"]["ownership_any"],
                    "mask_global_market": state["modality_masks"]["global_market"],
                }
                index_batch.append(index_row)
                state_batch.append(
                    {
                        "mint": mint,
                        "observation_id": oid,
                        "observation_time_t": t,
                        "model_state_json": json.dumps(
                            state, sort_keys=True, default=str, separators=(",", ":")
                        ),
                        "model_state_digest": digest,
                        "provenance_json": json.dumps(
                            result["provenance"],
                            sort_keys=True,
                            default=str,
                            separators=(",", ":"),
                        ),
                    }
                )
                observations_n += 1
                if len(leakage_candidates) < leakage_sample_n:
                    leakage_candidates.append(index_row)
                if len(index_batch) >= 500:
                    _flush()
        _flush()
    finally:
        if index_writer is not None:
            index_writer.close()
        if state_writer is not None:
            state_writer.close()

    if observations_n == 0:
        pq.write_table(
            pa.table({"observation_id": pa.array([], type=pa.string())}), index_path
        )
        pq.write_table(
            pa.table({"observation_id": pa.array([], type=pa.string())}), states_path
        )

    leakage_results = []
    for row in leakage_candidates:
        leak = assert_no_post_t_leakage(
            mint=row["mint"],
            observation_time_t=float(row["observation_time_t"]),
            indexes=indexes,
        )
        leakage_results.append(
            {
                "observation_id": row["observation_id"],
                "mint": row["mint"],
                "observation_time_t": row["observation_time_t"],
                "passed": leak["passed"],
                "cases": leak["cases"],
            }
        )

    obs_per_token_sorted = sorted(obs_per_token)

    def _pct(p: float) -> float | None:
        if not obs_per_token_sorted:
            return None
        idx = min(
            len(obs_per_token_sorted) - 1,
            max(0, int(round(p * (len(obs_per_token_sorted) - 1)))),
        )
        return float(obs_per_token_sorted[idx])

    disk_bytes = sum(p.stat().st_size for p in out_dir.glob("*.parquet") if p.is_file())
    audit = {
        "audit_id": OBSERVATION_DATASET_VERSION,
        "generated_at": utc_now(),
        "acquisition_freeze_id": FREEZE_ID,
        "pit_engine_version": PIT_ENGINE_VERSION,
        "tokens_n": len(mints),
        "observations_n": observations_n,
        "observations_by_trigger": dict(trigger_counts),
        "observations_by_age_or_event": dict(age_counts),
        "observations_per_token": {
            "min": obs_per_token_sorted[0] if obs_per_token_sorted else 0,
            "p50": _pct(0.5),
            "p90": _pct(0.9),
            "p99": _pct(0.99),
            "max": obs_per_token_sorted[-1] if obs_per_token_sorted else 0,
            "mean": (sum(obs_per_token) / len(obs_per_token) if obs_per_token else 0.0),
        },
        "modality_availability_by_age": {
            k: dict(v) for k, v in modality_by_age.items()
        },
        "ownership_evidence_by_age": {k: dict(v) for k, v in ownership_by_age.items()},
        "date_distribution_tokens": dict(day_counts),
        "regime_distribution_tokens": dict(regime_counts),
        "duplicate_time_collisions_removed": duplicate_time_collisions,
        "duplicate_rate": (
            duplicate_time_collisions
            / max(1, observations_n + duplicate_time_collisions)
        ),
        "leakage_tests": {
            "n_cases": len(leakage_results),
            "all_passed": all(r["passed"] for r in leakage_results),
            "results": leakage_results,
        },
        "estimated_disk_bytes": disk_bytes,
        "estimated_disk_mb": round(disk_bytes / (1024 * 1024), 3),
        "artifacts": {
            "observation_index_v1": str(index_path),
            "observation_model_states_v1": str(states_path),
        },
        "fixed_ages_sec": {k: v for k, v in PIT_AGE_OFFSETS_SEC},
        "event_trade_counts": list(EVENT_TRADE_COUNTS),
        "event_trade_count_cap": EVENT_TRADE_COUNT_CAP,
        "labels": "not_generated",
        "targets": "not_generated",
        "chronological_splits": "not_generated",
        "training": "not_started",
        "known_limitations": [
            "Model states store JSON payloads; wallet graphs can be large for "
            "high-activity mints.",
            "Event-driven points use atomic Pump trades (curve fallback only if "
            "atomic trades absent).",
            "Global trade context lags by <1 completed second by design.",
            "Ownership reconstructable remains rare under partition-coverage rules.",
            "No prediction labels or outcome fields are attached.",
        ],
    }
    write_json(out_dir / "OBSERVATION_DATASET_V1_AUDIT.json", audit)
    ensure_dir(manifest_dir)
    write_json(manifest_dir / "OBSERVATION_DATASET_V1_AUDIT.json", audit)
    (manifest_dir / "OBSERVATION_DATASET_V1_AUDIT.md").write_text(
        render_observation_dataset_audit_md(audit), encoding="utf-8"
    )
    write_json(
        out_dir / "observation_dataset_manifest.json",
        {
            "generated_at": utc_now(),
            "version": OBSERVATION_DATASET_VERSION,
            "tokens_n": len(mints),
            "observations_n": observations_n,
            "index_path": str(index_path),
            "states_path": str(states_path),
        },
    )
    return audit


def render_observation_dataset_audit_md(audit: dict[str, Any]) -> str:
    lines = [
        f"# Observation Dataset v1 (`{audit['audit_id']}`)",
        "",
        f"Generated at: `{audit['generated_at']}`",
        f"Freeze: `{audit['acquisition_freeze_id']}`",
        f"PIT engine: `{audit['pit_engine_version']}`",
        "",
        f"- tokens: **{audit['tokens_n']}**",
        f"- observations: **{audit['observations_n']}**",
        f"- by trigger: `{audit.get('observations_by_trigger')}`",
        f"- obs/token: `{audit.get('observations_per_token')}`",
        f"- duplicate collisions removed: "
        f"{audit.get('duplicate_time_collisions_removed')}",
        f"- disk MB: {audit.get('estimated_disk_mb')}",
        f"- leakage all_passed: "
        f"**{(audit.get('leakage_tests') or {}).get('all_passed')}**",
        "",
        "## Known limitations",
        "",
    ]
    for item in audit.get("known_limitations") or []:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
