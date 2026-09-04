"""Point-in-time world-state reconstruction engine (no labels / no training)."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.io import ensure_dir, utc_now, write_json


def _to_unix(ts: Any) -> float | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return ts.timestamp()
    text = str(ts)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()
    except ValueError:
        return None


def _load_subset_observation_times(world_root: Path) -> dict[str, float]:
    path = world_root / "universe" / "rich_subset_mints.parquet"
    table = pq.read_table(path)
    out: dict[str, float] = {}
    mints = table.column("mint").to_pylist()
    tcol = (
        "observation_time_t"
        if "observation_time_t" in table.column_names
        else "detected_at"
    )
    for mint, ts in zip(mints, table.column(tcol).to_pylist(), strict=True):
        unix = _to_unix(ts)
        if mint and unix is not None:
            out[mint] = unix
    return out


def load_pit_indexes(
    world_root: Path, *, mint_filter: set[str] | None = None
) -> dict[str, Any]:
    """Load frozen artifacts once for efficient multi-mint PIT reconstruction."""
    own_level: dict[str, str] = {}
    own_path = world_root / "universe" / "ownership_evidence.parquet"
    if own_path.exists() and pq.ParquetFile(own_path).metadata.num_rows:
        ot = pq.read_table(
            own_path, columns=["mint", "ownership_evidence_level"]
        )
        for m, level in zip(
            ot.column("mint").to_pylist(),
            ot.column("ownership_evidence_level").to_pylist(),
            strict=True,
        ):
            if mint_filter is None or m in mint_filter:
                own_level[m] = level

    curve_by_mint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exec_path = world_root / "execution" / "execution_curve_states.parquet"
    if exec_path.exists() and pq.ParquetFile(exec_path).metadata.num_rows:
        et = pq.read_table(exec_path)
        if "mint" in et.column_names and "event_time" in et.column_names:
            for row in et.to_pylist():
                mint = row.get("mint")
                if mint_filter is not None and mint not in mint_filter:
                    continue
                curve_by_mint[mint].append(row)

    micro_by_mint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rpc_path = world_root / "microstructure" / "rpc_transactions.parquet"
    if rpc_path.exists() and pq.ParquetFile(rpc_path).metadata.num_rows:
        for row in pq.read_table(rpc_path).to_pylist():
            mint = row.get("mint")
            if not row.get("found"):
                continue
            if mint_filter is not None and mint not in mint_filter:
                continue
            micro_by_mint[mint].append(row)

    graph_by_mint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    wt_path = world_root / "graph" / "edges_wallet_token.parquet"
    if wt_path.exists() and pq.ParquetFile(wt_path).metadata.num_rows:
        cols = [
            c
            for c in ["src", "dst", "first_t", "last_t", "buys", "sells"]
            if c in pq.read_schema(wt_path).names
        ]
        for row in pq.read_table(wt_path, columns=cols).to_pylist():
            mint = row.get("dst")
            if mint_filter is not None and mint not in mint_filter:
                continue
            graph_by_mint[mint].append(row)

    rpc_own_by_mint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rpc_own = world_root / "graph" / "edges_ownership_rpc_token_balances.parquet"
    if rpc_own.exists() and pq.ParquetFile(rpc_own).metadata.num_rows:
        for row in pq.read_table(rpc_own).to_pylist():
            mint = row.get("mint")
            if mint_filter is not None and mint not in mint_filter:
                continue
            rpc_own_by_mint[mint].append(row)

    ledger_by_mint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name in (
        "token_transfers_unbiased_days.parquet",
        "token_transfers_rich.parquet",
    ):
        bq_tt = world_root / "bigquery" / name
        if not bq_tt.exists() or pq.ParquetFile(bq_tt).metadata.num_rows == 0:
            continue
        cols = [
            c
            for c in [
                "mint",
                "source",
                "destination",
                "block_timestamp",
                "value",
                "tx_signature",
            ]
            if c in pq.read_schema(bq_tt).names
        ]
        for row in pq.read_table(bq_tt, columns=cols).to_pylist():
            mint = row.get("mint")
            if mint_filter is not None and mint not in mint_filter:
                continue
            ledger_by_mint[mint].append(row)

    global_rows = []
    gpath = world_root / "global_context" / "daily_pump_network_context.parquet"
    if gpath.exists():
        global_rows = pq.read_table(gpath).to_pylist()

    return {
        "own_level": own_level,
        "curve_by_mint": curve_by_mint,
        "micro_by_mint": micro_by_mint,
        "graph_by_mint": graph_by_mint,
        "rpc_own_by_mint": rpc_own_by_mint,
        "ledger_by_mint": ledger_by_mint,
        "global_rows": global_rows,
    }


def reconstruct_token_state_at_t(
    *,
    mint: str,
    observation_time_t: float,
    indexes: dict[str, Any],
) -> dict[str, Any]:
    """Build modality-masked state using only events with time <= T."""
    own_level = indexes["own_level"].get(mint, "missing")

    curve_rows = []
    for row in indexes["curve_by_mint"].get(mint, []):
        ts = _to_unix(row.get("event_time"))
        if ts is not None and ts <= observation_time_t:
            curve_rows.append(row)

    micro_rows = []
    for row in indexes["micro_by_mint"].get(mint, []):
        ts = row.get("block_time")
        if ts is not None and float(ts) <= observation_time_t:
            micro_rows.append(
                {
                    "tx_signature": row.get("tx_signature"),
                    "block_time": ts,
                    "slot": row.get("slot"),
                    "success": row.get("success"),
                    "fee_lamports": row.get("fee_lamports"),
                }
            )

    graph_edges = []
    for row in indexes["graph_by_mint"].get(mint, []):
        first = _to_unix(row.get("first_t"))
        if first is not None and first <= observation_time_t:
            graph_edges.append(
                {
                    "wallet": row.get("src"),
                    "first_t": first,
                    "buys": row.get("buys"),
                    "sells": row.get("sells"),
                }
            )

    ownership_obs = []
    for row in indexes["rpc_own_by_mint"].get(mint, []):
        ts = row.get("block_time")
        if ts is not None and float(ts) <= observation_time_t:
            ownership_obs.append(
                {
                    "kind": "rpc_balance_observed",
                    "owner": row.get("owner"),
                    "delta_ui_amount": row.get("delta_ui_amount"),
                    "tx_signature": row.get("tx_signature"),
                    "block_time": ts,
                }
            )
    for row in indexes["ledger_by_mint"].get(mint, []):
        ts = _to_unix(row.get("block_timestamp"))
        if ts is not None and ts <= observation_time_t:
            ownership_obs.append(
                {
                    "kind": "ledger_transfer",
                    "src": row.get("source"),
                    "dst": row.get("destination"),
                    "value": row.get("value"),
                    "tx_signature": row.get("tx_signature"),
                    "block_time": ts,
                }
            )

    global_rows = []
    for row in indexes["global_rows"]:
        day = row.get("day")
        if not day:
            continue
        try:
            day_end = datetime.fromisoformat(str(day)).replace(
                hour=23, minute=59, second=59, tzinfo=UTC
            ).timestamp()
        except ValueError:
            continue
        if day_end <= observation_time_t:
            global_rows.append(row)

    masks = {
        "event_curve": bool(curve_rows),
        "graph": bool(graph_edges),
        "microstructure": bool(micro_rows),
        "ownership_any": bool(ownership_obs),
        "ownership_ledger": any(o.get("kind") == "ledger_transfer" for o in ownership_obs),
        "ownership_rpc_balance": any(
            o.get("kind") == "rpc_balance_observed" for o in ownership_obs
        ),
        "global_market": bool(global_rows),
        "execution": bool(curve_rows),
    }
    return {
        "mint": mint,
        "observation_time_t": observation_time_t,
        "ownership_evidence_level": own_level,
        "modality_masks": masks,
        "quality_masks": {
            "ownership_is_full_ledger": own_level == "ledger_full_or_reconstructable"
            and masks["ownership_ledger"],
            "ownership_is_sampled_rpc_only": own_level == "rpc_balance_observed",
        },
        "token_curve_state_events_n": len(curve_rows),
        "wallet_graph_edges_n": len(graph_edges),
        "microstructure_txs_n": len(micro_rows),
        "ownership_observations_n": len(ownership_obs),
        "global_context_days_n": len(global_rows),
        "generated_at": utc_now(),
    }


def materialize_pit_states_for_subset(
    *,
    world_root: Path,
    max_mints: int | None = 50,
    observation_times: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Materialize PIT states for a sample (or all) subset mints at their T."""
    observation_times = observation_times or _load_subset_observation_times(world_root)
    mints = sorted(observation_times)
    if max_mints is not None:
        mints = mints[:max_mints]
    indexes = load_pit_indexes(world_root, mint_filter=set(mints))
    states = [
        reconstruct_token_state_at_t(
            mint=mint,
            observation_time_t=observation_times[mint],
            indexes=indexes,
        )
        for mint in mints
    ]
    out_dir = ensure_dir(world_root / "pit")
    flat = []
    for s in states:
        flat.append(
            {
                "mint": s["mint"],
                "observation_time_t": s["observation_time_t"],
                "ownership_evidence_level": s["ownership_evidence_level"],
                "mask_event_curve": s["modality_masks"]["event_curve"],
                "mask_graph": s["modality_masks"]["graph"],
                "mask_microstructure": s["modality_masks"]["microstructure"],
                "mask_ownership_any": s["modality_masks"]["ownership_any"],
                "mask_ownership_ledger": s["modality_masks"]["ownership_ledger"],
                "mask_ownership_rpc_balance": s["modality_masks"]["ownership_rpc_balance"],
                "mask_global_market": s["modality_masks"]["global_market"],
                "mask_execution": s["modality_masks"]["execution"],
                "quality_ownership_is_full_ledger": s["quality_masks"][
                    "ownership_is_full_ledger"
                ],
                "quality_ownership_is_sampled_rpc_only": s["quality_masks"][
                    "ownership_is_sampled_rpc_only"
                ],
                "token_curve_state_events_n": s["token_curve_state_events_n"],
                "wallet_graph_edges_n": s["wallet_graph_edges_n"],
                "microstructure_txs_n": s["microstructure_txs_n"],
                "ownership_observations_n": s["ownership_observations_n"],
                "global_context_days_n": s["global_context_days_n"],
            }
        )
    path = out_dir / "pit_states_sample.parquet"
    if flat:
        pq.write_table(pa.Table.from_pylist(flat), path)
    else:
        pq.write_table(pa.table({"mint": pa.array([], pa.string())}), path)
    manifest = {
        "generated_at": utc_now(),
        "mints_materialized": len(flat),
        "path": str(path),
        "policy": "input_only_observable_at_or_before_T",
        "labels": "not_generated",
    }
    write_json(out_dir / "pit_manifest.json", manifest)
    return manifest


def assert_no_post_t_leakage(
    *,
    mint: str,
    observation_time_t: float,
    indexes: dict[str, Any],
) -> dict[str, Any]:
    """Leakage test: post-T synthetic events must not alter state@T."""
    before = reconstruct_token_state_at_t(
        mint=mint,
        observation_time_t=observation_time_t,
        indexes=indexes,
    )
    # Inject post-T curve + micro + ledger + rpc ownership into indexes copy.
    import copy

    idx = copy.deepcopy(indexes)
    post_t = observation_time_t + 10_000
    idx["curve_by_mint"].setdefault(mint, []).append(
        {
            "mint": mint,
            "event_time": datetime.fromtimestamp(post_t, tz=UTC),
            "price_sol": 999.0,
            "leakage_marker": True,
        }
    )
    idx["micro_by_mint"].setdefault(mint, []).append(
        {
            "mint": mint,
            "found": True,
            "block_time": post_t,
            "tx_signature": "leak_sig",
            "slot": 0,
            "success": True,
            "fee_lamports": 1,
        }
    )
    idx["ledger_by_mint"].setdefault(mint, []).append(
        {
            "mint": mint,
            "source": "leak_src",
            "destination": "leak_dst",
            "block_timestamp": datetime.fromtimestamp(post_t, tz=UTC),
            "value": 1,
            "tx_signature": "leak_tt",
        }
    )
    idx["rpc_own_by_mint"].setdefault(mint, []).append(
        {
            "mint": mint,
            "owner": "leak_owner",
            "delta_ui_amount": 1.0,
            "tx_signature": "leak_own",
            "block_time": post_t,
        }
    )
    idx["graph_by_mint"].setdefault(mint, []).append(
        {
            "src": "leak_wallet",
            "dst": mint,
            "first_t": datetime.fromtimestamp(post_t, tz=UTC),
            "buys": 1,
            "sells": 0,
        }
    )
    after = reconstruct_token_state_at_t(
        mint=mint,
        observation_time_t=observation_time_t,
        indexes=idx,
    )
    keys = [
        "token_curve_state_events_n",
        "wallet_graph_edges_n",
        "microstructure_txs_n",
        "ownership_observations_n",
        "global_context_days_n",
        "ownership_evidence_level",
    ]
    unchanged = {k: before.get(k) == after.get(k) for k in keys}
    return {
        "mint": mint,
        "observation_time_t": observation_time_t,
        "state_unchanged": all(unchanged.values()),
        "field_equality": unchanged,
        "passed": all(unchanged.values()),
    }
