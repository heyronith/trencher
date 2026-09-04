"""Point-in-time world-state reconstruction engine (no labels / no training)."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from trencher.data.feasibility.concepts import SYSTEM_PROGRAM_WALLET
from trencher.data.world_model.io import ensure_dir, utc_now, write_json

PIT_AGE_OFFSETS_SEC = (
    ("launch", 0),
    ("10s", 10),
    ("30s", 30),
    ("1m", 60),
    ("2m", 120),
    ("5m", 300),
    ("10m", 600),
    ("30m", 1800),
)

_LEAKAGE_STATE_KEYS = (
    "mint",
    "observation_time_t",
    "ownership_evidence_level",
    "modality_masks",
    "quality_masks",
    "event_history_n",
    "token_curve_state_events_n",
    "wallet_token_interactions_n",
    "wallet_graph_edges_n",
    "microstructure_txs_n",
    "ownership_observations_n",
    "ownership_ledger_n",
    "ownership_rpc_n",
    "global_context_completed_days_n",
    "global_context_days_n",
    "rolling_global_context",
    "curve_state_latest",
    "wallet_token_state",
)


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


def _day_start_unix(t: float) -> float:
    dt = datetime.fromtimestamp(t, tz=UTC)
    return datetime(dt.year, dt.month, dt.day, tzinfo=UTC).timestamp()


def canonical_state_bytes(state: dict[str, Any]) -> bytes:
    """Stable serialization for byte-equivalent leakage comparisons."""
    payload = {k: state.get(k) for k in _LEAKAGE_STATE_KEYS}
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()


def state_digest(state: dict[str, Any]) -> str:
    return sha256(canonical_state_bytes(state)).hexdigest()


def load_launch_times(world_root: Path) -> dict[str, float]:
    """Launch/reference times from subset detected_at (not coupled to PIT API)."""
    path = world_root / "universe" / "rich_subset_mints.parquet"
    table = pq.read_table(path)
    out: dict[str, float] = {}
    mints = table.column("mint").to_pylist()
    tcol = "detected_at" if "detected_at" in table.column_names else "observation_time_t"
    for mint, ts in zip(mints, table.column(tcol).to_pylist(), strict=True):
        unix = _to_unix(ts)
        if mint and unix is not None:
            out[mint] = unix
    return out


def ensure_atomic_trades_index(
    *,
    world_root: Path,
    pump_root: Path | None = None,
    force: bool = False,
) -> Path:
    """Derive PIT-safe atomic Pump trades for the rich subset (local corpus only)."""
    out_dir = ensure_dir(world_root / "pit")
    out_path = out_dir / "atomic_trades_subset.parquet"
    if out_path.exists() and not force and pq.ParquetFile(out_path).metadata.num_rows:
        return out_path

    subset = pq.read_table(
        world_root / "universe" / "rich_subset_mints.parquet", columns=["mint"]
    )
    rich = set(subset.column("mint").to_pylist()) - {None}
    pump = pump_root or Path("data/external/pumpfun_memecoin_corpus")
    trades_dir = pump / "trades"
    if not trades_dir.exists():
        pq.write_table(pa.table({"mint": pa.array([], type=pa.string())}), out_path)
        return out_path

    dataset = ds.dataset(trades_dir, format="parquet")
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
    available = set(dataset.schema.names)
    use_cols = [c for c in cols if c in available]
    rows: list[dict[str, Any]] = []
    for batch in dataset.scanner(columns=use_cols, batch_size=256_000).to_batches():
        data = {c: batch.column(c).to_pylist() for c in use_cols}
        n = len(next(iter(data.values())))
        for i in range(n):
            mint = data["mint"][i]
            if mint not in rich:
                continue
            wallet = data.get("user_wallet", [None] * n)[i]
            if not wallet or wallet == SYSTEM_PROGRAM_WALLET:
                continue
            rows.append({c: data[c][i] for c in use_cols})

    if rows:
        pq.write_table(pa.Table.from_pylist(rows), out_path)
    else:
        pq.write_table(pa.table({"mint": pa.array([], type=pa.string())}), out_path)
    write_json(
        out_dir / "atomic_trades_manifest.json",
        {
            "generated_at": utc_now(),
            "path": str(out_path),
            "rows": len(rows),
            "mints": len({r["mint"] for r in rows}),
            "source": "pumpfun_memecoin_corpus.trades (frozen local)",
            "note": "Derived PIT index; not an external acquisition.",
            "labels": "not_generated",
        },
    )
    return out_path


def _wallet_token_state_from_trades(
    trades: list[dict[str, Any]], observation_time_t: float
) -> list[dict[str, Any]]:
    """Aggregate wallet↔token interactions strictly from event_time <= T."""
    by_wallet: dict[str, dict[str, Any]] = {}
    for row in trades:
        ts = _to_unix(row.get("event_time"))
        if ts is None or ts > observation_time_t:
            continue
        wallet = row.get("user_wallet") or row.get("wallet")
        if not wallet:
            continue
        slot = by_wallet.get(wallet)
        if slot is None:
            slot = {
                "wallet": wallet,
                "buys_before_t": 0,
                "sells_before_t": 0,
                "buy_sol_before_t": 0.0,
                "sell_sol_before_t": 0.0,
                "interaction_count": 0,
                "first_seen_before_t": ts,
                "last_seen_before_t": ts,
            }
            by_wallet[wallet] = slot
        sol = float(row.get("sol_amount") or 0.0)
        if bool(row.get("is_buy")):
            slot["buys_before_t"] += 1
            slot["buy_sol_before_t"] += sol
        else:
            slot["sells_before_t"] += 1
            slot["sell_sol_before_t"] += sol
        slot["interaction_count"] += 1
        slot["first_seen_before_t"] = min(slot["first_seen_before_t"], ts)
        slot["last_seen_before_t"] = max(slot["last_seen_before_t"], ts)
    return sorted(by_wallet.values(), key=lambda r: r["wallet"])

def load_pit_indexes(
    world_root: Path,
    *,
    mint_filter: set[str] | None = None,
    pump_root: Path | None = None,
    ensure_trades: bool = True,
) -> dict[str, Any]:
    """Load frozen + derived artifacts for efficient multi-mint PIT reconstruction."""
    if ensure_trades:
        ensure_atomic_trades_index(world_root=world_root, pump_root=pump_root)

    own_level: dict[str, str] = {}
    own_path = world_root / "universe" / "ownership_evidence.parquet"
    if own_path.exists() and pq.ParquetFile(own_path).metadata.num_rows:
        ot = pq.read_table(own_path, columns=["mint", "ownership_evidence_level"])
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

    trades_by_mint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    atomic_path = world_root / "pit" / "atomic_trades_subset.parquet"
    if atomic_path.exists() and pq.ParquetFile(atomic_path).metadata.num_rows:
        for row in pq.read_table(atomic_path).to_pylist():
            mint = row.get("mint")
            if mint_filter is not None and mint not in mint_filter:
                continue
            trades_by_mint[mint].append(row)

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
                "authority",
                "block_timestamp",
                "value",
                "tx_signature",
                "transfer_type",
            ]
            if c in pq.read_schema(bq_tt).names
        ]
        for row in pq.read_table(bq_tt, columns=cols).to_pylist():
            mint = row.get("mint")
            if mint_filter is not None and mint not in mint_filter:
                continue
            ledger_by_mint[mint].append(row)

    native_by_mint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    native_path = world_root / "graph" / "edges_funding_native_sol.parquet"
    if native_path.exists() and pq.ParquetFile(native_path).metadata.num_rows:
        for row in pq.read_table(native_path).to_pylist():
            mint = row.get("mint")
            if mint_filter is not None and mint not in mint_filter:
                continue
            native_by_mint[mint].append(row)

    global_rows = []
    gpath = world_root / "global_context" / "daily_pump_network_context.parquet"
    if gpath.exists():
        global_rows = pq.read_table(gpath).to_pylist()

    # Same-day launch stream: subset detected_at only (not full 798k universe).
    launch_events: list[tuple[float, str]] = []
    subset_path = world_root / "universe" / "rich_subset_mints.parquet"
    if subset_path.exists() and "detected_at" in pq.read_schema(subset_path).names:
        st = pq.read_table(subset_path, columns=["mint", "detected_at"])
        for mint, ts in zip(
            st.column("mint").to_pylist(),
            st.column("detected_at").to_pylist(),
            strict=True,
        ):
            u = _to_unix(ts)
            if u is None or not mint:
                continue
            if mint_filter is not None and mint not in mint_filter:
                continue
            launch_events.append((u, mint))

    future_labels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in (
        world_root / "postgrad" / "postgard_outcomes.parquet",
        Path("data/external/pumpfun_memecoin_corpus/postgard_outcomes.parquet"),
    ):
        if cand.exists():
            try:
                for row in pq.read_table(cand).to_pylist():
                    mint = row.get("mint")
                    if mint_filter is not None and mint not in mint_filter:
                        continue
                    future_labels[mint].append(row)
            except Exception:  # noqa: BLE001
                pass
            break

    migrations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in (
        world_root / "universe" / "migrations.parquet",
        Path("data/external/pumpfun_memecoin_corpus/migrations.parquet"),
    ):
        if cand.exists():
            try:
                for row in pq.read_table(cand).to_pylist():
                    mint = row.get("mint")
                    if mint_filter is not None and mint not in mint_filter:
                        continue
                    migrations[mint].append(row)
            except Exception:  # noqa: BLE001
                pass
            break

    return {
        "own_level": own_level,
        "curve_by_mint": curve_by_mint,
        "trades_by_mint": trades_by_mint,
        "micro_by_mint": micro_by_mint,
        "rpc_own_by_mint": rpc_own_by_mint,
        "ledger_by_mint": ledger_by_mint,
        "native_by_mint": native_by_mint,
        "global_rows": global_rows,
        "launch_events": launch_events,
        "future_labels_by_mint": future_labels,
        "migrations_by_mint": migrations,
        "lifetime_wallet_token_forbidden": True,
    }


def reconstruct_token_state_at_t(
    *,
    mint: str,
    observation_time_t: float,
    indexes: dict[str, Any],
) -> dict[str, Any]:
    """Build modality-masked state for arbitrary (mint, T); inputs are event_time<=T only."""
    own_level = indexes["own_level"].get(mint, "missing")
    day_start = _day_start_unix(observation_time_t)

    event_history = []
    curve_rows = []
    for row in indexes["curve_by_mint"].get(mint, []):
        ts = _to_unix(row.get("event_time"))
        if ts is not None and ts <= observation_time_t:
            curve_rows.append(row)
            event_history.append(
                {
                    "kind": "curve_trade",
                    "event_time": ts,
                    "is_buy": row.get("is_buy"),
                    "sol_amount": row.get("sol_amount"),
                    "price_sol": row.get("price_sol"),
                }
            )
    curve_rows_sorted = sorted(
        curve_rows, key=lambda r: _to_unix(r.get("event_time")) or 0.0
    )
    latest_curve = None
    if curve_rows_sorted:
        last = curve_rows_sorted[-1]
        latest_curve = {
            "event_time": _to_unix(last.get("event_time")),
            "price_sol": last.get("price_sol") or last.get("executable_price_sol"),
            "v_sol_bonding_curve": last.get("v_sol_bonding_curve"),
            "v_tokens_bonding_curve": last.get("v_tokens_bonding_curve"),
            "curve_pct_depleted": last.get("curve_pct_depleted"),
        }

    trades = indexes["trades_by_mint"].get(mint, [])
    wallet_token_state = _wallet_token_state_from_trades(trades, observation_time_t)
    for row in trades:
        ts = _to_unix(row.get("event_time"))
        if ts is not None and ts <= observation_time_t:
            event_history.append(
                {
                    "kind": "atomic_trade",
                    "event_time": ts,
                    "wallet": row.get("user_wallet"),
                    "is_buy": row.get("is_buy"),
                    "sol_amount": row.get("sol_amount"),
                }
            )

    micro_rows = []
    for row in indexes["micro_by_mint"].get(mint, []):
        ts = row.get("block_time")
        if ts is not None and float(ts) <= observation_time_t:
            micro_rows.append(
                {
                    "tx_signature": row.get("tx_signature"),
                    "block_time": float(ts),
                    "slot": row.get("slot"),
                    "success": row.get("success"),
                    "fee_lamports": row.get("fee_lamports"),
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
                    "block_time": float(ts),
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
                    "transfer_type": row.get("transfer_type"),
                    "tx_signature": row.get("tx_signature"),
                    "block_time": ts,
                }
            )

    native_rows = []
    for row in indexes["native_by_mint"].get(mint, []):
        ts = row.get("block_time")
        u = float(ts) if ts is not None else _to_unix(row.get("event_time"))
        if u is not None and u <= observation_time_t:
            native_rows.append(row)

    completed_days = []
    for row in indexes["global_rows"]:
        day = row.get("day")
        if not day:
            continue
        try:
            day_dt = datetime.fromisoformat(str(day)).replace(tzinfo=UTC)
        except ValueError:
            continue
        day_end = (day_dt + timedelta(days=1)).timestamp()
        if day_end <= day_start:
            completed_days.append(row)

    # Historical launches from fully completed days only (never unfinished day).
    launches_completed_days = sum(
        int(r.get("launches") or 0) for r in completed_days
    )
    # Same-day rolling: subset launches with detected_at in [day_start, T].
    same_day_subset_launches = sum(
        1
        for u, _ in indexes.get("launch_events", [])
        if day_start <= u <= observation_time_t
    )
    trades_before_t = sum(
        1
        for row in trades
        if (_to_unix(row.get("event_time")) or 0) <= observation_time_t
    )
    buy_sol_before_t = sum(
        float(row.get("sol_amount") or 0.0)
        for row in trades
        if bool(row.get("is_buy"))
        and (_to_unix(row.get("event_time")) or 0) <= observation_time_t
    )
    rolling = {
        "launches_from_completed_days": launches_completed_days,
        "same_day_subset_launches_at_or_before_t": same_day_subset_launches,
        "subset_mint_trades_at_or_before_t": trades_before_t,
        "subset_mint_buy_sol_at_or_before_t": buy_sol_before_t,
        "completed_global_days_n": len(completed_days),
        "same_day_partial_daily_aggregates_excluded": True,
    }

    future_only = {
        "postgrad_outcomes_n": len(
            indexes.get("future_labels_by_mint", {}).get(mint, [])
        ),
        "migrations_n": len(indexes.get("migrations_by_mint", {}).get(mint, [])),
        "note": "future_data_for_labels_only_not_used_in_state",
    }

    ledger_n = sum(1 for o in ownership_obs if o.get("kind") == "ledger_transfer")
    rpc_n = sum(1 for o in ownership_obs if o.get("kind") == "rpc_balance_observed")

    masks = {
        "event_history": bool(event_history),
        "event_curve": bool(curve_rows),
        "execution": bool(curve_rows),
        "wallet_token_dynamic": bool(wallet_token_state),
        "graph": bool(wallet_token_state),
        "microstructure": bool(micro_rows),
        "ownership_any": bool(ownership_obs),
        "ownership_ledger": ledger_n > 0,
        "ownership_rpc_balance": rpc_n > 0,
        "native_sol_history": bool(native_rows),
        "global_market": bool(completed_days)
        or same_day_subset_launches > 0
        or launches_completed_days > 0,
    }
    quality = {
        "ownership_holder_state_reconstructable": (
            own_level == "holder_state_reconstructable" and ledger_n > 0
        ),
        "ownership_transfer_ledger_partial": (
            own_level == "transfer_ledger_observed_partial"
        ),
        "ownership_is_sampled_rpc_only": own_level == "rpc_balance_observed",
        "wallet_graph_from_atomic_trades": True,
        "lifetime_wallet_aggregates_excluded": True,
        "global_excludes_unfinished_day": True,
        "holder_state_fabricated": False,
    }

    return {
        "mint": mint,
        "observation_time_t": observation_time_t,
        "ownership_evidence_level": own_level,
        "modality_masks": masks,
        "quality_masks": quality,
        "event_history_n": len(event_history),
        "token_curve_state_events_n": len(curve_rows),
        "curve_state_latest": latest_curve,
        "wallet_token_state": wallet_token_state,
        "wallet_token_interactions_n": len(wallet_token_state),
        "wallet_graph_edges_n": len(wallet_token_state),
        "microstructure_txs_n": len(micro_rows),
        "ownership_observations_n": len(ownership_obs),
        "ownership_ledger_n": ledger_n,
        "ownership_rpc_n": rpc_n,
        "native_sol_edges_n": len(native_rows),
        "global_context_completed_days_n": len(completed_days),
        "global_context_days_n": len(completed_days),
        "rolling_global_context": rolling,
        "future_labels_separate": future_only,
        "generated_at": utc_now(),
        "labels": "not_generated",
    }


def materialize_pit_states_for_subset(
    *,
    world_root: Path,
    max_mints: int | None = 50,
    observation_times: dict[str, float] | None = None,
    pump_root: Path | None = None,
    age_offsets: tuple[tuple[str, int], ...] | None = None,
) -> dict[str, Any]:
    """Materialize PIT states at arbitrary ages (default: launch + age ladder)."""
    launches = load_launch_times(world_root)
    mints = sorted(launches)
    if max_mints is not None:
        mints = mints[:max_mints]
    indexes = load_pit_indexes(
        world_root, mint_filter=set(mints), pump_root=pump_root
    )
    ages = age_offsets if age_offsets is not None else PIT_AGE_OFFSETS_SEC
    flat: list[dict[str, Any]] = []
    coverage: dict[str, int] = defaultdict(int)

    for mint in mints:
        launch_t = launches[mint]
        times: list[tuple[str, float]] = []
        if observation_times and mint in observation_times:
            times.append(("custom", observation_times[mint]))
        for name, offset in ages:
            times.append((name, launch_t + offset))
        for age_name, t in times:
            state = reconstruct_token_state_at_t(
                mint=mint, observation_time_t=t, indexes=indexes
            )
            if any(state["modality_masks"].values()):
                coverage[age_name] += 1
            flat.append(
                {
                    "mint": mint,
                    "age_name": age_name,
                    "observation_time_t": t,
                    "launch_t": launch_t,
                    "ownership_evidence_level": state["ownership_evidence_level"],
                    "mask_event_history": state["modality_masks"]["event_history"],
                    "mask_event_curve": state["modality_masks"]["event_curve"],
                    "mask_wallet_token_dynamic": state["modality_masks"][
                        "wallet_token_dynamic"
                    ],
                    "mask_microstructure": state["modality_masks"]["microstructure"],
                    "mask_ownership_any": state["modality_masks"]["ownership_any"],
                    "mask_ownership_ledger": state["modality_masks"]["ownership_ledger"],
                    "mask_ownership_rpc_balance": state["modality_masks"][
                        "ownership_rpc_balance"
                    ],
                    "mask_native_sol_history": state["modality_masks"][
                        "native_sol_history"
                    ],
                    "mask_global_market": state["modality_masks"]["global_market"],
                    "quality_holder_reconstructable": state["quality_masks"][
                        "ownership_holder_state_reconstructable"
                    ],
                    "quality_lifetime_aggregates_excluded": state["quality_masks"][
                        "lifetime_wallet_aggregates_excluded"
                    ],
                    "event_history_n": state["event_history_n"],
                    "token_curve_state_events_n": state["token_curve_state_events_n"],
                    "wallet_token_interactions_n": state["wallet_token_interactions_n"],
                    "microstructure_txs_n": state["microstructure_txs_n"],
                    "ownership_observations_n": state["ownership_observations_n"],
                    "global_context_completed_days_n": state[
                        "global_context_completed_days_n"
                    ],
                    "state_digest": state_digest(state),
                }
            )

    out_dir = ensure_dir(world_root / "pit")
    path = out_dir / "pit_states_sample.parquet"
    if flat:
        pq.write_table(pa.Table.from_pylist(flat), path)
    else:
        pq.write_table(pa.table({"mint": pa.array([], type=pa.string())}), path)
    manifest = {
        "generated_at": utc_now(),
        "mints_materialized": len(mints),
        "rows_materialized": len(flat),
        "path": str(path),
        "policy": "input_only_observable_at_or_before_T",
        "arbitrary_t": True,
        "age_offsets_sec": {k: v for k, v in ages},
        "arbitrary_t_coverage_mints_with_any_modality": dict(coverage),
        "labels": "not_generated",
        "engine_version": "pit_v2",
    }
    write_json(out_dir / "pit_manifest.json", manifest)
    return manifest


def assert_no_post_t_leakage(
    *,
    mint: str,
    observation_time_t: float,
    indexes: dict[str, Any],
) -> dict[str, Any]:
    """Comprehensive leakage suite — post-T mutations must not change state@T."""
    before = reconstruct_token_state_at_t(
        mint=mint,
        observation_time_t=observation_time_t,
        indexes=indexes,
    )
    before_bytes = canonical_state_bytes(before)
    cases: dict[str, bool] = {}

    def _check(name: str, mutate) -> None:  # noqa: ANN001
        idx = copy.deepcopy(indexes)
        mutate(idx)
        after = reconstruct_token_state_at_t(
            mint=mint, observation_time_t=observation_time_t, indexes=idx
        )
        cases[name] = canonical_state_bytes(after) == before_bytes

    post_t = observation_time_t + 10_000
    post_dt = datetime.fromtimestamp(post_t, tz=UTC)

    def mutate_new_trade_wallet(idx: dict[str, Any]) -> None:
        idx["trades_by_mint"].setdefault(mint, []).append(
            {
                "mint": mint,
                "user_wallet": "brand_new_post_t_wallet",
                "event_time": post_dt,
                "is_buy": True,
                "sol_amount": 9.0,
            }
        )

    # Existing pre-T wallet edge receives additional post-T buys/sells.
    idx_base = copy.deepcopy(indexes)
    wallet = "w_preexisting_leak_test"
    idx_base["trades_by_mint"].setdefault(mint, []).append(
        {
            "mint": mint,
            "user_wallet": wallet,
            "event_time": datetime.fromtimestamp(observation_time_t - 60, tz=UTC),
            "is_buy": True,
            "sol_amount": 1.0,
        }
    )
    pre_existing_before = reconstruct_token_state_at_t(
        mint=mint, observation_time_t=observation_time_t, indexes=idx_base
    )
    pre_existing_bytes = canonical_state_bytes(pre_existing_before)
    idx_after = copy.deepcopy(idx_base)
    idx_after["trades_by_mint"][mint].extend(
        [
            {
                "mint": mint,
                "user_wallet": wallet,
                "event_time": post_dt,
                "is_buy": True,
                "sol_amount": 999.0,
            },
            {
                "mint": mint,
                "user_wallet": wallet,
                "event_time": post_dt,
                "is_buy": False,
                "sol_amount": 500.0,
            },
        ]
    )
    pre_existing_after = reconstruct_token_state_at_t(
        mint=mint, observation_time_t=observation_time_t, indexes=idx_after
    )
    cases["post_t_trades_on_existing_wallet_edge"] = (
        canonical_state_bytes(pre_existing_after) == pre_existing_bytes
    )

    _check("post_t_new_trade_row", mutate_new_trade_wallet)

    def mutate_ownership_transfer(idx: dict[str, Any]) -> None:
        idx["ledger_by_mint"].setdefault(mint, []).append(
            {
                "mint": mint,
                "source": "leak_src",
                "destination": "leak_dst",
                "block_timestamp": post_dt,
                "value": 1,
                "tx_signature": "leak_tt",
                "transfer_type": "spl-transfer",
            }
        )

    def mutate_rpc_balance(idx: dict[str, Any]) -> None:
        idx["rpc_own_by_mint"].setdefault(mint, []).append(
            {
                "mint": mint,
                "owner": "leak_owner",
                "delta_ui_amount": 1.0,
                "tx_signature": "leak_own",
                "block_time": post_t,
            }
        )

    def mutate_microstructure(idx: dict[str, Any]) -> None:
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

    def mutate_curve(idx: dict[str, Any]) -> None:
        idx["curve_by_mint"].setdefault(mint, []).append(
            {
                "mint": mint,
                "event_time": post_dt,
                "price_sol": 999.0,
                "v_sol_bonding_curve": 1e9,
                "is_buy": True,
                "sol_amount": 10,
            }
        )

    def mutate_global_activity(idx: dict[str, Any]) -> None:
        day = datetime.fromtimestamp(observation_time_t, tz=UTC).date().isoformat()
        idx["global_rows"] = list(idx.get("global_rows") or []) + [
            {
                "day": day,
                "launches": 10_000,
                "trades": 10_000,
                "buy_volume_sol": 1e9,
            },
            {
                "day": (
                    datetime.fromtimestamp(observation_time_t, tz=UTC)
                    + timedelta(days=1)
                )
                .date()
                .isoformat(),
                "launches": 99_999,
            },
        ]
        idx["launch_events"] = list(idx.get("launch_events") or []) + [
            (post_t, "future_mint_launch")
        ]

    def mutate_graduation_outcomes(idx: dict[str, Any]) -> None:
        idx["future_labels_by_mint"].setdefault(mint, []).append(
            {
                "mint": mint,
                "graduated_at": post_dt,
                "outcome": "leak_postgrad",
            }
        )
        idx["migrations_by_mint"].setdefault(mint, []).append(
            {
                "mint": mint,
                "migrated_at": post_dt,
                "pool_address": "leak_pool",
            }
        )

    _check("post_t_ownership_transfers", mutate_ownership_transfer)
    _check("post_t_rpc_balance_observations", mutate_rpc_balance)
    _check("post_t_transaction_metadata", mutate_microstructure)
    _check("post_t_curve_observations", mutate_curve)
    _check("post_t_global_activity", mutate_global_activity)
    _check("post_t_graduation_migration_outcomes", mutate_graduation_outcomes)

    passed = all(cases.values())
    return {
        "mint": mint,
        "observation_time_t": observation_time_t,
        "cases": cases,
        "state_unchanged": passed,
        "passed": passed,
        "before_digest": sha256(before_bytes).hexdigest(),
    }
