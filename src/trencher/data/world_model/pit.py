"""Point-in-time world-state reconstruction engine v4 (no labels / no training)."""

from __future__ import annotations

import copy
import json
import math
from bisect import bisect_left, bisect_right
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
from trencher.data.world_model.ownership import (
    classify_ownership_at_t,
    load_queried_ledger_days,
)

PIT_ENGINE_VERSION = "pit_v4"
GLOBAL_CONTEXT_INDEX_VERSION = "global_context_v4"

PIT_AGE_OFFSETS_SEC = (
    ("launch", 0),
    ("10s", 10),
    ("30s", 30),
    ("1m", 60),
    ("2m", 120),
    ("5m", 300),
    ("10m", 600),
    ("30m", 1800),
    ("1h", 3600),
    ("6h", 21600),
)

GLOBAL_TRAILING_WINDOWS_SEC = (10, 30, 60, 300, 900)

# Field names that must never appear in model_state (recursive).
FORBIDDEN_FUTURE_FIELD_NAMES = frozenset(
    {
        "future_labels_separate",
        "future_labels_by_mint",
        "postgard",
        "postgard_outcomes",
        "postgard_outcomes_n",
        "post_grad",
        "postgrad",
        "graduated_at",
        "graduation_label",
        "outcome",
        "outcomes",
        "future_return",
        "future_returns",
        "survival",
        "future_survival",
        "ownership_evidence_level",  # full-history QA only
        "own_level",
        "own_level_full_history",
        "labels",
        "target",
        "targets",
        "y_true",
        "label",
    }
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


def last_completed_second(observation_time_t: float) -> int:
    """Latest fully completed unix second strictly before the second containing T.

    Second-level trade buckets aggregate ``int(event_time)`` in [sec, sec+1).
    At fractional T the current second is incomplete and must be excluded, so
    global trade context has intentional sub-second latency (<1s).
    Do not round T upward.
    """
    return int(math.floor(observation_time_t)) - 1



def model_state_bytes(model_state: dict[str, Any]) -> bytes:
    """Canonical serialization of the entire model-input payload."""
    return json.dumps(model_state, sort_keys=True, default=str, separators=(",", ":")).encode()


def model_state_digest(model_state: dict[str, Any]) -> str:
    return sha256(model_state_bytes(model_state)).hexdigest()


def find_forbidden_future_fields(obj: Any, *, path: str = "$") -> list[str]:
    """Return paths of forbidden future-only field names found recursively."""
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            here = f"{path}.{key}"
            if key in FORBIDDEN_FUTURE_FIELD_NAMES or key.lower() in {
                x.lower() for x in FORBIDDEN_FUTURE_FIELD_NAMES
            }:
                hits.append(here)
            hits.extend(find_forbidden_future_fields(v, path=here))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(find_forbidden_future_fields(v, path=f"{path}[{i}]"))
    return hits


def load_launch_times(world_root: Path) -> dict[str, float]:
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
        },
    )
    return out_path


def ensure_global_context_index(
    *,
    world_root: Path,
    pump_root: Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Derive timestamped global Pump context from frozen local corpus only."""
    out_dir = ensure_dir(world_root / "pit" / "global")
    launches_path = out_dir / "launches_timeline.parquet"
    buckets_path = out_dir / "trade_second_buckets.parquet"
    migrations_path = out_dir / "migrations_timeline.parquet"
    manifest_path = out_dir / "global_context_index_manifest.json"

    if not force and manifest_path.exists() and buckets_path.exists():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prev = {}
        if (
            prev.get("index_version") == GLOBAL_CONTEXT_INDEX_VERSION
            and launches_path.exists()
            and migrations_path.exists()
            and pq.ParquetFile(launches_path).metadata.num_rows
        ):
            return {
                "launches": launches_path,
                "trade_buckets": buckets_path,
                "migrations": migrations_path,
                "manifest": manifest_path,
            }

    pump = pump_root or Path("data/external/pumpfun_memecoin_corpus")

    # Launches from universe tokens.
    ut = world_root / "universe" / "universe_tokens.parquet"
    launch_rows: list[dict[str, Any]] = []
    if ut.exists():
        table = pq.read_table(ut, columns=["mint", "detected_at"])
        for mint, ts in zip(
            table.column("mint").to_pylist(),
            table.column("detected_at").to_pylist(),
            strict=True,
        ):
            u = _to_unix(ts)
            if mint and u is not None:
                launch_rows.append({"mint": mint, "event_time_unix": u})
    launch_rows.sort(key=lambda r: r["event_time_unix"])
    if launch_rows:
        pq.write_table(pa.Table.from_pylist(launch_rows), launches_path)
    else:
        pq.write_table(
            pa.table(
                {
                    "mint": pa.array([], type=pa.string()),
                    "event_time_unix": pa.array([], type=pa.float64()),
                }
            ),
            launches_path,
        )

    # Trade second buckets from full local trades corpus (volume/flow only;
    # exact window unique mint/wallet counts omitted for tractability).
    # Exclude SYSTEM_PROGRAM_WALLET; do not impute malformed rows.
    buckets: dict[int, list[float]] = defaultdict(lambda: [0, 0, 0, 0.0, 0.0])
    excluded_system_program = 0
    skipped_missing_time = 0
    trades_dir = pump / "trades"
    if trades_dir.exists():
        dataset = ds.dataset(trades_dir, format="parquet")
        use_cols = [
            c
            for c in ["event_time", "is_buy", "sol_amount", "user_wallet"]
            if c in set(dataset.schema.names)
        ]
        for batch in dataset.scanner(columns=use_cols, batch_size=512_000).to_batches():
            ets = batch.column("event_time").to_pylist()
            buys = batch.column("is_buy").to_pylist()
            sols = batch.column("sol_amount").to_pylist()
            wallets = (
                batch.column("user_wallet").to_pylist()
                if "user_wallet" in use_cols
                else [None] * len(ets)
            )
            for et, ib, sol, wallet in zip(ets, buys, sols, wallets, strict=True):
                if wallet == SYSTEM_PROGRAM_WALLET:
                    excluded_system_program += 1
                    continue
                u = _to_unix(et)
                if u is None:
                    skipped_missing_time += 1
                    continue
                sec = int(u)
                slot = buckets[sec]
                slot[0] += 1
                sval = float(sol or 0.0)
                if ib:
                    slot[1] += 1
                    slot[3] += sval
                else:
                    slot[2] += 1
                    slot[4] += sval

    bucket_rows = []
    for sec in sorted(buckets):
        n_tr, n_b, n_s, buy_sol, sell_sol = buckets[sec]
        bucket_rows.append(
            {
                "second_unix": sec,
                "trades": int(n_tr),
                "buys": int(n_b),
                "sells": int(n_s),
                "buy_sol": buy_sol,
                "sell_sol": sell_sol,
                "unique_mints": 0,
                "unique_wallets": 0,
            }
        )
    if bucket_rows:
        pq.write_table(pa.Table.from_pylist(bucket_rows), buckets_path)
    else:
        pq.write_table(
            pa.table({"second_unix": pa.array([], type=pa.int64())}),
            buckets_path,
        )

    # Network migrations/graduations that occurred (event times) — not mint labels.
    mig_rows: list[dict[str, Any]] = []
    for cand in (
        pump / "migrations.parquet",
        world_root / "universe" / "migrations.parquet",
    ):
        if not cand.exists():
            continue
        for row in pq.read_table(cand).to_pylist():
            u = _to_unix(row.get("migrated_at"))
            if u is None or not row.get("mint"):
                continue
            mig_rows.append({"mint": row["mint"], "event_time_unix": u})
        break
    mig_rows.sort(key=lambda r: r["event_time_unix"])
    if mig_rows:
        pq.write_table(pa.Table.from_pylist(mig_rows), migrations_path)
    else:
        pq.write_table(
            pa.table(
                {
                    "mint": pa.array([], type=pa.string()),
                    "event_time_unix": pa.array([], type=pa.float64()),
                }
            ),
            migrations_path,
        )

    write_json(
        manifest_path,
        {
            "generated_at": utc_now(),
            "index_version": GLOBAL_CONTEXT_INDEX_VERSION,
            "pit_engine_version": PIT_ENGINE_VERSION,
            "launches_rows": len(launch_rows),
            "trade_second_buckets": len(bucket_rows),
            "migrations_rows": len(mig_rows),
            "excluded_system_program_wallet_rows": excluded_system_program,
            "skipped_missing_event_time_rows": skipped_missing_time,
            "system_program_wallet": SYSTEM_PROGRAM_WALLET,
            "raw_local_artifacts_unchanged": True,
            "includes_user_wallet_filter": True,
            "bucket_policy": (
                "int(event_time) aggregates; consumers must use only fully "
                "completed seconds relative to T (last_completed_second); "
                "global trade context has intentional <1s latency"
            ),
            "source": "frozen local pumpfun corpus + universe_tokens",
            "external_queries": 0,
        },
    )
    return {
        "launches": launches_path,
        "trade_buckets": buckets_path,
        "migrations": migrations_path,
        "manifest": manifest_path,
    }


def _wallet_token_state_from_trades(
    trades: list[dict[str, Any]], observation_time_t: float
) -> list[dict[str, Any]]:
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


def _sum_buckets(
    seconds: list[int],
    rows_by_sec: dict[int, dict[str, Any]],
    start_sec: int,
    end_sec: int,
) -> dict[str, Any]:
    lo = bisect_left(seconds, start_sec)
    hi = bisect_right(seconds, end_sec)
    trades = buys = sells = 0
    buy_sol = sell_sol = 0.0
    unique_mints = 0
    unique_wallets = 0
    for sec in seconds[lo:hi]:
        r = rows_by_sec[sec]
        trades += int(r.get("trades") or 0)
        buys += int(r.get("buys") or 0)
        sells += int(r.get("sells") or 0)
        buy_sol += float(r.get("buy_sol") or 0.0)
        sell_sol += float(r.get("sell_sol") or 0.0)
        unique_mints += int(r.get("unique_mints") or 0)
        unique_wallets += int(r.get("unique_wallets") or 0)
    return {
        "trades": trades,
        "buys": buys,
        "sells": sells,
        "buy_sol": buy_sol,
        "sell_sol": sell_sol,
        "unique_mints_sum_of_seconds": unique_mints,
        "unique_wallets_sum_of_seconds": unique_wallets,
        "unique_counts_note": "exact_window_uniques_not_materialized_in_v3_buckets",
    }


def _count_events_in_window(times: list[float], t: float, window: int) -> int:
    if not times:
        return 0
    lo = bisect_left(times, t - window)
    hi = bisect_right(times, t)
    return max(0, hi - lo)


def load_pit_indexes(
    world_root: Path,
    *,
    mint_filter: set[str] | None = None,
    pump_root: Path | None = None,
    ensure_trades: bool = True,
    ensure_global: bool = True,
) -> dict[str, Any]:
    """Load frozen + derived artifacts. Does not load future label modules/files."""
    if ensure_trades:
        ensure_atomic_trades_index(world_root=world_root, pump_root=pump_root)
    global_paths = None
    if ensure_global:
        global_paths = ensure_global_context_index(
            world_root=world_root, pump_root=pump_root
        )

    # Full-history ownership levels for offline QA only — never copied into model_state.
    own_level_audit: dict[str, str] = {}
    own_path = world_root / "universe" / "ownership_evidence.parquet"
    if own_path.exists() and pq.ParquetFile(own_path).metadata.num_rows:
        ot = pq.read_table(own_path, columns=["mint", "ownership_evidence_level"])
        for m, level in zip(
            ot.column("mint").to_pylist(),
            ot.column("ownership_evidence_level").to_pylist(),
            strict=True,
        ):
            if mint_filter is None or m in mint_filter:
                own_level_audit[m] = level

    detected_at_by_mint: dict[str, Any] = {}
    subset_path = world_root / "universe" / "rich_subset_mints.parquet"
    if subset_path.exists() and "detected_at" in pq.read_schema(subset_path).names:
        st = pq.read_table(subset_path, columns=["mint", "detected_at"])
        for m, ts in zip(
            st.column("mint").to_pylist(),
            st.column("detected_at").to_pylist(),
            strict=True,
        ):
            if mint_filter is None or m in mint_filter:
                detected_at_by_mint[m] = ts

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

    global_daily = []
    gpath = world_root / "global_context" / "daily_pump_network_context.parquet"
    if gpath.exists():
        global_daily = pq.read_table(gpath).to_pylist()

    blocks_daily = []
    bpath = world_root / "bigquery" / "blocks_daily.parquet"
    if bpath.exists() and pq.ParquetFile(bpath).metadata.num_rows:
        blocks_daily = pq.read_table(bpath).to_pylist()

    launch_times: list[float] = []
    trade_seconds: list[int] = []
    trade_bucket_by_sec: dict[int, dict[str, Any]] = {}
    migration_times: list[float] = []
    if global_paths:
        if global_paths["launches"].exists():
            lt = pq.read_table(global_paths["launches"], columns=["event_time_unix"])
            launch_times = sorted(
                float(x) for x in lt.column(0).to_pylist() if x is not None
            )
        if global_paths["trade_buckets"].exists() and pq.ParquetFile(
            global_paths["trade_buckets"]
        ).metadata.num_rows:
            for row in pq.read_table(global_paths["trade_buckets"]).to_pylist():
                sec = int(row["second_unix"])
                trade_seconds.append(sec)
                trade_bucket_by_sec[sec] = row
            trade_seconds.sort()
        if global_paths["migrations"].exists() and pq.ParquetFile(
            global_paths["migrations"]
        ).metadata.num_rows:
            mt = pq.read_table(global_paths["migrations"], columns=["event_time_unix"])
            migration_times = sorted(
                float(x) for x in mt.column(0).to_pylist() if x is not None
            )

    return {
        "own_level_full_history_audit": own_level_audit,
        "detected_at_by_mint": detected_at_by_mint,
        "curve_by_mint": curve_by_mint,
        "trades_by_mint": trades_by_mint,
        "micro_by_mint": micro_by_mint,
        "rpc_own_by_mint": rpc_own_by_mint,
        "ledger_by_mint": ledger_by_mint,
        "native_by_mint": native_by_mint,
        "global_daily": global_daily,
        "blocks_daily": blocks_daily,
        "launch_times": launch_times,
        "trade_seconds": trade_seconds,
        "trade_bucket_by_sec": trade_bucket_by_sec,
        "migration_times": migration_times,
        "queried_ledger_days": load_queried_ledger_days(world_root),
        "lifetime_wallet_token_forbidden": True,
    }


def reconstruct_token_state_at_t(
    *,
    mint: str,
    observation_time_t: float,
    indexes: dict[str, Any],
) -> dict[str, Any]:
    """Return INPUT STATE ONLY for arbitrary (mint, T).

    Structure:
      model_state — exclusive model-input payload (leakage-compared)
      provenance — operational metadata (generated_at, engine version)
    """
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

    own_at_t = classify_ownership_at_t(
        transfer_rows=indexes["ledger_by_mint"].get(mint, []),
        rpc_rows=indexes["rpc_own_by_mint"].get(mint, []),
        trade_participation_before_t=bool(wallet_token_state),
        detected_at=indexes["detected_at_by_mint"].get(mint),
        observation_time_t=observation_time_t,
        queried_ledger_days=indexes.get("queried_ledger_days"),
    )

    native_rows = []
    for row in indexes["native_by_mint"].get(mint, []):
        ts = row.get("block_time")
        u = float(ts) if ts is not None else _to_unix(row.get("event_time"))
        if u is not None and u <= observation_time_t:
            native_rows.append(
                {
                    "src": row.get("src"),
                    "dst": row.get("dst"),
                    "lamports": row.get("lamports"),
                    "block_time": u,
                    "tx_signature": row.get("tx_signature"),
                }
            )

    completed_days = []
    for row in indexes.get("global_daily") or []:
        day = row.get("day")
        if not day:
            continue
        try:
            day_dt = datetime.fromisoformat(str(day)).replace(tzinfo=UTC)
        except ValueError:
            continue
        day_end = (day_dt + timedelta(days=1)).timestamp()
        if day_end <= day_start:
            completed_days.append(
                {
                    "day": day,
                    "launches": row.get("launches"),
                    "trades": row.get("trades"),
                    "buy_volume_sol": row.get("buy_volume_sol"),
                    "graduations": row.get("graduations"),
                }
            )

    completed_blocks = []
    for row in indexes.get("blocks_daily") or []:
        day = row.get("day")
        if not day:
            continue
        try:
            day_dt = datetime.fromisoformat(str(day)).replace(tzinfo=UTC)
        except ValueError:
            continue
        day_end = (day_dt + timedelta(days=1)).timestamp()
        if day_end <= day_start:
            completed_blocks.append(
                {
                    "day": day,
                    "block_count": row.get("block_count"),
                    "transaction_count_sum": row.get("transaction_count_sum"),
                    "transaction_count_avg": row.get("transaction_count_avg"),
                }
            )

    trailing: dict[str, Any] = {}
    launch_times = indexes.get("launch_times") or []
    migration_times = indexes.get("migration_times") or []
    trade_seconds = indexes.get("trade_seconds") or []
    trade_bucket_by_sec = indexes.get("trade_bucket_by_sec") or {}
    # Only fully completed seconds — never the incomplete second containing T.
    end_sec = last_completed_second(observation_time_t)
    for w in GLOBAL_TRAILING_WINDOWS_SEC:
        key = f"trailing_{w}s"
        trailing[key] = {
            "launches": _count_events_in_window(launch_times, observation_time_t, w),
            "migrations_graduations": _count_events_in_window(
                migration_times, observation_time_t, w
            ),
            "pump_trades": _sum_buckets(
                trade_seconds,
                trade_bucket_by_sec,
                end_sec - w + 1,
                end_sec,
            )
            if end_sec >= 0
            else _sum_buckets(trade_seconds, trade_bucket_by_sec, 1, 0),
        }

    rolling_global = {
        "completed_day_context": completed_days,
        "completed_block_context": completed_blocks,
        "trailing_windows": trailing,
        "same_day_partial_daily_aggregates_excluded": True,
        "global_trade_buckets_last_completed_second": end_sec,
        "global_trade_bucket_latency_note": (
            "second_buckets_exclude_incomplete_second_containing_T_lt_1s_latency"
        ),
        "launches_observed_at_or_before_t": bisect_right(
            launch_times, observation_time_t
        ),
        "migrations_graduations_at_or_before_t": bisect_right(
            migration_times, observation_time_t
        ),
    }

    ledger_n = int(own_at_t["ownership_rows_observed_at_t"])
    rpc_n = int(own_at_t["ownership_rpc_observed_at_t"])

    modality_masks = {
        "event_history": bool(event_history),
        "event_curve": bool(curve_rows),
        "execution": bool(curve_rows),
        "wallet_token_dynamic": bool(wallet_token_state),
        "graph": bool(wallet_token_state),
        "microstructure": bool(micro_rows),
        "ownership_any": bool(ownership_obs),
        "ownership_ledger": ledger_n > 0,
        "ownership_rpc_balance": rpc_n > 0,
        "ownership_holder_reconstructable_at_t": bool(
            own_at_t["holder_state_reconstructable_at_t"]
        ),
        "native_sol_history": bool(native_rows),
        "global_market": bool(completed_days)
        or bool(trailing)
        or bisect_right(launch_times, observation_time_t) > 0,
    }
    quality_masks = {
        "ownership_level_at_t": own_at_t["ownership_level_at_t"],
        "holder_state_reconstructable_at_t": bool(
            own_at_t["holder_state_reconstructable_at_t"]
        ),
        "wallet_graph_from_atomic_trades": True,
        "lifetime_wallet_aggregates_excluded": True,
        "global_excludes_unfinished_day": True,
        "global_trade_excludes_incomplete_second": True,
        "holder_state_fabricated": False,
        "full_history_ownership_level_excluded_from_inputs": True,
    }

    model_state = {
        "mint": mint,
        "observation_time_t": observation_time_t,
        "ownership_level_at_t": own_at_t["ownership_level_at_t"],
        "ownership_rows_observed_at_t": ledger_n,
        "ownership_rpc_observed_at_t": rpc_n,
        "holder_state_reconstructable_at_t": bool(
            own_at_t["holder_state_reconstructable_at_t"]
        ),
        "modality_masks": modality_masks,
        "quality_masks": quality_masks,
        "event_history_n": len(event_history),
        "token_curve_state_events_n": len(curve_rows),
        "curve_state_latest": latest_curve,
        "wallet_token_state": wallet_token_state,
        "wallet_token_interactions_n": len(wallet_token_state),
        "microstructure_txs_n": len(micro_rows),
        "ownership_observations_n": len(ownership_obs),
        "native_sol_edges_n": len(native_rows),
        "native_sol_history": native_rows,
        "global_context": rolling_global,
        "global_context_completed_days_n": len(completed_days),
    }

    forbidden = find_forbidden_future_fields(model_state)
    if forbidden:
        raise RuntimeError(f"forbidden future fields in model_state: {forbidden}")

    return {
        "model_state": model_state,
        "provenance": {
            "generated_at": utc_now(),
            "engine_version": PIT_ENGINE_VERSION,
            "policy": "input_only_observable_at_or_before_T",
            "full_history_ownership_audit_level": indexes.get(
                "own_level_full_history_audit", {}
            ).get(mint),
            "note": "full_history_ownership_audit_level is provenance only, not an input",
        },
    }


def materialize_pit_states_for_subset(
    *,
    world_root: Path,
    max_mints: int | None = 50,
    mint_list: list[str] | None = None,
    observation_times: dict[str, float] | None = None,
    pump_root: Path | None = None,
    age_offsets: tuple[tuple[str, int], ...] | None = None,
) -> dict[str, Any]:
    launches = load_launch_times(world_root)
    if mint_list is not None:
        mints = [m for m in mint_list if m in launches]
    else:
        mints = sorted(launches)
        if max_mints is not None:
            mints = mints[:max_mints]
    indexes = load_pit_indexes(
        world_root, mint_filter=set(mints), pump_root=pump_root
    )
    ages = age_offsets if age_offsets is not None else PIT_AGE_OFFSETS_SEC
    flat: list[dict[str, Any]] = []
    coverage: dict[str, int] = {name: 0 for name, _ in ages}
    own_by_age: dict[str, dict[str, int]] = {name: {} for name, _ in ages}
    holder_at_t_by_age: dict[str, int] = {name: 0 for name, _ in ages}
    global_cov: dict[str, int] = {name: 0 for name, _ in ages}

    for mint in mints:
        launch_t = launches[mint]
        times: list[tuple[str, float]] = []
        if observation_times and mint in observation_times:
            times.append(("custom", observation_times[mint]))
        for name, offset in ages:
            times.append((name, launch_t + offset))
        for age_name, t in times:
            result = reconstruct_token_state_at_t(
                mint=mint, observation_time_t=t, indexes=indexes
            )
            state = result["model_state"]
            if any(state["modality_masks"].values()):
                coverage[age_name] += 1
            level = state["ownership_level_at_t"]
            own_by_age.setdefault(age_name, {})
            own_by_age[age_name][level] = own_by_age[age_name].get(level, 0) + 1
            if state["holder_state_reconstructable_at_t"]:
                holder_at_t_by_age[age_name] = holder_at_t_by_age.get(age_name, 0) + 1
            if state["modality_masks"].get("global_market"):
                global_cov[age_name] = global_cov.get(age_name, 0) + 1
            flat.append(
                {
                    "mint": mint,
                    "age_name": age_name,
                    "observation_time_t": t,
                    "launch_t": launch_t,
                    "ownership_level_at_t": level,
                    "ownership_rows_observed_at_t": state["ownership_rows_observed_at_t"],
                    "ownership_rpc_observed_at_t": state["ownership_rpc_observed_at_t"],
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
                    "event_history_n": state["event_history_n"],
                    "wallet_token_interactions_n": state["wallet_token_interactions_n"],
                    "global_context_completed_days_n": state[
                        "global_context_completed_days_n"
                    ],
                    "model_state_digest": model_state_digest(state),
                }
            )

    out_dir = ensure_dir(world_root / "pit")
    path = out_dir / "pit_states_sample.parquet"
    if flat:
        pq.write_table(pa.Table.from_pylist(flat), path)
    else:
        pq.write_table(pa.table({"mint": pa.array([], pa.string())}), path)
    manifest = {
        "generated_at": utc_now(),
        "mints_materialized": len(mints),
        "rows_materialized": len(flat),
        "path": str(path),
        "policy": "input_only_observable_at_or_before_T",
        "arbitrary_t": True,
        "age_offsets_sec": {k: v for k, v in ages},
        "arbitrary_t_coverage_mints_with_any_modality": dict(coverage),
        "ownership_level_at_t_by_age": {
            age: dict(levels) for age, levels in own_by_age.items()
        },
        "holder_state_reconstructable_at_t_by_age": dict(holder_at_t_by_age),
        "global_context_coverage_by_age": dict(global_cov),
        "engine_version": PIT_ENGINE_VERSION,
    }
    write_json(out_dir / "pit_manifest.json", manifest)
    return manifest



def _indexes_for_mutation(indexes: dict[str, Any]) -> dict[str, Any]:
    """Copy mint-local collections; share immutable global timelines by reference."""
    shared = {
        "launch_times",
        "trade_seconds",
        "trade_bucket_by_sec",
        "migration_times",
        "global_daily",
        "blocks_daily",
        "own_level_full_history_audit",
        "detected_at_by_mint",
        "queried_ledger_days",
        "lifetime_wallet_token_forbidden",
    }
    out: dict[str, Any] = {}
    for k, v in indexes.items():
        if k in shared:
            out[k] = v
        elif isinstance(v, dict):
            out[k] = {mk: list(mv) if isinstance(mv, list) else mv for mk, mv in v.items()}
        else:
            out[k] = copy.deepcopy(v)
    return out

def assert_no_post_t_leakage(
    *,
    mint: str,
    observation_time_t: float,
    indexes: dict[str, Any],
) -> dict[str, Any]:
    """Mutate post-T sources; entire model_state payload must stay byte-identical."""
    before = reconstruct_token_state_at_t(
        mint=mint,
        observation_time_t=observation_time_t,
        indexes=indexes,
    )["model_state"]
    before_bytes = model_state_bytes(before)
    cases: dict[str, bool] = {}

    def _check(name: str, mutate) -> None:  # noqa: ANN001
        idx = _indexes_for_mutation(indexes)
        mutate(idx)
        after = reconstruct_token_state_at_t(
            mint=mint, observation_time_t=observation_time_t, indexes=idx
        )["model_state"]
        cases[name] = model_state_bytes(after) == before_bytes

    post_t = observation_time_t + 10_000
    post_dt = datetime.fromtimestamp(post_t, tz=UTC)

    # Existing pre-T wallet receives post-T buys/sells.
    idx_base = _indexes_for_mutation(indexes)
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
    pre_bytes = model_state_bytes(
        reconstruct_token_state_at_t(
            mint=mint, observation_time_t=observation_time_t, indexes=idx_base
        )["model_state"]
    )
    idx_after = _indexes_for_mutation(idx_base)
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
    cases["post_t_trades_on_existing_wallet_edge"] = (
        model_state_bytes(
            reconstruct_token_state_at_t(
                mint=mint, observation_time_t=observation_time_t, indexes=idx_after
            )["model_state"]
        )
        == pre_bytes
    )

    _check(
        "post_t_new_trade_row",
        lambda idx: idx["trades_by_mint"].setdefault(mint, []).append(
            {
                "mint": mint,
                "user_wallet": "brand_new_post_t_wallet",
                "event_time": post_dt,
                "is_buy": True,
                "sol_amount": 9.0,
            }
        ),
    )
    _check(
        "post_t_ownership_transfers",
        lambda idx: idx["ledger_by_mint"].setdefault(mint, []).append(
            {
                "mint": mint,
                "source": "leak_src",
                "destination": "leak_dst",
                "block_timestamp": post_dt,
                "value": 1,
                "tx_signature": "leak_tt",
                "transfer_type": "spl-transfer",
            }
        ),
    )
    _check(
        "post_t_rpc_balance_observations",
        lambda idx: idx["rpc_own_by_mint"].setdefault(mint, []).append(
            {
                "mint": mint,
                "owner": "leak_owner",
                "delta_ui_amount": 1.0,
                "tx_signature": "leak_own",
                "block_time": post_t,
            }
        ),
    )
    _check(
        "post_t_transaction_metadata",
        lambda idx: idx["micro_by_mint"].setdefault(mint, []).append(
            {
                "mint": mint,
                "found": True,
                "block_time": post_t,
                "tx_signature": "leak_sig",
                "slot": 0,
                "success": True,
                "fee_lamports": 1,
            }
        ),
    )
    _check(
        "post_t_curve_observations",
        lambda idx: idx["curve_by_mint"].setdefault(mint, []).append(
            {
                "mint": mint,
                "event_time": post_dt,
                "price_sol": 999.0,
                "v_sol_bonding_curve": 1e9,
                "is_buy": True,
                "sol_amount": 10,
            }
        ),
    )
    _check(
        "post_t_native_sol_history",
        lambda idx: idx["native_by_mint"].setdefault(mint, []).append(
            {
                "mint": mint,
                "src": "f_leak",
                "dst": "c_leak",
                "lamports": 99,
                "block_time": post_t,
                "tx_signature": "native_leak",
            }
        ),
    )

    def mutate_global(idx: dict[str, Any]) -> None:
        day = datetime.fromtimestamp(observation_time_t, tz=UTC).date().isoformat()
        idx["global_daily"] = list(idx.get("global_daily") or []) + [
            {"day": day, "launches": 10_000, "trades": 10_000, "buy_volume_sol": 1e9},
            {
                "day": (
                    datetime.fromtimestamp(observation_time_t, tz=UTC) + timedelta(days=1)
                )
                .date()
                .isoformat(),
                "launches": 99_999,
            },
        ]
        # Copy-on-write for shared global timelines before mutation.
        idx["launch_times"] = list(idx.get("launch_times") or []) + [post_t]
        idx["migration_times"] = list(idx.get("migration_times") or []) + [post_t]
        sec = int(post_t)
        idx["trade_seconds"] = sorted(list(idx.get("trade_seconds") or []) + [sec])
        buckets = dict(idx.get("trade_bucket_by_sec") or {})
        buckets[sec] = {
            "second_unix": sec,
            "trades": 999,
            "buys": 999,
            "sells": 0,
            "buy_sol": 1e6,
            "sell_sol": 0.0,
            "unique_mints": 50,
            "unique_wallets": 50,
        }
        idx["trade_bucket_by_sec"] = buckets

    _check("post_t_launches_global_trades_migrations", mutate_global)

    # Injecting future label-like structures into indexes must not affect model_state
    # (PIT ignores them; also proves reconstruct does not read them).
    def mutate_future_label_junk(idx: dict[str, Any]) -> None:
        idx["future_labels_by_mint"] = {
            mint: [{"graduated_at": post_dt, "outcome": "leak", "future_return": 9.0}]
        }
        idx["postgard_outcomes"] = [{"mint": mint, "survival": 0}]

    _check("post_t_graduation_migration_outcomes_ignored", mutate_future_label_junk)

    # As-of-T ownership must not upgrade from later ledger density.
    idx_own = _indexes_for_mutation(indexes)
    # Ensure early T has no reconstructable ledger; add only post-T dense ledger.
    idx_own["ledger_by_mint"][mint] = [
        r
        for r in idx_own["ledger_by_mint"].get(mint, [])
        if (_to_unix(r.get("block_timestamp")) or 0) <= observation_time_t
    ]
    early = reconstruct_token_state_at_t(
        mint=mint, observation_time_t=observation_time_t, indexes=idx_own
    )["model_state"]
    for i in range(30):
        idx_own["ledger_by_mint"].setdefault(mint, []).append(
            {
                "mint": mint,
                "source": "h0",
                "destination": f"h{i}",
                "block_timestamp": post_dt,
                "value": 10.0,
                "tx_signature": f"future_tt{i}",
                "transfer_type": "spl-transfer" if i else "mintTo",
            }
        )
    late = reconstruct_token_state_at_t(
        mint=mint, observation_time_t=observation_time_t, indexes=idx_own
    )["model_state"]
    cases["post_t_ledger_density_does_not_upgrade_ownership_at_t"] = (
        model_state_bytes(early) == model_state_bytes(late)
        and early["ownership_level_at_t"] == late["ownership_level_at_t"]
    )

    forbidden_hits = find_forbidden_future_fields(before)
    cases["model_state_has_no_forbidden_future_fields"] = not forbidden_hits

    # Same-second future trade must not alter state@T (bucket exclusion).
    def mutate_same_second_future_trade(idx: dict[str, Any]) -> None:
        sec = int(__import__("math").floor(observation_time_t))
        # Future event later in the same incomplete second as T.
        future_frac = sec + 0.9
        if future_frac <= observation_time_t:
            future_frac = observation_time_t + 0.001
            sec = int(__import__("math").floor(future_frac))
        idx["trade_seconds"] = sorted(set(list(idx.get("trade_seconds") or []) + [sec]))
        buckets = dict(idx.get("trade_bucket_by_sec") or {})
        prev = dict(buckets.get(sec) or {})
        buckets[sec] = {
            "second_unix": sec,
            "trades": int(prev.get("trades") or 0) + 1000,
            "buys": int(prev.get("buys") or 0) + 1000,
            "sells": int(prev.get("sells") or 0),
            "buy_sol": float(prev.get("buy_sol") or 0.0) + 1e6,
            "sell_sol": float(prev.get("sell_sol") or 0.0),
            "unique_mints": 0,
            "unique_wallets": 0,
        }
        idx["trade_bucket_by_sec"] = buckets

    _check("post_t_same_second_future_trade_bucket", mutate_same_second_future_trade)

    passed = all(cases.values())
    return {
        "mint": mint,
        "observation_time_t": observation_time_t,
        "cases": cases,
        "forbidden_field_hits": forbidden_hits,
        "state_unchanged": passed,
        "passed": passed,
        "before_digest": sha256(before_bytes).hexdigest(),
    }


# Back-compat aliases used by older tests/audit helpers.
def canonical_state_bytes(state: dict[str, Any]) -> bytes:
    if "model_state" in state:
        return model_state_bytes(state["model_state"])
    return model_state_bytes(state)


def state_digest(state: dict[str, Any]) -> str:
    return sha256(canonical_state_bytes(state)).hexdigest()
