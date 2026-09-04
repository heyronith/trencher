"""Acquire world-model layers from local PumpFun/RED corpora."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from trencher.data.feasibility.concepts import SYSTEM_PROGRAM_WALLET
from trencher.data.world_model.constants import CORPUS_END, CORPUS_START, SYNTHETIC_POOLS
from trencher.data.world_model.io import ensure_dir, file_provenance, utc_now, write_json
from trencher.data.world_model.subset import (
    select_unbiased_rich_subset,
    write_unbiased_subset,
)

logger = logging.getLogger(__name__)


def _day_key(ts: Any) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.astimezone(UTC).date().isoformat() if ts.tzinfo else ts.date().isoformat()
    text = str(ts)
    return text[:10] if len(text) >= 10 else None


def export_universe(pump_root: Path, out_dir: Path) -> dict[str, Any]:
    ensure_dir(out_dir)
    tokens = pq.read_table(
        pump_root / "tokens.parquet",
        columns=[
            "mint",
            "detected_at",
            "creator",
            "graduated_at",
            "is_mayhem_mode",
            "top10_pct_suspect",
            "initial_top10_pct_corrected",
            "dev_buy_pct_corrected",
            "supply_bug_corrected",
            "is_training_ready",
        ],
    )
    path = out_dir / "universe_tokens.parquet"
    pq.write_table(tokens, path)
    migrations = pq.read_table(pump_root / "migrations.parquet")
    mig_path = out_dir / "migrations.parquet"
    pq.write_table(migrations, mig_path)
    return {
        "tokens": file_provenance(path, source="pumpfun.tokens"),
        "migrations": file_provenance(mig_path, source="pumpfun.migrations"),
        "token_rows": tokens.num_rows,
        "graduated": int(
            tokens.num_rows
            - pc.sum(pc.is_null(tokens.column("graduated_at")).cast("int64")).as_py()
        ),
    }


def build_graduated_real_pool_subset(
    pump_root: Path, out_dir: Path, *, max_mints: int
) -> dict[str, Any]:
    """Keep graduated+real-pool list for postgrad/Gecko only — not fusion population."""
    ensure_dir(out_dir)
    migrations = pq.read_table(pump_root / "migrations.parquet")
    mints = migrations.column("mint").to_pylist()
    pools = migrations.column("pool_address").to_pylist()
    migrated = migrations.column("migrated_at").to_pylist()
    rows = []
    for mint, pool, when in zip(mints, pools, migrated, strict=True):
        if not pool or pool in SYNTHETIC_POOLS:
            continue
        rows.append({"mint": mint, "pool_address": pool, "migrated_at": when})
        if len(rows) >= max_mints:
            break
    table = (
        pa.Table.from_pylist(rows)
        if rows
        else pa.table(
            {
                "mint": pa.array([], type=pa.string()),
                "pool_address": pa.array([], type=pa.string()),
            }
        )
    )
    path = out_dir / "graduated_real_pool_mints.parquet"
    pq.write_table(table, path)
    return {
        "path": file_provenance(path, source="pumpfun.migrations.real_pools"),
        "mint_count": table.num_rows,
        "quality_flag": "outcome_selected_not_fusion_population",
    }


def build_rich_subset(pump_root: Path, out_dir: Path, *, max_mints: int) -> dict[str, Any]:
    """Outcome-independent fusion population. Writes rich_subset_mints.parquet."""
    ensure_dir(out_dir)
    selected = select_unbiased_rich_subset(
        pump_root / "tokens.parquet", max_mints=max_mints
    )
    written = write_unbiased_subset(selected, out_dir)
    graduated = build_graduated_real_pool_subset(
        pump_root, out_dir, max_mints=max_mints
    )
    written["graduated_real_pool"] = graduated
    return written


def _event_sort_key(ts: Any) -> float:
    if ts is None:
        return 0.0
    if isinstance(ts, datetime):
        return ts.timestamp()
    try:
        return float(ts)
    except (TypeError, ValueError):
        return 0.0


def select_stratified_trades_for_mint(
    trades: list[dict[str, Any]], *, cap: int
) -> list[dict[str, Any]]:
    """Pick earliest/buy/sell/size/curve/temporal trades. Deterministic."""
    if not trades or cap <= 0:
        return []
    ordered = sorted(
        trades,
        key=lambda t: (_event_sort_key(t.get("event_time")), t.get("tx_signature") or ""),
    )
    picks: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: dict[str, Any] | None) -> None:
        if row is None or len(picks) >= cap:
            return
        sig = row.get("tx_signature")
        if not sig or sig in seen:
            return
        seen.add(sig)
        picks.append(row)

    add(ordered[0])
    add(ordered[-1])
    buys = [t for t in ordered if t.get("is_buy")]
    sells = [t for t in ordered if not t.get("is_buy")]
    if buys:
        add(buys[len(buys) // 2])
    if sells:
        add(sells[len(sells) // 2])
    by_size = sorted(ordered, key=lambda t: float(t.get("sol_amount") or 0.0))
    add(by_size[0])
    add(by_size[-1])
    by_curve = sorted(
        ordered, key=lambda t: float(t.get("curve_pct_depleted") or 0.0)
    )
    add(by_curve[0])
    add(by_curve[-1])
    n = len(ordered)
    slots = max(cap, 4)
    for i in range(slots):
        idx = 0 if slots <= 1 else round(i * (n - 1) / (slots - 1))
        add(ordered[idx])
        if len(picks) >= cap:
            break
    return picks[:cap]


def extract_signature_targets(
    pump_root: Path,
    rich_mints: set[str],
    out_dir: Path,
    *,
    max_signatures: int,
    per_mint_cap: int | None = None,
    reservoir_per_mint: int = 80,
) -> dict[str, Any]:
    """Stratified per-token signatures. Not first-N globally."""
    ensure_dir(out_dir)
    n_mints = max(len(rich_mints), 1)
    cap = per_mint_cap or max(4, min(16, max_signatures // n_mints or 4))
    dataset = ds.dataset(pump_root / "trades", format="parquet")
    reservoirs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scanner = dataset.scanner(
        columns=[
            "mint",
            "tx_signature",
            "event_time",
            "user_wallet",
            "is_buy",
            "sol_amount",
            "curve_pct_depleted",
        ],
        batch_size=256_000,
    )
    for batch in scanner.to_batches():
        for mint, sig, ts, wallet, is_buy, sol, curve in zip(
            batch.column("mint").to_pylist(),
            batch.column("tx_signature").to_pylist(),
            batch.column("event_time").to_pylist(),
            batch.column("user_wallet").to_pylist(),
            batch.column("is_buy").to_pylist(),
            batch.column("sol_amount").to_pylist(),
            batch.column("curve_pct_depleted").to_pylist(),
            strict=True,
        ):
            if mint not in rich_mints or not sig:
                continue
            bucket = reservoirs[mint]
            row = {
                "mint": mint,
                "tx_signature": sig,
                "event_time": ts,
                "user_wallet": wallet,
                "is_buy": is_buy,
                "sol_amount": sol,
                "curve_pct_depleted": curve,
            }
            if len(bucket) < reservoir_per_mint:
                bucket.append(row)
                continue
            rank = int(hashlib.sha256(sig.encode()).hexdigest()[:8], 16)
            idx = rank % reservoir_per_mint
            if rank % 17 == 0:
                bucket[idx] = row

    selected_by_mint: dict[str, list[dict[str, Any]]] = {}
    for mint in sorted(rich_mints):
        selected_by_mint[mint] = select_stratified_trades_for_mint(
            reservoirs.get(mint, []), cap=cap
        )
    rows = []
    depth = 0
    while len(rows) < max_signatures:
        added = False
        for mint in sorted(rich_mints):
            picks = selected_by_mint.get(mint) or []
            if depth < len(picks):
                rows.append(picks[depth])
                added = True
            if len(rows) >= max_signatures:
                break
        if not added:
            break
        depth += 1
    per_mint: dict[str, int] = defaultdict(int)
    for row in rows:
        per_mint[row["mint"]] += 1

    table = (
        pa.Table.from_pylist(rows)
        if rows
        else pa.table(
            {"mint": pa.array([], pa.string()), "tx_signature": pa.array([], pa.string())}
        )
    )
    path = out_dir / "signature_targets.parquet"
    pq.write_table(table, path)
    return {
        "path": file_provenance(path, source="pumpfun.trades.tx_signature.stratified"),
        "rows": table.num_rows,
        "unique_mints": sum(1 for n in per_mint.values() if n),
        "per_mint_cap": cap,
        "max_signatures": max_signatures,
        "sampling": "stratified_per_token_lifecycle",
    }


def build_wallet_graph(
    pump_root: Path,
    out_dir: Path,
    *,
    rich_mints: set[str] | None = None,
    max_trade_batches: int | None = None,
    reuse_wallet_token: bool = True,
) -> dict[str, Any]:
    """Stream trades into wallet↔token and creator edges; co-buy on rich subset."""
    ensure_dir(out_dir)
    # Creator edges from tokens.
    tokens = pq.read_table(pump_root / "tokens.parquet", columns=["mint", "creator", "detected_at"])
    creator_rows = [
        {
            "edge_type": "creator_token",
            "src": c,
            "dst": m,
            "t": t,
            "weight": 1.0,
            "quality_flag": "observed",
        }
        for m, c, t in zip(
            tokens.column("mint").to_pylist(),
            tokens.column("creator").to_pylist(),
            tokens.column("detected_at").to_pylist(),
            strict=True,
        )
        if c and m
    ]
    creator_path = out_dir / "edges_creator_token.parquet"
    pq.write_table(pa.Table.from_pylist(creator_rows), creator_path)

    wt_path = out_dir / "edges_wallet_token.parquet"
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    reused_wt = reuse_wallet_token and wt_path.exists() and max_trade_batches is None
    if reused_wt:
        existing = pq.read_table(
            wt_path, columns=["src", "dst", "buys", "sells", "first_t", "last_t"]
        )
        wt_rows = existing.to_pylist()
        for row in wt_rows:
            agg[(row["src"], row["dst"])] = {
                "wallet": row["src"],
                "mint": row["dst"],
                "buys": row.get("buys") or 0,
                "sells": row.get("sells") or 0,
            }
    else:
        dataset = ds.dataset(pump_root / "trades", format="parquet")
        scanner = dataset.scanner(
            columns=["mint", "user_wallet", "event_time", "is_buy", "sol_amount"],
            batch_size=256_000,
        )
        for i, batch in enumerate(scanner.to_batches()):
            for mint, wallet, ts, is_buy, sol in zip(
                batch.column("mint").to_pylist(),
                batch.column("user_wallet").to_pylist(),
                batch.column("event_time").to_pylist(),
                batch.column("is_buy").to_pylist(),
                batch.column("sol_amount").to_pylist(),
                strict=True,
            ):
                if not wallet or wallet == SYSTEM_PROGRAM_WALLET or not mint:
                    continue
                key = (wallet, mint)
                slot = agg.get(key)
                if slot is None:
                    slot = {
                        "wallet": wallet,
                        "mint": mint,
                        "first_t": ts,
                        "last_t": ts,
                        "buys": 0,
                        "sells": 0,
                        "buy_sol": 0.0,
                        "sell_sol": 0.0,
                    }
                    agg[key] = slot
                else:
                    if ts is not None and (slot["first_t"] is None or ts < slot["first_t"]):
                        slot["first_t"] = ts
                    if ts is not None and (slot["last_t"] is None or ts > slot["last_t"]):
                        slot["last_t"] = ts
                if is_buy:
                    slot["buys"] += 1
                    if sol is not None:
                        slot["buy_sol"] += float(sol)
                else:
                    slot["sells"] += 1
                    if sol is not None:
                        slot["sell_sol"] += float(sol)
            if max_trade_batches is not None and i + 1 >= max_trade_batches:
                break

        wt_rows = [
            {
                "edge_type": "wallet_token",
                "src": v["wallet"],
                "dst": v["mint"],
                "first_t": v["first_t"],
                "last_t": v["last_t"],
                "buys": v["buys"],
                "sells": v["sells"],
                "buy_sol": v["buy_sol"],
                "sell_sol": v["sell_sol"],
                "quality_flag": "derived_from_trades_exclude_system_program",
            }
            for v in agg.values()
        ]
        if wt_rows:
            pq.write_table(pa.Table.from_pylist(wt_rows), wt_path)
        else:
            pq.write_table(
                pa.table(
                    {
                        "edge_type": pa.array([], pa.string()),
                        "src": pa.array([], pa.string()),
                    }
                ),
                wt_path,
            )

    # Co-buy edges for rich subset only (same mint buyers).
    co_path = out_dir / "edges_cobuy_rich.parquet"
    cobuy_count = 0
    if rich_mints:
        buyers: dict[str, set[str]] = defaultdict(set)
        for (wallet, mint), v in agg.items():
            if mint in rich_mints and v["buys"] > 0:
                buyers[mint].add(wallet)
        co_rows = []
        for mint, wallets in buyers.items():
            # Cap combinatorial explosion: pair only first 30 wallets lexicographically.
            ordered = sorted(wallets)[:30]
            for i, a in enumerate(ordered):
                for b in ordered[i + 1 :]:
                    co_rows.append(
                        {
                            "edge_type": "cobuy",
                            "src": a,
                            "dst": b,
                            "mint": mint,
                            "weight": 1.0,
                            "quality_flag": "rich_subset_capped_pairs",
                        }
                    )
        cobuy_count = len(co_rows)
        if co_rows:
            pq.write_table(pa.Table.from_pylist(co_rows), co_path)
        else:
            pq.write_table(
                pa.table({"edge_type": pa.array([], pa.string())}),
                co_path,
            )
    else:
        pq.write_table(pa.table({"edge_type": pa.array([], pa.string())}), co_path)

    # RED-COHORT membership edges.
    cohort_path = (
        pump_root.parent
        / "red_cohort_2026"
        / "extracted"
        / "RED-COHORT-2026-v1"
        / "sniper_cohorts.jsonl"
    )
    cohort_rows = []
    if cohort_path.exists():
        with cohort_path.open("rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj.get("cohort_id")
                for w in obj.get("wallets") or []:
                    if isinstance(w, str) and w:
                        cohort_rows.append(
                            {
                                "edge_type": "cohort_member",
                                "src": w,
                                "dst": cid,
                                "quality_flag": "red_cohort_observed",
                            }
                        )
    ch_path = out_dir / "edges_cohort_member.parquet"
    if cohort_rows:
        pq.write_table(pa.Table.from_pylist(cohort_rows), ch_path)
    else:
        pq.write_table(pa.table({"edge_type": pa.array([], pa.string())}), ch_path)

    return {
        "creator_token_edges": len(creator_rows),
        "wallet_token_edges": len(wt_rows),
        "cobuy_edges": cobuy_count,
        "cohort_member_edges": len(cohort_rows),
        "files": {
            "creator": file_provenance(creator_path, source="pumpfun.tokens"),
            "wallet_token": file_provenance(wt_path, source="pumpfun.trades"),
            "cobuy": file_provenance(co_path, source="pumpfun.trades.rich_subset"),
            "cohort": file_provenance(ch_path, source="red_cohort"),
        },
        "unique_wallets_in_wallet_token": len(
            {r.get("src") or r.get("wallet") for r in wt_rows}
        ),
        "reused_wallet_token": reused_wt,
        "quality_notes": [
            "wallet_stats aggregates not used",
            "System Program wallet excluded",
            "funding edges not produced here (see rpc/bq)",
        ],
    }


def build_global_context(pump_root: Path, out_dir: Path) -> dict[str, Any]:
    ensure_dir(out_dir)
    tokens = pq.read_table(
        pump_root / "tokens.parquet", columns=["mint", "detected_at", "graduated_at"]
    )
    launches: Counter[str] = Counter()
    grads: Counter[str] = Counter()
    for det, grad in zip(
        tokens.column("detected_at").to_pylist(),
        tokens.column("graduated_at").to_pylist(),
        strict=True,
    ):
        d = _day_key(det)
        if d:
            launches[d] += 1
        g = _day_key(grad)
        if g:
            grads[g] += 1

    # Trade volume by day (stream).
    trade_counts: Counter[str] = Counter()
    buy_sol: Counter[str] = Counter()
    dataset = ds.dataset(pump_root / "trades", format="parquet")
    scanner = dataset.scanner(columns=["event_time", "is_buy", "sol_amount"], batch_size=512_000)
    for batch in scanner.to_batches():
        for ts, is_buy, sol in zip(
            batch.column("event_time").to_pylist(),
            batch.column("is_buy").to_pylist(),
            batch.column("sol_amount").to_pylist(),
            strict=True,
        ):
            d = _day_key(ts)
            if not d:
                continue
            trade_counts[d] += 1
            if is_buy and sol is not None:
                buy_sol[d] += float(sol)

    days = sorted(set(launches) | set(grads) | set(trade_counts))
    rows = [
        {
            "day": d,
            "launches": launches.get(d, 0),
            "graduations": grads.get(d, 0),
            "trades": trade_counts.get(d, 0),
            "buy_volume_sol": buy_sol.get(d, 0.0),
            "graduation_rate": (
                grads.get(d, 0) / launches[d] if launches.get(d, 0) else None
            ),
            "in_corpus_window": (
                CORPUS_START.isoformat() <= d <= CORPUS_END.isoformat()
            ),
            "quality_flag": "derived_from_pumpfun",
        }
        for d in days
    ]
    path = out_dir / "daily_pump_network_context.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return {
        "path": file_provenance(path, source="pumpfun.tokens+trades"),
        "days": len(rows),
        "corpus_days_present": sum(1 for r in rows if r["in_corpus_window"]),
    }


def export_execution_state(pump_root: Path, out_dir: Path, rich_mints: set[str]) -> dict[str, Any]:
    """Curve-based executable state for rich-subset mints from trades."""
    ensure_dir(out_dir)
    dataset = ds.dataset(pump_root / "trades", format="parquet")
    rows: list[dict[str, Any]] = []
    per_mint = Counter()
    scanner = dataset.scanner(
        columns=[
            "mint",
            "event_time",
            "price_sol",
            "v_sol_bonding_curve",
            "v_tokens_bonding_curve",
            "curve_pct_depleted",
            "sol_amount",
            "token_amount",
            "is_buy",
        ],
        batch_size=256_000,
    )
    for batch in scanner.to_batches():
        for vals in zip(*[batch.column(c).to_pylist() for c in batch.schema.names], strict=True):
            row = dict(zip(batch.schema.names, vals, strict=True))
            mint = row["mint"]
            if mint not in rich_mints:
                continue
            if per_mint[mint] >= 200:
                continue
            per_mint[mint] += 1
            price = row["price_sol"]
            v_sol = row["v_sol_bonding_curve"]
            v_tok = row["v_tokens_bonding_curve"]
            rows.append(
                {
                    **row,
                    "executable_price_sol": price,
                    "has_reserves": v_sol is not None and v_tok is not None,
                    "quality_flag": (
                        "observed_curve_state"
                        if price is not None and v_sol is not None
                        else "partial_missing_price_or_reserves"
                    ),
                }
            )
    path = out_dir / "execution_curve_states.parquet"
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), path)
    else:
        pq.write_table(pa.table({"mint": pa.array([], pa.string())}), path)

    fee_path = out_dir / "pump_fee_regime.json"
    write_json(
        fee_path,
        {
            "generated_at": utc_now(),
            "note": (
                "Pump.fun fee schedule is protocol-documented; exact historical "
                "fee bps are not a column in the corpus. Flagged as metadata only."
            ),
            "fee_regime_quality_flag": "external_protocol_knowledge_not_row_level",
            "corpus_window": {
                "start": CORPUS_START.isoformat(),
                "end": CORPUS_END.isoformat(),
            },
        },
    )
    return {
        "path": file_provenance(path, source="pumpfun.trades"),
        "rows": len(rows),
        "mints": len(per_mint),
        "fee_regime": file_provenance(fee_path, source="protocol_docs"),
    }


def register_postgard(pump_root: Path, out_dir: Path) -> dict[str, Any]:
    ensure_dir(out_dir)
    meta = {
        "generated_at": utc_now(),
        "postgard_outcomes": str(pump_root / "postgard_outcomes.parquet"),
        "postgard_snapshots": str(pump_root / "postgard_snapshots.parquet"),
        "outcomes_rows": _safe_rows(pump_root / "postgard_outcomes.parquet"),
        "snapshots_rows": _safe_rows(pump_root / "postgard_snapshots.parquet"),
        "quality_flag": "authoritative_local_postgrad_corpus",
        "note": "Files remain in pumpfun_memecoin_corpus; registered, not duplicated.",
    }
    path = out_dir / "postgard_registry.json"
    write_json(path, meta)
    return meta


def _safe_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    return pq.ParquetFile(path).metadata.num_rows


def run_local_acquisition(
    *,
    external_root: Path,
    world_root: Path,
    rich_subset_max_mints: int = 4000,
    rpc_max_signatures: int = 25_000,
    max_trade_batches: int | None = None,
) -> dict[str, Any]:
    pump = external_root / "pumpfun_memecoin_corpus"
    summary: dict[str, Any] = {"generated_at": utc_now(), "layers": {}}
    summary["layers"]["universe"] = export_universe(pump, world_root / "universe")
    rich = build_rich_subset(
        pump, world_root / "universe", max_mints=rich_subset_max_mints
    )
    summary["layers"]["rich_subset"] = rich
    rich_mints = set(
        pq.read_table(world_root / "universe" / "rich_subset_mints.parquet", columns=["mint"])
        .column(0)
        .to_pylist()
    )
    summary["layers"]["signature_targets"] = extract_signature_targets(
        pump,
        rich_mints,
        world_root / "microstructure",
        max_signatures=rpc_max_signatures,
    )
    summary["layers"]["wallet_graph"] = build_wallet_graph(
        pump,
        world_root / "graph",
        rich_mints=rich_mints,
        max_trade_batches=max_trade_batches,
    )
    ctx_path = world_root / "global_context" / "daily_pump_network_context.parquet"
    if ctx_path.exists():
        days = pq.ParquetFile(ctx_path).metadata.num_rows
        summary["layers"]["global_context"] = {
            "reused": True,
            "days": days,
            "corpus_days_present": days,
        }
    else:
        summary["layers"]["global_context"] = build_global_context(
            pump, world_root / "global_context"
        )
    summary["layers"]["execution"] = export_execution_state(
        pump, world_root / "execution", rich_mints
    )
    summary["layers"]["postgard"] = register_postgard(pump, world_root / "postgrad")
    write_json(world_root / "local_acquisition_manifest.json", summary)
    return summary
