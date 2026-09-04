"""LABEL-ONLY future market path construction.

Completely separate from PIT/input reconstruction. Prefer reserve-consistent
prices; exclude System Program wallet and nonfinite prices; preserve distinct
same-signature events; do not fabricate post-grad continuity.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from trencher.data.feasibility.concepts import SYSTEM_PROGRAM_WALLET
from trencher.data.world_model.io import ensure_dir, file_provenance, utc_now, write_json
from trencher.data.world_model.targets.status import (
    CORPUS_END_UNIX,
    CORPUS_START_UNIX,
    OUTAGE_END_UNIX,
    OUTAGE_START_UNIX,
)
from trencher.data.world_model.targets.timeutil import is_finite_positive, to_unix

logger = logging.getLogger(__name__)

FUTURE_PATH_VERSION = "future_market_path_v1"
# Default dataset dirname used only when callers do not pass out_dir.
TARGET_DATASET_VERSION = "target_dataset_v2"


def _round_amt(x: Any, ndigits: int) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, ndigits)


def event_fingerprint(
    *,
    mint: str,
    tx_signature: str | None,
    event_time_unix: float,
    wallet: str | None,
    is_buy: bool | None,
    sol_amount: float | None,
    token_amount: float | None,
    v_sol: float | None = None,
    v_tok: float | None = None,
) -> str:
    side = None if is_buy is None else ("buy" if is_buy else "sell")
    return "|".join(
        [
            str(mint),
            str(tx_signature or ""),
            f"{float(event_time_unix):.6f}",
            str(wallet or ""),
            str(side or ""),
            str(_round_amt(sol_amount, 9)),
            str(_round_amt(token_amount, 6)),
            str(_round_amt(v_sol, 6)),
            str(_round_amt(v_tok, 3)),
        ]
    )


def _prefer_price(
    price_sol: Any,
    v_sol: Any,
    v_tok: Any,
) -> tuple[float | None, str]:
    """Prefer reserve-consistent price when reserves are positive and finite."""
    if is_finite_positive(v_sol) and is_finite_positive(v_tok):
        reserve_px = float(v_sol) / float(v_tok)
        if is_finite_positive(reserve_px):
            if is_finite_positive(price_sol):
                # Prefer reserves when they disagree materially, else observed.
                obs = float(price_sol)
                rel = abs(obs - reserve_px) / max(obs, reserve_px)
                if rel <= 0.05:
                    return obs, "trade_price_reserve_consistent"
                return reserve_px, "reserve_implied_preferred"
            return reserve_px, "reserve_implied"
    if is_finite_positive(price_sol):
        return float(price_sol), "trade_price_sol"
    return None, "unusable_price"


def _load_pregrad_rows(
    *,
    world_root: Path,
    pump_root: Path,
    mints: set[str],
) -> list[dict[str, Any]]:
    """Load pre-grad Pump trades for *mints* without hard-coding subset size."""
    atomic = world_root / "pit" / "atomic_trades_subset.parquet"
    rows: list[dict[str, Any]] = []
    seen_fp: set[str] = set()

    def _ingest(table_rows: list[dict[str, Any]], *, source: str) -> None:
        for row in table_rows:
            mint = row.get("mint")
            if mint not in mints:
                continue
            wallet = row.get("user_wallet")
            if wallet == SYSTEM_PROGRAM_WALLET:
                continue
            et = to_unix(row.get("event_time"))
            if et is None:
                continue
            v_sol = row.get("v_sol_bonding_curve")
            v_tok = row.get("v_tokens_bonding_curve")
            price, price_src = _prefer_price(row.get("price_sol"), v_sol, v_tok)
            if price is None:
                continue
            is_buy = row.get("is_buy")
            fp = event_fingerprint(
                mint=str(mint),
                tx_signature=row.get("tx_signature"),
                event_time_unix=et,
                wallet=wallet,
                is_buy=is_buy if isinstance(is_buy, bool) else None,
                sol_amount=row.get("sol_amount"),
                token_amount=row.get("token_amount"),
                v_sol=v_sol if is_finite_positive(v_sol) else None,
                v_tok=v_tok if is_finite_positive(v_tok) else None,
            )
            if fp in seen_fp:
                continue
            seen_fp.add(fp)
            side = None
            if isinstance(is_buy, bool):
                side = "buy" if is_buy else "sell"
            rows.append(
                {
                    "mint": mint,
                    "event_time_unix": float(et),
                    "price": float(price),
                    "price_source": price_src,
                    "side": side,
                    "is_buy": is_buy if isinstance(is_buy, bool) else None,
                    "sol_amount": (
                        float(row["sol_amount"])
                        if is_finite_positive(row.get("sol_amount"))
                        else None
                    ),
                    "token_amount": (
                        float(row["token_amount"])
                        if is_finite_positive(row.get("token_amount"))
                        else None
                    ),
                    "wallet": wallet,
                    "tx_signature": row.get("tx_signature"),
                    "v_sol_bonding_curve": (
                        float(v_sol) if is_finite_positive(v_sol) else None
                    ),
                    "v_tokens_bonding_curve": (
                        float(v_tok) if is_finite_positive(v_tok) else None
                    ),
                    "curve_pct_depleted": (
                        float(row["curve_pct_depleted"])
                        if row.get("curve_pct_depleted") is not None
                        and math.isfinite(float(row["curve_pct_depleted"]))
                        else None
                    ),
                    "quality": "pregrad_pump_trade",
                    "market_regime": "pre_graduation",
                    "source": source,
                    "event_fingerprint": fp,
                }
            )

    if atomic.exists() and pq.ParquetFile(atomic).metadata.num_rows:
        logger.info("Loading pre-grad path from atomic_trades_subset")
        _ingest(pq.read_table(atomic).to_pylist(), source="atomic_trades_subset")
    else:
        trades_dir = pump_root / "trades"
        if trades_dir.exists():
            logger.info("Scanning corpus trades for future path (%d mints)", len(mints))
            dataset = ds.dataset(str(trades_dir), format="parquet")
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
            for batch in dataset.scanner(columns=cols, batch_size=256_000).to_batches():
                batch_rows = [
                    dict(zip(batch.schema.names, vals, strict=True))
                    for vals in zip(
                        *[batch.column(c).to_pylist() for c in batch.schema.names],
                        strict=True,
                    )
                ]
                _ingest(batch_rows, source="pumpfun_trades_corpus")

    return rows


def _load_postgrad_rows(
    *,
    pump_root: Path,
    mints: set[str],
    migration_unix: dict[str, float],
) -> list[dict[str, Any]]:
    """Genuine frozen post-grad price trajectory only where coverage exists."""
    path = pump_root / "postgard_snapshots.parquet"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    # Only load rows for mints that both migrated and are in the request set.
    eligible = {m for m in mints if m in migration_unix}
    if not eligible:
        return []
    table = pq.read_table(
        path,
        columns=[
            "mint",
            "snapshot_time",
            "seconds_since_graduation",
            "price_native",
            "price_usd",
            "incomplete_data",
            "pair_address",
            "liquidity_usd",
        ],
    )
    for row in table.to_pylist():
        mint = row.get("mint")
        if mint not in eligible:
            continue
        et = to_unix(row.get("snapshot_time"))
        if et is None:
            continue
        mig_t = migration_unix[mint]
        if et <= mig_t:
            continue
        # Prefer native (SOL) price when available; else leave unavailable.
        price = None
        price_src = "postgrad_unusable"
        if is_finite_positive(row.get("price_native")):
            price = float(row["price_native"])
            price_src = "postgard_price_native"
        elif is_finite_positive(row.get("price_usd")):
            # USD-only marks are not SOL-executable continuity — keep separate.
            price = float(row["price_usd"])
            price_src = "postgard_price_usd_not_sol_continuous"
        if price is None:
            continue
        if row.get("incomplete_data"):
            quality = "postgrad_incomplete_snapshot"
        else:
            quality = "postgrad_snapshot"
        rows.append(
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
                "tx_signature": None,
                "v_sol_bonding_curve": None,
                "v_tokens_bonding_curve": None,
                "curve_pct_depleted": None,
                "quality": quality,
                "market_regime": "post_graduation",
                "source": "postgard_snapshots",
                "event_fingerprint": f"postgrad|{mint}|{et:.6f}|{price_src}",
                "liquidity_usd": (
                    float(row["liquidity_usd"])
                    if is_finite_positive(row.get("liquidity_usd"))
                    else None
                ),
            }
        )
    return rows


def load_migration_unix(
    *, world_root: Path, pump_root: Path, mints: set[str]
) -> dict[str, float]:
    candidates = [
        world_root / "universe" / "migrations.parquet",
        pump_root / "migrations.parquet",
        world_root / "pit" / "global" / "migrations_timeline.parquet",
    ]
    out: dict[str, float] = {}
    for path in candidates:
        if not path.exists():
            continue
        cols = pq.read_schema(path).names
        time_col = (
            "migrated_at"
            if "migrated_at" in cols
            else ("event_time_unix" if "event_time_unix" in cols else None)
        )
        if time_col is None or "mint" not in cols:
            continue
        for row in pq.read_table(path, columns=["mint", time_col]).to_pylist():
            mint = row.get("mint")
            if mint not in mints:
                continue
            et = to_unix(row.get(time_col))
            if et is None:
                continue
            # Prefer earliest genuine migration time if duplicates.
            prev = out.get(mint)
            if prev is None or et < prev:
                out[mint] = float(et)
        if out:
            break
    return out


def build_future_market_path_v1(
    *,
    world_root: Path,
    pump_root: Path,
    mints: set[str],
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Materialize LABEL-ONLY future market path for the given mint set."""
    out_dir = ensure_dir(out_dir or world_root / "targets" / TARGET_DATASET_VERSION)
    migration_unix = load_migration_unix(
        world_root=world_root, pump_root=pump_root, mints=mints
    )
    pre = _load_pregrad_rows(
        world_root=world_root, pump_root=pump_root, mints=mints
    )
    post = _load_postgrad_rows(
        pump_root=pump_root, mints=mints, migration_unix=migration_unix
    )

    # Do not carry stale pre-grad marks through unsupported post-grad regimes.
    # Keep pre-grad rows with et <= migration; post-grad only from genuine snapshots.
    filtered_pre: list[dict[str, Any]] = []
    for row in pre:
        mint = row["mint"]
        mig = migration_unix.get(mint)
        if mig is not None and row["event_time_unix"] > mig:
            continue
        filtered_pre.append(row)

    all_rows = filtered_pre + post
    all_rows.sort(
        key=lambda r: (
            r["mint"],
            r["event_time_unix"],
            r.get("event_fingerprint") or "",
        )
    )

    # Assign monotonic path_idx.
    for i, row in enumerate(all_rows):
        row["path_idx"] = i

    path = out_dir / f"{FUTURE_PATH_VERSION}.parquet"
    if all_rows:
        pq.write_table(pa.Table.from_pylist(all_rows), path)
    else:
        pq.write_table(
            pa.table(
                {
                    "mint": pa.array([], pa.string()),
                    "event_time_unix": pa.array([], pa.float64()),
                    "price": pa.array([], pa.float64()),
                    "path_idx": pa.array([], pa.int64()),
                }
            ),
            path,
        )

    # Entity index for fast mint slicing.
    entity_rows: list[dict[str, Any]] = []
    by_mint: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(all_rows):
        by_mint[row["mint"]].append(i)
    for mint in sorted(by_mint):
        idxs = by_mint[mint]
        entity_rows.append(
            {
                "mint": mint,
                "start_idx": idxs[0],
                "end_idx": idxs[-1] + 1,
                "n_rows": len(idxs),
                "n_pregrad": sum(
                    1
                    for j in idxs
                    if all_rows[j]["market_regime"] == "pre_graduation"
                ),
                "n_postgrad": sum(
                    1
                    for j in idxs
                    if all_rows[j]["market_regime"] == "post_graduation"
                ),
                "has_migration": mint in migration_unix,
                "migration_unix": migration_unix.get(mint),
                "first_event_unix": all_rows[idxs[0]]["event_time_unix"],
                "last_event_unix": all_rows[idxs[-1]]["event_time_unix"],
            }
        )
    entity_path = out_dir / f"{FUTURE_PATH_VERSION}_entity_index.parquet"
    if entity_rows:
        pq.write_table(pa.Table.from_pylist(entity_rows), entity_path)
    else:
        pq.write_table(
            pa.table({"mint": pa.array([], pa.string())}), entity_path
        )

    mig_path = out_dir / "migration_times_v1.parquet"
    mig_rows = [{"mint": m, "migration_unix": t} for m, t in sorted(migration_unix.items())]
    if mig_rows:
        pq.write_table(pa.Table.from_pylist(mig_rows), mig_path)
    else:
        pq.write_table(pa.table({"mint": pa.array([], pa.string())}), mig_path)

    meta = {
        "generated_at": utc_now(),
        "version": FUTURE_PATH_VERSION,
        "label_only": True,
        "mints_requested": len(mints),
        "mints_with_path": len(by_mint),
        "rows": len(all_rows),
        "pregrad_rows": len(filtered_pre),
        "postgrad_rows": len(post),
        "migrations_in_set": len(migration_unix),
        "excluded_system_program": True,
        "corpus_window_unix": {
            "start": CORPUS_START_UNIX,
            "end": CORPUS_END_UNIX,
        },
        "outage_window_unix": {
            "start": OUTAGE_START_UNIX,
            "end": OUTAGE_END_UNIX,
        },
        "path": file_provenance(path, source="label_only_future_path"),
        "entity_index": file_provenance(entity_path, source="label_only_future_path"),
        "note": (
            "Do not fabricate continuity across migration; post-grad marks only "
            "where genuine frozen snapshots exist."
        ),
    }
    write_json(out_dir / f"{FUTURE_PATH_VERSION}_manifest.json", meta)
    return meta


def load_path_arrays(
    path_parquet: Path,
    entity_index: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
    """Load per-mint numpy arrays for fast observation-level target compute."""
    entity = {
        r["mint"]: r for r in pq.read_table(entity_index).to_pylist() if r.get("mint")
    }
    if not entity:
        return {}, {}
    table = pq.read_table(path_parquet)
    rows = table.to_pylist()
    by_mint: dict[str, dict[str, np.ndarray]] = {}
    # Group contiguous rows by mint (path is sorted).
    i = 0
    n = len(rows)
    while i < n:
        mint = rows[i]["mint"]
        j = i + 1
        while j < n and rows[j]["mint"] == mint:
            j += 1
        chunk = rows[i:j]
        times = np.asarray([r["event_time_unix"] for r in chunk], dtype=np.float64)
        prices = np.asarray([r["price"] for r in chunk], dtype=np.float64)
        regimes = np.asarray(
            [1 if r["market_regime"] == "post_graduation" else 0 for r in chunk],
            dtype=np.int8,
        )
        is_buy = np.asarray(
            [
                1 if r.get("is_buy") is True else (0 if r.get("is_buy") is False else -1)
                for r in chunk
            ],
            dtype=np.int8,
        )
        sol = np.asarray(
            [
                r["sol_amount"] if r.get("sol_amount") is not None else np.nan
                for r in chunk
            ],
            dtype=np.float64,
        )
        tok = np.asarray(
            [
                r["token_amount"] if r.get("token_amount") is not None else np.nan
                for r in chunk
            ],
            dtype=np.float64,
        )
        wallets = np.asarray([r.get("wallet") or "" for r in chunk], dtype=object)
        v_sol = np.asarray(
            [
                r["v_sol_bonding_curve"]
                if r.get("v_sol_bonding_curve") is not None
                else np.nan
                for r in chunk
            ],
            dtype=np.float64,
        )
        v_tok = np.asarray(
            [
                r["v_tokens_bonding_curve"]
                if r.get("v_tokens_bonding_curve") is not None
                else np.nan
                for r in chunk
            ],
            dtype=np.float64,
        )
        curve = np.asarray(
            [
                r["curve_pct_depleted"]
                if r.get("curve_pct_depleted") is not None
                else np.nan
                for r in chunk
            ],
            dtype=np.float64,
        )
        price_src = np.asarray([r.get("price_source") or "" for r in chunk], dtype=object)
        path_idx = np.asarray([r["path_idx"] for r in chunk], dtype=np.int64)
        # SOL-continuous marks only (exclude USD-only postgrad as continuous path).
        usable = np.array(
            [
                (
                    r.get("market_regime") == "pre_graduation"
                    and is_finite_positive(r.get("price"))
                )
                or (
                    r.get("market_regime") == "post_graduation"
                    and r.get("price_source") == "postgard_price_native"
                    and is_finite_positive(r.get("price"))
                )
                for r in chunk
            ],
            dtype=bool,
        )
        by_mint[mint] = {
            "t": times,
            "price": prices,
            "regime": regimes,
            "is_buy": is_buy,
            "sol": sol,
            "token": tok,
            "wallet": wallets,
            "v_sol": v_sol,
            "v_tok": v_tok,
            "curve": curve,
            "price_source": price_src,
            "path_idx": path_idx,
            "usable_price": usable,
        }
        i = j
    return by_mint, entity
