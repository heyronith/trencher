"""Broad supervision populations (outcome-independent membership)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.pit import _to_unix

logger = logging.getLogger(__name__)

POPULATION_VERSION = "broad_population_v1"


def _mint_in_trades(pump_root: Path) -> set[str]:
    """Distinct mints appearing in Pump trade shards (streaming scan)."""
    trades_dir = pump_root / "trades"
    if not trades_dir.exists():
        return set()
    dataset = ds.dataset(str(trades_dir), format="parquet")
    seen: set[str] = set()
    for batch in dataset.scanner(columns=["mint"], batch_size=512_000).to_batches():
        for m in batch.column(0).to_pylist():
            if m:
                seen.add(str(m))
    return seen


def _migration_mints(pump_root: Path, world_root: Path) -> set[str]:
    candidates = [
        world_root / "universe" / "migrations.parquet",
        pump_root / "migrations.parquet",
    ]
    out: set[str] = set()
    for path in candidates:
        if not path.exists():
            continue
        cols = pq.read_schema(path).names
        if "mint" not in cols:
            continue
        for m in pq.read_table(path, columns=["mint"]).column(0).to_pylist():
            if m:
                out.add(str(m))
        if out:
            break
    return out


def build_broad_population_v1(
    *,
    world_root: Path,
    pump_root: Path,
    out_dir: Path | None = None,
    scan_trades_for_presence: bool = True,
) -> dict[str, Any]:
    """Build ``broad_population_v1.parquet`` with outcome-independent flags.

    Population classes (flags are additive; membership is never graduation-biased):
    - ``launch_population``: valid launches (mint + detected_at)
    - ``event_curve_population``: launches with usable Pump trade history
      (trade_count > 0 or mint appears in trades shards)
    - ``price_target_population``: initially False; filled during sequence build
      when finite trade prices are observed
    - ``graduation_event_population``: True for all launches under conservative
      censoring notes, with ``has_migration_evidence`` as metadata only
    """
    broad_root = ensure_dir(out_dir or world_root / "broad")
    tokens_path = pump_root / "tokens.parquet"
    if not tokens_path.exists():
        raise FileNotFoundError(f"missing tokens.parquet at {tokens_path}")

    cols = [
        "mint",
        "detected_at",
        "creator",
        "tracking_expires_at",
        "graduated_at",
        "trade_count",
    ]
    table = pq.read_table(tokens_path, columns=cols)
    logger.info("Loaded tokens.parquet rows=%d", table.num_rows)

    trade_mints: set[str] = set()
    if scan_trades_for_presence:
        logger.info("Scanning trade shards for mint presence")
        trade_mints = _mint_in_trades(pump_root)
        logger.info("Distinct mints in trades: %d", len(trade_mints))

    mig_mints = _migration_mints(pump_root, world_root)
    logger.info("Mints with migration evidence: %d", len(mig_mints))

    rows: list[dict[str, Any]] = []
    launch_n = event_n = mig_meta_n = 0
    for rec in table.to_pylist():
        mint = rec.get("mint")
        det = _to_unix(rec.get("detected_at"))
        if not mint or det is None:
            continue
        mint = str(mint)
        trade_count = rec.get("trade_count")
        try:
            tc = int(trade_count) if trade_count is not None else 0
        except (TypeError, ValueError):
            tc = 0
        in_trades = mint in trade_mints
        event_curve = tc > 0 or in_trades
        has_mig = mint in mig_mints
        # Graduation class includes ALL launches; migration is metadata only.
        # Preferential graduate sampling is forbidden.
        flags = {
            "launch_population": True,
            "event_curve_population": bool(event_curve),
            "price_target_population": False,
            "graduation_event_population": True,
        }
        rows.append(
            {
                "mint": mint,
                "detected_at_unix": float(det),
                "creator": rec.get("creator"),
                "tracking_expires_at_unix": _to_unix(rec.get("tracking_expires_at")),
                "graduated_at_unix": _to_unix(rec.get("graduated_at")),
                "trade_count": tc,
                "appears_in_trades": in_trades,
                "has_migration_evidence": has_mig,
                "population_flags": flags,
                "launch_population": True,
                "event_curve_population": bool(event_curve),
                "price_target_population": False,
                "graduation_event_population": True,
            }
        )
        launch_n += 1
        if event_curve:
            event_n += 1
        if has_mig:
            mig_meta_n += 1

    out_path = broad_root / f"{POPULATION_VERSION}.parquet"
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), out_path)
    else:
        pq.write_table(
            pa.table({"mint": pa.array([], type=pa.string())}), out_path
        )

    meta = {
        "generated_at": utc_now(),
        "version": POPULATION_VERSION,
        "path": str(out_path),
        "counts": {
            "launch_population": launch_n,
            "event_curve_population": event_n,
            "price_target_population": 0,
            "graduation_event_population": launch_n,
            "has_migration_evidence_metadata": mig_meta_n,
            "trade_mints_scanned": len(trade_mints),
        },
        "membership_policy": {
            "outcome_independent": True,
            "no_preferential_graduate_sampling": True,
            "graduated_at_unix_is_metadata_only": True,
            "graduation_event_population": (
                "All valid launches; migration evidence is metadata. "
                "Conservative v3 absence-censoring applies at target time."
            ),
            "event_curve_population": (
                "trade_count > 0 OR mint appears in Pump trades shards"
            ),
            "price_target_population": (
                "Filled during sequence build when finite price_sol / "
                "reserve-implied price is observed"
            ),
        },
    }
    write_json(broad_root / f"{POPULATION_VERSION}_manifest.json", meta)
    logger.info(
        "broad_population_v1 launches=%d event_curve=%d", launch_n, event_n
    )
    return meta


def load_population(path: Path) -> pa.Table:
    return pq.read_table(path)


def event_curve_mints(population_path: Path) -> list[str]:
    table = pq.read_table(population_path)
    mask = pc.equal(table.column("event_curve_population"), True)
    return [
        str(m)
        for m in table.filter(mask).column("mint").to_pylist()
        if m
    ]


def update_price_target_flags(
    population_path: Path, price_capable_mints: set[str]
) -> dict[str, Any]:
    """Set price_target_population for mints with finite-price trade evidence."""
    table = pq.read_table(population_path)
    rows = table.to_pylist()
    n = 0
    for row in rows:
        mint = row.get("mint")
        capable = bool(mint in price_capable_mints)
        row["price_target_population"] = capable
        flags = dict(row.get("population_flags") or {})
        flags["price_target_population"] = capable
        row["population_flags"] = flags
        if capable:
            n += 1
    pq.write_table(pa.Table.from_pylist(rows), population_path)
    return {"price_target_population": n}
