"""Quality checks including quantitative KNOWN_ISSUES validation."""

from __future__ import annotations

import gzip
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from trencher.data.feasibility.concepts import SYSTEM_PROGRAM_WALLET

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parquet_schema_report(path: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(path)
    cols = []
    for field in pf.schema_arrow:
        cols.append({"name": field.name, "type": str(field.type)})
    # Timestamp bounds from statistics when available.
    bounds: dict[str, dict[str, str | None]] = {}
    names = [c["name"] for c in cols]
    for name in names:
        if not any(
            tok in name.lower()
            for tok in ("time", "at", "timestamp", "detected", "graduated", "migrated")
        ):
            continue
        idx = names.index(name)
        mins: list[Any] = []
        maxs: list[Any] = []
        for rg in range(pf.metadata.num_row_groups):
            stats = pf.metadata.row_group(rg).column(idx).statistics
            if stats is None or not stats.has_min_max:
                continue
            mins.append(stats.min)
            maxs.append(stats.max)
        if mins and maxs:
            bounds[name] = {"min": str(min(mins)), "max": str(max(maxs))}
    return {
        "path": str(path),
        "rows": pf.metadata.num_rows,
        "columns": cols,
        "timestamp_bounds": bounds,
    }


def audit_tokens_table(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)
    n = table.num_rows
    mint = table.column("mint")
    n_unique_mint = len(pc.unique(mint))
    dup_mints = int(n - n_unique_mint)

    mayhem = int(pc.sum(table.column("is_mayhem_mode").cast("int64")).as_py() or 0)
    suspect_col = table.column("top10_pct_suspect")
    # bool/null object-like; cast carefully
    suspect_true = 0
    suspect_null = 0
    for v in suspect_col.to_pylist():
        if v is True:
            suspect_true += 1
        elif v is None:
            suspect_null += 1
    graduated = int(n - pc.sum(pc.is_null(table.column("graduated_at")).cast("int64")).as_py())
    training_ready = int(pc.sum(table.column("is_training_ready").cast("int64")).as_py() or 0)
    supply_bug = int(pc.sum(table.column("supply_bug_corrected").cast("int64")).as_py() or 0)

    # Uncorrected concentration > 100% among mayhem (should be rare after fix on corrected cols)
    top10 = table.column("initial_top10_pct")
    top10_corr = table.column("initial_top10_pct_corrected")
    bad_raw = int(pc.sum(pc.greater(top10, 100.0).cast("int64")).as_py() or 0)
    bad_corr = int(pc.sum(pc.greater(top10_corr, 100.0).cast("int64")).as_py() or 0)

    # Leakage fields null rates by graduation status
    leakage = {}
    grad_mask = pc.invert(pc.is_null(table.column("graduated_at")))
    non_grad_mask = pc.is_null(table.column("graduated_at"))
    for col in ("entry_price_20s_usd", "entry_price_30s_usd", "entry_price_1m_usd"):
        series = table.column(col)
        nulls = pc.is_null(series)
        g_n = int(pc.sum(grad_mask.cast("int64")).as_py() or 0)
        ng_n = int(pc.sum(non_grad_mask.cast("int64")).as_py() or 0)
        g_null = int(pc.sum(pc.and_(nulls, grad_mask).cast("int64")).as_py() or 0)
        ng_null = int(pc.sum(pc.and_(nulls, non_grad_mask).cast("int64")).as_py() or 0)
        leakage[col] = {
            "graduated_null_rate": (g_null / g_n) if g_n else None,
            "non_graduated_null_rate": (ng_null / ng_n) if ng_n else None,
        }

    detected = table.column("detected_at")
    return {
        "rows": n,
        "unique_mints": n_unique_mint,
        "duplicate_mint_rows": dup_mints,
        "mayhem_tokens": mayhem,
        "mayhem_rate": mayhem / n if n else None,
        "top10_pct_suspect_true": suspect_true,
        "top10_pct_suspect_null": suspect_null,
        "supply_bug_corrected_true": supply_bug,
        "graduated_tokens": graduated,
        "is_training_ready_true": training_ready,
        "initial_top10_pct_gt_100": bad_raw,
        "initial_top10_pct_corrected_gt_100": bad_corr,
        "leakage_entry_price_null_rates": leakage,
        "detected_at_min": str(pc.min(detected).as_py()),
        "detected_at_max": str(pc.max(detected).as_py()),
        "known_issues_checks": {
            "1.1_mayhem_count_near_109561": abs(mayhem - 109561) <= 5,
            "2.1_suspect_near_1486": abs(suspect_true - 1486) <= 5,
            "7_entry_price_leakage_pattern": (
                leakage["entry_price_1m_usd"]["graduated_null_rate"] is not None
                and leakage["entry_price_1m_usd"]["graduated_null_rate"] < 0.05
                and leakage["entry_price_1m_usd"]["non_graduated_null_rate"] is not None
                and leakage["entry_price_1m_usd"]["non_graduated_null_rate"] > 0.8
            ),
        },
    }


def audit_trades_quality(trades_path: str | Path) -> dict[str, Any]:
    """Stream trade shards for KNOWN_ISSUES quantitative checks."""
    path = Path(trades_path)
    if "*" in path.name or "?" in path.name:
        path = path.parent
    dataset = ds.dataset(path, format="parquet")
    total = 0
    null_price = 0
    null_sol = 0
    system_rows = 0
    curve_oob = 0
    inconsistent = 0
    checked_consistency = 0
    mints_with_null_price: set[str] = set()
    mints_total: set[str] = set()
    day_counts: Counter[str] = Counter()
    day_null_price: Counter[str] = Counter()

    cols = [
        "mint",
        "event_time",
        "sol_amount",
        "token_amount",
        "price_sol",
        "user_wallet",
        "curve_pct_depleted",
    ]
    scanner = dataset.scanner(columns=cols, batch_size=256_000)
    for batch in scanner.to_batches():
        n = batch.num_rows
        total += n
        mint = batch.column("mint")
        mints_total.update(mint.to_pylist())

        price = batch.column("price_sol")
        sol = batch.column("sol_amount")
        tok = batch.column("token_amount")
        wallet = batch.column("user_wallet")
        curve = batch.column("curve_pct_depleted")
        et = batch.column("event_time")

        price_null = pc.is_null(price)
        sol_null = pc.is_null(sol)
        null_price += int(pc.sum(price_null.cast("int64")).as_py() or 0)
        null_sol += int(pc.sum(sol_null.cast("int64")).as_py() or 0)

        # whole-token null-price tracking (approx via batch)
        if int(pc.sum(price_null.cast("int64")).as_py() or 0):
            for m, is_null in zip(mint.to_pylist(), price_null.to_pylist(), strict=True):
                if is_null:
                    mints_with_null_price.add(m)

        system_rows += int(
            pc.sum(pc.equal(wallet, SYSTEM_PROGRAM_WALLET).cast("int64")).as_py() or 0
        )

        # curve out of [0,100]
        curve_valid = pc.is_valid(curve)
        oob = pc.and_(
            curve_valid,
            pc.or_(pc.less(curve, 0.0), pc.greater(curve, 100.0)),
        )
        curve_oob += int(pc.sum(oob.cast("int64")).as_py() or 0)

        # sol_amount consistency where all present and token*price != 0
        present = pc.and_(pc.and_(pc.is_valid(sol), pc.is_valid(tok)), pc.is_valid(price))
        denom = pc.multiply(tok, price)
        nonzero = pc.and_(present, pc.not_equal(denom, 0))
        ratio = pc.divide(sol, denom)
        bad = pc.and_(
            nonzero,
            pc.or_(pc.less(ratio, 0.01), pc.greater(ratio, 100.0)),
        )
        inconsistent += int(pc.sum(bad.cast("int64")).as_py() or 0)
        checked_consistency += int(pc.sum(nonzero.cast("int64")).as_py() or 0)

        # daily null-price rates
        times = et.to_pylist()
        nulls = price_null.to_pylist()
        for ts, is_null in zip(times, nulls, strict=True):
            if ts is None:
                continue
            day = str(ts)[:10]
            day_counts[day] += 1
            if is_null:
                day_null_price[day] += 1

    jul3 = day_counts.get("2026-07-03", 0)
    day_null_rates = {
        day: (day_null_price[day] / day_counts[day]) if day_counts[day] else None
        for day in sorted(day_counts)
    }
    return {
        "rows": total,
        "unique_mints_in_trades": len(mints_total),
        "null_price_sol_rows": null_price,
        "null_price_sol_rate": null_price / total if total else None,
        "null_sol_amount_rows": null_sol,
        "null_sol_amount_rate": null_sol / total if total else None,
        "mints_with_any_null_price": len(mints_with_null_price),
        "system_program_trade_rows": system_rows,
        "system_program_trade_rate": system_rows / total if total else None,
        "curve_pct_depleted_out_of_0_100": curve_oob,
        "curve_pct_depleted_oob_rate": curve_oob / total if total else None,
        "sol_amount_inconsistent_rows": inconsistent,
        "sol_amount_inconsistent_rate_vs_checked": (
            inconsistent / checked_consistency if checked_consistency else None
        ),
        "checked_consistency_rows": checked_consistency,
        "jul3_trade_rows": jul3,
        "day_null_price_rates_sample": {
            k: day_null_rates[k]
            for k in ("2026-06-09", "2026-06-10", "2026-07-02", "2026-07-03", "2026-07-04")
            if k in day_null_rates
        },
        "known_issues_checks": {
            "3.1_inconsistency_near_3_38pct": (
                inconsistent / total if total else None,
                abs((inconsistent / total) - 0.0338) < 0.02 if total else False,
            ),
            "3.2_null_price_near_7_03pct": (
                null_price / total if total else None,
                abs((null_price / total) - 0.0703) < 0.02 if total else False,
            ),
            "3.3_system_program_near_19_92pct": (
                system_rows / total if total else None,
                abs((system_rows / total) - 0.1992) < 0.02 if total else False,
            ),
            "3.4_curve_oob_near_3_62pct": (
                curve_oob / total if total else None,
                abs((curve_oob / total) - 0.0362) < 0.02 if total else False,
            ),
            "5.5_jul3_outage_zero_or_near_zero_trades": jul3 == 0,
        },
    }


def audit_snapshots_quality(path: Path) -> dict[str, Any]:
    dataset = ds.dataset(str(path), format="parquet")
    total = 0
    curve_oob = 0
    scanner = dataset.scanner(columns=["curve_pct_depleted_eob"], batch_size=512_000)
    for batch in scanner.to_batches():
        total += batch.num_rows
        curve = batch.column(0)
        valid = pc.is_valid(curve)
        oob = pc.and_(valid, pc.or_(pc.less(curve, 0.0), pc.greater(curve, 100.0)))
        curve_oob += int(pc.sum(oob.cast("int64")).as_py() or 0)
    return {
        "rows": total,
        "curve_pct_depleted_eob_out_of_0_100": curve_oob,
        "curve_pct_depleted_eob_oob_rate": curve_oob / total if total else None,
        "known_issues_checks": {
            "3.4_snapshot_oob_near_2_71pct": (
                curve_oob / total if total else None,
                abs((curve_oob / total) - 0.0271) < 0.015 if total else False,
            )
        },
    }


def audit_migrations(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)
    n = table.num_rows
    pools = table.column("pool_address").to_pylist()
    synthetic = sum(1 for p in pools if p == "synthetic_graduation_queue")
    backfilled = sum(1 for p in pools if p == "backfilled_from_pumpswap_trade")
    return {
        "rows": n,
        "unique_mints": len(pc.unique(table.column("mint"))),
        "synthetic_graduation_queue": synthetic,
        "backfilled_from_pumpswap_trade": backfilled,
        "synthetic_or_backfilled_rate": (synthetic + backfilled) / n if n else None,
        "known_issues_checks": {
            "4.55_synthetic_near_1380": abs(synthetic - 1380) <= 20,
            "4.55_backfilled_near_69": abs(backfilled - 69) <= 5,
        },
    }


def audit_postgard_outcomes(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)
    labels = Counter(table.column("outcome_label").to_pylist())
    rugs = int(pc.sum(table.column("rug_detected").cast("int64")).as_py() or 0)
    return {
        "rows": table.num_rows,
        "unique_mints": len(pc.unique(table.column("mint"))),
        "outcome_label_counts": dict(labels),
        "has_rug_label_value": "rug" in labels,
        "rug_detected_true": rugs,
        "known_issues_checks": {
            "4.5_no_rug_outcome_label": "rug" not in labels,
        },
    }


def audit_red_pump(launches_path: Path, outcomes_path: Path) -> dict[str, Any]:
    mint_counts: Counter[str] = Counter()
    launch_rows = 0
    with gzip.open(launches_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            launch_rows += 1
            obj = json.loads(line)
            mint_counts[obj["mint"]] += 1
    dup_mints = sum(1 for _, c in mint_counts.items() if c > 1)

    import csv

    outcome_counts: Counter[str] = Counter()
    outcome_rows = 0
    mint_outcomes: dict[str, set[str]] = {}
    with gzip.open(outcomes_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            outcome_rows += 1
            outcome = (row.get("outcome") or "").strip()
            outcome_counts[outcome] += 1
            mint = row.get("mint") or ""
            mint_outcomes.setdefault(mint, set()).add(outcome)
    both = sum(
        1
        for outcomes in mint_outcomes.values()
        if "GRADUATED" in outcomes and "TIMEOUT" in outcomes
    )
    standard = outcome_counts.get("GRADUATED", 0) + outcome_counts.get("TIMEOUT", 0)
    nonstandard = outcome_rows - standard
    return {
        "launch_rows": launch_rows,
        "unique_launch_mints": len(mint_counts),
        "duplicate_launch_mints": dup_mints,
        "outcome_rows": outcome_rows,
        "outcome_counts": dict(outcome_counts),
        "nonstandard_or_blank_outcomes": nonstandard,
        "mints_with_both_graduated_and_timeout": both,
        "known_issues_checks": {
            "dup_launches_near_18": abs(dup_mints - 18) <= 5,
            "nonstandard_outcomes_near_215": abs(nonstandard - 215) <= 30,
            "both_grad_timeout_near_14": abs(both - 14) <= 5,
        },
    }


def load_known_issues_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")
