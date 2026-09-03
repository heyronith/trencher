"""Cross-dataset joinability and overlap analysis."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from trencher.data.feasibility.concepts import SYSTEM_PROGRAM_WALLET


def _set_stats(a: set[str], b: set[str]) -> dict[str, Any]:
    inter = a & b
    return {
        "left": len(a),
        "right": len(b),
        "intersection": len(inter),
        "jaccard": (len(inter) / len(a | b)) if (a or b) else None,
        "coverage_of_left": (len(inter) / len(a)) if a else None,
        "coverage_of_right": (len(inter) / len(b)) if b else None,
    }


def load_pumpfun_mints(tokens_path: Path) -> set[str]:
    table = pq.read_table(tokens_path, columns=["mint"])
    return set(table.column("mint").to_pylist())


def load_pumpfun_creators(tokens_path: Path) -> set[str]:
    table = pq.read_table(tokens_path, columns=["creator"])
    return {c for c in table.column("creator").to_pylist() if c}


def load_graduated_mints(tokens_path: Path) -> set[str]:
    table = pq.read_table(tokens_path, columns=["mint", "graduated_at"])
    mints = table.column("mint").to_pylist()
    grads = table.column("graduated_at").to_pylist()
    return {m for m, g in zip(mints, grads, strict=True) if g is not None}


def load_migration_mints(path: Path) -> set[str]:
    table = pq.read_table(path, columns=["mint"])
    return set(table.column("mint").to_pylist())


def load_trade_wallets(trades_path: str | Path, *, max_batches: int | None = None) -> set[str]:
    path = Path(trades_path)
    if "*" in path.name or "?" in path.name:
        path = path.parent
    dataset = ds.dataset(path, format="parquet")
    wallets: set[str] = set()
    scanner = dataset.scanner(columns=["user_wallet"], batch_size=512_000)
    for i, batch in enumerate(scanner.to_batches()):
        wallets.update(batch.column(0).to_pylist())
        if max_batches is not None and i + 1 >= max_batches:
            break
    wallets.discard(None)
    wallets.discard(SYSTEM_PROGRAM_WALLET)
    return wallets


def load_red_pump_mints(launches_path: Path) -> set[str]:
    mints: set[str] = set()
    with gzip.open(launches_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            mints.add(json.loads(line)["mint"])
    return mints


def load_red_pump_graduated(outcomes_path: Path) -> set[str]:
    import csv

    mints: set[str] = set()
    with gzip.open(outcomes_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("outcome") or "").strip() == "GRADUATED":
                mints.add(row["mint"])
    return mints


def load_cohort_entities(cohorts_path: Path) -> dict[str, Any]:
    wallets: set[str] = set()
    mints: set[str] = set()
    n = 0
    with cohorts_path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            n += 1
            obj = json.loads(line)
            for w in obj.get("wallets") or []:
                if isinstance(w, str) and w:
                    wallets.add(w)
            for item in obj.get("mints_hit") or []:
                if isinstance(item, str) and item:
                    mints.add(item)
                elif isinstance(item, dict):
                    mint = item.get("mint")
                    if isinstance(mint, str) and mint:
                        mints.add(mint)
    return {"cohorts": n, "wallets": wallets, "mints": mints}


def compute_joinability(
    *,
    pumpfun_root: Path,
    red_pump_root: Path,
    red_cohort_root: Path,
    trade_wallet_max_batches: int | None = None,
) -> dict[str, Any]:
    tokens = pumpfun_root / "tokens.parquet"
    migrations = pumpfun_root / "migrations.parquet"
    trades_dir = pumpfun_root / "trades"
    launches = red_pump_root / "red_pump_2026_v1_launches.jsonl.gz"
    outcomes = red_pump_root / "red_pump_2026_v1_outcomes.csv.gz"
    cohorts = (
        red_cohort_root
        / "extracted"
        / "RED-COHORT-2026-v1"
        / "sniper_cohorts.jsonl"
    )

    pf_mints = load_pumpfun_mints(tokens)
    pf_creators = load_pumpfun_creators(tokens)
    pf_grad = load_graduated_mints(tokens)
    mig_mints = load_migration_mints(migrations)
    rp_mints = load_red_pump_mints(launches)
    rp_grad = load_red_pump_graduated(outcomes)
    cohort = load_cohort_entities(cohorts)
    trade_wallets = load_trade_wallets(
        trades_dir, max_batches=trade_wallet_max_batches
    )

    postgard = pq.read_table(
        pumpfun_root / "postgard_outcomes.parquet", columns=["mint"]
    )
    postgard_mints = set(postgard.column("mint").to_pylist())

    return {
        "token_mint_overlap": {
            "pumpfun_vs_red_pump": _set_stats(pf_mints, rp_mints),
            "pumpfun_graduated_vs_migrations": _set_stats(pf_grad, mig_mints),
            "pumpfun_graduated_vs_postgard_outcomes": _set_stats(pf_grad, postgard_mints),
            "pumpfun_graduated_vs_red_pump_graduated": _set_stats(pf_grad, rp_grad),
            "red_pump_graduated_vs_migrations": _set_stats(rp_grad, mig_mints),
            "cohort_mints_vs_pumpfun": _set_stats(cohort["mints"], pf_mints),
            "cohort_mints_vs_red_pump": _set_stats(cohort["mints"], rp_mints),
        },
        "wallet_overlap": {
            "cohort_wallets_vs_trade_wallets": _set_stats(
                cohort["wallets"], trade_wallets
            ),
            "trade_wallets_scanned": len(trade_wallets),
            "trade_wallet_scan_capped": trade_wallet_max_batches is not None,
        },
        "creator_overlap": {
            "unique_creators": len(pf_creators),
            "creators_also_in_trade_wallets": len(pf_creators & trade_wallets),
            "creators_also_in_cohort_wallets": len(pf_creators & cohort["wallets"]),
        },
        "cohort_summary": {
            "cohorts": cohort["cohorts"],
            "unique_wallets": len(cohort["wallets"]),
            "unique_mints_hit": len(cohort["mints"]),
        },
    }
