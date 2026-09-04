"""Ownership evidence levels and stratified fusion coverage (single source of truth)."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.io import ensure_dir, utc_now, write_json

OwnershipLevel = Literal[
    "ledger_full_or_reconstructable",
    "rpc_balance_observed",
    "trade_derived_proxy",
    "missing",
]

OWNERSHIP_LEVEL_ORDER = (
    "ledger_full_or_reconstructable",
    "rpc_balance_observed",
    "trade_derived_proxy",
    "missing",
)


def _mint_set(path: Path, column: str = "mint") -> set[str]:
    if not path.exists() or not path.is_file():
        return set()
    try:
        if pq.ParquetFile(path).metadata.num_rows == 0:
            return set()
        return set(pq.read_table(path, columns=[column]).column(0).to_pylist()) - {None}
    except Exception:  # noqa: BLE001
        return set()


def _mint_row_counts(path: Path, column: str = "mint") -> dict[str, int]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        if pq.ParquetFile(path).metadata.num_rows == 0:
            return {}
        counts: dict[str, int] = defaultdict(int)
        for m in pq.read_table(path, columns=[column]).column(0).to_pylist():
            if m:
                counts[m] += 1
        return dict(counts)
    except Exception:  # noqa: BLE001
        return {}


def classify_ownership_evidence(world_root: Path) -> dict[str, Any]:
    """Assign the strongest ownership evidence level per unbiased-subset mint."""
    subset_path = world_root / "universe" / "rich_subset_mints.parquet"
    subset = pq.read_table(subset_path)
    mints = subset.column("mint").to_pylist()
    S = set(mints)

    bq_paths = [
        world_root / "bigquery" / "token_transfers_unbiased_days.parquet",
        world_root / "bigquery" / "token_transfers_rich.parquet",
    ]
    bq_counts: dict[str, int] = defaultdict(int)
    for path in bq_paths:
        for mint, n in _mint_row_counts(path).items():
            if mint in S:
                bq_counts[mint] = max(bq_counts[mint], n)
    accounts = _mint_set(world_root / "bigquery" / "accounts_rich.parquet") & S

    rpc_path = world_root / "graph" / "edges_ownership_rpc_token_balances.parquet"
    rpc_counts = _mint_row_counts(rpc_path)
    rpc_sig_counts: dict[str, int] = defaultdict(int)
    if rpc_path.exists() and pq.ParquetFile(rpc_path).metadata.num_rows:
        t = pq.read_table(rpc_path, columns=["mint", "tx_signature"])
        by_mint: dict[str, set[str]] = defaultdict(set)
        for mint, sig in zip(
            t.column("mint").to_pylist(), t.column("tx_signature").to_pylist(), strict=True
        ):
            if mint in S and sig:
                by_mint[mint].add(sig)
        rpc_sig_counts = {m: len(v) for m, v in by_mint.items()}

    # Trade participation as proxy (wallet_token edges).
    trade_proxy = _mint_set(world_root / "graph" / "edges_wallet_token.parquet", "dst") & S

    rows = []
    for i, mint in enumerate(mints):
        bq_n = int(bq_counts.get(mint, 0))
        rpc_n = int(rpc_counts.get(mint, 0))
        rpc_sigs = int(rpc_sig_counts.get(mint, 0))
        has_accounts = mint in accounts
        # Ledger: Token Transfers history sufficient to reconstruct balances.
        # Account snapshots alone are not a transfer ledger.
        if bq_n > 0:
            level: OwnershipLevel = "ledger_full_or_reconstructable"
            source = "bigquery.Token Transfers"
            quality = (
                "sparse_ledger"
                if bq_n < 3 and not has_accounts
                else "ledger_with_account_snapshots"
                if has_accounts
                else "ledger"
            )
        elif rpc_n > 0:
            level = "rpc_balance_observed"
            source = "solana_rpc.getTransaction.meta.pre_post_token_balances"
            quality = (
                "single_sampled_tx"
                if rpc_sigs <= 1
                else "multi_sampled_tx_not_full_ledger"
            )
        elif mint in trade_proxy:
            level = "trade_derived_proxy"
            source = "pumpfun.trades / edges_wallet_token"
            quality = "trade_participation_only"
        else:
            level = "missing"
            source = None
            quality = "missing"

        rows.append(
            {
                "mint": mint,
                "launch_day": subset.column("launch_day")[i].as_py()
                if "launch_day" in subset.column_names
                else None,
                "launch_regime": subset.column("launch_regime")[i].as_py()
                if "launch_regime" in subset.column_names
                else None,
                "graduated": bool(subset.column("graduated")[i].as_py())
                if "graduated" in subset.column_names
                else None,
                "ownership_evidence_level": level,
                "ownership_source": source,
                "ownership_quality": quality,
                "bq_token_transfer_rows": bq_n,
                "has_account_snapshots": has_accounts,
                "rpc_token_balance_edges": rpc_n,
                "rpc_distinct_tx_signatures": rpc_sigs,
                "has_trade_participation": mint in trade_proxy,
                "availability_mask_ledger": level == "ledger_full_or_reconstructable",
                "availability_mask_rpc_balance": level
                in {"ledger_full_or_reconstructable", "rpc_balance_observed"},
                "availability_mask_any_genuine": level
                in {"ledger_full_or_reconstructable", "rpc_balance_observed"},
                "availability_mask_trade_proxy": level != "missing",
            }
        )

    out_dir = ensure_dir(world_root / "universe")
    path = out_dir / "ownership_evidence.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    counts = Counter(r["ownership_evidence_level"] for r in rows)
    summary = {
        "generated_at": utc_now(),
        "path": str(path),
        "subset_n": len(rows),
        "counts_by_level": {k: counts.get(k, 0) for k in OWNERSHIP_LEVEL_ORDER},
        "notes": [
            "ledger_full_or_reconstructable requires BigQuery Token Transfers "
            "for the mint (Accounts snapshots alone are insufficient).",
            "rpc_balance_observed is genuine on-chain meta but only for sampled txs "
            "(typically 1 signature/mint in current freeze) — not a full holder ledger.",
            "trade_derived_proxy is participation only and must not be treated as "
            "genuine ownership reconstruction.",
        ],
    }
    write_json(out_dir / "ownership_evidence_summary.json", summary)
    return {"rows": rows, "summary": summary, "path": path}


def build_fusion_coverage_truth(
    world_root: Path, *, manifest_dir: Path | None = None
) -> dict[str, Any]:
    """Deterministic stratified fusion coverage — single source of truth."""
    own = classify_ownership_evidence(world_root)
    own_by_mint = {r["mint"]: r for r in own["rows"]}

    subset = pq.read_table(world_root / "universe" / "rich_subset_mints.parquet")
    mints = subset.column("mint").to_pylist()
    S = set(mints)

    event = _mint_set(world_root / "execution" / "execution_curve_states.parquet") & S
    graph = _mint_set(world_root / "graph" / "edges_wallet_token.parquet", "dst") & S
    graph |= _mint_set(world_root / "graph" / "edges_creator_token.parquet", "dst") & S
    exec_m = event
    market_ok = True

    rt_path = world_root / "microstructure" / "rpc_transactions.parquet"
    micro: set[str] = set()
    if rt_path.exists() and pq.ParquetFile(rt_path).metadata.num_rows:
        rt = pq.read_table(rt_path, columns=["mint", "found"])
        mcol = rt.column("mint").to_pylist()
        fcol = rt.column("found").to_pylist()
        micro = {m for m, f in zip(mcol, fcol, strict=True) if m and f} & S

    native_path = world_root / "graph" / "edges_funding_native_sol.parquet"
    native_mints = _mint_set(native_path) & S

    rows = []
    for i, mint in enumerate(mints):
        o = own_by_mint[mint]
        has_event = mint in event
        has_graph = mint in graph
        has_micro = mint in micro
        has_exec = mint in exec_m
        has_market = market_ok
        core = has_event and has_graph and has_micro and has_exec and has_market
        any_own = bool(o["availability_mask_any_genuine"])
        ledger = bool(o["availability_mask_ledger"])
        rows.append(
            {
                "mint": mint,
                "launch_day": subset.column("launch_day")[i].as_py()
                if "launch_day" in subset.column_names
                else None,
                "launch_regime": subset.column("launch_regime")[i].as_py()
                if "launch_regime" in subset.column_names
                else None,
                "graduated": bool(subset.column("graduated")[i].as_py())
                if "graduated" in subset.column_names
                else None,
                "has_event_curve": has_event,
                "has_graph": has_graph,
                "has_microstructure": has_micro,
                "has_execution_state": has_exec,
                "has_market_context": has_market,
                "has_native_sol_history": mint in native_mints,
                "ownership_evidence_level": o["ownership_evidence_level"],
                "ownership_quality": o["ownership_quality"],
                "core_modalities_without_ownership": core,
                "fusion_any_genuine_ownership": core and any_own,
                "fusion_reconstructable_ownership": core and ledger,
                "availability_combo": "+".join(
                    k
                    for k, ok in [
                        ("event", has_event),
                        ("graph", has_graph),
                        ("micro", has_micro),
                        ("exec", has_exec),
                        ("market", has_market),
                        ("own_" + o["ownership_evidence_level"], True),
                        ("native_sol", mint in native_mints),
                    ]
                    if ok
                ),
            }
        )

    out_dir = ensure_dir(world_root / "universe")
    path = out_dir / "fusion_coverage_truth.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)

    def _count(flag: str) -> int:
        return sum(1 for r in rows if r[flag])

    def _grad_split(flag: str) -> dict[str, int]:
        sel = [r for r in rows if r[flag]]
        return {
            "n": len(sel),
            "graduated": sum(1 for r in sel if r.get("graduated")),
            "non_graduated": sum(1 for r in sel if not r.get("graduated")),
        }

    combos = Counter(r["availability_combo"] for r in rows)
    truth = {
        "generated_at": utc_now(),
        "source_of_truth": str(path),
        "definition_version": "fusion_coverage_v2_ownership_evidence",
        "prior_disagreement_note": (
            "coverage_report.json previously reported ~3116 while modality_matrix "
            "reported ~3132 due to a stale report snapshot; both used the same "
            "live intersection rule. This file is now the only authoritative count."
        ),
        "subset_n": len(rows),
        "counts": {
            "core_event_graph_micro_exec_market": _count(
                "core_modalities_without_ownership"
            ),
            "core_plus_any_genuine_ownership": _count("fusion_any_genuine_ownership"),
            "core_plus_reconstructable_ownership": _count(
                "fusion_reconstructable_ownership"
            ),
            "with_true_native_sol_history": _count("has_native_sol_history"),
            "core_plus_native_sol": sum(
                1
                for r in rows
                if r["core_modalities_without_ownership"] and r["has_native_sol_history"]
            ),
        },
        "composition": {
            "core": _grad_split("core_modalities_without_ownership"),
            "any_genuine_ownership": _grad_split("fusion_any_genuine_ownership"),
            "reconstructable_ownership": _grad_split(
                "fusion_reconstructable_ownership"
            ),
        },
        "ownership_evidence_counts": own["summary"]["counts_by_level"],
        "availability_combo_top": combos.most_common(30),
        "fusion_readiness_policy": {
            "full_world_state_fusion_requires": "core_plus_reconstructable_ownership",
            "rpc_balance_observed_usable_as": (
                "partial ownership evidence / modality mask, not full holder-state "
                "reconstruction"
            ),
            "do_not_equate": (
                "rpc_balance_observed != ledger_full_or_reconstructable"
            ),
        },
    }
    write_json(out_dir / "fusion_coverage_truth.json", truth)
    md = manifest_dir or Path("manifests") / "world_model_data"
    write_json(md / "fusion_coverage_truth.json", truth)
    return {"rows": rows, "truth": truth, "ownership": own, "path": path}
