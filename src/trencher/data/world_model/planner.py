"""Extraction planner for world-model historical layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trencher.data.world_model.constants import (
    CORPUS_END,
    CORPUS_START,
    GECKO_SOL_USDC_POOL,
    SYNTHETIC_POOLS,
)
from trencher.data.world_model.io import utc_now, write_json


def _count_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    return pq.ParquetFile(path).metadata.num_rows


def build_extraction_plan(
    *,
    external_root: Path,
    probe: dict[str, Any],
    rich_subset_max_mints: int = 2721,
    rpc_max_signatures: int = 25_000,
    gecko_max_pools: int = 2721,
) -> dict[str, Any]:
    pump = external_root / "pumpfun_memecoin_corpus"
    migrations = pump / "migrations.parquet"
    real_pools = 0
    synthetic = 0
    if migrations.exists():
        table = pq.read_table(migrations, columns=["pool_address"])
        for p in table.column(0).to_pylist():
            if p in SYNTHETIC_POOLS or not p:
                synthetic += 1
            else:
                real_pools += 1

    bq_ok = probe.get("bigquery", {}).get("decision") == "usable_with_partition_filters"
    sol_decision = probe.get("solarchive", {}).get("decision")
    gecko_ok = probe.get("geckoterminal", {}).get("decision") in {
        "usable",
        "degraded_or_rate_limited",
        "skipped_for_ci",
    }

    plan: dict[str, Any] = {
        "generated_at": utc_now(),
        "anchor_universe": {
            "source": "pumpfun_memecoin_corpus.tokens",
            "token_rows": _count_rows(pump / "tokens.parquet"),
            "trade_rows": None,
            "graduated_rows": None,
            "window": {
                "start": CORPUS_START.isoformat(),
                "end": CORPUS_END.isoformat(),
            },
        },
        "layers": {
            "transaction_microstructure": {
                "primary": "rpc_getTransaction" if not bq_ok else "bigquery_Transactions",
                "fallback": ["rpc_getTransaction", "pumpfun.trades.tx_signature"],
                "solarchive": sol_decision,
                "scope": (
                    f"rich multimodal subset up to {rpc_max_signatures} signatures "
                    "prioritized by graduated+real-pool mints"
                ),
                "fields": [
                    "signature",
                    "slot",
                    "index",
                    "success",
                    "fee",
                    "compute_units_consumed",
                    "account_keys",
                    "pre_balances",
                    "post_balances",
                    "log_messages",
                    "instructions",
                    "priority_fee_derivable",
                ],
                "actions": [
                    "extract_signature_universe_from_pump_trades",
                    "rpc_enrich_rich_subset" if not bq_ok else "bq_enrich_by_signature",
                ],
            },
            "token_ownership_transfers": {
                "primary": "bigquery_Token_Transfers" if bq_ok else "derived_from_pump_trades",
                "fallback": ["pumpfun.trades token/sol amounts", "tokens concentration corrected"],
                "scope": "full trade-derived ownership flows; BQ SPL transfers if auth",
                "actions": [
                    "build_wallet_mint_flow_from_trades",
                    "export_creator_holdings_from_tokens",
                    "bq_token_transfers_for_rich_mints" if bq_ok else "flag_bq_unavailable",
                ],
            },
            "wallet_history_graph": {
                "primary": "pumpfun.trades + tokens.creator + red_cohort",
                "funding_primary": "bq_or_rpc_balance_deltas",
                "actions": [
                    "wallet_token_participation_edges",
                    "creator_token_edges",
                    "co_buy_edges_rich_subset",
                    "cohort_membership_edges",
                    "funding_edges_from_rpc_balance_changes",
                ],
            },
            "global_market_network_context": {
                "primary": "pumpfun_aggregates + geckoterminal_sol_usdc",
                "bq_blocks": bq_ok,
                "actions": [
                    "daily_launch_trade_graduation_aggregates",
                    f"gecko_ohlcv_sol_usdc:{GECKO_SOL_USDC_POOL}",
                    "rpc_block_samples_from_enriched_slots",
                ],
            },
            "post_graduation": {
                "primary": "pumpfun.postgard_*",
                "supplement": "geckoterminal_pool_ohlcv",
                "real_pools": real_pools,
                "synthetic_or_null_pools": synthetic,
                "gecko_max_pools": gecko_max_pools,
                "actions": [
                    "register_postgard_as_authoritative_postgrad",
                    "gecko_ohlcv_for_real_pools",
                    "gecko_search_for_synthetic_pool_discovery",
                ],
            },
            "execution_state": {
                "primary": "pumpfun.trades/snapshots curve reserves",
                "postgrad_liquidity": "postgard_snapshots + gecko",
                "actions": [
                    "export_curve_execution_snapshots",
                    "flag_fee_regime_from_known_pump_schedule",
                ],
            },
        },
        "rich_subset_policy": {
            "definition": (
                "Graduated tokens with real migration pool_address, "
                f"capped at {rich_subset_max_mints} mints"
            ),
            "rpc_max_signatures": rpc_max_signatures,
            "rationale": (
                "Broad event/curve corpus remains full PumpFun population; "
                "microstructure/graph funding enrichment concentrates on richest multimodal slice."
            ),
        },
        "source_decisions": {
            "bigquery": probe.get("bigquery", {}).get("decision"),
            "solarchive": sol_decision,
            "geckoterminal": probe.get("geckoterminal", {}).get("decision"),
            "geckoterminal_planned": gecko_ok,
        },
        "do_not": [
            "download_full_solarchive_archive",
            "issue_unpartitioned_bigquery_scans",
            "synthesize_missing_values",
        ],
    }

    trades_dir = pump / "trades"
    if trades_dir.exists():
        total = 0
        for f in trades_dir.glob("*.parquet"):
            total += pq.ParquetFile(f).metadata.num_rows
        plan["anchor_universe"]["trade_rows"] = total

    tokens = pump / "tokens.parquet"
    if tokens.exists():
        t = pq.read_table(tokens, columns=["graduated_at"])
        plan["anchor_universe"]["graduated_rows"] = sum(
            1 for g in t.column(0).to_pylist() if g is not None
        )

    return plan


def write_plan(plan: dict[str, Any], manifest_dir: Path) -> Path:
    path = manifest_dir / "extraction_plan.json"
    write_json(path, plan)
    return path
