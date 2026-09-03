"""Coverage report and encoder readiness assessment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trencher.data.world_model.io import utc_now, write_json


def _rows(path: Path) -> int:
    if not path.exists():
        return 0
    return int(pq.ParquetFile(path).metadata.num_rows)


def build_coverage_report(
    *,
    world_root: Path,
    probe: dict[str, Any],
    plan: dict[str, Any],
    local: dict[str, Any],
    gecko: dict[str, Any],
    rpc: dict[str, Any],
    bq: dict[str, Any],
) -> dict[str, Any]:
    graph = world_root / "graph"
    post = local.get("layers", {}).get("postgard", {})
    rich_mints = local.get("layers", {}).get("rich_subset", {}).get("mint_count", 0)
    wt_edges = local.get("layers", {}).get("wallet_graph", {}).get("wallet_token_edges", 0)
    creator_edges = local.get("layers", {}).get("wallet_graph", {}).get("creator_token_edges", 0)
    cobuy = local.get("layers", {}).get("wallet_graph", {}).get("cobuy_edges", 0)
    cohort = local.get("layers", {}).get("wallet_graph", {}).get("cohort_member_edges", 0)
    funding = _rows(graph / "edges_funding_rpc.parquet")
    wallets = local.get("layers", {}).get("wallet_graph", {}).get(
        "unique_wallets_in_wallet_token", 0
    )
    exec_rows = local.get("layers", {}).get("execution", {}).get("rows", 0)
    exec_mints = local.get("layers", {}).get("execution", {}).get("mints", 0)
    global_days = local.get("layers", {}).get("global_context", {}).get(
        "corpus_days_present", 0
    )
    gecko_sol = gecko.get("sol_usdc", {}) if not gecko.get("skipped") else {}
    gecko_pools = gecko.get("graduated_pools", {}) if not gecko.get("skipped") else {}
    match_rate = rpc.get("signature_match_rate")
    rpc_found = rpc.get("found", 0)
    bq_bytes = bq.get("bytes_processed_actual", 0)
    sol_bytes = 0  # never downloaded

    layers = [
        {
            "layer": "transaction_microstructure",
            "tokens_covered": local.get("layers", {})
            .get("signature_targets", {})
            .get("unique_mints", 0),
            "events_rows": rpc.get("total_enriched") or 0,
            "date_coverage": "2026-06-05..2026-07-14 (subset)",
            "quality": (
                "high"
                if match_rate and match_rate > 0.9
                else "partial" if rpc_found else "signature_index_only"
            ),
            "source": "pumpfun.trades.tx_signature + solana_rpc.getTransaction",
        },
        {
            "layer": "token_ownership_transfers",
            "tokens_covered": plan.get("anchor_universe", {}).get("token_rows"),
            "events_rows": wt_edges,
            "date_coverage": "corpus window via trades",
            "quality": (
                "trade_derived_only"
                if bq.get("decision") in {"auth_missing", "skipped_for_ci", "skipped"}
                else "bq_plus_trades"
            ),
            "source": "pumpfun.trades (+ BigQuery Token Transfers if auth)",
        },
        {
            "layer": "wallet_history_graph",
            "tokens_covered": rich_mints,
            "events_rows": wt_edges + creator_edges + cobuy + cohort + funding,
            "date_coverage": "corpus window",
            "quality": "strong_participation_partial_funding",
            "source": "pumpfun + red_cohort + rpc balance-delta proxies",
        },
        {
            "layer": "global_market_network_context",
            "tokens_covered": plan.get("anchor_universe", {}).get("token_rows"),
            "events_rows": global_days + int(gecko_sol.get("corpus_window_rows") or 0),
            "date_coverage": "2026-06-05..2026-07-14",
            "quality": "high_pump_aggregates + gecko_sol_usdc"
            if gecko_sol.get("covers_corpus")
            else "pump_aggregates_partial_gecko",
            "source": "pumpfun daily aggregates + geckoterminal SOL/USDC",
        },
        {
            "layer": "post_graduation",
            "tokens_covered": post.get("outcomes_rows") or 0,
            "events_rows": post.get("snapshots_rows") or 0,
            "date_coverage": "graduated subset",
            "quality": "high_local_postgard",
            "source": "pumpfun.postgard_* (+ gecko pool OHLCV supplement)",
        },
        {
            "layer": "execution_state",
            "tokens_covered": exec_mints,
            "events_rows": exec_rows,
            "date_coverage": "rich subset pre-grad",
            "quality": "high_curve_reserves",
            "source": "pumpfun.trades/snapshots curve fields",
        },
    ]

    # Encoder readiness
    event_ready = (plan.get("anchor_universe", {}).get("trade_rows") or 0) > 1_000_000
    wallet_ready = wt_edges > 100_000 and wallets > 10_000
    # Funding is partial without BQ — still enough for participation graph training
    wallet_status = "READY" if wallet_ready else "PARTIAL"
    if funding == 0 and bq.get("decision") == "auth_missing":
        wallet_status = "PARTIAL"

    token_ready = event_ready and exec_mints > 100
    global_ready = global_days >= 30 and (
        gecko_sol.get("covers_corpus") or gecko.get("skipped") is not True
    )
    # If gecko skipped in live run without coverage, still have pump aggregates
    global_status = "READY" if global_days >= 30 else "PARTIAL"

    postgrad_ok = (post.get("outcomes_rows") or 0) > 1000
    fusion_ok = event_ready and token_ready and global_days >= 30 and postgrad_ok
    # Require meaningful multimodal overlap for fusion
    fusion_status = (
        "READY"
        if fusion_ok and rpc_found > 100
        else ("PARTIAL" if fusion_ok else "BLOCKED")
    )
    # Prediction heads need labels + paths — feasibility already proved labels
    heads_status = "READY" if event_ready and postgrad_ok else "PARTIAL"

    encoders = {
        "Event Encoder": "READY" if event_ready else "BLOCKED",
        "Wallet/Graph Encoder": wallet_status,
        "Token/Ownership/Curve Encoder": "READY" if token_ready else "PARTIAL",
        "Global Market Encoder": (
            global_status if global_ready or global_days >= 30 else "PARTIAL"
        ),
        "Fusion/world-state model": fusion_status,
        "prediction heads": heads_status,
    }

    irrecoverable = [
        "Full SPL/Token-2022 transfer ledger without BigQuery auth or Solarchive 2026",
        (
            "Parsed System-program funding graph for all wallets "
            "(only RPC balance-delta proxies on subset)"
        ),
        "Exact historical priority-fee/Jito tip fields when RPC meta lacks them",
        "Solarchive txs/accounts for June–July 2026 (index ends 2025-12-05)",
        "Real pool addresses for synthetic_graduation_queue / null migrations without discovery",
        "Row-level Pump fee bps history (protocol metadata only)",
    ]

    funding_wallets = 0
    funding_path = graph / "edges_funding_rpc.parquet"
    if funding_path.exists() and funding:
        ft = pq.read_table(funding_path, columns=["src", "dst"])
        funding_wallets = len(
            set(ft.column("src").to_pylist()) | set(ft.column("dst").to_pylist())
        )

    report = {
        "generated_at": utc_now(),
        "layers": layers,
        "metrics": {
            "transaction_signature_match_rate": match_rate,
            "mints_with_transfer_or_ownership_history": plan.get(
                "anchor_universe", {}
            ).get("token_rows"),
            "wallet_mint_flow_edges": wt_edges,
            "wallets_with_usable_prior_history": wallets,
            "wallets_with_real_funding_edges": funding_wallets,
            "funding_edge_count": funding,
            "graph_edge_counts_by_type": {
                "wallet_token": wt_edges,
                "creator_token": creator_edges,
                "cobuy": cobuy,
                "cohort_member": cohort,
                "sol_balance_delta_proxy": funding,
            },
            "tokens_with_postgrad_history": post.get("outcomes_rows"),
            "network_context_corpus_days": global_days,
            "execution_state_mints": exec_mints,
            "execution_state_rows": exec_rows,
            "bigquery_bytes_processed": bq_bytes,
            "solarchive_bytes_downloaded": sol_bytes,
            "gecko_api_calls": gecko.get("api_calls", 0),
            "gecko_pools_completed": gecko_pools.get("pools_completed_total", 0),
            "rpc_transactions_found": rpc_found,
            "rich_subset_mints": rich_mints,
        },
        "source_probe_decisions": {
            "bigquery": probe.get("bigquery", {}).get("decision"),
            "solarchive": probe.get("solarchive", {}).get("decision"),
            "geckoterminal": probe.get("geckoterminal", {}).get("decision"),
        },
        "encoders": encoders,
        "irrecoverable_fields": irrecoverable,
        "world_model_dataset_ready": all(
            encoders[k] == "READY"
            for k in (
                "Event Encoder",
                "Wallet/Graph Encoder",
                "Token/Ownership/Curve Encoder",
                "Global Market Encoder",
            )
        )
        and encoders["Fusion/world-state model"] in {"READY", "PARTIAL"},
    }
    return report


def render_world_model_md(report: dict[str, Any]) -> str:
    lines = [
        "# World Model Historical Data",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        "## Coverage matrix",
        "",
        "| World-model layer | tokens covered | events/rows | date coverage | quality | source |",
        "| ----------------- | -------------: | ----------: | ------------- | ------- | ------ |",
    ]
    for row in report["layers"]:
        lines.append(
            f"| {row['layer']} | {row['tokens_covered']} | {row['events_rows']} | "
            f"{row['date_coverage']} | {row['quality']} | {row['source']} |"
        )
    m = report["metrics"]
    lines.extend(
        [
            "",
            "## Key metrics",
            "",
            f"- Transaction-signature match rate: {m.get('transaction_signature_match_rate')}",
            (
                f"- Wallet–mint ownership/flow edges: "
                f"{m.get('wallet_mint_flow_edges')}"
            ),
            (
                f"- Tokens with trade-derived ownership history: "
                f"{m.get('mints_with_transfer_or_ownership_history')}"
            ),
            f"- Wallets with usable prior history: {m.get('wallets_with_usable_prior_history')}",
            f"- Wallets with funding-proxy edges: {m.get('wallets_with_real_funding_edges')}",
            f"- Funding proxy edges: {m.get('funding_edge_count')}",
            f"- Graph edge counts: `{m.get('graph_edge_counts_by_type')}`",
            f"- Tokens with post-grad history: {m.get('tokens_with_postgrad_history')}",
            f"- Network-context corpus days: {m.get('network_context_corpus_days')}",
            f"- Execution-state coverage: {m.get('execution_state_mints')} mints / "
            f"{m.get('execution_state_rows')} rows",
            f"- BigQuery bytes processed: {m.get('bigquery_bytes_processed')}",
            f"- Solarchive bytes downloaded: {m.get('solarchive_bytes_downloaded')}",
            f"- Gecko API calls / pools completed: {m.get('gecko_api_calls')} / "
            f"{m.get('gecko_pools_completed')}",
            f"- RPC transactions found: {m.get('rpc_transactions_found')}",
            "",
            "## Encoder readiness",
            "",
        ]
    )
    for name, status in report["encoders"].items():
        lines.append(f"- **{name}**: `{status}`")
    lines.extend(
        [
            "",
            f"**World-model dataset ready (core encoders):** "
            f"{report['world_model_dataset_ready']}",
            "",
            "## Remaining irrecoverable / blocked without new sources",
            "",
        ]
    )
    lines.extend(f"- {x}" for x in report["irrecoverable_fields"])
    lines.extend(
        [
            "",
            "## Source decisions",
            "",
            f"- BigQuery: `{report['source_probe_decisions'].get('bigquery')}`",
            f"- Solarchive: `{report['source_probe_decisions'].get('solarchive')}`",
            f"- GeckoTerminal: `{report['source_probe_decisions'].get('geckoterminal')}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(manifest_dir: Path, report: dict[str, Any]) -> None:
    write_json(manifest_dir / "coverage_report.json", report)
    (manifest_dir / "WORLD_MODEL_DATA.md").write_text(
        render_world_model_md(report), encoding="utf-8"
    )
