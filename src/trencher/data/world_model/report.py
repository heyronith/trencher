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


def _mint_set(path: Path, column: str = "mint") -> set[str]:
    if not path.exists() or _rows(path) == 0:
        return set()
    try:
        return set(pq.read_table(path, columns=[column]).column(0).to_pylist()) - {None}
    except Exception:  # noqa: BLE001
        return set()


def _unique_count(path: Path, column: str) -> int:
    return len(_mint_set(path, column))


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
    bq_dir = world_root / "bigquery"
    post = local.get("layers", {}).get("postgard", {})
    rich_mints = local.get("layers", {}).get("rich_subset", {}).get("mint_count", 0)
    wt_edges = local.get("layers", {}).get("wallet_graph", {}).get("wallet_token_edges", 0)
    creator_edges = local.get("layers", {}).get("wallet_graph", {}).get("creator_token_edges", 0)
    cobuy = local.get("layers", {}).get("wallet_graph", {}).get("cobuy_edges", 0)
    cohort = local.get("layers", {}).get("wallet_graph", {}).get("cohort_member_edges", 0)
    rpc_funding = _rows(graph / "edges_funding_rpc.parquet")
    spl_transfers = _rows(graph / "edges_funding_spl.parquet")
    native_sol = _rows(graph / "edges_funding_native_sol.parquet")
    shared_funder = _rows(graph / "edges_shared_funder.parquet")
    shared_native = _rows(graph / "edges_shared_native_funder.parquet")
    creator_own = _rows(graph / "edges_creator_ownership_transfers.parquet")
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
    rpc_found = int(rpc.get("found") or 0)
    rpc_tx_path = world_root / "microstructure" / "rpc_transactions.parquet"
    micro_mints = set()
    if rpc_tx_path.exists() and _rows(rpc_tx_path):
        try:
            rt = pq.read_table(rpc_tx_path, columns=["mint", "found"])
            mints = rt.column("mint").to_pylist()
            founds = rt.column("found").to_pylist()
            micro_mints = {m for m, f in zip(mints, founds, strict=True) if m and f}
        except Exception:  # noqa: BLE001
            micro_mints = _mint_set(rpc_tx_path, "mint")
    micro_tokens = len(micro_mints) or int(
        local.get("layers", {}).get("signature_targets", {}).get("unique_mints") or 0
    )
    bq_bytes = int(bq.get("bytes_processed_actual") or 0)
    bq_billed = int(bq.get("bytes_billed_actual") or 0)
    sol_bytes = 0

    tt_path = bq_dir / "token_transfers_rich.parquet"
    tt_unbiased_path = bq_dir / "token_transfers_unbiased_days.parquet"
    accounts_path = bq_dir / "accounts_rich.parquet"
    blocks_path = bq_dir / "blocks_daily.parquet"
    true_spl_rows = _rows(tt_path)
    true_spl_mints = _unique_count(tt_path, "mint")
    unbiased_spl_mints = _unique_count(tt_unbiased_path, "mint") or true_spl_mints
    unbiased_spl_rows = _rows(tt_unbiased_path) or true_spl_rows
    true_spl_wallets = 0
    if true_spl_rows:
        try:
            t = pq.read_table(tt_path, columns=["source", "destination"])
            true_spl_wallets = len(
                (
                    set(t.column("source").to_pylist())
                    | set(t.column("destination").to_pylist())
                )
                - {None}
            )
        except Exception:  # noqa: BLE001
            derived = bq.get("artifacts", {}).get("transfer_derived_graph", {})
            true_spl_wallets = int(derived.get("unique_transfer_wallets") or 0)

    accounts_rows = _rows(accounts_path)
    accounts_mints = _unique_count(accounts_path, "mint")
    blocks_rows = _rows(blocks_path)

    spl_wallets = 0
    if spl_transfers and (graph / "edges_funding_spl.parquet").exists():
        ft = pq.read_table(graph / "edges_funding_spl.parquet", columns=["src", "dst"])
        spl_wallets = len(
            set(ft.column("src").to_pylist()) | set(ft.column("dst").to_pylist()) - {None}
        )
    native_wallets = 0
    if native_sol and (graph / "edges_funding_native_sol.parquet").exists():
        nt = pq.read_table(graph / "edges_funding_native_sol.parquet", columns=["src", "dst"])
        native_wallets = len(
            set(nt.column("src").to_pylist()) | set(nt.column("dst").to_pylist()) - {None}
        )

    subset_path = world_root / "universe" / "rich_subset_mints.parquet"
    subset_mints = _mint_set(subset_path)
    subset_grad = 0
    subset_nongrad = 0
    bias_proof = local.get("layers", {}).get("rich_subset", {}).get("bias_proof") or {}
    if subset_path.exists() and _rows(subset_path):
        try:
            st = pq.read_table(subset_path)
            if "graduated" in st.column_names:
                flags = st.column("graduated").to_pylist()
                subset_grad = sum(1 for g in flags if g)
                subset_nongrad = len(flags) - subset_grad
        except Exception:  # noqa: BLE001
            pass
    if not bias_proof:
        proof_path = world_root / "universe" / "unbiased_subset_bias_proof.json"
        if proof_path.exists():
            from trencher.data.world_model.io import read_json

            bias_proof = read_json(proof_path)

    wt_path = graph / "edges_wallet_token.parquet"
    graph_mints = _mint_set(wt_path, "dst")
    ownership_mints = _mint_set(tt_unbiased_path, "mint") or _mint_set(tt_path, "mint")
    ownership_mints &= subset_mints
    exec_path = world_root / "execution" / "execution_curve_states.parquet"
    event_mints = _mint_set(exec_path, "mint") or subset_mints
    market_ok = global_days >= 20
    fusion_overlap = subset_mints
    if event_mints:
        fusion_overlap &= event_mints
    if graph_mints:
        fusion_overlap &= graph_mints
    if ownership_mints:
        fusion_overlap &= ownership_mints
    if micro_mints:
        fusion_overlap &= micro_mints
    if not market_ok:
        fusion_overlap = set()
    overlap_n = len(fusion_overlap)
    overlap_grad = 0
    if subset_path.exists() and "graduated" in (
        pq.read_schema(subset_path).names if subset_path.exists() else []
    ):
        st = pq.read_table(subset_path, columns=["mint", "graduated"])
        gmap = dict(
            zip(st.column("mint").to_pylist(), st.column("graduated").to_pylist(), strict=True)
        )
        overlap_grad = sum(1 for m in fusion_overlap if gmap.get(m))
    overlap_nongrad = overlap_n - overlap_grad
    not_grad_proxy = bool(bias_proof.get("selection_is_not_graduation_proxy"))

    layers = [
        {
            "layer": "transaction_microstructure",
            "tokens": micro_tokens,
            "wallets": None,
            "rows_edges": rpc_found,
            "date_coverage": "2026-06-05..2026-07-14 (enriched subset)",
            "source": "pumpfun.tx_signature + solana_rpc.getTransaction",
            "quality": (
                "high_subset"
                if match_rate and match_rate > 0.95 and rpc_found >= 1000
                else "narrow_subset"
            ),
        },
        {
            "layer": "token_ownership_transfers",
            "tokens": true_spl_mints or 0,
            "wallets": true_spl_wallets or 0,
            "rows_edges": true_spl_rows,
            "date_coverage": "representative corpus days (partition-pruned)",
            "source": "bigquery.Token Transfers (unbiased rich mints)"
            if true_spl_rows
            else "pumpfun.trades (trade-derived only)",
            "quality": "ledger" if true_spl_rows else "trade_derived_only",
        },
        {
            "layer": "wallet_history_graph",
            "tokens": rich_mints,
            "wallets": wallets,
            "rows_edges": (
                wt_edges
                + creator_edges
                + cobuy
                + cohort
                + spl_transfers
                + rpc_funding
                + native_sol
                + shared_funder
                + shared_native
            ),
            "date_coverage": "corpus window",
            "source": "pumpfun + red_cohort + bq SPL + native SOL system transfers",
            "quality": (
                "strong_with_spl_and_native_sol"
                if native_sol > 0 and spl_transfers > 100_000
                else "strong_with_spl_transfers"
                if spl_transfers > 100_000
                else "participation_strong_funding_partial"
            ),
        },
        {
            "layer": "global_market_network_context",
            "tokens": plan.get("anchor_universe", {}).get("token_rows"),
            "wallets": None,
            "rows_edges": global_days + blocks_rows + int(gecko_sol.get("corpus_window_rows") or 0),
            "date_coverage": "2026-06-05..2026-07-14",
            "source": "pumpfun daily + geckoterminal SOL/USDC + bigquery.Blocks",
            "quality": "high" if global_days >= 30 and blocks_rows else "partial",
        },
        {
            "layer": "post_graduation",
            "tokens": post.get("outcomes_rows") or 0,
            "wallets": None,
            "rows_edges": post.get("snapshots_rows") or 0,
            "date_coverage": "graduated subset (label, not fusion inclusion)",
            "source": "pumpfun.postgard_* (+ gecko pool OHLCV)",
            "quality": "high_local_postgard",
        },
        {
            "layer": "execution_state",
            "tokens": exec_mints,
            "wallets": None,
            "rows_edges": exec_rows,
            "date_coverage": "unbiased rich subset",
            "source": "pumpfun.trades curve reserves",
            "quality": "curve_observed_subset",
        },
        {
            "layer": "account_ownership_snapshots",
            "tokens": accounts_mints,
            "wallets": _unique_count(accounts_path, "owner") if accounts_rows else 0,
            "rows_edges": accounts_rows,
            "date_coverage": "2026-06..2026-07",
            "source": "bigquery.Accounts",
            "quality": "ledger_snapshots" if accounts_rows else "missing",
        },
        {
            "layer": "native_sol_funding",
            "tokens": _unique_count(graph / "edges_funding_native_sol.parquet", "mint")
            if native_sol
            else 0,
            "wallets": native_wallets,
            "rows_edges": native_sol,
            "date_coverage": "pre-T wallet history",
            "source": "solana_rpc.getSignaturesForAddress + parsed System Program transfer",
            "quality": "native_sol" if native_sol else "missing",
        },
    ]

    trade_rows = int(plan.get("anchor_universe", {}).get("trade_rows") or 0)
    event_status = "READY" if trade_rows >= 10_000_000 else "PARTIAL"

    if wallets >= 100_000 and wt_edges >= 1_000_000 and spl_transfers >= 100_000:
        wallet_status = "READY"
    elif wallets >= 10_000 and wt_edges >= 100_000:
        wallet_status = "PARTIAL"
    else:
        wallet_status = "BLOCKED"

    if true_spl_mints >= 1_000 and true_spl_rows >= 100_000 and exec_mints >= 500:
        token_status = "READY"
    elif exec_mints >= 100 or true_spl_rows > 0:
        token_status = "PARTIAL"
    else:
        token_status = "BLOCKED"

    global_status = (
        "READY"
        if global_days >= 30 and (gecko_sol.get("covers_corpus") or blocks_rows >= 30)
        else "PARTIAL" if global_days >= 20 else "BLOCKED"
    )

    postgrad_ok = (post.get("outcomes_rows") or 0) >= 1000
    fusion_trainable = (
        overlap_n >= 2_000
        and overlap_nongrad >= 500
        and overlap_grad >= 1
        and not_grad_proxy
        and native_sol > 0
        and micro_tokens >= 2_000
        and event_status == "READY"
        and token_status == "READY"
        and wallet_status in {"READY", "PARTIAL"}
        and global_status == "READY"
    )
    if fusion_trainable:
        fusion_status = "READY"
    elif event_status == "READY" and postgrad_ok:
        fusion_status = "PARTIAL"
    else:
        fusion_status = "BLOCKED"

    heads_status = "READY" if event_status == "READY" and postgrad_ok else "PARTIAL"

    encoders = {
        "Event Encoder": event_status,
        "Wallet/Graph Encoder": wallet_status,
        "Token/Ownership/Curve Encoder": token_status,
        "Global Market Encoder": global_status,
        "Fusion/world-state model": fusion_status,
        "prediction heads": heads_status,
    }

    irrecoverable = [
        (
            "Full-universe native SOL Instructions table "
            "(SELECT ~7.7 TiB; recovered only via RPC System Program parses on subset)"
        ),
        (
            "Full Transactions table scans for microstructure "
            "(~370+ GiB/day for signature column alone; use RPC subset instead)"
        ),
        "Solarchive txs/accounts for June–July 2026 (index ends 2025-12-05)",
        "Real pool addresses for synthetic/null migrations without discovery",
        "Row-level Pump fee bps history (protocol metadata only)",
        (
            f"Complete fusion population is the multimodal overlap "
            f"({overlap_n} tokens), not the full 798k universe"
        ),
    ]

    core_ready = all(
        encoders[k] == "READY"
        for k in (
            "Event Encoder",
            "Wallet/Graph Encoder",
            "Token/Ownership/Curve Encoder",
            "Global Market Encoder",
        )
    )

    report = {
        "generated_at": utc_now(),
        "layers": layers,
        "metrics": {
            "transaction_signature_match_rate": match_rate,
            "true_spl_transfer_count": true_spl_rows,
            "mints_with_genuine_ownership_histories": true_spl_mints,
            "unbiased_day_spl_transfer_count": unbiased_spl_rows,
            "unbiased_day_spl_mints": unbiased_spl_mints,
            "wallets_with_prior_history": wallets,
            "spl_transfer_edges": spl_transfers,
            "wallets_in_spl_transfers": spl_wallets,
            "true_native_sol_funding_edges": native_sol,
            "wallets_with_native_sol_funding": native_wallets,
            "shared_funder_graph_edges": shared_funder,
            "shared_native_funder_edges": shared_native,
            "creator_ownership_transfer_rows": creator_own,
            "transaction_microstructure_tokens": micro_tokens,
            "transaction_microstructure_txs": rpc_found,
            "execution_state_mints": exec_mints,
            "execution_state_rows": exec_rows,
            "account_snapshot_rows": accounts_rows,
            "account_snapshot_mints": accounts_mints,
            "network_context_corpus_days": global_days,
            "blocks_daily_rows": blocks_rows,
            "tokens_with_postgrad_history": post.get("outcomes_rows"),
            "bigquery_bytes_processed": bq_bytes,
            "bigquery_bytes_billed": bq_billed,
            "solarchive_bytes_downloaded": sol_bytes,
            "gecko_pools_completed": gecko_pools.get("pools_completed_total", 0),
            "graph_edge_counts_by_type": {
                "wallet_token": wt_edges,
                "creator_token": creator_edges,
                "cobuy": cobuy,
                "cohort_member": cohort,
                "spl_transfer": spl_transfers,
                "sol_balance_delta_proxy": rpc_funding,
                "system_sol_transfer": native_sol,
                "shared_funder": shared_funder,
                "shared_native_funder": shared_native,
                "creator_related_transfer": creator_own,
            },
            "unbiased_rich_subset_mints": rich_mints or len(subset_mints),
            "unbiased_subset_graduated": subset_grad,
            "unbiased_subset_non_graduated": subset_nongrad,
            "fusion_overlap_tokens": overlap_n,
            "fusion_overlap_graduated": overlap_grad,
            "fusion_overlap_non_graduated": overlap_nongrad,
            "selection_is_not_graduation_proxy": not_grad_proxy,
        },
        "bias_proof": bias_proof,
        "fusion_trainability": {
            "overlap_definition": (
                "tokens in unbiased subset with event (curve/trades) + graph "
                "+ SPL ownership + microstructure + global market context"
            ),
            "overlap_n": overlap_n,
            "ready_threshold": 2000,
            "status": fusion_status,
        },
        "source_probe_decisions": {
            "bigquery": probe.get("bigquery", {}).get("decision"),
            "solarchive": probe.get("solarchive", {}).get("decision"),
            "geckoterminal": probe.get("geckoterminal", {}).get("decision"),
        },
        "encoders": encoders,
        "irrecoverable_fields": irrecoverable,
        "world_model_dataset_ready": core_ready
        and encoders["Fusion/world-state model"] in {"READY", "PARTIAL"},
        "readiness_notes": [
            "READY requires meaningful trainability, not merely non-empty tables.",
            "Fusion READY requires a large unbiased multimodal overlap, not graduated-only tokens.",
            "SPL transfers are labeled spl_transfer, not genuine native-SOL funding.",
        ],
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
        "| Layer | tokens | wallets | rows/edges | date coverage | source | quality |",
        "| ----- | -----: | ------: | ---------: | ------------- | ------ | ------- |",
    ]
    for row in report["layers"]:
        wallets = row.get("wallets")
        wallets_s = "" if wallets is None else f"{wallets}"
        lines.append(
            f"| {row['layer']} | {row.get('tokens')} | {wallets_s} | "
            f"{row.get('rows_edges')} | {row.get('date_coverage')} | "
            f"{row.get('source')} | {row.get('quality')} |"
        )
    m = report["metrics"]
    lines.extend(
        [
            "",
            "## Key metrics",
            "",
            f"- True SPL transfer count: {m.get('true_spl_transfer_count')}",
            f"- Mints with genuine ownership histories: "
            f"{m.get('mints_with_genuine_ownership_histories')}",
            f"- Unbiased rich subset: {m.get('unbiased_rich_subset_mints')} "
            f"(graduated={m.get('unbiased_subset_graduated')}, "
            f"non-graduated={m.get('unbiased_subset_non_graduated')})",
            f"- Fusion multimodal overlap: {m.get('fusion_overlap_tokens')} "
            f"(graduated={m.get('fusion_overlap_graduated')}, "
            f"non-graduated={m.get('fusion_overlap_non_graduated')})",
            f"- Selection is not a graduation proxy: "
            f"{m.get('selection_is_not_graduation_proxy')}",
            f"- Wallets with prior history: {m.get('wallets_with_prior_history')}",
            f"- SPL transfer edges (not native SOL funding): {m.get('spl_transfer_edges')}",
            f"- True native-SOL funding edges: {m.get('true_native_sol_funding_edges')}",
            f"- Wallets with native-SOL funding: {m.get('wallets_with_native_sol_funding')}",
            f"- Shared-funder graph edges (SPL): {m.get('shared_funder_graph_edges')}",
            f"- Shared native funder edges: {m.get('shared_native_funder_edges')}",
            f"- Creator ownership transfer rows: {m.get('creator_ownership_transfer_rows')}",
            f"- Transaction microstructure: {m.get('transaction_microstructure_tokens')} tokens / "
            f"{m.get('transaction_microstructure_txs')} txs "
            f"(match rate={m.get('transaction_signature_match_rate')})",
            f"- Execution-state coverage: {m.get('execution_state_mints')} mints / "
            f"{m.get('execution_state_rows')} rows",
            f"- Account snapshots: {m.get('account_snapshot_mints')} mints / "
            f"{m.get('account_snapshot_rows')} rows",
            f"- BigQuery bytes processed / billed: "
            f"{m.get('bigquery_bytes_processed')} / {m.get('bigquery_bytes_billed')}",
            f"- Solarchive bytes downloaded: {m.get('solarchive_bytes_downloaded')}",
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
            f"**World-model dataset ready (strict core encoders):** "
            f"{report['world_model_dataset_ready']}",
            "",
            "## Remaining missing / irrecoverable under current constraints",
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
