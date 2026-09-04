"""Append-only historical data freeze manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trencher.data.world_model.io import ensure_dir, file_provenance, read_json, utc_now, write_json
from trencher.data.world_model.ownership import build_fusion_coverage_truth

FREEZE_ID = "world_model_freeze_v1"


def _rows(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:  # noqa: BLE001
        return 0


def _unique(path: Path, column: str) -> int:
    if _rows(path) == 0:
        return 0
    try:
        return len(set(pq.read_table(path, columns=[column]).column(0).to_pylist()) - {None})
    except Exception:  # noqa: BLE001
        return 0


def _artifact(
    path: Path,
    *,
    source: str,
    quality: str,
    ownership_evidence_level: str | None = None,
    mint_column: str | None = "mint",
    wallet_column: str | None = None,
    date_coverage: str | None = None,
    known_limitations: list[str] | None = None,
) -> dict[str, Any]:
    prov = file_provenance(path, source=source)
    rec: dict[str, Any] = {
        **prov,
        "row_count": _rows(path),
        "quality": quality,
        "ownership_evidence_level": ownership_evidence_level,
        "date_coverage": date_coverage,
        "known_limitations": known_limitations or [],
        "append_only": True,
    }
    if mint_column:
        rec["mint_count"] = _unique(path, mint_column)
    if wallet_column:
        rec["wallet_count"] = _unique(path, wallet_column)
    return rec


def build_data_freeze(
    *,
    world_root: Path,
    manifest_dir: Path,
    external_root: Path | None = None,  # noqa: ARG001 — reserved for provenance
) -> dict[str, Any]:
    """Create versioned freeze manifest; does not mutate raw acquisition files."""
    coverage = build_fusion_coverage_truth(world_root, manifest_dir=manifest_dir)
    bq_manifest = world_root / "bigquery" / "bigquery_manifest.json"
    lifetime_billed = 0
    if bq_manifest.exists():
        prev = read_json(bq_manifest)
        lifetime_billed = int(
            prev.get("lifetime_bytes_billed") or prev.get("bytes_billed_actual") or 0
        )

    artifacts = {
        "universe_tokens": _artifact(
            world_root / "universe" / "universe_tokens.parquet",
            source="pumpfun.tokens",
            quality="authoritative_local",
            date_coverage="2026-06-05..2026-07-14",
        ),
        "rich_subset_mints": _artifact(
            world_root / "universe" / "rich_subset_mints.parquet",
            source="outcome_independent_stratified",
            quality="outcome_independent",
            date_coverage="representative corpus days excl Jul 3",
            known_limitations=["Not the full 798k universe"],
        ),
        "ownership_evidence": _artifact(
            world_root / "universe" / "ownership_evidence.parquet",
            source="classified_multi_source",
            quality="evidence_leveled",
            ownership_evidence_level="mixed",
        ),
        "fusion_coverage_truth": _artifact(
            world_root / "universe" / "fusion_coverage_truth.parquet",
            source="deterministic_intersection",
            quality="source_of_truth",
        ),
        "token_transfers_bq": _artifact(
            world_root / "bigquery" / "token_transfers_rich.parquet",
            source="bigquery.Token Transfers",
            quality="ledger",
            ownership_evidence_level="ledger_full_or_reconstructable",
            wallet_column="source",
            date_coverage="corpus window / day extracts",
            known_limitations=[
                "Sparse for non-graduated Pump mints",
                "Includes prior graduated-window union rows",
            ],
        ),
        "token_transfers_unbiased_days": _artifact(
            world_root / "bigquery" / "token_transfers_unbiased_days.parquet",
            source="bigquery.Token Transfers.day_partitions",
            quality="ledger_day_extracts",
            ownership_evidence_level="ledger_full_or_reconstructable",
            date_coverage="representative/unqueried day extracts",
        ),
        "rpc_token_balance_ownership": _artifact(
            world_root / "graph" / "edges_ownership_rpc_token_balances.parquet",
            source="solana_rpc.getTransaction.token_balances",
            quality="sampled_tx_observations",
            ownership_evidence_level="rpc_balance_observed",
            wallet_column="owner",
            known_limitations=[
                "Typically one sampled trade signature per mint",
                "Not a complete holder-distribution history",
            ],
        ),
        "native_sol_funding": _artifact(
            world_root / "graph" / "edges_funding_native_sol.parquet",
            source="solana_rpc.system_program_transfer",
            quality="native_sol_pre_T",
            wallet_column="src",
            known_limitations=["Creators/early wallets subset only"],
        ),
        "rpc_transactions": _artifact(
            world_root / "microstructure" / "rpc_transactions.parquet",
            source="solana_rpc.getTransaction",
            quality="archival_rpc_subset",
            date_coverage="stratified signatures on rich subset",
            known_limitations=["Not full tx universe"],
        ),
        "signature_targets": _artifact(
            world_root / "microstructure" / "signature_targets.parquet",
            source="pumpfun.trades.stratified",
            quality="stratified_per_token",
        ),
        "execution_curve_states": _artifact(
            world_root / "execution" / "execution_curve_states.parquet",
            source="pumpfun.trades.curve",
            quality="curve_observed_subset",
        ),
        "edges_wallet_token": _artifact(
            world_root / "graph" / "edges_wallet_token.parquet",
            source="pumpfun.trades",
            quality="trade_participation",
            ownership_evidence_level="trade_derived_proxy",
            mint_column="dst",
            wallet_column="src",
        ),
        "daily_pump_network_context": _artifact(
            world_root / "global_context" / "daily_pump_network_context.parquet",
            source="pumpfun.aggregates",
            quality="derived_daily",
            mint_column=None,
            date_coverage="corpus window",
        ),
        "blocks_daily": _artifact(
            world_root / "bigquery" / "blocks_daily.parquet",
            source="bigquery.Blocks",
            quality="network_context",
            mint_column=None,
            date_coverage="2026-06..2026-07",
        ),
    }

    freeze = {
        "freeze_id": FREEZE_ID,
        "generated_at": utc_now(),
        "policy": {
            "append_only_raw": True,
            "no_further_bulk_acquisition": True,
            "external_bulk_requires_explicit_request": True,
        },
        "schema_protocol_versions": {
            "ownership_evidence": "v1",
            "fusion_coverage_truth": "fusion_coverage_v2_ownership_evidence",
            "pit_engine": "v1",
        },
        "bigquery_lifetime_bytes_billed": lifetime_billed,
        "bigquery_lifetime_gib_billed": round(lifetime_billed / (1024**3), 3),
        "fusion_coverage": coverage["truth"]["counts"],
        "ownership_evidence_counts": coverage["truth"]["ownership_evidence_counts"],
        "artifacts": artifacts,
        "known_limitations": [
            "RPC ownership observations are sampled-tx facts, not full ledgers.",
            "Reconstructable ownership multimodal overlap is much smaller than "
            "any-genuine-ownership overlap.",
            "Solarchive has no June–July 2026 coverage.",
            "Native SOL funding covers a wallet subset only.",
        ],
    }
    ensure_dir(manifest_dir)
    out = manifest_dir / f"{FREEZE_ID}.json"
    write_json(out, freeze)
    write_json(manifest_dir / "DATA_FREEZE.json", freeze)
    (manifest_dir / "DATA_FREEZE.md").write_text(render_freeze_md(freeze), encoding="utf-8")
    # Also park a copy under world_root for local consumers.
    write_json(ensure_dir(world_root / "freeze") / f"{FREEZE_ID}.json", freeze)
    return freeze


def render_freeze_md(freeze: dict[str, Any]) -> str:
    lines = [
        f"# Data Freeze `{freeze['freeze_id']}`",
        "",
        f"Generated at: `{freeze['generated_at']}`",
        "",
        "## Policy",
        "",
        "- Historical raw/canonical acquisition artifacts are **append-only**.",
        "- No further external bulk acquisition unless explicitly requested.",
        "",
        "## Fusion coverage (authoritative)",
        "",
    ]
    for k, v in freeze["fusion_coverage"].items():
        lines.append(f"- `{k}`: **{v}**")
    lines.extend(["", "## Ownership evidence levels", ""])
    for k, v in freeze["ownership_evidence_counts"].items():
        lines.append(f"- `{k}`: **{v}**")
    lines.extend(
        [
            "",
            f"## BigQuery lifetime billed: **{freeze['bigquery_lifetime_gib_billed']} GiB**",
            "",
            "## Artifacts",
            "",
            "| artifact | rows | mints | quality | ownership level |",
            "| -------- | ---: | ----: | ------- | --------------- |",
        ]
    )
    for name, art in freeze["artifacts"].items():
        lines.append(
            f"| {name} | {art.get('row_count')} | {art.get('mint_count', '')} | "
            f"{art.get('quality')} | {art.get('ownership_evidence_level') or ''} |"
        )
    lines.extend(["", "## Known limitations", ""])
    lines.extend(f"- {x}" for x in freeze["known_limitations"])
    lines.append("")
    return "\n".join(lines)
