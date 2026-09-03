"""BigQuery acquisition with mandatory dry-run and byte caps."""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.constants import (
    BQ_DATASET,
    BQ_DRY_RUN_ABORT_BYTES,
    BQ_PROJECT_ENV,
    BQ_SESSION_BUDGET_BYTES,
    BQ_TT_ABORT_BYTES,
    CORPUS_END,
    CORPUS_START,
)
from trencher.data.world_model.io import ensure_dir, file_provenance, utc_now, write_json

logger = logging.getLogger(__name__)

BQ_LOCATION = "us-central1"
TMP_DATASET = "world_model_tmp"


def _client(project: str | None):
    from google.cloud import bigquery

    project = project or os.environ.get(BQ_PROJECT_ENV) or os.environ.get("GCP_PROJECT")
    return bigquery.Client(project=project, location=BQ_LOCATION), bigquery


def _sha16(sql: str) -> str:
    return hashlib.sha256(sql.encode()).hexdigest()[:16]


def _ensure_tmp_dataset(client: Any, bigquery: Any) -> None:
    ds = bigquery.Dataset(f"{client.project}.{TMP_DATASET}")
    ds.location = BQ_LOCATION
    client.create_dataset(ds, exists_ok=True)


def _upload_mints(client: Any, bigquery: Any, mints: list[str]) -> str:
    table_id = f"{client.project}.{TMP_DATASET}.rich_mints"
    job = client.load_table_from_json(
        [{"mint": m} for m in mints],
        table_id,
        job_config=bigquery.LoadJobConfig(
            schema=[bigquery.SchemaField("mint", "STRING")],
            write_disposition="WRITE_TRUNCATE",
        ),
    )
    job.result()
    return table_id


def _dry_run(
    client: Any,
    bigquery: Any,
    sql: str,
    *,
    max_bytes_billed: int,
) -> dict[str, Any]:
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            dry_run=True,
            use_query_cache=False,
            maximum_bytes_billed=max_bytes_billed,
        ),
        location=BQ_LOCATION,
    )
    est = int(job.total_bytes_processed or 0)
    return {
        "sql_sha16": _sha16(sql),
        "estimated_bytes": est,
        "estimated_gib": round(est / (1024**3), 4),
        "within_abort": est <= BQ_DRY_RUN_ABORT_BYTES,
        "within_max_billed": est <= max_bytes_billed,
    }


def _download_table(
    client: Any,
    table_id: str,
    out_path: Path,
) -> dict[str, Any]:
    ensure_dir(out_path.parent)
    if out_path.exists():
        out_path.unlink()
    table = client.get_table(table_id)
    # Prefer Arrow download (BQ Storage API when available).
    job = client.query(
        f"SELECT * FROM `{table_id}`",
        location=BQ_LOCATION,
    )
    try:
        arrow = job.result().to_arrow(create_bqstorage_client=True)
    except Exception:  # noqa: BLE001
        arrow = job.result().to_arrow(create_bqstorage_client=False)
    pq.write_table(arrow, out_path)
    return {
        "path": file_provenance(out_path, source=f"bigquery:{table_id}"),
        "rows": arrow.num_rows,
        "bq_table_rows": table.num_rows,
        "download_bytes_processed": int(job.total_bytes_processed or 0),
    }


def _build_funding_and_shared(
    transfers_path: Path,
    out_dir: Path,
    creators: set[str] | None = None,
) -> dict[str, Any]:
    """Derive funding-topology edges from genuine SPL transfer ledger rows."""
    ensure_dir(out_dir)
    fund_path = out_dir / "edges_funding_spl.parquet"
    shared_path = out_dir / "edges_shared_funder.parquet"
    creator_path = out_dir / "edges_creator_ownership_transfers.parquet"
    if not transfers_path.exists() or pq.ParquetFile(transfers_path).metadata.num_rows == 0:
        empty = pa.table({"edge_type": pa.array([], pa.string())})
        pq.write_table(empty, fund_path)
        pq.write_table(empty, shared_path)
        pq.write_table(empty, creator_path)
        return {"funding_edges": 0, "shared_funder_edges": 0, "creator_ownership_rows": 0}

    # Funding edges are the ledger itself with edge metadata.
    table = pq.read_table(
        transfers_path,
        columns=[
            "source",
            "destination",
            "mint",
            "block_timestamp",
            "value",
            "tx_signature",
        ],
    )
    n = table.num_rows
    edge_type = pa.array(["spl_transfer"] * n, pa.string())
    quality = pa.array(["bigquery_token_transfers_observed"] * n, pa.string())
    funding = pa.table(
        {
            "edge_type": edge_type,
            "src": table.column("source"),
            "dst": table.column("destination"),
            "mint": table.column("mint"),
            "value": table.column("value"),
            "block_timestamp": table.column("block_timestamp"),
            "tx_signature": table.column("tx_signature"),
            "quality_flag": quality,
        }
    )
    # Drop self-transfers / null endpoints
    mask = []
    src = table.column("source").to_pylist()
    dst = table.column("destination").to_pylist()
    for s, d in zip(src, dst, strict=True):
        mask.append(bool(s and d and s != d))
    funding = funding.filter(pa.array(mask))
    pq.write_table(funding, fund_path)

    from collections import defaultdict

    funder_dests: dict[str, set[str]] = defaultdict(set)
    src2 = funding.column("src").to_pylist()
    dst2 = funding.column("dst").to_pylist()
    for s, d in zip(src2, dst2, strict=True):
        funder_dests[s].add(d)
    shared = []
    for funder, dests in funder_dests.items():
        if len(dests) < 2:
            continue
        ordered = sorted(dests)[:40]
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                shared.append(
                    {
                        "edge_type": "shared_funder",
                        "funder": funder,
                        "dst_a": a,
                        "dst_b": b,
                        "n_destinations": len(dests),
                        "quality_flag": "derived_from_bq_spl_transfers",
                    }
                )
    if shared:
        pq.write_table(pa.Table.from_pylist(shared), shared_path)
    else:
        pq.write_table(pa.table({"edge_type": pa.array([], pa.string())}), shared_path)

    creators = creators or set()
    creator_rows = []
    if creators:
        # Prefer authority matches (common for creator/dev token ops), then src/dst.
        cols = ["source", "destination", "mint", "block_timestamp", "value", "tx_signature"]
        auth_col = None
        try:
            full = pq.read_table(transfers_path, columns=[*cols, "authority"])
            auth_col = full.column("authority").to_pylist()
        except Exception:  # noqa: BLE001
            full = funding
            auth_col = [None] * funding.num_rows
        src_f = full.column("source").to_pylist()
        dst_f = full.column("destination").to_pylist()
        mint_f = full.column("mint").to_pylist()
        ts_f = full.column("block_timestamp").to_pylist()
        val_f = full.column("value").to_pylist()
        sig_f = full.column("tx_signature").to_pylist()
        for i, (s, d, m, t, v, g) in enumerate(
            zip(src_f, dst_f, mint_f, ts_f, val_f, sig_f, strict=True)
        ):
            a = auth_col[i] if auth_col else None
            role = None
            if a in creators:
                role = "authority"
            elif s in creators:
                role = "src"
            elif d in creators:
                role = "dst"
            if not role:
                continue
            creator_rows.append(
                {
                    "edge_type": "creator_related_transfer",
                    "src": s,
                    "dst": d,
                    "authority": a,
                    "mint": m,
                    "value": float(v) if v is not None else None,
                    "block_timestamp": t,
                    "tx_signature": g,
                    "creator_role": role,
                    "quality_flag": "bigquery_token_transfers_observed",
                }
            )
    if creator_rows:
        pq.write_table(pa.Table.from_pylist(creator_rows), creator_path)
    else:
        pq.write_table(pa.table({"edge_type": pa.array([], pa.string())}), creator_path)

    return {
        "funding_edges": funding.num_rows,
        "shared_funder_edges": len(shared),
        "creator_ownership_rows": len(creator_rows),
        "unique_transfer_mints": len(set(funding.column("mint").to_pylist()) - {None}),
        "unique_transfer_wallets": len((set(src2) | set(dst2)) - {None}),
        "files": {
            "funding": file_provenance(fund_path, source="bigquery.Token Transfers"),
            "shared_funder": file_provenance(shared_path, source="derived"),
            "creator": file_provenance(creator_path, source="bigquery+pumpfun.creators"),
        },
    }


def run_bigquery_acquisition(
    *,
    world_root: Path,
    project: str | None = None,
    max_bytes_billed: int = 650 * 1024**3,
    session_budget: int = BQ_SESSION_BUDGET_BYTES,
    skip_live: bool = False,
    execute: bool = False,
    rich_mints_path: Path | None = None,
    creators_path: Path | None = None,
) -> dict[str, Any]:
    """Acquire Token Transfers / Accounts / Blocks for the corpus with cost caps."""
    out_dir = ensure_dir(world_root / "bigquery")
    project = project or os.environ.get(BQ_PROJECT_ENV) or os.environ.get("GCP_PROJECT")
    payload: dict[str, Any] = {
        "generated_at": utc_now(),
        "project": project,
        "location": BQ_LOCATION,
        "execute": execute,
        "skip_live": skip_live,
        "bytes_processed_actual": 0,
        "bytes_billed_actual": 0,
        "queries": [],
        "artifacts": {},
        "decision": "skipped",
        "notes": [],
    }
    if skip_live:
        payload["decision"] = "skipped_for_ci"
        write_json(out_dir / "bigquery_manifest.json", payload)
        return payload

    try:
        from google.auth.exceptions import DefaultCredentialsError
    except ImportError:
        payload["decision"] = "library_missing"
        write_json(out_dir / "bigquery_manifest.json", payload)
        return payload

    try:
        client, bigquery = _client(project)
        payload["project"] = client.project
    except DefaultCredentialsError as exc:
        payload["decision"] = "auth_missing"
        payload["error"] = str(exc)
        write_json(out_dir / "bigquery_manifest.json", payload)
        return payload
    except Exception as exc:  # noqa: BLE001
        payload["decision"] = "auth_missing"
        payload["error"] = str(exc)
        write_json(out_dir / "bigquery_manifest.json", payload)
        return payload

    if not execute:
        # Dry-run suite only
        start = f"{CORPUS_START.isoformat()} 00:00:00"
        end = f"{(CORPUS_END + timedelta(days=1)).isoformat()} 00:00:00"
        suite = {
            "token_transfers_day_counts": f"""
                SELECT DATE(block_timestamp) d, COUNT(*) n
                FROM `{BQ_DATASET}.Token Transfers`
                WHERE block_timestamp >= TIMESTAMP('{start}')
                  AND block_timestamp < TIMESTAMP('{end}')
                GROUP BY 1 ORDER BY 1
            """,
            "blocks_window": f"""
                SELECT DATE(block_timestamp) d, COUNT(*) n, SUM(transaction_count) txs
                FROM `{BQ_DATASET}.Blocks`
                WHERE block_timestamp >= TIMESTAMP('2026-06-01')
                  AND block_timestamp < TIMESTAMP('2026-08-01')
                GROUP BY 1
            """,
        }
        for name, sql in suite.items():
            try:
                rec = {
                    "name": name,
                    **_dry_run(
                        client, bigquery, sql, max_bytes_billed=max_bytes_billed
                    ),
                }
            except Exception as exc:  # noqa: BLE001
                rec = {"name": name, "error": str(exc)}
            payload["queries"].append(rec)
        payload["decision"] = "dry_run_only_ready"
        write_json(out_dir / "bigquery_manifest.json", payload)
        return payload

    # --- execute path ---
    _ensure_tmp_dataset(client, bigquery)
    rich_path = rich_mints_path or (world_root / "universe" / "rich_subset_mints.parquet")
    if not rich_path.exists():
        payload["decision"] = "missing_rich_subset"
        write_json(out_dir / "bigquery_manifest.json", payload)
        return payload
    mints = [m for m in pq.read_table(rich_path, columns=["mint"]).column(0).to_pylist() if m]
    mint_table = _upload_mints(client, bigquery, mints)
    payload["notes"].append(f"uploaded {len(mints)} rich mints to {mint_table}")

    session_used = 0
    start = f"{CORPUS_START.isoformat()} 00:00:00"
    end_excl = f"{(CORPUS_END + timedelta(days=1)).isoformat()} 00:00:00"

    queries_plan = [
        (
            "token_transfers_rich",
            f"""
            SELECT
              t.block_timestamp,
              t.block_slot,
              t.tx_signature,
              t.source,
              t.destination,
              t.authority,
              t.value,
              t.decimals,
              t.mint,
              t.transfer_type
            FROM `{BQ_DATASET}.Token Transfers` t
            INNER JOIN `{mint_table}` m USING (mint)
            WHERE t.block_timestamp >= TIMESTAMP('{start}')
              AND t.block_timestamp < TIMESTAMP('{end_excl}')
            """,
            f"{client.project}.{TMP_DATASET}.token_transfers_rich",
            650 * 1024**3,
        ),
        (
            "accounts_rich_jun_jul",
            f"""
            SELECT
              block_timestamp,
              block_slot,
              pubkey,
              mint,
              owner,
              lamports,
              token_amount,
              token_amount_decimals,
              account_type,
              tx_signature
            FROM `{BQ_DATASET}.Accounts`
            WHERE block_timestamp >= TIMESTAMP('2026-06-01')
              AND block_timestamp < TIMESTAMP('2026-08-01')
              AND mint IN (SELECT mint FROM `{mint_table}`)
            """,
            f"{client.project}.{TMP_DATASET}.accounts_rich",
            25 * 1024**3,
        ),
        (
            "blocks_daily",
            """
            SELECT
              DATE(block_timestamp) AS day,
              COUNT(*) AS block_count,
              SUM(transaction_count) AS transaction_count_sum,
              AVG(transaction_count) AS transaction_count_avg
            FROM `bigquery-public-data.crypto_solana_mainnet_us.Blocks`
            WHERE block_timestamp >= TIMESTAMP('2026-06-01')
              AND block_timestamp < TIMESTAMP('2026-08-01')
            GROUP BY 1
            ORDER BY 1
            """,
            f"{client.project}.{TMP_DATASET}.blocks_daily",
            5 * 1024**3,
        ),
    ]

    for name, sql, dest, billed_cap in queries_plan:
        dry = _dry_run(client, bigquery, sql, max_bytes_billed=billed_cap)
        rec = {
            "name": name,
            "dest_table": dest,
            "sql": sql.strip(),
            **dry,
            "executed": False,
            "bytes_processed": 0,
            "bytes_billed": 0,
            "aborted": False,
            "error": None,
        }
        abort_limit = (
            BQ_TT_ABORT_BYTES if name.startswith("token_transfers") else BQ_DRY_RUN_ABORT_BYTES
        )
        if dry["estimated_bytes"] > abort_limit or dry["estimated_bytes"] > billed_cap:
            rec["aborted"] = True
            rec["error"] = "dry_run_exceeded_threshold"
            payload["queries"].append(rec)
            continue
        if session_used + dry["estimated_bytes"] > session_budget:
            rec["aborted"] = True
            rec["error"] = "session_budget_exceeded"
            payload["queries"].append(rec)
            continue
        try:
            job_config = bigquery.QueryJobConfig(
                destination=dest,
                write_disposition="WRITE_TRUNCATE",
                maximum_bytes_billed=billed_cap,
                use_query_cache=False,
            )
            job = client.query(sql, job_config=job_config, location=BQ_LOCATION)
            job.result()
            rec["executed"] = True
            rec["bytes_processed"] = int(job.total_bytes_processed or 0)
            rec["bytes_billed"] = int(job.total_bytes_billed or 0)
            rec["dest_rows"] = client.get_table(dest).num_rows
            session_used += rec["bytes_processed"]
            payload["bytes_processed_actual"] += rec["bytes_processed"]
            payload["bytes_billed_actual"] += rec["bytes_billed"]
        except Exception as exc:  # noqa: BLE001
            rec["error"] = str(exc)
            rec["aborted"] = True
        payload["queries"].append(rec)
        write_json(out_dir / "bigquery_manifest.json", payload)  # checkpoint

    # Download executed tables
    mapping = {
        "token_transfers_rich": out_dir / "token_transfers_rich.parquet",
        "accounts_rich_jun_jul": out_dir / "accounts_rich.parquet",
        "blocks_daily": out_dir / "blocks_daily.parquet",
    }
    for rec in payload["queries"]:
        if not rec.get("executed"):
            continue
        name = rec["name"]
        if name not in mapping:
            continue
        try:
            # Fresh download path
            if mapping[name].exists():
                mapping[name].unlink()
            payload["artifacts"][name] = _download_table(
                client, rec["dest_table"], mapping[name]
            )
        except Exception as exc:  # noqa: BLE001
            payload["artifacts"][name] = {"error": str(exc)}

    # Derive funding / shared-funder / creator ownership from genuine TT ledger
    creators: set[str] = set()
    cpath = creators_path or (world_root / "universe" / "universe_tokens.parquet")
    if cpath.exists():
        creators = {
            c
            for c in pq.read_table(cpath, columns=["creator"]).column(0).to_pylist()
            if c
        }
    transfers_local = mapping["token_transfers_rich"]
    graph_dir = world_root / "graph"
    payload["artifacts"]["transfer_derived_graph"] = _build_funding_and_shared(
        transfers_local, graph_dir, creators=creators
    )

    # Explicit note: native SOL system transfers not extracted (Instructions SELECT ~7TB)
    payload["notes"].append(
        "Native SOL system-program transfer ledger not extracted: "
        "Instructions SELECT dry-run ~7.7 TiB for corpus window; aborted by cost controls."
    )
    payload["notes"].append(
        "Transactions full-day scans aborted: signature-only one day ~370 GiB; "
        "microstructure continues via RPC getTransaction enrichment."
    )
    payload["session_bytes_processed"] = session_used
    payload["decision"] = (
        "executed"
        if any(q.get("executed") for q in payload["queries"])
        else "aborted_or_failed"
    )
    write_json(out_dir / "bigquery_manifest.json", payload)
    write_json(
        out_dir / "queries.json",
        {
            "generated_at": utc_now(),
            "queries": [
                {
                    "name": q.get("name"),
                    "sql_sha16": q.get("sql_sha16"),
                    "sql": q.get("sql"),
                    "estimated_bytes": q.get("estimated_bytes"),
                    "bytes_processed": q.get("bytes_processed"),
                    "bytes_billed": q.get("bytes_billed"),
                    "executed": q.get("executed"),
                    "aborted": q.get("aborted"),
                    "error": q.get("error"),
                    "dest_rows": q.get("dest_rows"),
                }
                for q in payload["queries"]
            ],
        },
    )
    return payload
