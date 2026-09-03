"""Optional BigQuery acquisition with mandatory dry-run and byte caps."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

from trencher.data.world_model.constants import (
    BQ_DATASET,
    BQ_DRY_RUN_ABORT_BYTES,
    BQ_MAX_BYTES_BILLED_DEFAULT,
    BQ_PROJECT_ENV,
    BQ_SESSION_BUDGET_BYTES,
    CORPUS_END,
    CORPUS_START,
)
from trencher.data.world_model.io import ensure_dir, utc_now, write_json

logger = logging.getLogger(__name__)


def run_bigquery_acquisition(
    *,
    world_root: Path,
    project: str | None = None,
    max_bytes_billed: int = BQ_MAX_BYTES_BILLED_DEFAULT,
    skip_live: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    """Probe/plan BQ pulls. Execute only when execute=True and dry-run is cheap."""
    out_dir = ensure_dir(world_root / "bigquery")
    project = project or os.environ.get(BQ_PROJECT_ENV) or os.environ.get("GCP_PROJECT")
    payload: dict[str, Any] = {
        "generated_at": utc_now(),
        "project": project,
        "execute": execute,
        "skip_live": skip_live,
        "bytes_processed_actual": 0,
        "queries": [],
        "decision": "skipped",
    }
    if skip_live:
        payload["decision"] = "skipped_for_ci"
        write_json(out_dir / "bigquery_manifest.json", payload)
        return payload

    try:
        from google.auth.exceptions import DefaultCredentialsError
        from google.cloud import bigquery
    except ImportError:
        payload["decision"] = "library_missing"
        write_json(out_dir / "bigquery_manifest.json", payload)
        return payload

    try:
        client = bigquery.Client(project=project) if project else bigquery.Client()
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

    start = f"{CORPUS_START.isoformat()} 00:00:00"
    end = f"{(CORPUS_END).isoformat()} 23:59:59"
    # Intentionally tiny coverage confirmation query (partition-filtered).
    sql = f"""
    SELECT COUNT(*) AS n
    FROM `{BQ_DATASET}.Transactions`
    WHERE block_timestamp >= TIMESTAMP('{start}')
      AND block_timestamp < TIMESTAMP('{end}')
      AND DATE(block_timestamp) = DATE('{CORPUS_START.isoformat()}')
    """
    sha = hashlib.sha256(sql.encode()).hexdigest()[:16]
    job_config = bigquery.QueryJobConfig(
        dry_run=True,
        use_query_cache=False,
        maximum_bytes_billed=max_bytes_billed,
    )
    try:
        dry = client.query(sql, job_config=job_config)
        est = int(dry.total_bytes_processed or 0)
    except Exception as exc:  # noqa: BLE001
        payload["decision"] = "dry_run_failed"
        payload["error"] = str(exc)
        write_json(out_dir / "bigquery_manifest.json", payload)
        return payload

    qrec = {
        "name": "transactions_one_day_count",
        "sql_sha16": sha,
        "estimated_bytes": est,
        "within_abort": est <= BQ_DRY_RUN_ABORT_BYTES,
        "executed": False,
        "bytes_processed": 0,
    }
    payload["queries"].append(qrec)
    if est > BQ_DRY_RUN_ABORT_BYTES:
        payload["decision"] = "aborted_too_expensive"
        write_json(out_dir / "bigquery_manifest.json", payload)
        return payload

    if not execute:
        payload["decision"] = "dry_run_only_ready"
        write_json(out_dir / "bigquery_manifest.json", payload)
        return payload

    if est > BQ_SESSION_BUDGET_BYTES:
        payload["decision"] = "aborted_session_budget"
        write_json(out_dir / "bigquery_manifest.json", payload)
        return payload

    run_config = bigquery.QueryJobConfig(
        dry_run=False,
        maximum_bytes_billed=max_bytes_billed,
    )
    job = client.query(sql, job_config=run_config)
    rows = list(job.result())
    processed = int(job.total_bytes_processed or 0)
    qrec["executed"] = True
    qrec["bytes_processed"] = processed
    qrec["result_n"] = int(rows[0]["n"]) if rows else None
    payload["bytes_processed_actual"] = processed
    payload["decision"] = "executed_coverage_sample"
    write_json(out_dir / "bigquery_manifest.json", payload)
    write_json(
        out_dir / "transactions_day_count.json",
        {"day": CORPUS_START.isoformat(), "n": qrec.get("result_n"), "bytes": processed},
    )
    return payload
