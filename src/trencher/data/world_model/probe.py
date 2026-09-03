"""Source capability probes for BigQuery, Solarchive, and GeckoTerminal."""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from trencher.data.world_model.constants import (
    BQ_DATASET,
    BQ_DRY_RUN_ABORT_BYTES,
    BQ_MAX_BYTES_BILLED_DEFAULT,
    BQ_PROJECT_ENV,
    BQ_TABLES,
    CORPUS_END,
    CORPUS_START,
    GECKO_BASE_URL,
    GECKO_MIN_INTERVAL_SEC,
    GECKO_NETWORK,
    GECKO_SOL_USDC_POOL,
    SOLARCHIVE_ACCOUNTS_REL,
    SOLARCHIVE_TXS_REL,
)
from trencher.data.world_model.io import utc_now, write_json

logger = logging.getLogger(__name__)


def _daterange(start: date, end: date) -> list[date]:
    out: list[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def probe_solarchive(solarchive_root: Path) -> dict[str, Any]:
    """Use downloaded indexes only — never download archive shards."""
    txs_root = solarchive_root / SOLARCHIVE_TXS_REL
    accounts_root = solarchive_root / SOLARCHIVE_ACCOUNTS_REL
    txs_index = txs_root / "index.json"
    result: dict[str, Any] = {
        "available_locally": solarchive_root.exists(),
        "txs_index_present": txs_index.exists(),
        "corpus_start": CORPUS_START.isoformat(),
        "corpus_end": CORPUS_END.isoformat(),
        "txs_partitions_for_corpus": [],
        "accounts_partitions_for_corpus": [],
        "txs_coverage_complete": False,
        "accounts_coverage_complete": False,
        "estimated_download_bytes_if_present": None,
        "estimated_download_gib_if_present": None,
        "decision": "unavailable_for_corpus_dates",
        "notes": [],
    }
    if not txs_index.exists():
        result["notes"].append("txs/index.json missing; solarchive light download incomplete")
        return result

    import json

    txs_meta = json.loads(txs_index.read_text(encoding="utf-8"))
    partitions = {p["partition"]: p for p in txs_meta.get("partitions", [])}
    all_dates = sorted(partitions)
    result["txs_partition_min"] = all_dates[0] if all_dates else None
    result["txs_partition_max"] = all_dates[-1] if all_dates else None

    needed_days = _daterange(CORPUS_START, CORPUS_END)
    present = []
    missing = []
    est_bytes = 0
    # If corpus dates absent, estimate using recent median day size.
    recent_sizes = [
        int(p.get("total_size_bytes") or 0)
        for p in txs_meta.get("partitions", [])[-30:]
        if p.get("total_size_bytes")
    ]
    median_day = sorted(recent_sizes)[len(recent_sizes) // 2] if recent_sizes else 0

    for d in needed_days:
        key = d.isoformat()
        if key in partitions:
            present.append(key)
            est_bytes += int(partitions[key].get("total_size_bytes") or 0)
        else:
            missing.append(key)
            est_bytes += median_day

    result["txs_partitions_for_corpus"] = present
    result["txs_partitions_missing"] = missing
    result["txs_coverage_complete"] = len(missing) == 0
    result["estimated_download_bytes_if_present"] = est_bytes
    result["estimated_download_gib_if_present"] = round(est_bytes / (1024**3), 2)

    # Accounts are monthly.
    months = {
        f"{CORPUS_START.year}-{CORPUS_START.month:02d}",
        f"{CORPUS_END.year}-{CORPUS_END.month:02d}",
    }
    acct_present = []
    acct_missing = []
    acct_bytes = 0
    for month in sorted(months):
        idx = accounts_root / month / "index.json"
        if idx.exists():
            meta = json.loads(idx.read_text(encoding="utf-8"))
            acct_present.append(month)
            acct_bytes += int(meta.get("total_size_bytes") or 0)
        else:
            acct_missing.append(month)
    result["accounts_partitions_for_corpus"] = acct_present
    result["accounts_partitions_missing"] = acct_missing
    result["accounts_coverage_complete"] = len(acct_missing) == 0
    result["accounts_estimated_bytes_if_present"] = acct_bytes or None

    if not result["txs_coverage_complete"]:
        result["notes"].append(
            "Solarchive txs index has no June–July 2026 daily partitions "
            f"(max={result['txs_partition_max']}). Do not download archive for this corpus."
        )
        result["decision"] = "skip_no_corpus_coverage"
    elif est_bytes > 500 * 1024**3:
        result["notes"].append(
            "Full-day Solarchive txs for corpus window would exceed 500 GiB; "
            "prefer BigQuery signature/mint-filtered queries instead."
        )
        result["decision"] = "uneconomical_full_day_download"
    else:
        result["decision"] = "candidate_targeted_download"

    return result


def _bq_client(project: str | None):
    from google.cloud import bigquery

    return bigquery.Client(project=project) if project else bigquery.Client()


def _normalize_table_id(name: str) -> str:
    return name.replace(" ", "_").lower()


def probe_bigquery(
    *,
    project: str | None = None,
    max_bytes_billed: int = BQ_MAX_BYTES_BILLED_DEFAULT,
    skip_live: bool = False,
) -> dict[str, Any]:
    """Probe BigQuery public Solana dataset with dry-runs only for cost estimates."""
    project = project or os.environ.get(BQ_PROJECT_ENV) or os.environ.get("GCP_PROJECT")
    out: dict[str, Any] = {
        "dataset": BQ_DATASET,
        "project": project,
        "auth_available": False,
        "skip_live": skip_live,
        "tables": {},
        "corpus_coverage": {},
        "dry_runs": [],
        "bytes_estimated_total": 0,
        "decision": "auth_missing",
        "notes": [],
    }
    if skip_live:
        out["decision"] = "skipped_for_ci"
        out["notes"].append("Live BigQuery probe skipped (CI / --skip-live).")
        return out

    # Avoid long metadata-server hangs when no credentials are configured.
    if not project and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        if not adc.exists():
            out["notes"].append(
                "No GOOGLE_CLOUD_PROJECT / GOOGLE_APPLICATION_CREDENTIALS / ADC file; "
                "skipping live BigQuery probe."
            )
            out["decision"] = "auth_missing"
            return out

    try:
        from google.auth.exceptions import DefaultCredentialsError
        from google.cloud import bigquery
    except ImportError:
        out["notes"].append("google-cloud-bigquery not installed")
        return out

    try:
        client = _bq_client(project)
        # Force credential resolution.
        _ = client.project
        out["auth_available"] = True
        out["project"] = client.project
    except DefaultCredentialsError as exc:
        out["notes"].append(f"ADC missing: {exc}")
        out["decision"] = "auth_missing"
        return out
    except Exception as exc:  # noqa: BLE001
        out["notes"].append(f"client init failed: {exc}")
        out["decision"] = "auth_missing"
        return out

    for table_name in BQ_TABLES:
        table_id = f"{BQ_DATASET}.{table_name}"
        alt_id = f"{BQ_DATASET}.{_normalize_table_id(table_name)}"
        entry: dict[str, Any] = {"requested": table_name, "resolved": None, "schema": []}
        for candidate in (table_id, alt_id, f"{BQ_DATASET}.{table_name.title()}"):
            try:
                table = client.get_table(candidate)
                entry["resolved"] = candidate
                entry["num_rows"] = table.num_rows
                entry["num_bytes"] = table.num_bytes
                entry["schema"] = [
                    {"name": f.name, "type": f.field_type, "mode": f.mode}
                    for f in table.schema
                ]
                break
            except Exception as exc:  # noqa: BLE001
                entry.setdefault("errors", []).append(f"{candidate}: {exc}")
        out["tables"][table_name] = entry

    # Dry-run coverage queries with strict partition filters.
    start_ts = f"{CORPUS_START.isoformat()} 00:00:00"
    end_ts = f"{(CORPUS_END + timedelta(days=1)).isoformat()} 00:00:00"
    coverage_sql = {
        "transactions_day_counts": f"""
            SELECT DATE(block_timestamp) AS d, COUNT(*) AS n
            FROM `{out['tables'].get('Transactions', {}).get('resolved')
                or f'{BQ_DATASET}.Transactions'}`
            WHERE block_timestamp >= TIMESTAMP('{start_ts}')
              AND block_timestamp < TIMESTAMP('{end_ts}')
            GROUP BY 1
            ORDER BY 1
        """,
        "token_transfers_day_counts": f"""
            SELECT DATE(block_timestamp) AS d, COUNT(*) AS n
            FROM `{out['tables'].get('Token Transfers', {}).get('resolved')
                or f'{BQ_DATASET}.Token Transfers'}`
            WHERE block_timestamp >= TIMESTAMP('{start_ts}')
              AND block_timestamp < TIMESTAMP('{end_ts}')
            GROUP BY 1
            ORDER BY 1
        """,
        "blocks_day_counts": f"""
            SELECT DATE(block_timestamp) AS d, COUNT(*) AS n
            FROM `{out['tables'].get('Blocks', {}).get('resolved')
                or f'{BQ_DATASET}.Blocks'}`
            WHERE block_timestamp >= TIMESTAMP('{start_ts}')
              AND block_timestamp < TIMESTAMP('{end_ts}')
            GROUP BY 1
            ORDER BY 1
        """,
    }

    total_est = 0
    for name, sql in coverage_sql.items():
        job_config = bigquery.QueryJobConfig(
            dry_run=True,
            use_query_cache=False,
            maximum_bytes_billed=max_bytes_billed,
        )
        try:
            job = client.query(sql, job_config=job_config)
            est = int(job.total_bytes_processed or 0)
            total_est += est
            record = {
                "name": name,
                "dry_run": True,
                "estimated_bytes": est,
                "estimated_gib": round(est / (1024**3), 4),
                "within_abort_threshold": est <= BQ_DRY_RUN_ABORT_BYTES,
                "sql_sha16": __import__("hashlib").sha256(sql.encode()).hexdigest()[:16],
            }
            out["dry_runs"].append(record)
            out["corpus_coverage"][name] = {
                "status": "dry_run_ok" if record["within_abort_threshold"] else "too_expensive",
                "estimated_bytes": est,
            }
        except Exception as exc:  # noqa: BLE001
            out["dry_runs"].append({"name": name, "error": str(exc)})
            out["corpus_coverage"][name] = {"status": "error", "error": str(exc)}

    out["bytes_estimated_total"] = total_est
    if any(r.get("within_abort_threshold") for r in out["dry_runs"] if "estimated_bytes" in r):
        out["decision"] = "usable_with_partition_filters"
    else:
        out["decision"] = "unavailable_or_uneconomical"
    return out


def probe_geckoterminal(
    *,
    sample_pool: str | None = None,
    skip_live: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "base_url": GECKO_BASE_URL,
        "network": GECKO_NETWORK,
        "sol_usdc_pool": GECKO_SOL_USDC_POOL,
        "skip_live": skip_live,
        "rate_limit_hint_per_min": 10,
        "decision": "skipped_for_ci" if skip_live else "unknown",
        "checks": {},
        "notes": [],
    }
    if skip_live:
        out["notes"].append("Live GeckoTerminal probe skipped (CI / --skip-live).")
        return out

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    def _get(path: str) -> tuple[int, Any]:
        time.sleep(GECKO_MIN_INTERVAL_SEC)
        url = f"{GECKO_BASE_URL}{path}"
        resp = session.get(url, timeout=60)
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            body = {"raw": resp.text[:500]}
        return resp.status_code, body

    code, body = _get(f"/networks/{GECKO_NETWORK}/pools/{GECKO_SOL_USDC_POOL}")
    out["checks"]["network"] = {
        "status_code": code,
        "ok": code == 200,
        "via": "sol_usdc_pool_lookup",
    }

    code, body = _get(
        f"/networks/{GECKO_NETWORK}/pools/{GECKO_SOL_USDC_POOL}/ohlcv/day"
        f"?aggregate=1&limit=100&currency=usd"
    )
    ohlcv = (
        body.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
        if isinstance(body, dict)
        else []
    )
    out["checks"]["sol_usdc_ohlcv"] = {
        "status_code": code,
        "ok": code == 200 and bool(ohlcv),
        "candles": len(ohlcv),
        "covers_corpus_window": _ohlcv_covers_corpus(ohlcv),
    }

    pool = sample_pool or GECKO_SOL_USDC_POOL
    code, body = _get(f"/networks/{GECKO_NETWORK}/pools/{pool}")
    out["checks"]["pool_lookup"] = {
        "status_code": code,
        "ok": code == 200,
        "pool": pool,
    }

    ok = (
        out["checks"]["network"]["ok"]
        and out["checks"]["sol_usdc_ohlcv"]["ok"]
        and out["checks"]["pool_lookup"]["ok"]
    )
    out["decision"] = "usable" if ok else "degraded_or_rate_limited"
    if not ok:
        out["notes"].append("One or more GeckoTerminal checks failed; may be rate-limited.")
    return out


def _ohlcv_covers_corpus(ohlcv: list[Any]) -> bool:
    if not ohlcv:
        return False
    times = [int(row[0]) for row in ohlcv if row]
    if not times:
        return False
    start = datetime(CORPUS_START.year, CORPUS_START.month, CORPUS_START.day).timestamp()
    end = datetime(CORPUS_END.year, CORPUS_END.month, CORPUS_END.day).timestamp()
    return min(times) <= start and max(times) >= end


def run_source_probe(
    *,
    external_root: Path,
    manifest_dir: Path,
    skip_live: bool = False,
    bq_project: str | None = None,
    sample_pool: str | None = None,
) -> dict[str, Any]:
    solarchive = probe_solarchive(external_root / "solarchive")
    bq = probe_bigquery(project=bq_project, skip_live=skip_live)
    gecko = probe_geckoterminal(sample_pool=sample_pool, skip_live=skip_live)
    pumpfun = external_root / "pumpfun_memecoin_corpus"
    payload = {
        "generated_at": utc_now(),
        "corpus_window": {
            "start": CORPUS_START.isoformat(),
            "end": CORPUS_END.isoformat(),
        },
        "local_anchors": {
            "pumpfun_present": (pumpfun / "tokens.parquet").exists(),
            "red_pump_present": (external_root / "red_pump_2026").exists(),
            "red_cohort_present": (external_root / "red_cohort_2026").exists(),
            "postgard_snapshots_present": (pumpfun / "postgard_snapshots.parquet").exists(),
        },
        "bigquery": bq,
        "solarchive": solarchive,
        "geckoterminal": gecko,
        "source_priority": [
            "pumpfun_local_for_events_curve_execution",
            "bigquery_for_microstructure_transfers_funding_if_auth",
            "rpc_getTransaction_for_signature_microstructure_subset",
            "geckoterminal_for_sol_market_and_postgrad_pools",
            "solarchive_skip_no_2026_coverage",
        ],
    }
    write_json(manifest_dir / "source_probe.json", payload)
    return payload
