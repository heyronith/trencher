"""Resumable idempotent Dune backfill engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trencher.dune.checkpoint import mark_partition_complete
from trencher.dune.client import DuneRunner
from trencher.dune.config import DatasetConfig, load_datasets_config, load_sql_template, render_sql
from trencher.dune.manifest import (
    PartitionManifest,
    is_partition_complete,
    read_manifest,
    sql_hash,
    utc_now_iso,
    write_manifest,
)
from trencher.schemas.registry import get_schema

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path("data/raw/dune")


@dataclass
class BackfillResult:
    dataset: str
    partition_date: date
    skipped: bool
    dry_run: bool
    row_count: int
    parquet_path: Path | None
    manifest_path: Path | None


def _default_data_root() -> Path:
    import os

    return Path(os.environ.get("TRENCHER_DATA_ROOT", DEFAULT_DATA_ROOT))


def iter_date_chunks(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    if end < start:
        raise ValueError("end date must be >= start date")
    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")

    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def partition_dir(data_root: Path, dataset: str, partition_date: date) -> Path:
    return data_root / dataset / f"dt={partition_date.isoformat()}"


def _ts_bounds(partition_start: date, partition_end: date) -> tuple[str, str]:
    start_ts = datetime.combine(partition_start, datetime.min.time(), tzinfo=UTC).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    end_exclusive = datetime.combine(
        partition_end + timedelta(days=1), datetime.min.time(), tzinfo=UTC
    ).strftime("%Y-%m-%d %H:%M:%S")
    return start_ts, end_exclusive


def _coerce_to_schema(df: pd.DataFrame, schema_dataset: str) -> pd.DataFrame:
    schema = get_schema(schema_dataset)
    if df.empty:
        dtypes = _pandas_dtypes(schema)
        return pd.DataFrame(
            {name: pd.Series(dtype=pa_dtype) for name, pa_dtype in dtypes.items()}
        )
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=False)
    return table.to_pandas(types_mapper=pd.ArrowDtype)


def _pandas_dtypes(schema: pa.Schema) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for field in schema:
        if pa.types.is_timestamp(field.type):
            mapping[field.name] = "datetime64[ns, UTC]"
        elif pa.types.is_date(field.type):
            mapping[field.name] = "object"
        elif pa.types.is_boolean(field.type):
            mapping[field.name] = "boolean"
        elif pa.types.is_integer(field.type):
            mapping[field.name] = "Int64"
        elif pa.types.is_floating(field.type):
            mapping[field.name] = "float64"
        elif pa.types.is_decimal(field.type):
            mapping[field.name] = "object"
        else:
            mapping[field.name] = "string"
    return mapping


def _write_parquet(df: pd.DataFrame, out_dir: Path, schema_dataset: str) -> tuple[Path, str]:
    schema = get_schema(schema_dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_name = "part-00000.parquet"
    out_path = out_dir / rel_name
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=False)
    pq.write_table(table, out_path, compression="zstd")
    from trencher.dune.manifest import file_sha256

    return out_path, file_sha256(out_path)


def backfill_partition(
    runner: DuneRunner,
    dataset_cfg: DatasetConfig,
    partition_date: date,
    *,
    data_root: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    schema_version: str = "1.0.0",
) -> BackfillResult:
    root = data_root or _default_data_root()
    out_dir = partition_dir(root, dataset_cfg.name, partition_date)

    if not force and is_partition_complete(out_dir):
        manifest = read_manifest(out_dir)
        logger.info(
            "Skipping complete partition %s/%s (%d rows)",
            dataset_cfg.name,
            partition_date,
            manifest.row_count if manifest else 0,
        )
        parquet_name = manifest.parquet_files[0] if manifest else "part-00000.parquet"
        return BackfillResult(
            dataset=dataset_cfg.name,
            partition_date=partition_date,
            skipped=True,
            dry_run=dry_run,
            row_count=manifest.row_count if manifest else 0,
            parquet_path=out_dir / parquet_name,
            manifest_path=out_dir / "manifest.json",
        )

    start_ts, end_ts = _ts_bounds(partition_date, partition_date)
    template = load_sql_template(dataset_cfg)
    sql = render_sql(template, start_ts=start_ts, end_ts=end_ts)
    hashed = sql_hash(sql)

    if dry_run:
        logger.info(
            "Dry run %s %s: would execute SQL hash %s",
            dataset_cfg.name,
            partition_date,
            hashed[:12],
        )
        return BackfillResult(
            dataset=dataset_cfg.name,
            partition_date=partition_date,
            skipped=False,
            dry_run=True,
            row_count=0,
            parquet_path=None,
            manifest_path=None,
        )

    result = runner.run_sql(sql, query_id=dataset_cfg.query_id)
    df = _coerce_to_schema(result.dataframe, dataset_cfg.schema_dataset)

    parquet_path, checksum = _write_parquet(df, out_dir, dataset_cfg.schema_dataset)
    rel_name = parquet_path.name

    manifest = PartitionManifest(
        dataset=dataset_cfg.name,
        schema_version=schema_version,
        partition_date=partition_date.isoformat(),
        sql_hash=hashed,
        sql=sql,
        parameters={"start_ts": start_ts, "end_ts": end_ts},
        dune_execution_id=result.execution_id,
        dune_query_id=result.query_id,
        extracted_at=utc_now_iso(),
        row_count=len(df),
        source_time_start=start_ts,
        source_time_end=end_ts,
        source_tables=list(dataset_cfg.source_tables),
        parquet_files=[rel_name],
        file_checksums={rel_name: checksum},
        schema_dataset=dataset_cfg.schema_dataset,
    )
    manifest_path = write_manifest(out_dir, manifest)
    mark_partition_complete(root, dataset_cfg.name, partition_date)

    logger.info(
        "Wrote %s/%s: %d rows -> %s",
        dataset_cfg.name,
        partition_date,
        len(df),
        parquet_path,
    )
    return BackfillResult(
        dataset=dataset_cfg.name,
        partition_date=partition_date,
        skipped=False,
        dry_run=False,
        row_count=len(df),
        parquet_path=parquet_path,
        manifest_path=manifest_path,
    )


def backfill_range(
    runner: DuneRunner,
    dataset_name: str,
    start: date,
    end: date,
    *,
    chunk_days: int = 1,
    data_root: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    config_path: Path | None = None,
) -> list[BackfillResult]:
    datasets = load_datasets_config(config_path)
    if dataset_name not in datasets:
        known = ", ".join(sorted(datasets))
        raise KeyError(f"Unknown dataset {dataset_name!r}. Known: {known}")
    dataset_cfg = datasets[dataset_name]
    root = data_root or _default_data_root()
    results: list[BackfillResult] = []

    for chunk_start, chunk_end in iter_date_chunks(start, end, chunk_days):
        # One output partition per calendar day within the chunk.
        day = chunk_start
        while day <= chunk_end:
            results.append(
                backfill_partition(
                    runner,
                    dataset_cfg,
                    day,
                    data_root=root,
                    force=force,
                    dry_run=dry_run,
                )
            )
            day += timedelta(days=1)
    return results
