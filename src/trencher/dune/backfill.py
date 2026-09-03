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
from trencher.dune.config import (
    DatasetConfig,
    load_datasets_config,
    load_enabled_tables,
    load_sql_template,
    render_sql,
)
from trencher.dune.manifest import (
    PartitionManifest,
    is_partition_complete,
    read_manifest,
    sql_hash,
    utc_now_iso,
    write_manifest,
)
from trencher.queries.constants import SCHEMA_VERSION
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


def iter_days(start: date, end: date) -> list[date]:
    days: list[date] = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


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


def _empty_frame(schema_dataset: str) -> pd.DataFrame:
    schema = get_schema(schema_dataset)
    dtypes = _pandas_dtypes(schema)
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in dtypes.items()})


def _coerce_to_schema(df: pd.DataFrame, schema_dataset: str) -> pd.DataFrame:
    schema = get_schema(schema_dataset)
    if df.empty:
        return _empty_frame(schema_dataset)
    aligned = df.copy()
    for field in schema:
        if field.name not in aligned.columns:
            aligned[field.name] = None
    aligned = aligned[[field.name for field in schema]]
    table = pa.Table.from_pandas(aligned, schema=schema, preserve_index=False, safe=False)
    return table.to_pandas(types_mapper=pd.ArrowDtype)


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").date()
    return ts.date()


def _split_by_block_date(
    df: pd.DataFrame,
    days: list[date],
    schema_dataset: str,
) -> dict[date, pd.DataFrame]:
    empty = _empty_frame(schema_dataset)
    grouped: dict[date, pd.DataFrame] = {day: empty.copy() for day in days}
    if df.empty:
        return grouped
    if "block_date" not in df.columns:
        if len(days) == 1:
            grouped[days[0]] = df
        else:
            logger.warning("Result missing block_date; assigning all rows to %s", days[0])
            grouped[days[0]] = df
        return grouped
    labels = df["block_date"].map(_as_date)
    for day in days:
        subset = df.loc[labels == day]
        grouped[day] = subset if not subset.empty else empty.copy()
    return grouped


def _write_parquet(df: pd.DataFrame, out_dir: Path, schema_dataset: str) -> tuple[Path, str]:
    schema = get_schema(schema_dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_name = "part-00000.parquet"
    out_path = out_dir / rel_name
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=False)
    pq.write_table(table, out_path, compression="zstd")
    from trencher.dune.manifest import file_sha256

    return out_path, file_sha256(out_path)


def _skipped_result(
    dataset: str,
    partition_date: date,
    out_dir: Path,
    dry_run: bool,
) -> BackfillResult:
    manifest = read_manifest(out_dir)
    parquet_name = manifest.parquet_files[0] if manifest else "part-00000.parquet"
    return BackfillResult(
        dataset=dataset,
        partition_date=partition_date,
        skipped=True,
        dry_run=dry_run,
        row_count=manifest.row_count if manifest else 0,
        parquet_path=out_dir / parquet_name,
        manifest_path=out_dir / "manifest.json",
    )


def _write_day_partition(
    *,
    dataset_cfg: DatasetConfig,
    partition_date: date,
    day_df: pd.DataFrame,
    sql: str,
    hashed: str,
    query_start_ts: str,
    query_end_ts: str,
    result_execution_id: str | None,
    result_query_id: int | None,
    data_root: Path,
    schema_version: str,
) -> BackfillResult:
    out_dir = partition_dir(data_root, dataset_cfg.name, partition_date)
    day_start, day_end = _ts_bounds(partition_date, partition_date)
    coerced = _coerce_to_schema(day_df, dataset_cfg.schema_dataset)
    parquet_path, checksum = _write_parquet(coerced, out_dir, dataset_cfg.schema_dataset)
    rel_name = parquet_path.name
    manifest = PartitionManifest(
        dataset=dataset_cfg.name,
        schema_version=schema_version,
        partition_date=partition_date.isoformat(),
        sql_hash=hashed,
        sql=sql,
        parameters={
            "start_ts": query_start_ts,
            "end_ts": query_end_ts,
            "partition_start_ts": day_start,
            "partition_end_ts": day_end,
        },
        dune_execution_id=result_execution_id,
        dune_query_id=result_query_id,
        extracted_at=utc_now_iso(),
        row_count=len(coerced),
        source_time_start=day_start,
        source_time_end=day_end,
        source_tables=list(dataset_cfg.source_tables),
        parquet_files=[rel_name],
        file_checksums={rel_name: checksum},
        schema_dataset=dataset_cfg.schema_dataset,
    )
    manifest_path = write_manifest(out_dir, manifest)
    mark_partition_complete(data_root, dataset_cfg.name, partition_date)
    logger.info(
        "Wrote %s/%s: %d rows -> %s",
        dataset_cfg.name,
        partition_date,
        len(coerced),
        parquet_path,
    )
    return BackfillResult(
        dataset=dataset_cfg.name,
        partition_date=partition_date,
        skipped=False,
        dry_run=False,
        row_count=len(coerced),
        parquet_path=parquet_path,
        manifest_path=manifest_path,
    )


def backfill_chunk(
    runner: DuneRunner,
    dataset_cfg: DatasetConfig,
    chunk_start: date,
    chunk_end: date,
    *,
    data_root: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    schema_version: str = SCHEMA_VERSION,
    enabled_tables: set[str] | None = None,
) -> list[BackfillResult]:
    """Execute one Dune query for [chunk_start, chunk_end] and write daily partitions."""
    root = data_root or _default_data_root()
    days = iter_days(chunk_start, chunk_end)
    if not force:
        incomplete = [
            day
            for day in days
            if not is_partition_complete(partition_dir(root, dataset_cfg.name, day))
        ]
        if not incomplete:
            return [
                _skipped_result(
                    dataset_cfg.name,
                    day,
                    partition_dir(root, dataset_cfg.name, day),
                    dry_run,
                )
                for day in days
            ]
        query_start, query_end = min(incomplete), max(incomplete)
        write_days = incomplete
    else:
        query_start, query_end = chunk_start, chunk_end
        write_days = days

    start_ts, end_ts = _ts_bounds(query_start, query_end)
    tables = enabled_tables if enabled_tables is not None else load_enabled_tables(root)
    template = load_sql_template(dataset_cfg)
    sql = render_sql(template, start_ts=start_ts, end_ts=end_ts, enabled_tables=tables)
    hashed = sql_hash(sql)

    if dry_run:
        logger.info(
            "Dry run %s %s..%s: would execute SQL hash %s",
            dataset_cfg.name,
            query_start,
            query_end,
            hashed[:12],
        )
        return [
            BackfillResult(
                dataset=dataset_cfg.name,
                partition_date=day,
                skipped=False,
                dry_run=True,
                row_count=0,
                parquet_path=None,
                manifest_path=None,
            )
            for day in write_days
        ]

    result = runner.run_sql(
        sql,
        query_id=dataset_cfg.query_id,
        parameters={"start_ts": start_ts, "end_ts": end_ts},
    )
    df = _coerce_to_schema(result.dataframe, dataset_cfg.schema_dataset)
    by_day = _split_by_block_date(df, write_days, dataset_cfg.schema_dataset)
    return [
        _write_day_partition(
            dataset_cfg=dataset_cfg,
            partition_date=day,
            day_df=by_day[day],
            sql=sql,
            hashed=hashed,
            query_start_ts=start_ts,
            query_end_ts=end_ts,
            result_execution_id=result.execution_id,
            result_query_id=result.query_id,
            data_root=root,
            schema_version=schema_version,
        )
        for day in write_days
    ]


def backfill_partition(
    runner: DuneRunner,
    dataset_cfg: DatasetConfig,
    partition_date: date,
    *,
    data_root: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    schema_version: str = SCHEMA_VERSION,
    enabled_tables: set[str] | None = None,
) -> BackfillResult:
    results = backfill_chunk(
        runner,
        dataset_cfg,
        partition_date,
        partition_date,
        data_root=data_root,
        force=force,
        dry_run=dry_run,
        schema_version=schema_version,
        enabled_tables=enabled_tables,
    )
    return results[0]


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
    enabled_tables: set[str] | None = None,
) -> list[BackfillResult]:
    datasets = load_datasets_config(config_path)
    if dataset_name not in datasets:
        known = ", ".join(sorted(datasets))
        raise KeyError(f"Unknown dataset {dataset_name!r}. Known: {known}")
    dataset_cfg = datasets[dataset_name]
    root = data_root or _default_data_root()
    results: list[BackfillResult] = []
    for chunk_start, chunk_end in iter_date_chunks(start, end, chunk_days):
        results.extend(
            backfill_chunk(
                runner,
                dataset_cfg,
                chunk_start,
                chunk_end,
                data_root=root,
                force=force,
                dry_run=dry_run,
                enabled_tables=enabled_tables,
            )
        )
    return results
