"""Manifest and checksum tests."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.dune.manifest import (
    PartitionManifest,
    file_sha256,
    is_partition_complete,
    read_manifest,
    sql_hash,
    write_manifest,
)


def test_sql_hash_stable() -> None:
    assert sql_hash("SELECT 1") == sql_hash("SELECT 1")
    assert sql_hash("SELECT 1") != sql_hash("SELECT 2")


def test_manifest_round_trip(tmp_path: Path) -> None:
    partition = tmp_path / "dt=2025-01-01"
    partition.mkdir(parents=True)
    parquet = partition / "part-00000.parquet"
    table = pa.table({"block_slot": [1]}, schema=pa.schema([("block_slot", pa.int64())]))
    pq.write_table(table, parquet)

    manifest = PartitionManifest(
        dataset="market_context",
        schema_version="1.0.0",
        partition_date="2025-01-01",
        sql_hash=sql_hash("x"),
        sql="SELECT 1",
        parameters={"start_ts": "a", "end_ts": "b"},
        dune_execution_id="exec1",
        dune_query_id=None,
        extracted_at="2025-01-01T00:00:00+00:00",
        row_count=1,
        source_time_start="2025-01-01",
        source_time_end="2025-01-02",
        source_tables=["solana.blocks"],
        parquet_files=["part-00000.parquet"],
        file_checksums={"part-00000.parquet": file_sha256(parquet)},
        schema_dataset="market_context",
    )
    write_manifest(partition, manifest)
    loaded = read_manifest(partition)
    assert loaded is not None
    assert loaded.dataset == "market_context"
    assert loaded.row_count == 1


def test_is_partition_complete_detects_tampering(tmp_path: Path) -> None:
    partition = tmp_path / "dt=2025-01-01"
    partition.mkdir()
    parquet = partition / "part-00000.parquet"
    parquet.write_bytes(b"parquet")
    manifest = PartitionManifest(
        dataset="x",
        schema_version="1.0.0",
        partition_date="2025-01-01",
        sql_hash="abc",
        sql="SELECT 1",
        parameters={},
        dune_execution_id=None,
        dune_query_id=None,
        extracted_at="t",
        row_count=1,
        source_time_start="a",
        source_time_end="b",
        source_tables=[],
        parquet_files=["part-00000.parquet"],
        file_checksums={"part-00000.parquet": "wrong"},
        schema_dataset="market_context",
    )
    write_manifest(partition, manifest)
    assert not is_partition_complete(partition)
