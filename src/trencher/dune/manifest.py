"""Partition manifest and provenance tracking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class PartitionManifest:
    dataset: str
    schema_version: str
    partition_date: str
    sql_hash: str
    sql: str
    parameters: dict[str, Any]
    dune_execution_id: str | None
    dune_query_id: int | None
    extracted_at: str
    row_count: int
    source_time_start: str
    source_time_end: str
    source_tables: list[str]
    parquet_files: list[str]
    file_checksums: dict[str, str]
    schema_dataset: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PartitionManifest:
        return cls(**data)


def sql_hash(sql: str) -> str:
    return hashlib.sha256(sql.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(partition_dir: Path) -> Path:
    return partition_dir / "manifest.json"


def write_manifest(partition_dir: Path, manifest: PartitionManifest) -> Path:
    partition_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(partition_dir)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def read_manifest(partition_dir: Path) -> PartitionManifest | None:
    path = manifest_path(partition_dir)
    if not path.exists():
        return None
    return PartitionManifest.from_dict(json.loads(path.read_text()))


def is_partition_complete(partition_dir: Path) -> bool:
    manifest = read_manifest(partition_dir)
    if manifest is None or not manifest.parquet_files:
        return False
    for rel in manifest.parquet_files:
        parquet = partition_dir / rel
        if not parquet.exists():
            return False
        if file_sha256(parquet) != manifest.file_checksums.get(rel):
            return False
    return True


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
