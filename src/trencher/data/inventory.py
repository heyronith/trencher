"""Lightweight inventory of downloaded external datasets."""

from __future__ import annotations

import csv
import gzip
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trencher.data.paths import JOIN_KEY_CANDIDATES, TIMESTAMP_CANDIDATES

logger = logging.getLogger(__name__)


def _format_of(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith(".csv.gz"):
        return "csv.gz"
    if name.endswith(".jsonl.gz"):
        return "jsonl.gz"
    if name.endswith(".jsonl"):
        return "jsonl"
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".json"):
        return "json"
    if name.endswith(".md"):
        return "markdown"
    if name.endswith(".py"):
        return "python"
    if name.endswith(".zip"):
        return "zip"
    if name.endswith(".cff"):
        return "citation"
    if name.endswith(".svg"):
        return "svg"
    if name.endswith(".txt"):
        return "text"
    return path.suffix.lstrip(".") or "unknown"


def _parquet_inventory(path: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    columns = [field.name for field in schema]
    types = {field.name: str(field.type) for field in schema}
    join_keys = [c for c in columns if c.lower() in {k.lower() for k in JOIN_KEY_CANDIDATES}]
    ts_cols = [c for c in columns if c.lower() in {k.lower() for k in TIMESTAMP_CANDIDATES}]

    time_bounds: dict[str, dict[str, str | None]] = {}
    for col in ts_cols:
        col_idx = columns.index(col)
        mins: list[Any] = []
        maxs: list[Any] = []
        for rg_idx in range(pf.metadata.num_row_groups):
            stats = pf.metadata.row_group(rg_idx).column(col_idx).statistics
            if stats is None or not stats.has_min_max:
                continue
            mins.append(stats.min)
            maxs.append(stats.max)
        if mins and maxs:
            time_bounds[col] = {"min": str(min(mins)), "max": str(max(maxs))}

    return {
        "row_count": pf.metadata.num_rows,
        "columns": columns,
        "column_types": types,
        "join_keys": join_keys,
        "timestamp_bounds": time_bounds,
    }


def _count_text_rows(path: Path, *, gzipped: bool) -> int:
    opener = gzip.open if gzipped else open
    count = 0
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:  # type: ignore[arg-type]
        for _ in handle:
            count += 1
    return count


def _csv_inventory(path: Path, *, gzipped: bool) -> dict[str, Any]:
    opener = gzip.open if gzipped else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:  # type: ignore[arg-type]
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return {
                "row_count": 0,
                "columns": [],
                "column_types": {},
                "join_keys": [],
                "timestamp_bounds": {},
            }
        rows = sum(1 for _ in reader)
    columns = [c.strip() for c in header]
    join_keys = [c for c in columns if c.lower() in {k.lower() for k in JOIN_KEY_CANDIDATES}]
    return {
        "row_count": rows,
        "columns": columns,
        "column_types": {c: "string" for c in columns},
        "join_keys": join_keys,
        "timestamp_bounds": {},
    }


def _jsonl_inventory(path: Path, *, gzipped: bool) -> dict[str, Any]:
    opener = gzip.open if gzipped else open
    columns: list[str] = []
    seen: set[str] = set()
    rows = 0
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:  # type: ignore[arg-type]
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows += 1
            if rows <= 20:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    for key in obj:
                        if key not in seen:
                            seen.add(key)
                            columns.append(key)
    join_keys = [c for c in columns if c.lower() in {k.lower() for k in JOIN_KEY_CANDIDATES}]
    return {
        "row_count": rows,
        "columns": columns,
        "column_types": {c: "json" for c in columns},
        "join_keys": join_keys,
        "timestamp_bounds": {},
    }


def _json_inventory(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "row_count": None,
            "columns": [],
            "column_types": {},
            "join_keys": [],
            "timestamp_bounds": {},
            "error": str(exc),
        }
    if isinstance(payload, list):
        columns: list[str] = []
        if payload and isinstance(payload[0], dict):
            columns = list(payload[0].keys())
        return {
            "row_count": len(payload),
            "columns": columns,
            "column_types": {c: "json" for c in columns},
            "join_keys": [
                c
                for c in columns
                if c.lower() in {k.lower() for k in JOIN_KEY_CANDIDATES}
            ],
            "timestamp_bounds": {},
        }
    if isinstance(payload, dict):
        return {
            "row_count": 1,
            "columns": list(payload.keys()),
            "column_types": {c: "json" for c in payload},
            "join_keys": [
                c
                for c in payload
                if c.lower() in {k.lower() for k in JOIN_KEY_CANDIDATES}
            ],
            "timestamp_bounds": {},
        }
    return {
        "row_count": 1,
        "columns": [],
        "column_types": {},
        "join_keys": [],
        "timestamp_bounds": {},
    }


def inventory_file(path: Path, *, relative: str) -> dict[str, Any]:
    fmt = _format_of(path)
    base = {
        "path": relative,
        "format": fmt,
        "size": path.stat().st_size if path.exists() else 0,
        "row_count": None,
        "columns": [],
        "column_types": {},
        "join_keys": [],
        "timestamp_bounds": {},
        "openable": False,
        "error": None,
    }
    try:
        if fmt == "parquet":
            base.update(_parquet_inventory(path))
            base["openable"] = True
        elif fmt == "csv.gz":
            base.update(_csv_inventory(path, gzipped=True))
            base["openable"] = True
        elif fmt == "csv":
            base.update(_csv_inventory(path, gzipped=False))
            base["openable"] = True
        elif fmt == "jsonl.gz":
            base.update(_jsonl_inventory(path, gzipped=True))
            base["openable"] = True
        elif fmt == "jsonl":
            base.update(_jsonl_inventory(path, gzipped=False))
            base["openable"] = True
        elif fmt == "json":
            base.update(_json_inventory(path))
            base["openable"] = True
        elif fmt in {"markdown", "python", "text", "citation", "svg", "zip"}:
            if fmt == "zip":
                import zipfile

                with zipfile.ZipFile(path) as zf:
                    bad = zf.testzip()
                    if bad is not None:
                        raise ValueError(f"corrupt zip member: {bad}")
                    base["row_count"] = len(zf.namelist())
            else:
                # Ensure readable.
                with path.open("rb") as handle:
                    handle.read(64)
            base["openable"] = True
        else:
            with path.open("rb") as handle:
                handle.read(16)
            base["openable"] = True
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inventory failed for %s", path)
        base["error"] = str(exc)
        base["openable"] = False
    return base


def build_source_inventory(
    source_id: str,
    data_dir: Path,
    file_records: list[dict[str, Any]],
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for record in file_records:
        rel = str(record["path"])
        path = data_dir / rel
        if not path.exists() or not path.is_file():
            files.append(
                {
                    "path": rel,
                    "format": _format_of(path),
                    "size": 0,
                    "row_count": None,
                    "columns": [],
                    "column_types": {},
                    "join_keys": [],
                    "timestamp_bounds": {},
                    "openable": False,
                    "error": "missing",
                }
            )
            continue
        item = inventory_file(path, relative=rel)
        item["sha256"] = record.get("sha256")
        item["verified"] = record.get("verified")
        item["download_status"] = record.get("status")
        files.append(item)

    return {
        "source_id": source_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "file_count": len(files),
        "total_bytes": sum(int(f.get("size") or 0) for f in files),
        "files": files,
    }
