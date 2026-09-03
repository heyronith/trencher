"""Dune dataset configuration loader."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

PACKAGE = "trencher.queries"

OPTIONAL_BLOCK = re.compile(
    r"/\* optional:(.+?) \*/\n?(.*?)\n?/\* /optional:\1 \*/",
    re.DOTALL,
)
OPTIONAL_COL_BLOCK = re.compile(
    r"/\* optional-col:((?:[A-Za-z0-9_]+\.){2}[A-Za-z0-9_]+):([A-Za-z0-9_]+) \*/"
    r"\s*(.*?)\s*"
    r"/\* /optional-col:\1 \*/",
    re.DOTALL,
)

PREFLIGHT_TABLES_FILE = ".preflight-tables.json"
DEFAULT_DATA_ROOT = Path("data/raw/dune")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    schema_dataset: str
    description: str
    sql_file: str
    partition_column: str
    source_tables: tuple[str, ...]
    optional_source_tables: tuple[str, ...] = ()
    query_id: int | None = None
    preflight_cost: str = "low"
    coverage_start: str | None = None

    @property
    def sql_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "queries" / "sql" / self.sql_file

    @property
    def required_source_tables(self) -> tuple[str, ...]:
        optional = set(self.optional_source_tables)
        return tuple(t for t in self.source_tables if t not in optional)


def _queries_root() -> Path:
    return Path(__file__).resolve().parents[1] / "queries"


def load_datasets_config(path: Path | None = None) -> dict[str, DatasetConfig]:
    config_path = path or (_queries_root() / "datasets.yaml")
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text())
    datasets: dict[str, DatasetConfig] = {}
    for name, spec in raw["datasets"].items():
        datasets[name] = DatasetConfig(
            name=name,
            schema_dataset=spec["schema_dataset"],
            description=spec["description"],
            sql_file=spec["sql_file"],
            partition_column=spec["partition_column"],
            source_tables=tuple(spec["source_tables"]),
            optional_source_tables=tuple(spec.get("optional_source_tables", ())),
            query_id=spec.get("query_id"),
            preflight_cost=spec.get("preflight_cost", "low"),
            coverage_start=spec.get("coverage_start"),
        )
    return datasets


def load_sql_template(dataset: DatasetConfig) -> str:
    sql_path = dataset.sql_path
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL template missing for {dataset.name}: {sql_path}")
    return sql_path.read_text()


def resolve_data_root(data_root: Path | None = None) -> Path:
    """Single effective data root for preflight cache and backfill output."""
    if data_root is not None:
        return Path(data_root)
    return Path(os.environ.get("TRENCHER_DATA_ROOT", DEFAULT_DATA_ROOT))


def strip_optional_blocks(template: str, enabled_tables: set[str] | None) -> str:
    """Keep optional SQL only for tables Dune actually exposes."""

    def repl(match: re.Match[str]) -> str:
        table = match.group(1).strip()
        body = match.group(2)
        if enabled_tables is None or table in enabled_tables:
            return body
        return ""

    return OPTIONAL_BLOCK.sub(repl, template)


def _table_and_column(full_name: str) -> tuple[str, str]:
    schema, table, column = full_name.rsplit(".", 2)
    return f"{schema}.{table}", column


def strip_optional_columns(
    template: str,
    enabled_columns: dict[str, set[str]] | None,
) -> str:
    """Replace optional-col blocks with NULL when the catalog lacks the column.

    ``enabled_columns is None`` keeps the original expression (full-SQL preview).
    A missing table key or column name is treated as absent.
    """

    def repl(match: re.Match[str]) -> str:
        full_name = match.group(1)
        sqltype = match.group(2)
        body = match.group(3)
        if enabled_columns is None:
            return body
        table, column = _table_and_column(full_name)
        have = {c.lower() for c in enabled_columns.get(table, set())}
        if column.lower() in have:
            return body
        return f"CAST(NULL AS {sqltype})"

    return OPTIONAL_COL_BLOCK.sub(repl, template)


def render_sql(
    template: str,
    *,
    start_ts: str,
    end_ts: str,
    enabled_tables: set[str] | None = None,
    enabled_columns: dict[str, set[str]] | None = None,
) -> str:
    sql = strip_optional_blocks(template, enabled_tables)
    sql = strip_optional_columns(sql, enabled_columns)
    return sql.replace("{{start_ts}}", start_ts).replace("{{end_ts}}", end_ts)


def preflight_tables_path(data_root: Path) -> Path:
    return resolve_data_root(data_root) / PREFLIGHT_TABLES_FILE


def _read_preflight_payload(data_root: Path | None) -> dict[str, Any] | None:
    if data_root is None:
        return None
    path = preflight_tables_path(data_root)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else None


def load_enabled_tables(data_root: Path | None) -> set[str] | None:
    payload = _read_preflight_payload(data_root)
    if not payload:
        return None
    tables = payload.get("existing_tables")
    if not tables:
        return None
    return set(tables)


def load_enabled_columns(data_root: Path | None) -> dict[str, set[str]] | None:
    payload = _read_preflight_payload(data_root)
    if not payload:
        return None
    columns = payload.get("columns")
    if not columns or not isinstance(columns, dict):
        return None
    return {str(table): {str(col) for col in cols} for table, cols in columns.items()}


def save_enabled_tables(
    data_root: Path,
    tables: set[str],
    columns: dict[str, set[str]] | None = None,
) -> Path:
    path = preflight_tables_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"existing_tables": sorted(tables)}
    if columns is not None:
        payload["columns"] = {
            table: sorted(cols) for table, cols in sorted(columns.items())
        }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def package_sql_exists() -> bool:
    try:
        return resources.files(PACKAGE).joinpath("sql").is_dir()
    except (ModuleNotFoundError, TypeError):
        return False
