"""Dune dataset configuration loader."""

from __future__ import annotations

import json
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

PREFLIGHT_TABLES_FILE = ".preflight-tables.json"


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


def strip_optional_blocks(template: str, enabled_tables: set[str] | None) -> str:
    """Keep optional SQL only for tables Dune actually exposes."""

    def repl(match: re.Match[str]) -> str:
        table = match.group(1).strip()
        body = match.group(2)
        if enabled_tables is None or table in enabled_tables:
            return body
        return ""

    return OPTIONAL_BLOCK.sub(repl, template)


def render_sql(
    template: str,
    *,
    start_ts: str,
    end_ts: str,
    enabled_tables: set[str] | None = None,
) -> str:
    sql = strip_optional_blocks(template, enabled_tables)
    return sql.replace("{{start_ts}}", start_ts).replace("{{end_ts}}", end_ts)


def preflight_tables_path(data_root: Path) -> Path:
    return data_root / PREFLIGHT_TABLES_FILE


def load_enabled_tables(data_root: Path | None) -> set[str] | None:
    if data_root is None:
        return None
    path = preflight_tables_path(data_root)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    tables = payload.get("existing_tables")
    if not tables:
        return None
    return set(tables)


def save_enabled_tables(data_root: Path, tables: set[str]) -> Path:
    path = preflight_tables_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"existing_tables": sorted(tables)}, indent=2) + "\n")
    return path


def package_sql_exists() -> bool:
    try:
        return resources.files(PACKAGE).joinpath("sql").is_dir()
    except (ModuleNotFoundError, TypeError):
        return False
