"""Dune dataset configuration loader."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

PACKAGE = "trencher.queries"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    schema_dataset: str
    description: str
    sql_file: str
    partition_column: str
    source_tables: tuple[str, ...]
    query_id: int | None = None
  # optional saved Dune query ID override

    @property
    def sql_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "queries" / "sql" / self.sql_file


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
            query_id=spec.get("query_id"),
        )
    return datasets


def load_sql_template(dataset: DatasetConfig) -> str:
    sql_path = dataset.sql_path
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL template missing for {dataset.name}: {sql_path}")
    return sql_path.read_text()


def render_sql(template: str, *, start_ts: str, end_ts: str) -> str:
    return template.replace("{{start_ts}}", start_ts).replace("{{end_ts}}", end_ts)


def package_sql_exists() -> bool:
    try:
        return resources.files(PACKAGE).joinpath("sql").is_dir()
    except (ModuleNotFoundError, TypeError):
        return False
