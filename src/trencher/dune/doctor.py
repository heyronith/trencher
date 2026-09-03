"""Environment and configuration validation (no Dune credits)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from trencher import __version__
from trencher.dune.config import load_datasets_config, load_sql_template, resolve_data_root
from trencher.schemas.registry import DATASET_SCHEMAS, get_schema


@dataclass
class DoctorReport:
    ok: bool
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_ok(self, msg: str) -> None:
        self.checks.append(msg)

    def add_warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False


def run_doctor(*, config_path: Path | None = None, data_root: Path | None = None) -> DoctorReport:
    report = DoctorReport(ok=True)

    report.add_ok(f"trencher {__version__}")

    api_key = os.environ.get("DUNE_API_KEY")
    if api_key:
        report.add_ok("DUNE_API_KEY is set")
    else:
        report.add_warn("DUNE_API_KEY is not set (required for backfill, not for tests)")

    perf = os.environ.get("TRENCHER_DUNE_PERFORMANCE", "medium")
    if perf not in {"medium", "large"}:
        report.add_error(f"Invalid TRENCHER_DUNE_PERFORMANCE={perf!r}; use medium or large")
    else:
        report.add_ok(f"Dune performance tier: {perf}")

    # Schema registry
    for name in sorted(DATASET_SCHEMAS):
        schema = get_schema(name)
        report.add_ok(f"schema {name}: {len(schema)} fields")

    # Dataset config + SQL templates
    try:
        datasets = load_datasets_config(config_path)
    except Exception as exc:
        report.add_error(f"Failed to load datasets config: {exc}")
        return report

    report.add_ok(f"Loaded {len(datasets)} dataset definitions")

    for name, cfg in sorted(datasets.items()):
        if cfg.schema_dataset not in DATASET_SCHEMAS:
            report.add_error(
                f"Dataset {name} references unknown schema {cfg.schema_dataset!r}"
            )
            continue
        try:
            sql = load_sql_template(cfg)
        except FileNotFoundError as exc:
            report.add_error(str(exc))
            continue
        if "{{start_ts}}" not in sql or "{{end_ts}}" not in sql:
            report.add_error(f"SQL for {name} missing {{start_ts}}/{{end_ts}} placeholders")
        else:
            report.add_ok(f"SQL template {cfg.sql_file} ({len(sql)} bytes)")
        for table in cfg.source_tables:
            report.add_ok(f"  source table documented: {table}")

    root = resolve_data_root(data_root)
    report.add_ok(f"Data root: {root} (writable check skipped in doctor)")
    report.add_ok("live preflight: uv run trencher dune validate-queries")

    return report
