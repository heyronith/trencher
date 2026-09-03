"""Preflight tests (mocked Dune)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from trencher.cli import main
from trencher.dune.client import QueryExecutionResult
from trencher.dune.preflight import REQUIRED_COLUMNS, run_preflight


class QueueRunner:
    def __init__(self, frames: list[pd.DataFrame]) -> None:
        self.frames = list(frames)
        self.calls: list[dict] = []

    def run_sql(self, sql: str, **kwargs) -> QueryExecutionResult:
        self.calls.append({"sql": sql, **kwargs})
        df = self.frames.pop(0) if self.frames else pd.DataFrame()
        return QueryExecutionResult(
            execution_id="preflight-1",
            query_id=None,
            row_count=len(df),
            dataframe=df,
        )


def _catalog_frame() -> pd.DataFrame:
    rows = []
    for table, cols in REQUIRED_COLUMNS.items():
        schema, name = table.split(".", 1)
        for col in cols:
            rows.append({"table_schema": schema, "table_name": name, "column_name": col})
    return pd.DataFrame(rows)


def test_preflight_catalog_and_cache(tmp_data_root: Path) -> None:
    runner = QueueRunner([_catalog_frame()])
    report = run_preflight(
        runner,
        data_root=tmp_data_root,
        include_samples=False,
    )
    assert report.ok
    assert "pumpdotfun_solana.pump_call_create" in report.existing_tables
    assert (tmp_data_root / ".preflight-tables.json").exists()
    assert "information_schema" in runner.calls[0]["sql"]


def test_preflight_missing_required_table_fails() -> None:
    runner = QueueRunner([pd.DataFrame()])
    report = run_preflight(runner, include_samples=False)
    assert not report.ok
    assert any("Dune table missing" in err for err in report.errors)


def test_cli_validate_queries_is_wired() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["dune", "validate-queries", "--help"])
    assert result.exit_code == 0
    assert "Low-cost" in result.output
