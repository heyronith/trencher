"""Preflight tests (mocked Dune)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from trencher.cli import main
from trencher.dune.backfill import backfill_partition
from trencher.dune.client import DuneExecutionError, QueryExecutionResult
from trencher.dune.config import load_datasets_config, load_enabled_tables, save_enabled_tables
from trencher.dune.preflight import REQUIRED_COLUMNS, explain_validate_sql, run_preflight


class QueueRunner:
    def __init__(self, frames: list[pd.DataFrame]) -> None:
        self.frames = list(frames)
        self.calls: list[dict] = []
        self.explain_error: Exception | None = None

    def run_sql(self, sql: str, **kwargs) -> QueryExecutionResult:
        self.calls.append({"sql": sql, **kwargs})
        if sql.lstrip().upper().startswith("EXPLAIN"):
            if self.explain_error is not None:
                raise self.explain_error
            return QueryExecutionResult(
                execution_id="explain-1",
                query_id=None,
                row_count=1,
                dataframe=pd.DataFrame({"Valid": [True]}),
            )
        df = self.frames.pop(0) if self.frames else pd.DataFrame()
        return QueryExecutionResult(
            execution_id="preflight-1",
            query_id=None,
            row_count=len(df),
            dataframe=df,
        )


def _catalog_frame(*, extra: dict[str, tuple[str, ...]] | None = None) -> pd.DataFrame:
    rows = []
    for table, cols in REQUIRED_COLUMNS.items():
        schema, name = table.split(".", 1)
        for col in cols:
            rows.append({"table_schema": schema, "table_name": name, "column_name": col})
    if extra:
        for table, cols in extra.items():
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
    cache = tmp_data_root / ".preflight-tables.json"
    assert cache.exists()
    assert "columns" in cache.read_text()
    assert "information_schema" in runner.calls[0]["sql"]
    explain_calls = [c for c in runner.calls if c["sql"].lstrip().upper().startswith("EXPLAIN")]
    assert len(explain_calls) == len(load_datasets_config())
    assert report.data_root == str(tmp_data_root)
    assert all(v["method"] == "explain_validate" for v in report.sql_validations)


def test_preflight_persists_when_data_root_is_default(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "from-env"
    monkeypatch.setenv("TRENCHER_DATA_ROOT", str(root))
    runner = QueueRunner([_catalog_frame()])
    report = run_preflight(runner, data_root=None, include_samples=False)
    assert report.ok
    assert (root / ".preflight-tables.json").exists()
    assert load_enabled_tables(root)


def test_preflight_missing_required_table_fails() -> None:
    runner = QueueRunner([pd.DataFrame()])
    report = run_preflight(runner, include_samples=False, include_sql_validation=False)
    assert not report.ok
    assert any("Dune table missing" in err for err in report.errors)


def test_preflight_fails_on_unguarded_missing_optional_column(tmp_data_root: Path) -> None:
    runner = QueueRunner([_catalog_frame()])
    report = run_preflight(
        runner,
        data_root=tmp_data_root,
        include_samples=False,
        include_sql_validation=True,
    )
    # Production templates are guarded, so a missing optional ix_name is only a warning.
    assert report.ok
    assert any("ix_name" in warn for warn in report.warnings)

    unguarded = """
SELECT ix_name
FROM pumpdotfun_solana.pump_evt_tradeevent
WHERE evt_block_time >= TIMESTAMP '{{start_ts}}'
  AND evt_block_time < TIMESTAMP '{{end_ts}}'
"""
    from trencher.dune.preflight import unguarded_optional_column_refs

    refs = unguarded_optional_column_refs(
        unguarded, ["pumpdotfun_solana.pump_evt_tradeevent"]
    )
    assert "pumpdotfun_solana.pump_evt_tradeevent.ix_name" in refs


def test_preflight_errors_when_rendered_sql_uses_missing_optional_column(
    tmp_data_root: Path, monkeypatch
) -> None:
    from trencher.dune import preflight as preflight_mod
    from trencher.dune.config import DatasetConfig

    fake = DatasetConfig(
        name="leaky",
        schema_dataset="token_lifecycle",
        description="unguarded optional column",
        sql_file="token_lifecycle.sql",
        partition_column="evt_block_time",
        source_tables=("pumpdotfun_solana.pump_evt_tradeevent",),
    )
    monkeypatch.setattr(
        preflight_mod,
        "load_datasets_config",
        lambda path=None: {"leaky": fake},
    )
    monkeypatch.setattr(
        preflight_mod,
        "load_sql_template",
        lambda cfg: (
            "SELECT ix_name FROM pumpdotfun_solana.pump_evt_tradeevent "
            "WHERE evt_block_time >= TIMESTAMP '{{start_ts}}' "
            "AND evt_block_time < TIMESTAMP '{{end_ts}}'"
        ),
    )
    runner = QueueRunner(
        [
            pd.DataFrame(
                [
                    {
                        "table_schema": "pumpdotfun_solana",
                        "table_name": "pump_evt_tradeevent",
                        "column_name": col,
                    }
                    for col in REQUIRED_COLUMNS["pumpdotfun_solana.pump_evt_tradeevent"]
                ]
            )
        ]
    )
    report = run_preflight(
        runner,
        data_root=tmp_data_root,
        include_samples=False,
        include_sql_validation=False,
    )
    assert not report.ok
    assert any("unavailable column" in err and "ix_name" in err for err in report.errors)


def test_preflight_fails_when_explain_rejects(tmp_data_root: Path) -> None:
    runner = QueueRunner([_catalog_frame()])
    runner.explain_error = DuneExecutionError("Column 'ix_name' cannot be resolved")
    report = run_preflight(
        runner,
        data_root=tmp_data_root,
        include_samples=False,
    )
    assert not report.ok
    assert any("EXPLAIN (TYPE VALIDATE) failed" in err for err in report.errors)


def test_preflight_falls_back_when_explain_unsupported(tmp_data_root: Path) -> None:
    class FallbackRunner(QueueRunner):
        def run_sql(self, sql: str, **kwargs) -> QueryExecutionResult:
            self.calls.append({"sql": sql, **kwargs})
            if sql.lstrip().upper().startswith("EXPLAIN"):
                raise DuneExecutionError("line 1:1: mismatched input 'EXPLAIN'")
            if " _trencher_validate " in sql:
                return QueryExecutionResult(
                    execution_id="zero-row",
                    query_id=None,
                    row_count=0,
                    dataframe=pd.DataFrame(),
                )
            df = self.frames.pop(0) if self.frames else pd.DataFrame()
            return QueryExecutionResult(
                execution_id="preflight-1",
                query_id=None,
                row_count=len(df),
                dataframe=df,
            )

    runner = FallbackRunner([_catalog_frame()])
    report = run_preflight(
        runner,
        data_root=tmp_data_root,
        include_samples=False,
    )
    assert report.ok
    assert any("EXPLAIN (TYPE VALIDATE) unsupported" in w for w in report.warnings)
    assert any(v["method"] == "zero_row" for v in report.sql_validations)


def test_backfill_reads_preflight_cache_to_strip_optional_tables(
    tmp_data_root: Path, sample_token_lifecycle_df
) -> None:
    save_enabled_tables(
        tmp_data_root,
        {"pumpdotfun_solana.pump_call_create", "pumpdotfun_solana.pump_call_migrate"},
        {
            "pumpdotfun_solana.pump_call_create": {"creator", "account_mint"},
            "pumpdotfun_solana.pump_call_migrate": {"account_mint"},
        },
    )
    from trencher.dune.client import MockDuneRunner

    runner = MockDuneRunner()
    runner.default_response = sample_token_lifecycle_df
    cfg = load_datasets_config()["token_lifecycle"]
    backfill_partition(runner, cfg, date(2025, 1, 1), data_root=tmp_data_root)
    sql = runner.calls[0]["sql"]
    assert "pump_call_create_v2" not in sql
    assert "pump_call_create" in sql


def test_explain_validate_sql_wraps_select() -> None:
    wrapped = explain_validate_sql("SELECT 1")
    assert wrapped.startswith("EXPLAIN (TYPE VALIDATE)")
    assert "SELECT 1" in wrapped


def test_cli_validate_queries_is_wired() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["dune", "validate-queries", "--help"])
    assert result.exit_code == 0
    assert "Low-cost" in result.output
