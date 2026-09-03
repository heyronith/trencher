"""Backfill engine tests (mocked Dune)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pyarrow.parquet as pq
import pytest

from trencher.dune.backfill import (
    backfill_partition,
    backfill_range,
    iter_date_chunks,
    partition_dir,
)
from trencher.dune.client import MockDuneRunner
from trencher.dune.config import load_datasets_config
from trencher.dune.manifest import is_partition_complete, read_manifest
from trencher.schemas.registry import get_schema


def test_iter_date_chunks() -> None:
    chunks = iter_date_chunks(date(2025, 1, 1), date(2025, 1, 3), chunk_days=2)
    assert chunks == [(date(2025, 1, 1), date(2025, 1, 2)), (date(2025, 1, 3), date(2025, 1, 3))]


def test_iter_date_chunks_invalid() -> None:
    with pytest.raises(ValueError):
        iter_date_chunks(date(2025, 1, 3), date(2025, 1, 1), chunk_days=1)


def _empty_df_for_dataset(schema_dataset: str) -> pd.DataFrame:
    schema = get_schema(schema_dataset)
    return pd.DataFrame({name: pd.Series(dtype="object") for name in schema.names})


def test_backfill_writes_parquet_and_manifest(tmp_data_root, sample_token_lifecycle_df) -> None:
    datasets = load_datasets_config()
    cfg = datasets["token_lifecycle"]
    runner = MockDuneRunner(responses={"": sample_token_lifecycle_df})

    result = backfill_partition(
        runner,
        cfg,
        date(2025, 1, 1),
        data_root=tmp_data_root,
    )
    assert not result.skipped
    assert result.row_count == 1
    out_dir = partition_dir(tmp_data_root, "token_lifecycle", date(2025, 1, 1))
    assert is_partition_complete(out_dir)
    manifest = read_manifest(out_dir)
    assert manifest is not None
    assert manifest.sql_hash
    assert manifest.row_count == 1
    table = pq.read_table(out_dir / "part-00000.parquet")
    assert table.num_rows == 1


def test_backfill_idempotent_skip(tmp_data_root, sample_token_lifecycle_df) -> None:
    datasets = load_datasets_config()
    cfg = datasets["token_lifecycle"]
    runner = MockDuneRunner(responses={"": sample_token_lifecycle_df})
    backfill_partition(runner, cfg, date(2025, 1, 1), data_root=tmp_data_root)
    result = backfill_partition(runner, cfg, date(2025, 1, 1), data_root=tmp_data_root)
    assert result.skipped
    assert len(runner.calls) == 1


def test_backfill_force_requeries(tmp_data_root, sample_token_lifecycle_df) -> None:
    datasets = load_datasets_config()
    cfg = datasets["token_lifecycle"]
    runner = MockDuneRunner(responses={"": sample_token_lifecycle_df})
    backfill_partition(runner, cfg, date(2025, 1, 1), data_root=tmp_data_root)
    backfill_partition(runner, cfg, date(2025, 1, 1), data_root=tmp_data_root, force=True)
    assert len(runner.calls) == 2


def test_backfill_dry_run_no_writes(tmp_data_root) -> None:
    datasets = load_datasets_config()
    cfg = datasets["market_context"]
    runner = MockDuneRunner()
    result = backfill_partition(
        runner,
        cfg,
        date(2025, 1, 1),
        data_root=tmp_data_root,
        dry_run=True,
    )
    assert result.dry_run
    assert len(runner.calls) == 0
    assert not partition_dir(tmp_data_root, "market_context", date(2025, 1, 1)).exists()


def test_backfill_range_multiple_days(tmp_data_root, sample_token_lifecycle_df) -> None:
    runner = MockDuneRunner(responses={"": sample_token_lifecycle_df})
    results = backfill_range(
        runner,
        "token_lifecycle",
        date(2025, 1, 1),
        date(2025, 1, 2),
        data_root=tmp_data_root,
    )
    assert len(results) == 2
    assert all(not r.skipped for r in results)


def test_all_datasets_have_sql_templates() -> None:
    datasets = load_datasets_config()
    for name, cfg in datasets.items():
        assert cfg.sql_path.exists(), f"Missing SQL for {name}"
