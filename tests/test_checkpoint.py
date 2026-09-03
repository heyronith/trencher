"""Checkpoint tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from trencher.dune.checkpoint import load_checkpoint, mark_partition_complete, save_checkpoint


def test_checkpoint_tracks_partitions(tmp_data_root: Path) -> None:
    cp = mark_partition_complete(tmp_data_root, "token_lifecycle", date(2025, 1, 1))
    assert "2025-01-01" in cp.completed_partitions
    reloaded = load_checkpoint(tmp_data_root, "token_lifecycle")
    assert reloaded.last_success_date == "2025-01-01"


def test_checkpoint_idempotent(tmp_data_root: Path) -> None:
    mark_partition_complete(tmp_data_root, "trades_events", date(2025, 1, 2))
    cp = mark_partition_complete(tmp_data_root, "trades_events", date(2025, 1, 2))
    assert cp.completed_partitions.count("2025-01-02") == 1


def test_save_and_load(tmp_data_root: Path) -> None:
    from trencher.dune.checkpoint import DatasetCheckpoint

    cp = DatasetCheckpoint(dataset="x", completed_partitions=["2025-01-01"])
    save_checkpoint(tmp_data_root, cp)
    loaded = load_checkpoint(tmp_data_root, "x")
    assert loaded.completed_partitions == ["2025-01-01"]
