"""Dataset-level checkpoint state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass
class DatasetCheckpoint:
    dataset: str
    completed_partitions: list[str] = field(default_factory=list)
    last_success_date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "completed_partitions": sorted(self.completed_partitions),
            "last_success_date": self.last_success_date,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetCheckpoint:
        return cls(
            dataset=data["dataset"],
            completed_partitions=list(data.get("completed_partitions", [])),
            last_success_date=data.get("last_success_date"),
            metadata=dict(data.get("metadata", {})),
        )


def checkpoint_path(data_root: Path, dataset: str) -> Path:
    return data_root / ".checkpoints" / f"{dataset}.json"


def load_checkpoint(data_root: Path, dataset: str) -> DatasetCheckpoint:
    path = checkpoint_path(data_root, dataset)
    if not path.exists():
        return DatasetCheckpoint(dataset=dataset)
    return DatasetCheckpoint.from_dict(json.loads(path.read_text()))


def save_checkpoint(data_root: Path, checkpoint: DatasetCheckpoint) -> Path:
    path = checkpoint_path(data_root, checkpoint.dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def mark_partition_complete(
    data_root: Path,
    dataset: str,
    partition_date: date,
) -> DatasetCheckpoint:
    cp = load_checkpoint(data_root, dataset)
    key = partition_date.isoformat()
    if key not in cp.completed_partitions:
        cp.completed_partitions.append(key)
    cp.last_success_date = key
    save_checkpoint(data_root, cp)
    return cp
