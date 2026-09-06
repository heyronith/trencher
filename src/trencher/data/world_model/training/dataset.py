"""Streaming observation dataset over frozen split assignments."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from trencher.data.world_model.splits.cutoffs import SPLIT_TRAIN
from trencher.data.world_model.training.forbid import strip_forbidden


class StreamingSplitDataset:
    """Stream observation assignment rows for a base split without loading all into RAM."""

    def __init__(
        self,
        *,
        assignment_paths: list[Path],
        split: str = SPLIT_TRAIN,
        seed: int = 0,
        shuffle: bool = False,
        require_in_interval: bool = True,
        population: str = "broad",
        shuffle_paths: bool | None = None,
    ) -> None:
        self.assignment_paths = list(assignment_paths)
        self.split = split
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.require_in_interval = require_in_interval
        self.population = population
        # When False, keep caller-provided path order (used by worker-sharded loaders).
        self.shuffle_paths = bool(shuffle) if shuffle_paths is None else bool(shuffle_paths)

    def _row_ok(self, row: dict[str, Any]) -> bool:
        if row.get("base_split") != self.split:
            return False
        if self.require_in_interval and not row.get("obs_in_split_interval"):
            return False
        return not (self.population == "rich" and not row.get("in_rich_subset"))

    def iter_rows(self, *, max_examples: int | None = None) -> Iterator[dict[str, Any]]:
        paths = list(self.assignment_paths)
        if self.shuffle_paths:
            rng = np.random.default_rng(self.seed)
            order = rng.permutation(len(paths))
            paths = [paths[i] for i in order]
        n = 0
        for path in paths:
            rows = pq.read_table(path).to_pylist()
            if self.shuffle:
                rng = np.random.default_rng(
                    deterministic_shard_seed(self.seed, path.name)
                )
                order = rng.permutation(len(rows))
                rows = [rows[i] for i in order]
            for row in rows:
                if not self._row_ok(row):
                    continue
                yield dict(row)
                n += 1
                if max_examples is not None and n >= max_examples:
                    return

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self.iter_rows()

    def iter_batches(
        self, batch_size: int, *, max_batches: int | None = None
    ) -> Iterator[list[dict[str, Any]]]:
        max_examples = None if max_batches is None else batch_size * max_batches
        batch: list[dict[str, Any]] = []
        n_batches = 0
        for row in self.iter_rows(max_examples=max_examples):
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                n_batches += 1
                batch = []
                if max_batches is not None and n_batches >= max_batches:
                    return
        if batch:
            yield batch


def synthetic_example_from_assignment(row: dict[str, Any]) -> dict[str, Any]:
    """Minimal feature/target payload for smoke + unit tests (no identities)."""
    T = float(row["observation_time_t"])
    age = float(row.get("token_age_seconds") or 0.0)
    features = strip_forbidden(
        {
            "observation_time_t": T,
            "token_age_seconds": age,
            "time_delta_from_launch": age,
            "modality_mask_events": 1.0,
            "modality_mask_curve": 1.0,
            "modality_mask_creator": 1.0,
            "modality_mask_graph": 1.0 if row.get("in_rich_subset") else 0.0,
            "modality_mask_micro": 1.0 if row.get("in_rich_subset") else 0.0,
            "modality_mask_ownership": 1.0 if row.get("in_rich_subset") else 0.0,
        }
    )
    targets = {
        "returns_h60": None,
        "returns_h60_status": "modality_unavailable",
        "eligible_returns_h60": bool(row.get("eligible_returns_h60", False)),
        "explosion_within_60s": None,
        "explosion_within_60s_status": "modality_unavailable",
        "eligible_explosion_imminence_h60": bool(
            row.get("eligible_explosion_imminence_h60", False)
        ),
    }
    meta = {
        "observation_id": row.get("observation_id"),
        "base_split": row.get("base_split"),
        "population": "rich" if row.get("in_rich_subset") else "broad",
    }
    return {"features": features, "targets": targets, "meta": meta, "padding_mask": None}


def deterministic_shard_seed(base_seed: int, shard: str) -> int:
    h = hashlib.sha256(f"{base_seed}:{shard}".encode()).hexdigest()
    return int(h[:8], 16)
