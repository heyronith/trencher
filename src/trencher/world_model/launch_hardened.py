"""Hardened world-model launch with deterministic finite-loader cycling and fast resume.

The frozen V4 loader is deterministic but finite. Production phases are step-budgeted,
so a phase may legitimately require more optimizer updates than one loader pass provides.
This entrypoint preserves the existing training loop and model semantics while wrapping
the frozen DataLoader in a deterministic cycling stream.

For resume, the historical trainer cursor is measured in consumed DataLoader batches.
Rebuilding those batches is very expensive because each skipped example still constructs
all production modalities. The wrapper reconstructs the same per-worker assignment-row
cursor before ProductionExampleBuilder runs, then resumes normal example construction.
An optional expected epoch-batch count fails closed if the cheap raw-row accounting does
not match a previously observed full loader pass.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset

from trencher.data.world_model.training.dataset import StreamingSplitDataset
from trencher.data.world_model.training.example import ProductionExampleBuilder
from trencher.world_model.batch_adapter import example_to_tensors
from trencher.world_model.trainer import ThreePhaseTrainer
from trencher.world_model.v4_loader import FrozenV4IterableDataset, shard_assignment_paths


def _accepted_raw_rows(ds: FrozenV4IterableDataset, paths: list[Any]) -> int:
    columns = ["base_split", "obs_in_split_interval"]
    if ds.population == "rich":
        columns.append("in_rich_subset")
    if ds._fold is not None:
        columns.append("detected_at_unix")

    total = 0
    for path in paths:
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        use_columns = [c for c in columns if c in schema_names]
        rows = pq.read_table(path, columns=use_columns).to_pylist()
        for row in rows:
            if row.get("base_split") != ds.base_split:
                continue
            if not row.get("obs_in_split_interval"):
                continue
            if ds.population == "rich" and not row.get("in_rich_subset"):
                continue
            if not ds._accept(row):
                continue
            total += 1
    return total


def _worker_row_counts(ds: FrozenV4IterableDataset, num_workers: int) -> list[int]:
    workers = max(int(num_workers), 1)
    counts: list[int] = []
    for worker_id in range(workers):
        paths = shard_assignment_paths(
            ds.assignment_paths,
            worker_id=worker_id,
            num_workers=workers,
            seed=ds.seed,
            shuffle=ds.shuffle,
        )
        counts.append(_accepted_raw_rows(ds, paths))
    return counts


def _worker_batch_capacities(row_counts: list[int], batch_size: int) -> list[int]:
    return [math.ceil(n / batch_size) if n else 0 for n in row_counts]


def _successful_batches_by_worker(capacities: list[int], take: int) -> list[int]:
    """Simulate DataLoader's round-robin worker assignment over successful batches."""
    remaining = list(capacities)
    used = [0 for _ in capacities]
    if take < 0 or take > sum(capacities):
        raise ValueError(f"invalid batch offset {take} for epoch size {sum(capacities)}")
    next_worker = 0
    for _ in range(take):
        chosen = None
        for delta in range(len(remaining)):
            candidate = (next_worker + delta) % len(remaining)
            if remaining[candidate] > 0:
                chosen = candidate
                break
        if chosen is None:
            raise RuntimeError("worker batch simulation exhausted unexpectedly")
        used[chosen] += 1
        remaining[chosen] -= 1
        next_worker = (chosen + 1) % len(remaining)
    return used


class _FastSkipFrozenDataset(IterableDataset):
    """Frozen dataset replica that skips assignment rows before expensive building."""

    def __init__(
        self,
        base: FrozenV4IterableDataset,
        *,
        skip_rows_by_worker: list[int],
    ) -> None:
        super().__init__()
        self.base = base
        self.skip_rows_by_worker = list(skip_rows_by_worker)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = torch.utils.data.get_worker_info()
        worker_id = 0 if info is None else int(info.id)
        target_skip = int(self.skip_rows_by_worker[worker_id])
        paths = self.base._worker_paths()
        stream = StreamingSplitDataset(
            assignment_paths=paths,
            split=self.base.base_split,
            seed=self.base.seed,
            shuffle=self.base.shuffle,
            shuffle_paths=False,
            population=self.base.population,
        )
        builder = ProductionExampleBuilder(
            world_root=self.base.world_root,
            split_root=self.base.split_root,
            population=self.base.population,
            use_cache=self.base.use_cache,
        )

        skipped = 0
        yielded = 0
        pending: list[dict[str, Any]] = []

        def emit(rows: list[dict[str, Any]]) -> Iterator[dict[str, torch.Tensor]]:
            nonlocal yielded
            builder.prefetch(rows)
            for row in rows:
                ex = builder.build(row)
                if not ex.get("causal_ok", True):
                    raise RuntimeError(
                        "fast resume encountered causal_ok=False after raw-row cursor; "
                        "refusing a potentially divergent resume sequence"
                    )
                item = example_to_tensors(ex, path_steps=self.base.path_steps)
                item["loader_worker_id"] = torch.tensor(worker_id, dtype=torch.long)
                yielded += 1
                yield item
                if self.base.max_examples is not None and yielded >= self.base.max_examples:
                    return

        for row in stream.iter_rows(max_examples=None):
            if not self.base._accept(row):
                continue
            if skipped < target_skip:
                skipped += 1
                continue
            pending.append(row)
            if len(pending) < 32:
                continue
            yield from emit(pending)
            if self.base.max_examples is not None and yielded >= self.base.max_examples:
                return
            pending = []

        if skipped != target_skip:
            raise RuntimeError(
                f"fast resume worker {worker_id} could skip only {skipped}/{target_skip} rows"
            )
        if pending:
            yield from emit(pending)


def _clone_loader_with_dataset(
    base: DataLoader,
    dataset: IterableDataset,
    *,
    persistent_workers: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": base.batch_size,
        "num_workers": base.num_workers,
        "collate_fn": base.collate_fn,
        "pin_memory": base.pin_memory,
        "drop_last": base.drop_last,
        "timeout": base.timeout,
    }
    if base.num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = base.prefetch_factor
    return DataLoader(dataset, **kwargs)


class _CyclingFastResumeLoader:
    def __init__(self, base: DataLoader, *, consumed_batches: int) -> None:
        if not isinstance(base.dataset, FrozenV4IterableDataset):
            raise TypeError("cycling fast resume requires FrozenV4IterableDataset")
        if base.batch_size is None:
            raise TypeError("cycling fast resume requires automatic batching")

        self.base = base
        self.dataset = base.dataset
        self.batch_size = int(base.batch_size)
        self.consumed_batches = max(int(consumed_batches), 0)

        self.row_counts = _worker_row_counts(self.dataset, int(base.num_workers))
        self.capacities = _worker_batch_capacities(self.row_counts, self.batch_size)
        self.epoch_batches = sum(self.capacities)
        if self.epoch_batches <= 0:
            raise RuntimeError("frozen loader has zero training batches")

        expected_raw = os.environ.get("TRENCHER_WM_EXPECT_EPOCH_BATCHES", "").strip()
        if expected_raw:
            expected = int(expected_raw)
            if self.epoch_batches != expected:
                raise RuntimeError(
                    f"raw frozen epoch has {self.epoch_batches} batches, expected {expected}; "
                    "refusing fast resume because causal filtering/order may differ"
                )

        self.offset = self.consumed_batches % self.epoch_batches
        worker_skip_batches = _successful_batches_by_worker(self.capacities, self.offset)
        self.skip_rows_by_worker = [
            min(batch_count * self.batch_size, row_count)
            for batch_count, row_count in zip(
                worker_skip_batches, self.row_counts, strict=True
            )
        ]
        print(
            "[resume_stream] "
            f"epoch_batches={self.epoch_batches} consumed={self.consumed_batches} "
            f"epoch_offset={self.offset} row_counts={self.row_counts} "
            f"worker_skip_batches={worker_skip_batches}"
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.offset:
            fast_ds = _FastSkipFrozenDataset(
                self.dataset,
                skip_rows_by_worker=self.skip_rows_by_worker,
            )
            fast_loader = _clone_loader_with_dataset(
                self.base,
                fast_ds,
                persistent_workers=False,
            )
            yielded = 0
            for batch in fast_loader:
                yielded += 1
                yield batch
            if yielded == 0 and self.offset < self.epoch_batches:
                raise RuntimeError("fast-resume tail yielded zero batches")
        else:
            for batch in self.base:
                yield batch

        while True:
            yielded = 0
            for batch in self.base:
                yielded += 1
                yield batch
            if yielded == 0:
                raise RuntimeError("frozen loader produced zero batches on cycle restart")


_ORIGINAL_TRAIN_PHASE = ThreePhaseTrainer.train_phase


def _hardened_train_phase(
    self: ThreePhaseTrainer,
    phase: str,
    batches: Any,
    *,
    max_steps: int | None = None,
    resume_step: int | None = None,
    skip_microbatches: int | None = None,
):
    if isinstance(batches, DataLoader) and isinstance(
        batches.dataset, FrozenV4IterableDataset
    ):
        consumed = (
            int(skip_microbatches)
            if skip_microbatches is not None
            else int(self.consumed_microbatches)
        )
        wrapped = _CyclingFastResumeLoader(batches, consumed_batches=consumed)
        return _ORIGINAL_TRAIN_PHASE(
            self,
            phase,
            wrapped,
            max_steps=max_steps,
            resume_step=resume_step,
            skip_microbatches=0,
        )
    return _ORIGINAL_TRAIN_PHASE(
        self,
        phase,
        batches,
        max_steps=max_steps,
        resume_step=resume_step,
        skip_microbatches=skip_microbatches,
    )


def _install() -> None:
    ThreePhaseTrainer.train_phase = _hardened_train_phase  # type: ignore[method-assign]


def main() -> None:
    _install()
    from trencher.world_model.launch import main as launch_main

    launch_main()


if __name__ == "__main__":
    main()
