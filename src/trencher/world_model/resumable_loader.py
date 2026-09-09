"""Deterministic frozen-loader cycling and cheap resume cursor planning.

Production phases are step-budgeted; the frozen V4 DataLoader is finite. This module
wraps a frozen loader so training can cycle the same deterministic epoch ordering until
the configured step budget is reached, and can resume from ``consumed_microbatches``
without rebuilding already-consumed examples through ``ProductionExampleBuilder``.

Cursor contract: ``FROZEN_V4_CYCLIC_CURSOR_V1``.
"""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict, deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset

from trencher.data.world_model.training.dataset import deterministic_shard_seed
from trencher.data.world_model.training.example import ProductionExampleBuilder
from trencher.world_model.batch_adapter import example_to_tensors
from trencher.world_model.v4_loader import FrozenV4IterableDataset, shard_assignment_paths

LOADER_CURSOR_CONTRACT = "FROZEN_V4_CYCLIC_CURSOR_V1"


class LoaderCursorError(RuntimeError):
    """Fail-closed cursor / epoch accounting error."""


@dataclass(frozen=True)
class LoaderCursorPlan:
    """Deterministic resume plan for a frozen cyclic stream."""

    contract: str
    epoch_batches: int
    consumed_batches: int
    cycle_index: int
    epoch_offset: int
    row_counts: tuple[int, ...]
    capacities: tuple[int, ...]
    worker_skip_batches: tuple[int, ...]
    skip_rows_by_worker: tuple[int, ...]
    remaining_worker_schedule: tuple[int, ...]
    batch_size: int
    num_workers: int

    def as_checkpoint_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "epoch_batches": int(self.epoch_batches),
            "consumed_batches": int(self.consumed_batches),
            "cycle_index": int(self.cycle_index),
            "epoch_offset": int(self.epoch_offset),
        }


def expected_epoch_batches_from_env() -> int | None:
    raw = os.environ.get("TRENCHER_WM_EXPECT_EPOCH_BATCHES", "").strip()
    if not raw:
        return None
    return int(raw)


def worker_batch_capacities(row_counts: Sequence[int], batch_size: int) -> list[int]:
    """Per-worker successful batch counts with ``drop_last=False`` (ceil)."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [math.ceil(n / batch_size) if n else 0 for n in row_counts]


def successful_batches_by_worker(capacities: Sequence[int], take: int) -> list[int]:
    """Simulate DataLoader round-robin over successful worker batches.

    PyTorch's multi-worker DataLoader advances a shared round-robin index, skipping
    exhausted workers, for each successfully yielded batch.
    """
    remaining = list(capacities)
    used = [0 for _ in capacities]
    total = sum(capacities)
    if take < 0 or take > total:
        raise ValueError(f"invalid batch offset {take} for epoch size {total}")
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


def remaining_worker_schedule(capacities: Sequence[int], offset: int) -> list[int]:
    """Worker ids for each remaining successful batch after ``offset``."""
    remaining = list(capacities)
    total = sum(capacities)
    if offset < 0 or offset > total:
        raise ValueError(f"invalid batch offset {offset} for epoch size {total}")
    next_worker = 0
    for _ in range(offset):
        chosen = None
        for delta in range(len(remaining)):
            candidate = (next_worker + delta) % len(remaining)
            if remaining[candidate] > 0:
                chosen = candidate
                break
        if chosen is None:
            raise RuntimeError("worker schedule burn-in exhausted unexpectedly")
        remaining[chosen] -= 1
        next_worker = (chosen + 1) % len(remaining)

    schedule: list[int] = []
    while sum(remaining) > 0:
        chosen = None
        for delta in range(len(remaining)):
            candidate = (next_worker + delta) % len(remaining)
            if remaining[candidate] > 0:
                chosen = candidate
                break
        if chosen is None:
            break
        schedule.append(chosen)
        remaining[chosen] -= 1
        next_worker = (chosen + 1) % len(remaining)
    return schedule


def _filter_columns(ds: FrozenV4IterableDataset) -> list[str]:
    columns = ["base_split", "obs_in_split_interval"]
    if ds.population == "rich":
        columns.append("in_rich_subset")
    if ds._fold is not None:
        columns.append("detected_at_unix")
    return columns


def _row_passes_raw_filters(ds: FrozenV4IterableDataset, row: dict[str, Any]) -> bool:
    if row.get("base_split") != ds.base_split:
        return False
    if not row.get("obs_in_split_interval"):
        return False
    if ds.population == "rich" and not row.get("in_rich_subset"):
        return False
    return bool(ds._accept(row))


def count_accepted_raw_rows(
    ds: FrozenV4IterableDataset,
    paths: Sequence[Path],
    *,
    progress_every_files: int = 64,
    progress_label: str | None = None,
) -> int:
    """Count assignment rows that pass split/population/walk-forward filters.

    Does not run ``ProductionExampleBuilder`` and does not apply ``causal_ok``.
    Production must fail-closed via ``TRENCHER_WM_EXPECT_EPOCH_BATCHES`` when this
    cheap count must match an observed post-causal DataLoader epoch size.
    """
    columns = _filter_columns(ds)
    total = 0
    t0 = time.perf_counter()
    for file_i, path in enumerate(paths):
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        use_columns = [c for c in columns if c in schema_names]
        rows = pq.read_table(path, columns=use_columns).to_pylist()
        for row in rows:
            if _row_passes_raw_filters(ds, row):
                total += 1
        if (
            progress_label
            and progress_every_files > 0
            and (file_i + 1) % progress_every_files == 0
        ):
            elapsed = time.perf_counter() - t0
            print(
                f"[resume_stream] {progress_label} files={file_i + 1}/{len(paths)} "
                f"accepted_rows={total} elapsed={elapsed:.1f}s"
            )
    return total


def worker_row_counts(
    ds: FrozenV4IterableDataset,
    num_workers: int,
    *,
    log_progress: bool = False,
) -> list[int]:
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
        label = f"worker_{worker_id}_row_scan" if log_progress else None
        counts.append(
            count_accepted_raw_rows(ds, paths, progress_label=label)
        )
    return counts


def _accepted_indices_for_file(
    path: Path,
    ds: FrozenV4IterableDataset,
    *,
    shuffle: bool,
) -> list[int]:
    """Return shuffled-order indices of rows that pass raw filters."""
    columns = _filter_columns(ds)
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    use_columns = [c for c in columns if c in schema_names]
    rows = pq.read_table(path, columns=use_columns).to_pylist()
    order = list(range(len(rows)))
    if shuffle:
        rng = np.random.default_rng(deterministic_shard_seed(ds.seed, path.name))
        order = [order[i] for i in rng.permutation(len(order))]
    accepted: list[int] = []
    for idx in order:
        if _row_passes_raw_filters(ds, rows[idx]):
            accepted.append(idx)
    return accepted


def plan_loader_cursor(
    ds: FrozenV4IterableDataset,
    *,
    batch_size: int,
    num_workers: int,
    consumed_batches: int,
    expected_epoch_batches: int | None = None,
    log: bool = True,
) -> LoaderCursorPlan:
    """Build a fail-closed cursor plan for ``consumed_batches``."""
    if log:
        print("[resume_stream] planning frozen cursor")
    t0 = time.perf_counter()
    workers = max(int(num_workers), 1)
    row_counts = worker_row_counts(ds, workers, log_progress=log)
    capacities = worker_batch_capacities(row_counts, batch_size)
    epoch_batches = sum(capacities)
    if epoch_batches <= 0:
        raise LoaderCursorError("frozen loader has zero training batches")

    if expected_epoch_batches is not None and epoch_batches != int(expected_epoch_batches):
        raise LoaderCursorError(
            f"raw frozen epoch has {epoch_batches} batches, "
            f"expected {expected_epoch_batches}; refusing resume because "
            "causal filtering/order may differ"
        )

    consumed = max(int(consumed_batches), 0)
    cycle_index = consumed // epoch_batches
    epoch_offset = consumed % epoch_batches
    worker_skip = successful_batches_by_worker(capacities, epoch_offset)
    skip_rows = [
        min(batch_count * batch_size, row_count)
        for batch_count, row_count in zip(worker_skip, row_counts, strict=True)
    ]
    schedule = remaining_worker_schedule(capacities, epoch_offset)
    plan = LoaderCursorPlan(
        contract=LOADER_CURSOR_CONTRACT,
        epoch_batches=epoch_batches,
        consumed_batches=consumed,
        cycle_index=cycle_index,
        epoch_offset=epoch_offset,
        row_counts=tuple(row_counts),
        capacities=tuple(capacities),
        worker_skip_batches=tuple(worker_skip),
        skip_rows_by_worker=tuple(skip_rows),
        remaining_worker_schedule=tuple(schedule),
        batch_size=int(batch_size),
        num_workers=workers,
    )
    if log:
        elapsed = time.perf_counter() - t0
        print(f"[resume_stream] epoch_batches={plan.epoch_batches}")
        print(f"[resume_stream] consumed_batches={plan.consumed_batches}")
        print(f"[resume_stream] cycle_index={plan.cycle_index}")
        print(f"[resume_stream] epoch_offset={plan.epoch_offset}")
        print(f"[resume_stream] worker_row_counts={list(plan.row_counts)}")
        print(f"[resume_stream] worker_skip_batches={list(plan.worker_skip_batches)}")
        print(f"[resume_stream] cursor planning completed elapsed={elapsed:.1f}s")
    return plan


def validate_checkpoint_loader_cursor(
    *,
    payload: dict[str, Any] | None,
    plan: LoaderCursorPlan,
    consumed_batches: int,
) -> None:
    """Fail closed when saved cursor metadata disagrees with live accounting."""
    extra = (payload or {}).get("extra") or {}
    cursor = extra.get("loader_cursor")
    if not cursor:
        return
    if not isinstance(cursor, dict):
        raise LoaderCursorError("checkpoint loader_cursor must be a mapping")
    contract = cursor.get("contract")
    if contract != LOADER_CURSOR_CONTRACT:
        raise LoaderCursorError(
            f"unsupported loader_cursor contract {contract!r}; "
            f"expected {LOADER_CURSOR_CONTRACT!r}"
        )
    saved_epoch = int(cursor["epoch_batches"])
    saved_consumed = int(cursor["consumed_batches"])
    saved_cycle = int(cursor["cycle_index"])
    saved_offset = int(cursor["epoch_offset"])
    if saved_epoch != plan.epoch_batches:
        raise LoaderCursorError(
            f"loader_cursor epoch_batches mismatch: ckpt={saved_epoch} live={plan.epoch_batches}"
        )
    if saved_consumed != int(consumed_batches):
        raise LoaderCursorError(
            f"loader_cursor consumed_batches mismatch: "
            f"ckpt={saved_consumed} telemetry={consumed_batches}"
        )
    if saved_cycle != plan.cycle_index or saved_offset != plan.epoch_offset:
        raise LoaderCursorError(
            f"loader_cursor cycle/offset mismatch: "
            f"ckpt=({saved_cycle},{saved_offset}) live=({plan.cycle_index},{plan.epoch_offset})"
        )


def assert_phase_global_consistent(
    *,
    phase: str,
    global_step: int,
    phase_step: int,
    ssl_steps: int,
    broad_steps: int,
) -> None:
    """Fail closed when phase-local and global counters disagree."""
    phase = str(phase)
    if phase == "ssl":
        expected = int(phase_step)
    elif phase == "broad":
        expected = int(ssl_steps) + int(phase_step)
    elif phase == "rich":
        expected = int(ssl_steps) + int(broad_steps) + int(phase_step)
    else:
        raise LoaderCursorError(f"unknown phase for counter consistency: {phase!r}")
    if int(global_step) != expected:
        raise LoaderCursorError(
            f"phase/global counter mismatch for phase={phase}: "
            f"global_step={global_step} expected={expected} "
            f"(ssl_steps={ssl_steps}, broad_steps={broad_steps}, phase_step={phase_step})"
        )


def batch_worker_id(batch: dict[str, Any]) -> int:
    wid = batch.get("loader_worker_id")
    if wid is None:
        raise LoaderCursorError("batch missing loader_worker_id required for resume ordering")
    if torch.is_tensor(wid):
        return int(wid.reshape(-1)[0].item())
    return int(wid)


def emit_batches_in_worker_order(
    batch_iter: Iterator[dict[str, Any]],
    schedule: Sequence[int],
) -> Iterator[dict[str, Any]]:
    """Re-order DataLoader output to match an uninterrupted worker schedule."""
    buffers: dict[int, deque[dict[str, Any]]] = defaultdict(deque)
    src = iter(batch_iter)
    source_exhausted = False

    def _pull() -> bool:
        nonlocal source_exhausted
        try:
            batch = next(src)
        except StopIteration:
            source_exhausted = True
            return False
        buffers[batch_worker_id(batch)].append(batch)
        return True

    for expected in schedule:
        while not buffers[expected] and not source_exhausted:
            if not _pull():
                break
        if not buffers[expected]:
            buffered = {k: len(v) for k, v in buffers.items()}
            raise LoaderCursorError(
                f"resume ordering missing batch from worker {expected}; "
                f"buffered={buffered}"
            )
        yield buffers[expected].popleft()

    while not source_exhausted:
        _pull()
    leftovers = {wid: len(q) for wid, q in buffers.items() if q}
    if leftovers:
        raise LoaderCursorError(
            f"resume ordering leftover batches after schedule drained: {leftovers}"
        )


class FastSkipFrozenDataset(IterableDataset):
    """Frozen dataset replica that skips assignment rows before expensive building.

    Whole parquet files before the worker-local cursor are skipped using only the
    minimal filter columns. Inside the cursor file, only rows at/after the cursor
    are built. A post-cursor ``causal_ok=False`` aborts loudly.
    """

    def __init__(
        self,
        base: FrozenV4IterableDataset,
        *,
        skip_rows_by_worker: Sequence[int],
    ) -> None:
        super().__init__()
        self.base = base
        self.skip_rows_by_worker = [int(x) for x in skip_rows_by_worker]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = torch.utils.data.get_worker_info()
        worker_id = 0 if info is None else int(info.id)
        target_skip = int(self.skip_rows_by_worker[worker_id])
        paths = self.base._worker_paths()
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

        for path in paths:
            accepted_idx = _accepted_indices_for_file(
                path, self.base, shuffle=self.base.shuffle
            )
            n_accepted = len(accepted_idx)
            if skipped + n_accepted <= target_skip:
                skipped += n_accepted
                continue

            local_skip = target_skip - skipped
            skipped = target_skip
            # Need full rows for example construction from the cursor onward.
            all_rows = pq.read_table(path).to_pylist()
            for abs_i, row_i in enumerate(accepted_idx):
                if abs_i < local_skip:
                    continue
                row = dict(all_rows[row_i])
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


def clone_loader_with_dataset(
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


class CyclingResumableLoader:
    """Infinite deterministic cyclic stream over a finite frozen DataLoader.

    The first epoch continues from ``consumed_batches % epoch_batches`` via a cheap
    assignment-level seek. Subsequent epochs restart the base loader from the beginning
    of the same frozen order. Batches are never cached across cycles.
    """

    def __init__(
        self,
        base: DataLoader,
        *,
        consumed_batches: int,
        expected_epoch_batches: int | None = None,
        plan: LoaderCursorPlan | None = None,
        log: bool = True,
    ) -> None:
        if not isinstance(base.dataset, FrozenV4IterableDataset):
            raise TypeError("cycling resume requires FrozenV4IterableDataset")
        if base.batch_size is None:
            raise TypeError("cycling resume requires automatic batching")

        self.base = base
        self.dataset = base.dataset
        self.batch_size = int(base.batch_size)
        self.consumed_batches = max(int(consumed_batches), 0)
        self.expected_epoch_batches = expected_epoch_batches
        self.plan = plan or plan_loader_cursor(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=int(base.num_workers),
            consumed_batches=self.consumed_batches,
            expected_epoch_batches=expected_epoch_batches,
            log=log,
        )
        self.epoch_batches = self.plan.epoch_batches
        self.offset = self.plan.epoch_offset

    def __iter__(self) -> Iterator[dict[str, Any]]:
        first = True
        if self.offset:
            fast_ds = FastSkipFrozenDataset(
                self.dataset,
                skip_rows_by_worker=self.plan.skip_rows_by_worker,
            )
            fast_loader = clone_loader_with_dataset(
                self.base,
                fast_ds,
                persistent_workers=False,
            )
            schedule = self.plan.remaining_worker_schedule
            yielded = 0
            for batch in emit_batches_in_worker_order(iter(fast_loader), schedule):
                if first:
                    print("[resume_stream] first resumed batch ready")
                    first = False
                yielded += 1
                yield batch
            if yielded != len(schedule):
                raise LoaderCursorError(
                    f"fast-resume tail yielded {yielded} batches, expected {len(schedule)}"
                )
        else:
            for batch in self.base:
                if first:
                    print("[resume_stream] first resumed batch ready")
                    first = False
                yield batch

        while True:
            yielded = 0
            for batch in self.base:
                yielded += 1
                yield batch
            if yielded == 0:
                raise RuntimeError("frozen loader produced zero batches on cycle restart")


def wrap_frozen_dataloader(
    loader: DataLoader,
    *,
    consumed_batches: int,
    expected_epoch_batches: int | None = None,
    checkpoint_payload: dict[str, Any] | None = None,
    require_expected_for_legacy: bool = True,
    log: bool = True,
) -> CyclingResumableLoader:
    """Build the production cyclic/fast-resume stream around a frozen DataLoader."""
    if not isinstance(loader.dataset, FrozenV4IterableDataset):
        raise TypeError("wrap_frozen_dataloader requires FrozenV4IterableDataset")

    consumed = max(int(consumed_batches), 0)
    expected = expected_epoch_batches
    if expected is None:
        expected = expected_epoch_batches_from_env()

    extra = (checkpoint_payload or {}).get("extra") or {}
    has_cursor = isinstance(extra.get("loader_cursor"), dict)
    if (
        require_expected_for_legacy
        and consumed > 0
        and not has_cursor
        and expected is None
    ):
        raise LoaderCursorError(
            "legacy checkpoint resume requires TRENCHER_WM_EXPECT_EPOCH_BATCHES "
            "to prove cheap cursor accounting matches the observed epoch size"
        )

    plan = plan_loader_cursor(
        loader.dataset,
        batch_size=int(loader.batch_size or 0),
        num_workers=int(loader.num_workers),
        consumed_batches=consumed,
        expected_epoch_batches=expected,
        log=log,
    )
    validate_checkpoint_loader_cursor(
        payload=checkpoint_payload,
        plan=plan,
        consumed_batches=consumed,
    )
    return CyclingResumableLoader(
        loader,
        consumed_batches=consumed,
        expected_epoch_batches=expected,
        plan=plan,
        log=False,
    )


def audit_frozen_loader_cursor(
    *,
    world_root: Path,
    split_root: Path,
    mode: str,
    population: str,
    seed: int,
    num_workers: int,
    batch_size: int,
    shuffle: bool = True,
) -> dict[str, Any]:
    """Read-only cursor audit using the same planner as production resume."""
    from trencher.world_model.v4_loader import assert_dataset_mode

    mode_n = assert_dataset_mode(mode)
    ds = FrozenV4IterableDataset(
        world_root=Path(world_root),
        split_root=Path(split_root),
        mode=mode_n,
        population=population,  # type: ignore[arg-type]
        seed=int(seed),
        shuffle=bool(shuffle),
        use_cache=True,
    )
    workers = max(int(num_workers), 1)
    row_counts = worker_row_counts(ds, workers, log_progress=True)
    capacities = worker_batch_capacities(row_counts, int(batch_size))
    payload: dict[str, Any] = {
        "mode": mode_n,
        "population": population,
        "seed": int(seed),
        "num_workers": workers,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "worker_row_counts": row_counts,
        "worker_batch_capacities": capacities,
        "accepted_rows_total": int(sum(row_counts)),
        "expected_dataloader_batches": int(sum(capacities)),
    }
    for i, (rows, batches) in enumerate(zip(row_counts, capacities, strict=True)):
        payload[f"worker_{i}_accepted_rows"] = int(rows)
        payload[f"worker_{i}_batches"] = int(batches)
    return payload
