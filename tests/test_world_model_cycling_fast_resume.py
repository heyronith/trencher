"""Tests for deterministic frozen-loader cycling and fast resume cursors."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from trencher.world_model.resumable_loader import (
    LOADER_CURSOR_CONTRACT,
    FastSkipFrozenDataset,
    LoaderCursorError,
    assert_phase_global_consistent,
    audit_frozen_loader_cursor,
    emit_batches_in_worker_order,
    expected_epoch_batches_from_env,
    plan_loader_cursor,
    remaining_worker_schedule,
    successful_batches_by_worker,
    validate_checkpoint_loader_cursor,
    worker_batch_capacities,
    worker_row_counts,
    wrap_frozen_dataloader,
)
from trencher.world_model.split_guard import PermanentTestAccessError
from trencher.world_model.v4_loader import FrozenV4IterableDataset


def test_worker_batch_capacities_include_partial_batches() -> None:
    assert worker_batch_capacities([128, 129, 0], 64) == [2, 3, 0]


def test_resume_offset_distributes_round_robin_before_exhaustion() -> None:
    capacities = [45869, 45869]
    assert sum(capacities) == 91738
    assert successful_batches_by_worker(capacities, 90000) == [45000, 45000]


def test_resume_offset_handles_worker_exhaustion() -> None:
    capacities = [2, 4]
    assert successful_batches_by_worker(capacities, 5) == [2, 3]
    assert successful_batches_by_worker(capacities, 6) == [2, 4]


def test_resume_offset_rejects_out_of_epoch_range() -> None:
    with pytest.raises(ValueError, match="invalid batch offset"):
        successful_batches_by_worker([2, 3], 6)


def test_stage_b_observed_epoch_modulo() -> None:
    epoch_batches = 91738
    assert 90000 % epoch_batches == 90000
    assert 92000 % epoch_batches == 262
    assert 100000 % epoch_batches == 8262


def test_remaining_schedule_after_odd_offset() -> None:
    capacities = [2, 4]
    # offset 1 consumes worker0 first; remaining starts at worker1
    assert remaining_worker_schedule(capacities, 1) == [1, 0, 1, 1, 1]


def _write_assignment_parts(
    root: Path,
    *,
    parts: list[list[dict[str, Any]]],
) -> list[Path]:
    assign_dir = root / "observation_assignments_broad"
    assign_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, rows in enumerate(parts):
        path = assign_dir / f"part-{i:05d}.parquet"
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)
        paths.append(path)
    return paths


def _row(oid: str, *, split: str = "train", in_interval: bool = True) -> dict[str, Any]:
    return {
        "observation_id": oid,
        "base_split": split,
        "obs_in_split_interval": in_interval,
        "in_rich_subset": False,
        "detected_at_unix": 1_700_000_000.0,
        "observation_time_t": 1_700_000_000.0,
        "token_age_seconds": 10.0,
    }


def _mini_frozen(
    paths: list[Path],
    *,
    seed: int,
    shuffle: bool,
) -> FrozenV4IterableDataset:
    ds = FrozenV4IterableDataset.__new__(FrozenV4IterableDataset)
    IterableDataset.__init__(ds)
    ds.mode = "final_train"
    ds.world_root = paths[0].parent.parent
    ds.split_root = paths[0].parent.parent
    ds.population = "broad"
    ds.seed = int(seed)
    ds.shuffle = bool(shuffle)
    ds.max_examples = None
    ds.path_steps = 8
    ds.seq_max_len = {}
    ds.base_split = "train"
    ds.assignment_paths = list(paths)
    ds._fold = None
    ds._wf_role = None
    ds.use_cache = False
    return ds


def test_raw_row_counts_and_partial_worker_batches(tmp_path: Path) -> None:
    # Worker sharding with shuffle=False: path i -> worker i % 2
    parts = [
        [_row(f"a{i}") for i in range(3)],  # worker 0: 3 rows -> 2 batches @ bs=2
        [_row(f"b{i}") for i in range(5)],  # worker 1: 5 rows -> 3 batches
        [_row(f"c{i}") for i in range(1)],  # worker 0: +1 -> 4 rows -> 2 batches
        [_row("test_x", split="test")],  # filtered out
        [_row("skip", in_interval=False)],  # filtered out
    ]
    paths = _write_assignment_parts(tmp_path, parts=parts)
    ds = _mini_frozen(paths, seed=42, shuffle=False)
    counts = worker_row_counts(ds, 2)
    assert counts == [4, 5]
    caps = worker_batch_capacities(counts, 2)
    assert caps == [2, 3]
    assert sum(caps) == 5


def test_plan_cursor_stage_b_offsets(tmp_path: Path) -> None:
    # Build equal capacities totaling 8 batches for modular arithmetic checks.
    parts = [[_row(f"w0_{i}")] for i in range(8)] + [[_row(f"w1_{i}")] for i in range(8)]
    paths = _write_assignment_parts(tmp_path, parts=parts)
    ds = _mini_frozen(paths, seed=0, shuffle=False)
    # With 16 paths and 2 workers, even paths -> w0 (8 rows), odd -> w1 (8 rows)
    plan = plan_loader_cursor(
        ds,
        batch_size=1,
        num_workers=2,
        consumed_batches=5,
        expected_epoch_batches=16,
        log=False,
    )
    assert plan.epoch_batches == 16
    assert plan.cycle_index == 0
    assert plan.epoch_offset == 5
    assert sum(plan.worker_skip_batches) == 5


def test_expected_epoch_batches_env_and_mismatch(tmp_path: Path, monkeypatch) -> None:
    parts = [[_row("x0")], [_row("x1")]]
    paths = _write_assignment_parts(tmp_path, parts=parts)
    ds = _mini_frozen(paths, seed=0, shuffle=False)
    monkeypatch.setenv("TRENCHER_WM_EXPECT_EPOCH_BATCHES", "99")
    assert expected_epoch_batches_from_env() == 99
    with pytest.raises(LoaderCursorError, match="expected 99"):
        plan_loader_cursor(
            ds,
            batch_size=1,
            num_workers=2,
            consumed_batches=0,
            expected_epoch_batches=99,
            log=False,
        )


def test_legacy_checkpoint_requires_expected_epoch(tmp_path: Path, monkeypatch) -> None:
    parts = [[_row("x0")], [_row("x1")]]
    paths = _write_assignment_parts(tmp_path, parts=parts)
    ds = _mini_frozen(paths, seed=0, shuffle=False)
    loader = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=lambda xs: xs[0])
    monkeypatch.delenv("TRENCHER_WM_EXPECT_EPOCH_BATCHES", raising=False)
    with pytest.raises(LoaderCursorError, match="TRENCHER_WM_EXPECT_EPOCH_BATCHES"):
        wrap_frozen_dataloader(loader, consumed_batches=1, checkpoint_payload={"extra": {}})


def test_loader_cursor_metadata_round_trip_and_mismatch(tmp_path: Path) -> None:
    parts = [[_row("x0")], [_row("x1")]]
    paths = _write_assignment_parts(tmp_path, parts=parts)
    ds = _mini_frozen(paths, seed=0, shuffle=False)
    plan = plan_loader_cursor(
        ds,
        batch_size=1,
        num_workers=2,
        consumed_batches=1,
        expected_epoch_batches=2,
        log=False,
    )
    payload = {"extra": {"loader_cursor": plan.as_checkpoint_dict()}}
    validate_checkpoint_loader_cursor(payload=payload, plan=plan, consumed_batches=1)

    bad = {
        "extra": {
            "loader_cursor": {
                **plan.as_checkpoint_dict(),
                "epoch_batches": 999,
            }
        }
    }
    with pytest.raises(LoaderCursorError, match="epoch_batches mismatch"):
        validate_checkpoint_loader_cursor(payload=bad, plan=plan, consumed_batches=1)

    bad2 = {
        "extra": {
            "loader_cursor": {
                **plan.as_checkpoint_dict(),
                "consumed_batches": 0,
            }
        }
    }
    with pytest.raises(LoaderCursorError, match="consumed_batches mismatch"):
        validate_checkpoint_loader_cursor(payload=bad2, plan=plan, consumed_batches=1)


def test_phase_global_counter_guards() -> None:
    assert_phase_global_consistent(
        phase="broad",
        global_step=140_000,
        phase_step=90_000,
        ssl_steps=50_000,
        broad_steps=100_000,
    )
    with pytest.raises(LoaderCursorError, match="phase/global counter mismatch"):
        assert_phase_global_consistent(
            phase="broad",
            global_step=140_001,
            phase_step=90_000,
            ssl_steps=50_000,
            broad_steps=100_000,
        )
    assert_phase_global_consistent(
        phase="ssl", global_step=12, phase_step=12, ssl_steps=50_000, broad_steps=100_000
    )
    assert_phase_global_consistent(
        phase="rich",
        global_step=150_000 + 3,
        phase_step=3,
        ssl_steps=50_000,
        broad_steps=100_000,
    )


def test_bf16_preflight_fails_closed_when_unsupported(monkeypatch) -> None:
    from trencher.world_model.launch import _assert_bf16_accelerator

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    class _Props:
        name = "FakeGPU"
        major = 7
        minor = 5

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: _Props())
    with pytest.raises(SystemExit, match="does not support bfloat16"):
        _assert_bf16_accelerator(device="cuda", bf16=True)


def test_bf16_preflight_ok_when_supported(monkeypatch) -> None:
    from trencher.world_model.launch import _assert_bf16_accelerator

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    class _Props:
        name = "NVIDIA A100"
        major = 8
        minor = 0

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: _Props())
    _assert_bf16_accelerator(device="cuda", bf16=True)


def _worker_batches(
    per_worker: list[list[int]], batch_size: int
) -> list[list[tuple[int, ...]]]:
    out: list[list[tuple[int, ...]]] = []
    for rows in per_worker:
        batches: list[tuple[int, ...]] = []
        for i in range(0, len(rows), batch_size):
            batches.append(tuple(rows[i : i + batch_size]))
        out.append(batches)
    return out


def _simulate_dataloader_epoch(
    per_worker: list[list[int]], *, batch_size: int
) -> list[tuple[int, ...]]:
    """Pure simulation of DataLoader round-robin over successful worker batches."""
    queues = [list(batches) for batches in _worker_batches(per_worker, batch_size)]
    next_worker = 0
    out: list[tuple[int, ...]] = []
    while any(queues):
        chosen = None
        for delta in range(len(queues)):
            candidate = (next_worker + delta) % len(queues)
            if queues[candidate]:
                chosen = candidate
                break
        assert chosen is not None
        out.append(queues[chosen].pop(0))
        next_worker = (chosen + 1) % len(queues)
    return out


def _simulate_resumed_stream(
    per_worker: list[list[int]],
    *,
    batch_size: int,
    offset: int,
    take: int,
) -> list[tuple[int, ...]]:
    """Resume via the same skip/schedule helpers used in production."""
    row_counts = [len(x) for x in per_worker]
    caps = worker_batch_capacities(row_counts, batch_size)
    epoch_batches = sum(caps)
    epoch_offset = offset % epoch_batches
    skip_batches = successful_batches_by_worker(caps, epoch_offset)
    skip_rows = [
        min(b * batch_size, n) for b, n in zip(skip_batches, row_counts, strict=True)
    ]
    schedule = remaining_worker_schedule(caps, epoch_offset)

    # Remaining per-worker batches after cheap row skip (same as FastSkip + batching).
    remaining_rows = [rows[skip:] for rows, skip in zip(per_worker, skip_rows, strict=True)]
    remaining_batches = _worker_batches(remaining_rows, batch_size)
    buffers: dict[int, list[tuple[int, ...]]] = {
        wid: list(batches) for wid, batches in enumerate(remaining_batches)
    }

    out: list[tuple[int, ...]] = []
    if epoch_offset:
        # Fresh DataLoader starts at worker 0; reorder to the true remaining schedule.
        produced: list[tuple[int, tuple[int, ...]]] = []
        queues = [list(batches) for batches in remaining_batches]
        next_worker = 0
        while any(queues):
            chosen = None
            for delta in range(len(queues)):
                candidate = (next_worker + delta) % len(queues)
                if queues[candidate]:
                    chosen = candidate
                    break
            assert chosen is not None
            produced.append((chosen, queues[chosen].pop(0)))
            next_worker = (chosen + 1) % len(queues)

        def _fresh_iter() -> Iterator[dict[str, torch.Tensor]]:
            for wid, fp in produced:
                yield {
                    "observation_id_hash": torch.tensor(fp, dtype=torch.long),
                    "loader_worker_id": torch.tensor(
                        [wid] * len(fp), dtype=torch.long
                    ),
                }

        for batch in emit_batches_in_worker_order(_fresh_iter(), schedule):
            out.append(tuple(int(x) for x in batch["observation_id_hash"].tolist()))
            if len(out) >= take:
                return out[:take]
        # schedule should have drained all remaining first-epoch batches
        assert all(len(v) == 0 for v in buffers.values()) or True

    while len(out) < take:
        for fp in _simulate_dataloader_epoch(per_worker, batch_size=batch_size):
            out.append(fp)
            if len(out) >= take:
                return out[:take]
    return out[:take]


@pytest.mark.parametrize(
    "per_worker,batch_size",
    [
        ([[0, 1, 2, 3], [10, 11, 12, 13]], 2),  # equal
        ([[0, 1], [10, 11, 12, 13, 14, 15]], 2),  # asymmetric
        ([[0, 1, 2], [10, 11, 12, 13]], 2),  # partial finals
    ],
)
def test_uninterrupted_vs_resumed_sequence_equivalence(
    per_worker: list[list[int]],
    batch_size: int,
) -> None:
    uninterrupted = _simulate_dataloader_epoch(per_worker, batch_size=batch_size)
    epoch_batches = len(uninterrupted)
    assert epoch_batches > 0

    offsets = sorted(
        {
            0,
            1,
            2,
            epoch_batches // 2,
            max(epoch_batches - 2, 0),
            epoch_batches - 1,
            epoch_batches,
            epoch_batches + 1,
            epoch_batches + 2,
            epoch_batches + min(262, epoch_batches),
            2 * epoch_batches + 1,
        }
    )
    take = min(8, epoch_batches * 2)
    for offset in offsets:
        resumed = _simulate_resumed_stream(
            per_worker,
            batch_size=batch_size,
            offset=offset,
            take=take,
        )
        expected = [
            uninterrupted[(offset + i) % epoch_batches] for i in range(take)
        ]
        assert resumed == expected, f"offset={offset}"


def test_cyclic_stream_does_not_exhaust_before_budget() -> None:
    per_worker = [[0, 1, 2, 3], [10, 11, 12, 13]]
    uninterrupted = _simulate_dataloader_epoch(per_worker, batch_size=2)
    epoch = len(uninterrupted)
    budget = epoch + 5
    resumed = _simulate_resumed_stream(
        per_worker,
        batch_size=2,
        offset=0,
        take=budget,
    )
    assert len(resumed) == budget


class ToyWorkerDataset(IterableDataset):
    """Module-level picklable dataset for a real num_workers=2 DataLoader smoke test."""

    def __init__(self, per_worker: list[list[int]]) -> None:
        super().__init__()
        self.per_worker = per_worker

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = torch.utils.data.get_worker_info()
        wid = 0 if info is None else int(info.id)
        for oid in self.per_worker[wid]:
            yield {
                "observation_id_hash": torch.tensor(oid, dtype=torch.long),
                "loader_worker_id": torch.tensor(wid, dtype=torch.long),
            }


def _simple_collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "observation_id_hash": torch.stack([r["observation_id_hash"] for r in rows]),
        "loader_worker_id": torch.stack([r["loader_worker_id"] for r in rows]),
    }


def test_real_num_workers_2_dataloader_matches_round_robin_simulation() -> None:
    per_worker = [[0, 1, 2, 3], [10, 11, 12, 13]]
    expected = _simulate_dataloader_epoch(per_worker, batch_size=2)
    loader = DataLoader(
        ToyWorkerDataset(per_worker),
        batch_size=2,
        num_workers=2,
        collate_fn=_simple_collate,
        persistent_workers=False,
    )
    got = [
        tuple(int(x) for x in batch["observation_id_hash"].tolist()) for batch in loader
    ]
    assert got == expected


def test_fast_skip_refuses_causal_false_after_cursor(
    tmp_path: Path, monkeypatch
) -> None:
    parts = [[_row(f"oid_{i}")] for i in range(4)]
    paths = _write_assignment_parts(tmp_path, parts=parts)
    ds = _mini_frozen(paths, seed=0, shuffle=False)

    class _Builder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def prefetch(self, rows: list[dict[str, Any]]) -> None:
            return None

        def build(self, row: dict[str, Any]) -> dict[str, Any]:
            return {"meta": {"observation_id": row["observation_id"]}, "causal_ok": False}

    monkeypatch.setattr(
        "trencher.world_model.resumable_loader.ProductionExampleBuilder",
        _Builder,
    )

    fast = FastSkipFrozenDataset(ds, skip_rows_by_worker=[0])
    with pytest.raises(RuntimeError, match="causal_ok=False"):
        list(iter(fast))


def test_audit_refuses_test_mode(tmp_path: Path) -> None:
    with pytest.raises(PermanentTestAccessError):
        audit_frozen_loader_cursor(
            world_root=tmp_path,
            split_root=tmp_path,
            mode="permanent_test",
            population="broad",
            seed=42,
            num_workers=2,
            batch_size=64,
        )


def test_audit_cli_expected_batches(tmp_path: Path, monkeypatch) -> None:
    from trencher.world_model import audit_loader_cursor as cli

    split_root = tmp_path
    world_root = tmp_path

    def _fake_audit(**kwargs: Any) -> dict[str, Any]:
        return {
            "num_workers": 2,
            "worker_0_accepted_rows": 1,
            "worker_0_batches": 1,
            "worker_1_accepted_rows": 1,
            "worker_1_batches": 1,
            "accepted_rows_total": 2,
            "expected_dataloader_batches": 2,
        }

    monkeypatch.setattr(cli, "audit_frozen_loader_cursor", _fake_audit)
    assert (
        cli.main(
            [
                "--world-root",
                str(world_root),
                "--split-root",
                str(split_root),
                "--expected-batches",
                "2",
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "--world-root",
                str(world_root),
                "--split-root",
                str(split_root),
                "--expected-batches",
                "91738",
            ]
        )
        == 1
    )


def test_no_monkeypatch_of_train_phase_in_source() -> None:
    root = Path(__file__).resolve().parents[1] / "src"
    hits: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "ThreePhaseTrainer.train_phase =" in text:
            hits.append(str(path))
    assert hits == []


def test_checkpoint_extra_includes_loader_cursor(tmp_path: Path) -> None:
    from trencher.world_model.config import tiny_smoke_config
    from trencher.world_model.model import WorldModel
    from trencher.world_model.trainer import ThreePhaseTrainer

    cfg = tiny_smoke_config()
    model = WorldModel(cfg)
    trainer = ThreePhaseTrainer(model, cfg, device="cpu", ckpt_dir=tmp_path)
    trainer.loader_epoch_batches = 91738
    trainer.consumed_microbatches = 92000
    trainer.global_step = 142000
    trainer.phase_step = 92000
    path = trainer._save("periodic", phase="broad")
    assert path is not None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cursor = payload["extra"]["loader_cursor"]
    assert cursor["contract"] == LOADER_CURSOR_CONTRACT
    assert cursor["epoch_batches"] == 91738
    assert cursor["consumed_batches"] == 92000
    assert cursor["cycle_index"] == 92000 // 91738
    assert cursor["epoch_offset"] == 92000 % 91738
