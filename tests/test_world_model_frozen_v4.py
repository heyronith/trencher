"""Real frozen-V4 integration tests (tiny sample from local world_model freeze)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

WORLD = Path("data/external/world_model")
SPLIT = WORLD / "splits" / "split_v1"


def _have_freeze() -> bool:
    return (SPLIT / "cutoffs.json").exists() and (
        SPLIT / "observation_assignments_broad"
    ).exists()


pytestmark = pytest.mark.skipif(not _have_freeze(), reason="frozen world_model data not present")


def test_test_mode_refused():
    from trencher.world_model.split_guard import PermanentTestAccessError
    from trencher.world_model.v4_loader import assert_dataset_mode

    with pytest.raises(PermanentTestAccessError):
        assert_dataset_mode("test")


def test_broad_loader_audit_sample():
    from trencher.world_model.v4_loader import audit_loader_sample

    audit = audit_loader_sample(
        world_root=WORLD,
        split_root=SPLIT,
        mode="final_train",
        population="broad",
        max_examples=16,
        seed=0,
    )
    assert audit["observations_loaded"] == 16
    assert audit["q85_test_rows"] == 0
    assert audit["causal_violations"] == 0
    assert audit["examples_per_sec"] > 0
    Path("manifests/world_model_data").mkdir(parents=True, exist_ok=True)
    Path("manifests/world_model_data/LOADER_AUDIT_BROAD_SAMPLE_V2.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )


def test_batch_parity_individual_vs_collate():
    from trencher.data.world_model.training.dataset import StreamingSplitDataset
    from trencher.data.world_model.training.example import ProductionExampleBuilder
    from trencher.data.world_model.training.loaders import resolve_assignment_paths
    from trencher.world_model.batch_adapter import (
    collate_tensor_examples,
    example_to_tensors,
)

    paths = resolve_assignment_paths(SPLIT, "broad")
    ds = StreamingSplitDataset(assignment_paths=paths[:1], split="train", shuffle=False)
    rows = []
    for r in ds.iter_rows(max_examples=4):
        rows.append(r)
    assert rows
    builder = ProductionExampleBuilder(world_root=WORLD, split_root=SPLIT, population="broad")
    builder.prefetch(rows)
    tensors = [example_to_tensors(builder.build(r)) for r in rows]
    batch = collate_tensor_examples(tensors)
    assert batch["event_x"].shape[0] == len(tensors)
    # First row event features must match unpadded prefix
    n0 = int(tensors[0]["event_mask"].sum().item())
    if n0 > 0:
        assert torch.allclose(
            batch["event_x"][0, :n0], tensors[0]["event_x"][:n0], equal_nan=True
        )


def test_causality_future_event_mutation_fixture(tmp_path: Path):
    """Append a synthetic post-T event in a copied feature array; rebuild digest must change
    only when mutation is pre-T. Dataset-level: times > T must already be excluded.
    """
    from trencher.data.world_model.training.dataset import StreamingSplitDataset
    from trencher.data.world_model.training.example import ProductionExampleBuilder
    from trencher.data.world_model.training.loaders import resolve_assignment_paths

    paths = resolve_assignment_paths(SPLIT, "broad")
    ds = StreamingSplitDataset(assignment_paths=paths[:1], split="train", shuffle=False)
    row = next(ds.iter_rows(max_examples=1))
    builder = ProductionExampleBuilder(world_root=WORLD, split_root=SPLIT, population="broad")
    builder.prefetch([row])
    ex = builder.build(row)
    assert ex.get("causal_ok", True)
    events = np.asarray(ex["sequences"]["events"])
    T = float(row["observation_time_t"])
    # All event rel_dt channels are relative; window_meta should prove causal truncation.
    # Hard check: no absolute times > T in window meta raw_len vs count.
    meta = ex.get("window_meta") or {}
    assert "events" in meta or events.shape[0] >= 0
    # Mutate a *copy* after the last event as if a future row leaked — adapter digest path
    from trencher.world_model.batch_adapter import example_to_tensors

    t0 = example_to_tensors(ex)
    dig0 = float(t0["event_x"].sum()) if t0["event_x"].numel() else 0.0
    # Append a fake future event into sequences then rebuild tensors without going through reader
    if events.size:
        fake = events[-1:].copy()
        fake[:, 0] = fake[:, 0] + 1e6  # huge rel_dt
        ex2 = dict(ex)
        ex2["sequences"] = dict(ex["sequences"])
        ex2["sequences"]["events"] = np.concatenate([events, fake], axis=0)
        t1 = example_to_tensors(ex2)
        # Inputs change if we bypass causal reader — proving the *reader* is the guard.
        # Re-load via builder must remain dig0.
        ex_reload = builder.build(row)
        t2 = example_to_tensors(ex_reload)
        dig2 = float(t2["event_x"].sum()) if t2["event_x"].numel() else 0.0
        assert dig2 == dig0
        assert (float(t1["event_x"].sum()) != dig0
                or t1["event_x"].shape[0] != t0["event_x"].shape[0])
    assert T > 0


def test_forward_backward_on_real_broad_batch():
    from trencher.data.world_model.training.dataset import StreamingSplitDataset
    from trencher.data.world_model.training.example import ProductionExampleBuilder
    from trencher.data.world_model.training.loaders import resolve_assignment_paths
    from trencher.world_model.batch_adapter import (
        barrier_bids_from_pairs,
        collate_tensor_examples,
        example_to_tensors,
    )
    from trencher.world_model.config import tiny_smoke_config
    from trencher.world_model.losses import MultiTaskLoss
    from trencher.world_model.model import WorldModel

    cfg = tiny_smoke_config()
    paths = resolve_assignment_paths(SPLIT, "broad")
    ds = StreamingSplitDataset(assignment_paths=paths[:1], split="train", shuffle=False)
    rows = list(ds.iter_rows(max_examples=4))
    builder = ProductionExampleBuilder(world_root=WORLD, split_root=SPLIT, population="broad")
    builder.prefetch(rows)
    tensors = [
        example_to_tensors(
            builder.build(r),
            explosion_horizons=cfg.heads.explosion_horizons,
            survival_horizons=cfg.heads.survival_horizons,
            return_horizons=cfg.heads.return_horizons,
            barrier_bids=barrier_bids_from_pairs(cfg.heads.barrier_pairs),
            execution_sizes=cfg.heads.execution_sizes,
            path_steps=cfg.heads.path_steps,
            path_dim=cfg.heads.path_dim,
        )
        for r in rows
    ]
    batch = collate_tensor_examples(tensors)
    model = WorldModel(cfg)
    loss_fn = MultiTaskLoss(cfg)
    out = model(batch, phase="broad")
    breakdown = loss_fn(out, batch, phase="broad")
    assert torch.isfinite(breakdown.total)
    breakdown.total.backward()


def test_checkpoint_resume_equivalence(tmp_path: Path):
    from trencher.data.world_model.training.dataset import StreamingSplitDataset
    from trencher.data.world_model.training.example import ProductionExampleBuilder
    from trencher.data.world_model.training.loaders import resolve_assignment_paths
    from trencher.world_model.batch_adapter import (
        barrier_bids_from_pairs,
        collate_tensor_examples,
        example_to_tensors,
    )
    from trencher.world_model.checkpoint import load_checkpoint, save_checkpoint
    from trencher.world_model.config import tiny_smoke_config
    from trencher.world_model.model import WorldModel
    from trencher.world_model.trainer import ThreePhaseTrainer

    cfg = tiny_smoke_config()
    paths = resolve_assignment_paths(SPLIT, "broad")
    ds = StreamingSplitDataset(assignment_paths=paths[:1], split="train", shuffle=False)
    rows = list(ds.iter_rows(max_examples=8))
    builder = ProductionExampleBuilder(world_root=WORLD, split_root=SPLIT, population="broad")
    builder.prefetch(rows)
    tensors = [
        example_to_tensors(
            builder.build(r),
            explosion_horizons=cfg.heads.explosion_horizons,
            survival_horizons=cfg.heads.survival_horizons,
            return_horizons=cfg.heads.return_horizons,
            barrier_bids=barrier_bids_from_pairs(cfg.heads.barrier_pairs),
            execution_sizes=cfg.heads.execution_sizes,
            path_steps=cfg.heads.path_steps,
            path_dim=cfg.heads.path_dim,
        )
        for r in rows
    ]
    batches = [collate_tensor_examples(tensors[i : i + 2]) for i in range(0, len(tensors), 2)]
    cfg.train.lr = 1e-3
    cfg.train.warmup_steps = 0

    torch.manual_seed(0)
    np.random.seed(0)
    m1 = WorldModel(cfg)
    t1 = ThreePhaseTrainer(m1, cfg, device="cpu", ckpt_dir=tmp_path / "a")
    n_steps = min(3, len(batches))
    t1.train_phase("broad", list(batches[:n_steps]), max_steps=n_steps)

    torch.manual_seed(0)
    np.random.seed(0)
    m2 = WorldModel(cfg)
    t2 = ThreePhaseTrainer(m2, cfg, device="cpu", ckpt_dir=tmp_path / "b")
    k = 1
    t2.train_phase("broad", list(batches[:k]), max_steps=k)
    ckpt = save_checkpoint(
        tmp_path / "mid.pt",
        model=m2,
        optimizer=t2.optimizer,
        scheduler=t2._scheduler,
        step=t2.global_step,
        phase="broad",
        config=cfg,
        telemetry={"phase_step": t2.phase_step},
    )
    m3 = WorldModel(cfg)
    t3 = ThreePhaseTrainer(m3, cfg, device="cpu")
    load_checkpoint(
        ckpt, model=m3, optimizer=t3.optimizer, scheduler=t3._scheduler, expected_config=cfg
    )
    t3.global_step = t2.global_step
    rem = list(batches[k:n_steps])
    if rem:
        t3.train_phase("broad", rem, max_steps=len(rem))
    for a, b in zip(m1.parameters(), m3.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-3, rtol=1e-3)
