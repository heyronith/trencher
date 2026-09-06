"""Finite deep-input sanitization + SSL mask + GCS latest resume tests."""

from __future__ import annotations

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


def _synthetic_event_row(*, price=1.0, v_sol=10.0, sol=0.5, inter=0.2) -> np.ndarray:
    # EVENT_CHANNELS length 13
    return np.asarray(
        [
            0.0,  # rel_dt
            inter,  # interarrival
            1.0,  # is_buy
            1.0,  # side_id
            sol,  # sol_amount
            100.0,  # token_amount
            price,  # price_sol
            v_sol,  # v_sol
            1000.0,  # v_tok
            0.1,  # curve_pct
            1.0 if np.isfinite(v_sol) else 0.0,  # has_reserves
            1.0 if np.isfinite(price) else 0.0,  # has_finite_price
            1.0,  # source_quality
        ],
        dtype=np.float32,
    )


def _example_from_events(events: np.ndarray) -> dict:
    return {
        "meta": {"observation_id": "synth_missing", "base_split": "train"},
        "sequences": {"events": events},
        "modality_masks": {"mask_events": 1.0},
        "targets": {},
        "features": {},
        "causal_ok": True,
    }


def test_missing_price_sanitized_keeps_has_finite_price_flag():
    from trencher.world_model.batch_adapter import example_to_tensors

    e0 = _synthetic_event_row(price=np.nan)
    e1 = _synthetic_event_row(price=1.2, inter=0.3)
    events = np.stack([e0, e1], axis=0)
    # Force has_finite_price=0 on first row
    events[0, 11] = 0.0
    out = example_to_tensors(_example_from_events(events))
    assert torch.isfinite(out["event_x"]).all()
    assert float(out["event_x"][0, 6]) == 0.0  # sanitized price
    assert float(out["event_x"][0, 11]) == 0.0  # has_finite_price retained
    assert not bool(out["event_x_orig_finite"][0, 6])


def test_missing_reserve_sanitized_and_excluded_from_ssl_reserve():
    from trencher.world_model.batch_adapter import example_to_tensors

    e0 = _synthetic_event_row(v_sol=np.nan)
    e1 = _synthetic_event_row(v_sol=12.0, inter=0.4)
    events = np.stack([e0, e1], axis=0)
    events[0, 10] = 0.0
    out = example_to_tensors(_example_from_events(events))
    assert torch.isfinite(out["event_x"]).all()
    assert float(out["event_x"][0, 7]) == 0.0
    assert not bool(out["ssl_reserve_mask"][0])


def test_missing_next_amount_excluded_from_size_ssl():
    from trencher.world_model.batch_adapter import example_to_tensors

    e0 = _synthetic_event_row()
    e1 = _synthetic_event_row(sol=np.nan, inter=0.5)
    events = np.stack([e0, e1], axis=0)
    out = example_to_tensors(_example_from_events(events))
    assert not bool(out["ssl_size_mask"][0])
    assert bool(out["ssl_side_mask"][0])


def test_recon_never_selects_originally_missing_cell():
    from trencher.world_model.batch_adapter import collate_tensor_examples, example_to_tensors
    from trencher.world_model.ssl_masking import apply_ssl_event_masking

    e0 = _synthetic_event_row(price=np.nan)
    e1 = _synthetic_event_row(price=1.0, inter=0.2)
    e2 = _synthetic_event_row(price=1.1, inter=0.3)
    events = np.stack([e0, e1, e2], axis=0)
    events[0, 11] = 0.0
    out = example_to_tensors(_example_from_events(events))
    batch = collate_tensor_examples([out])
    # Force mask ratio 1.0 on all eligible cells
    masked = apply_ssl_event_masking(batch, seed=0, mask_ratio=1.0)
    # Position 0 price channel was originally non-finite → must not be in recon_mask
    assert not bool(masked["ssl_recon_mask"][0, 0, 6])
    assert torch.isfinite(masked["ssl_recon_target"]).all()


def test_complete_valid_event_unchanged_except_sanitize_noop():
    from trencher.world_model.batch_adapter import example_to_tensors

    e0 = _synthetic_event_row(price=1.5, v_sol=9.0, sol=0.7, inter=0.25)
    e1 = _synthetic_event_row(price=1.6, v_sol=9.5, sol=0.8, inter=0.3)
    events = np.stack([e0, e1], axis=0)
    out = example_to_tensors(_example_from_events(events))
    assert torch.allclose(out["event_x"], torch.from_numpy(events))
    assert bool(out["event_x_orig_finite"].all())


def test_forward_backward_finite_on_missing_event():
    from trencher.world_model.batch_adapter import collate_tensor_examples, example_to_tensors
    from trencher.world_model.config import tiny_smoke_config
    from trencher.world_model.losses import MultiTaskLoss
    from trencher.world_model.model import WorldModel

    rows = []
    for price, vsol, sol in [(np.nan, 10.0, 0.5), (1.0, np.nan, 0.5), (1.0, 10.0, np.nan)]:
        e0 = _synthetic_event_row(price=price if price == price else 1.0, v_sol=10.0, sol=0.5)
        e1 = _synthetic_event_row(price=1.0, v_sol=vsol, sol=sol, inter=0.2)
        if not np.isfinite(price):
            e0[6] = np.nan
            e0[11] = 0.0
        events = np.stack([e0, e1], axis=0)
        rows.append(example_to_tensors(_example_from_events(events)))
    batch = collate_tensor_examples(rows)
    cfg = tiny_smoke_config()
    model = WorldModel(cfg)
    loss_fn = MultiTaskLoss(cfg)
    out = model(batch, phase="ssl")
    bd = loss_fn(out, batch, phase="ssl")
    assert torch.isfinite(bd.total)
    bd.total.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


@pytest.mark.skipif(not _have_freeze(), reason="frozen world_model data not present")
def test_real_broad_batch_with_missing_is_finite():
    from trencher.data.world_model.training.dataset import StreamingSplitDataset
    from trencher.data.world_model.training.example import ProductionExampleBuilder
    from trencher.data.world_model.training.loaders import resolve_assignment_paths
    from trencher.world_model.batch_adapter import (
        barrier_bids_from_pairs,
        collate_tensor_examples,
        example_to_tensors,
    )
    from trencher.world_model.config import tiny_smoke_config
    from trencher.world_model.input_sanitize import assert_finite_model_inputs
    from trencher.world_model.losses import MultiTaskLoss
    from trencher.world_model.model import WorldModel

    cfg = tiny_smoke_config()
    paths = resolve_assignment_paths(SPLIT, "broad")
    ds = StreamingSplitDataset(
        assignment_paths=paths[:4], split="train", shuffle=False, shuffle_paths=False
    )
    builder = ProductionExampleBuilder(world_root=WORLD, split_root=SPLIT, population="broad")
    rows = []
    missing_hits = 0
    for r in ds.iter_rows(max_examples=64):
        builder.prefetch([r])
        t = example_to_tensors(
            builder.build(r),
            explosion_horizons=cfg.heads.explosion_horizons,
            survival_horizons=cfg.heads.survival_horizons,
            return_horizons=cfg.heads.return_horizons,
            barrier_bids=barrier_bids_from_pairs(cfg.heads.barrier_pairs),
            execution_sizes=cfg.heads.execution_sizes,
            path_steps=cfg.heads.path_steps,
            path_dim=cfg.heads.path_dim,
        )
        if float(t["had_original_nonfinite"]) > 0.5:
            missing_hits += 1
            rows.append(t)
        if len(rows) >= 4:
            break
    assert missing_hits >= 1, "expected at least one real observation with original missing values"
    batch = collate_tensor_examples(rows)
    assert_finite_model_inputs(batch)
    model = WorldModel(cfg)
    loss_fn = MultiTaskLoss(cfg)
    out = model(batch, phase="ssl")
    bd = loss_fn(out, batch, phase="ssl")
    assert torch.isfinite(bd.total)
    bd.total.backward()


def test_gcs_latest_resume_via_local_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from trencher.world_model.durable import LocalMirrorStore
    from trencher.world_model.launch import _maybe_fetch_durable_resume

    run_id = "finite_resume_test"
    mirror = tmp_path / "durable"
    store = LocalMirrorStore(mirror / run_id)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    payload = {"step": 7, "marker": "ok"}
    local = tmp_path / "src.pt"
    torch.save(payload, local)
    store.upload_file(local, "latest.pt")

    monkeypatch.setenv("TRENCHER_DURABLE_MIRROR", str(mirror))
    monkeypatch.setenv("TRENCHER_WM_RUN_ID", run_id)
    monkeypatch.delenv("TRENCHER_GCS_URI", raising=False)
    resolved = _maybe_fetch_durable_resume("latest", ckpt_dir)
    assert Path(resolved).is_file()
    loaded = torch.load(resolved, map_location="cpu", weights_only=False)
    assert loaded["step"] == 7

    # Explicit latest with neither backend must fail closed
    monkeypatch.delenv("TRENCHER_DURABLE_MIRROR", raising=False)
    from trencher.world_model.durable import DurableSyncError

    with pytest.raises(DurableSyncError):
        _maybe_fetch_durable_resume("latest", ckpt_dir)


@pytest.mark.skipif(not _have_freeze(), reason="frozen world_model data not present")
def test_finite_audit_smoke_small():
    from trencher.world_model.finite_audit import audit_deep_input_finite

    audit = audit_deep_input_finite(
        world_root=WORLD,
        split_root=SPLIT,
        population="broad",
        max_examples=64,
        seed=0,
    )
    assert audit["observations_scanned"] == 64
    assert audit["model_input_nonfinite_after_tensorization_total"] == 0
    assert audit["observations_with_original_missing"] >= 1
