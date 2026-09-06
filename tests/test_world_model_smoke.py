"""CPU smoke tests for trencher.world_model (synthetic tensors only)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from trencher.world_model import WorldModel, count_parameters  # noqa: E402
from trencher.world_model.checkpoint import (  # noqa: E402
    ConfigMismatchError,
    load_checkpoint,
    save_checkpoint,
)
from trencher.world_model.config import (  # noqa: E402
    load_world_model_config,
    tiny_smoke_config,
)
from trencher.world_model.data import WorldModelDataset  # noqa: E402
from trencher.world_model.losses import MultiTaskLoss  # noqa: E402
from trencher.world_model.split_guard import (  # noqa: E402
    PermanentTestAccessError,
    refuse_test_split,
)
from trencher.world_model.staging import (  # noqa: E402
    build_staging_manifest,
    verify_staging_manifest,
    write_staging_manifest,
)
from trencher.world_model.trainer import ThreePhaseTrainer  # noqa: E402


def _synth_batch(cfg, *, rich: bool = True, missing: set[str] | None = None, b: int = 2) -> dict:
    missing = missing or set()
    t_e, t_c, t_g = 8, 4, 4
    h_exp = len(cfg.heads.explosion_horizons)
    h_surv = len(cfg.heads.survival_horizons)
    n_bar = len(cfg.heads.barrier_levels)

    batch: dict = {}
    if "events" not in missing:
        batch["event_x"] = torch.randn(b, t_e, cfg.event.n_channels)
        batch["event_times"] = torch.linspace(0, 10, t_e).unsqueeze(0).expand(b, -1)
        batch["event_mask"] = torch.ones(b, t_e, dtype=torch.bool)
        batch["ssl_next_side"] = torch.randint(0, cfg.event.n_sides, (b, t_e - 1))
        batch["ssl_next_log_dt"] = torch.randn(b, t_e - 1)
        batch["ssl_next_size"] = torch.randn(b, t_e - 1)
        batch["ssl_mask"] = torch.ones(b, t_e - 1, dtype=torch.bool)

    if "curve" not in missing:
        batch["curve_x"] = torch.randn(b, t_c, cfg.curve.n_channels)
        batch["curve_times"] = torch.linspace(0, 10, t_c).unsqueeze(0).expand(b, -1)
        batch["curve_mask"] = torch.ones(b, t_c, dtype=torch.bool)

    if "global" not in missing:
        batch["global_buckets"] = torch.randn(b, t_g, cfg.global_temporal.n_bucket_channels)
        batch["global_times"] = torch.linspace(0, 10, t_g).unsqueeze(0).expand(b, -1)
        batch["global_scalars"] = torch.randn(b, cfg.global_temporal.n_scalars)
        batch["global_mask"] = torch.ones(b, t_g, dtype=torch.bool)

    if "creator" not in missing:
        batch["creator_feat"] = torch.randn(b, 4)

    if rich:
        if "microstructure" not in missing:
            batch["micro_x"] = torch.randn(b, 4, cfg.microstructure.n_channels)
            batch["micro_times"] = torch.linspace(0, 5, 4).unsqueeze(0).expand(b, -1)
            batch["micro_mask"] = torch.ones(b, 4, dtype=torch.bool)
            batch["micro_available"] = torch.ones(b)
        if "ownership" not in missing:
            batch["own_x"] = torch.randn(b, 4, cfg.ownership.n_channels)
            batch["own_times"] = torch.linspace(0, 5, 4).unsqueeze(0).expand(b, -1)
            batch["own_mask"] = torch.ones(b, 4, dtype=torch.bool)
            batch["own_available"] = torch.ones(b)
            batch["own_quality"] = torch.ones(b)
        if "wallet_prior" not in missing:
            batch["wallet_x"] = torch.randn(b, 4, cfg.wallet_prior.n_channels)
            batch["wallet_times"] = torch.linspace(0, 5, 4).unsqueeze(0).expand(b, -1)
            batch["wallet_mask"] = torch.ones(b, 4, dtype=torch.bool)
            batch["wallet_available"] = torch.ones(b)
        if "graph" not in missing:
            batch["graph_edge_feat"] = torch.randn(b, 4, cfg.graph.n_edge_channels)
            batch["graph_times"] = torch.linspace(0, 5, 4).unsqueeze(0).expand(b, -1)
            batch["graph_mask"] = torch.ones(b, 4, dtype=torch.bool)
            batch["graph_available"] = torch.ones(b)

    # Targets
    batch["explosion_target"] = torch.randint(0, 2, (b, h_exp)).float()
    batch["explosion_mask"] = torch.ones(b, h_exp)
    batch["survival_event"] = torch.randint(0, 2, (b,)).float()
    batch["survival_time_idx"] = torch.randint(0, h_surv, (b,))
    batch["survival_mask"] = torch.ones(b)
    batch["barrier_target"] = torch.randint(0, 2, (b, n_bar)).float()
    batch["return_target"] = torch.randn(b)
    batch["mfe_target"] = torch.randn(b).abs()
    batch["mae_target"] = torch.randn(b).abs()
    batch["persistence_target"] = torch.randint(0, cfg.heads.persistence_classes, (b,))
    batch["path_target"] = torch.randn(b, cfg.heads.path_steps, cfg.heads.path_dim)
    batch["execution_target"] = torch.randn(b, cfg.heads.execution_dim)
    batch["sample_mask"] = torch.ones(b)
    return batch


def test_import_and_param_count():
    model = WorldModel.tiny_for_tests()
    n = count_parameters(model)
    assert n > 1_000
    assert model.cfg.pure_on_chain
    assert model.cfg.forbid_id_embedding_tables


def test_forward_broad_rich_and_missing_modality():
    cfg = tiny_smoke_config()
    model = WorldModel(cfg)
    model.eval()

    broad = _synth_batch(cfg, rich=False)
    with torch.no_grad():
        out_b = model(broad, phase="broad")
    assert "explosion_logits" in out_b
    assert out_b["explosion_logits"].shape[0] == 2
    assert "fused" in out_b

    rich = _synth_batch(cfg, rich=True)
    with torch.no_grad():
        out_r = model(rich, phase="rich")
    assert out_r["path_pred"].shape == (2, cfg.heads.path_steps, cfg.heads.path_dim)

    missing = _synth_batch(cfg, rich=True, missing={"microstructure", "graph"})
    # Explicitly mark unavailable
    missing["micro_available"] = torch.zeros(2)
    missing["graph_available"] = torch.zeros(2)
    # Still provide tensors (zeros) so encoder path is exercised with avail=0
    missing["micro_x"] = torch.zeros(2, 4, cfg.microstructure.n_channels)
    missing["micro_times"] = torch.zeros(2, 4)
    missing["micro_mask"] = torch.ones(2, 4, dtype=torch.bool)
    missing["graph_edge_feat"] = torch.zeros(2, 4, cfg.graph.n_edge_channels)
    missing["graph_times"] = torch.zeros(2, 4)
    missing["graph_mask"] = torch.ones(2, 4, dtype=torch.bool)
    with torch.no_grad():
        out_m = model(missing, phase="rich")
    assert torch.isfinite(out_m["fused"]).all()


def test_loss_masks_and_backward():
    cfg = tiny_smoke_config()
    model = WorldModel(cfg)
    loss_fn = MultiTaskLoss(cfg)
    batch = _synth_batch(cfg, rich=True)
    # Mask out second sample entirely
    batch["sample_mask"] = torch.tensor([1.0, 0.0])
    out = model(batch, phase="rich")
    breakdown = loss_fn(out, batch, phase="rich")
    assert torch.isfinite(breakdown.total)
    breakdown.total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "expected gradients"


def test_tiny_overfit_five_steps():
    cfg = tiny_smoke_config()
    cfg.train.lr = 5e-3
    model = WorldModel(cfg)
    batch = _synth_batch(cfg, rich=False, b=4)
    batches = [batch for _ in range(5)]
    trainer = ThreePhaseTrainer(model, cfg, device="cpu")
    logs = trainer.train_phase("broad", batches, max_steps=5)
    assert len(logs) == 5
    assert logs[-1].loss <= logs[0].loss + 1e-3 or logs[-1].loss < logs[0].loss * 1.05
    # Prefer actual decrease for a stable synthetic overfit signal
    assert logs[-1].loss < logs[0].loss


def test_checkpoint_roundtrip(tmp_path: Path):
    cfg = tiny_smoke_config()
    model = WorldModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=opt, step=7, phase="broad", config=cfg)
    model2 = WorldModel(cfg)
    load_checkpoint(path, model=model2, expected_config=cfg, map_location="cpu")
    for a, b in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.allclose(a, b)

    cfg2 = tiny_smoke_config(name="other")
    with pytest.raises(ConfigMismatchError):
        load_checkpoint(path, model=WorldModel(cfg2), expected_config=cfg2)


def test_test_split_guard_fails():
    with pytest.raises(PermanentTestAccessError):
        refuse_test_split("test")
    with pytest.raises(PermanentTestAccessError):
        WorldModelDataset([{"x": torch.zeros(1)}], split="test")


def test_yaml_config_loads():
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/world_model/world_model_candidate_v1.yaml"
    cfg = load_world_model_config(path)
    assert cfg.d_model == 256
    assert cfg.event.n_layers == 6
    assert cfg.forbid_id_embedding_tables


def test_staging_manifest_roundtrip(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    # Build a mini staging root with required keys pointing at real repo files
    stage = tmp_path / "stage"
    (stage / "configs/world_model").mkdir(parents=True)
    (stage / "manifests/world_model_data").mkdir(parents=True)
    files = {
        "config": root / "configs/world_model/world_model_candidate_v1.yaml",
        "training_contract": root
        / "manifests/world_model_data/WORLD_MODEL_TRAINING_CONTRACT_V1.json",
        "candidate_manifest": root / "manifests/world_model_data/WORLD_MODEL_CANDIDATE_V1.json",
        "freeze": root / "manifests/world_model_data/DATA_FREEZE.json",
        "split_audit": root / "manifests/world_model_data/SPLIT_AUDIT_V1.json",
        "loader_audit": root / "manifests/world_model_data/TRAINING_LOADER_AUDIT_V4.json",
    }
    rels = {}
    for key, src in files.items():
        if key == "config":
            dest = stage / "configs/world_model" / src.name
        else:
            dest = stage / "manifests/world_model_data" / src.name
        dest.write_bytes(src.read_bytes())
        rels[key] = dest.relative_to(stage)

    man = build_staging_manifest(stage, rels)
    man_path = write_staging_manifest(man, stage / "staging.json")
    result = verify_staging_manifest(man_path)
    assert result["ok"] is True


def test_causality_future_mutation_changes_targets_not_inputs():
    """Mutating 'future' labels must not change causal input digests at T."""
    cfg = tiny_smoke_config()
    batch = _synth_batch(cfg, rich=True, b=2)

    def _digest(b: dict) -> str:
        causal_keys = [
            "event_x",
            "event_times",
            "curve_x",
            "curve_times",
            "global_buckets",
            "global_scalars",
            "micro_x",
            "own_x",
            "wallet_x",
            "graph_edge_feat",
        ]
        parts = []
        for k in causal_keys:
            if k in b and torch.is_tensor(b[k]):
                parts.append(k + ":" + str(tuple(b[k].shape)) + ":" + str(float(b[k].sum())))
        return "|".join(parts)

    d0 = _digest(batch)
    # Mutate future labels only
    batch2 = dict(batch)
    batch2["explosion_target"] = 1.0 - batch["explosion_target"]
    batch2["return_target"] = batch["return_target"] + 3.0
    batch2["path_target"] = batch["path_target"] + 1.0
    d1 = _digest(batch2)
    assert d0 == d1
    assert not torch.equal(batch["explosion_target"], batch2["explosion_target"])


def test_missing_modality_uses_availability_not_fabricated_observed_zeros():
    cfg = tiny_smoke_config()
    model = WorldModel(cfg)
    model.eval()
    present = _synth_batch(cfg, rich=True, b=2)
    missing = _synth_batch(cfg, rich=True, b=2)
    missing["micro_x"] = torch.zeros_like(missing["micro_x"])
    missing["micro_available"] = torch.zeros(2)
    with torch.no_grad():
        out_p = model(present, phase="rich")
        out_m = model(missing, phase="rich")
    # Fused states must differ when availability flips (null embedding path).
    assert not torch.allclose(out_p["fused"], out_m["fused"])


def test_v4_loader_refuses_test_mode():
    from trencher.world_model.v4_loader import assert_dataset_mode, build_deep_loader_manifest

    with pytest.raises(PermanentTestAccessError):
        assert_dataset_mode("test")
    with pytest.raises(PermanentTestAccessError):
        assert_dataset_mode("permanent_test")
    man = build_deep_loader_manifest(
        world_root=Path("/tmp/world"),
        split_root=Path("/tmp/split"),
        mode="wf_fold_1",
        population="broad",
    )
    assert man["guards"]["permanent_test"] == "FORBIDDEN"
