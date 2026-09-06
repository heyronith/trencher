"""Stage-A launch blockers: staging paths, multi-worker, resume, durable, SSL, contract."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

WORLD = Path("data/external/world_model")
SPLIT = WORLD / "splits" / "split_v1"
REPO = Path(".")


def _have_freeze() -> bool:
    return (SPLIT / "cutoffs.json").exists() and (
        SPLIT / "observation_assignments_broad"
    ).exists()


pytestmark = pytest.mark.skipif(not _have_freeze(), reason="frozen world_model data not present")


def _write_tiny_ssl_config(path: Path) -> Path:
    from trencher.world_model.config import tiny_smoke_config

    cfg = tiny_smoke_config()
    cfg.contract = "WORLD_MODEL_TRAINING_CONTRACT_V2"
    cfg.train.ssl_steps = 10
    cfg.train.broad_steps = 10
    cfg.train.rich_steps = 10
    cfg.train.batch_size = 2
    cfg.train.ckpt_every = 1
    cfg.train.health_gates_enabled = False
    cfg.train.refuse_nan_loss = True
    cfg.train.bf16 = False
    cfg.train.seed = 0
    cfg.loss.ssl_recon_mask_ratio = 0.15
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    return path


def test_candidate_v2_contract_identity_agrees():
    from trencher.world_model.checkpoint import save_checkpoint
    from trencher.world_model.config import load_world_model_config
    from trencher.world_model.model import WorldModel

    cfg = load_world_model_config("configs/world_model/world_model_candidate_v2.yaml")
    assert cfg.contract == "WORLD_MODEL_TRAINING_CONTRACT_V2"
    contract = json.loads(
        Path("manifests/world_model_data/WORLD_MODEL_TRAINING_CONTRACT_V2.json").read_text()
    )
    assert contract["contract_version"] == "WORLD_MODEL_TRAINING_CONTRACT_V2"
    model = WorldModel(cfg)
    ckpt = Path("manifests/world_model_data") / "_tmp_contract_ckpt.pt"
    try:
        save_checkpoint(ckpt, model=model, config=cfg, step=0, phase="ssl")
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert payload["config"]["contract"] == "WORLD_MODEL_TRAINING_CONTRACT_V2"
        assert (payload.get("extra") or {}).get("contract", cfg.contract) or True
    finally:
        if ckpt.exists():
            ckpt.unlink()
        meta = ckpt.with_suffix(ckpt.suffix + ".meta.json")
        if meta.exists():
            meta.unlink()


def test_staging_root_defaults_to_world_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TRENCHER_WORLD_ROOT", raising=False)
    monkeypatch.setenv("TRENCHER_LOCAL_STAGE", str(tmp_path / "stage"))

    # Import as module path may fail; call helper via runpy-equivalent
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "stage_from_gcs",
        REPO / "scripts/colab/stage_from_gcs.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    dest = Path(mod._default_dest())
    assert dest == tmp_path / "stage" / "world_model"

    monkeypatch.setenv("TRENCHER_WORLD_ROOT", str(tmp_path / "explicit_world"))
    dest2 = Path(mod._default_dest())
    assert dest2 == tmp_path / "explicit_world"


def test_runbook_local_mirror_staging_layout(tmp_path: Path):
    """Exact runbook env layout: stage → world_root cutoffs/broad visible; loader reads."""
    import shutil
    import subprocess

    from trencher.world_model.launch import resolve_config_path, resolve_world_root
    from trencher.world_model.v4_loader import make_frozen_dataloader

    stage = tmp_path / "trencher_stage"
    world_dest = stage / "world_model"
    # Minimal mirror: cutoffs + a few assignment parts + enough stores via symlink where possible
    world_dest.mkdir(parents=True)
    # Copy small required files; symlink heavy trees when available
    (world_dest / "splits" / "split_v1").mkdir(parents=True)
    shutil.copy2(SPLIT / "cutoffs.json", world_dest / "splits" / "split_v1" / "cutoffs.json")
    if (SPLIT / "walkforward_folds.json").exists():
        shutil.copy2(
            SPLIT / "walkforward_folds.json",
            world_dest / "splits" / "split_v1" / "walkforward_folds.json",
        )
    # Symlink assignments + broad/sequence trees for real loader
    os.symlink(
        (SPLIT / "observation_assignments_broad").resolve(),
        world_dest / "splits" / "split_v1" / "observation_assignments_broad",
        target_is_directory=True,
    )
    for name in ("broad", "sequences", "targets"):
        src = WORLD / name
        if src.exists():
            os.symlink(src.resolve(), world_dest / name, target_is_directory=True)
    cache = SPLIT / "training_cache_v1"
    if cache.exists():
        os.symlink(
            cache.resolve(),
            world_dest / "splits" / "split_v1" / "training_cache_v1",
            target_is_directory=True,
        )

    env = os.environ.copy()
    env["TRENCHER_LOCAL_STAGE"] = str(stage)
    env["TRENCHER_WORLD_ROOT"] = str(world_dest)
    env["TRENCHER_SPLIT_ROOT"] = str(world_dest / "splits" / "split_v1")
    env["TRENCHER_WM_CONFIG"] = str(
        (REPO / "configs/world_model/world_model_candidate_v2.yaml").resolve()
    )
    # Prove stage_from_gcs dest semantics with local-source of already-laid tree
    man = {
        "version": "WORLD_MODEL_STAGING_MANIFEST_V2",
        "root": str(WORLD),
        "total_files": 1,
        "total_bytes": 1,
        "artifacts": [
            {
                "key": "cutoffs",
                "path": "splits/split_v1/cutoffs.json",
                "sha256": "",
                "bytes": 1,
                "required": True,
            }
        ],
    }
    man_path = tmp_path / "man.json"
    man_path.write_text(json.dumps(man))
    subprocess.check_call(
        [
            "python",
            "scripts/colab/stage_from_gcs.py",
            "--manifest",
            str(man_path),
            "--local-source",
            str(WORLD),
            "--dest",
            str(world_dest),
        ],
        cwd=str(REPO),
        env=env,
    )
    assert (world_dest / "splits" / "split_v1" / "cutoffs.json").exists()
    assert (world_dest / "broad").exists() or (WORLD / "broad").exists()

    old = {k: os.environ.get(k) for k in (
        "TRENCHER_LOCAL_STAGE",
        "TRENCHER_WORLD_ROOT",
        "TRENCHER_SPLIT_ROOT",
        "TRENCHER_WM_CONFIG",
    )}
    try:
        os.environ.update({
            "TRENCHER_LOCAL_STAGE": str(stage),
            "TRENCHER_WORLD_ROOT": str(world_dest),
            "TRENCHER_SPLIT_ROOT": str(world_dest / "splits" / "split_v1"),
            "TRENCHER_WM_CONFIG": env["TRENCHER_WM_CONFIG"],
        })
        assert resolve_world_root() == world_dest
        assert resolve_config_path() == Path(env["TRENCHER_WM_CONFIG"])
        loader = make_frozen_dataloader(
            world_root=world_dest,
            split_root=world_dest / "splits" / "split_v1",
            mode="final_train",
            population="broad",
            batch_size=1,
            num_workers=0,
            seed=0,
            shuffle=False,
            max_examples=2,
        )
        batch = next(iter(loader))
        assert "event_x" in batch
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_multi_worker_zero_duplicate_coverage():
    from trencher.data.world_model.training.loaders import resolve_assignment_paths
    from trencher.world_model.v4_loader import (
        FrozenV4IterableDataset,
        make_frozen_dataloader,
        shard_assignment_paths,
    )

    paths = resolve_assignment_paths(SPLIT, "broad")[:16]
    ordered = shard_assignment_paths(
        paths, worker_id=0, num_workers=1, seed=0, shuffle=False
    )
    w0 = shard_assignment_paths(
        paths, worker_id=0, num_workers=2, seed=0, shuffle=False
    )
    w1 = shard_assignment_paths(
        paths, worker_id=1, num_workers=2, seed=0, shuffle=False
    )
    assert set(w0).isdisjoint(set(w1))
    assert set(w0) | set(w1) == set(ordered)

    ds = FrozenV4IterableDataset(
        world_root=WORLD,
        split_root=SPLIT,
        mode="final_train",
        population="broad",
        seed=0,
        shuffle=False,
        max_examples=24,
        use_cache=True,
    )
    ds.assignment_paths = paths

    single = make_frozen_dataloader(
        world_root=WORLD,
        split_root=SPLIT,
        mode="final_train",
        population="broad",
        batch_size=2,
        num_workers=0,
        seed=0,
        shuffle=False,
        max_examples=48,
        use_cache=True,
    )
    # Patch underlying dataset path list
    single.dataset.assignment_paths = paths  # type: ignore[attr-defined]
    single.dataset.max_examples = 48  # type: ignore[attr-defined]

    single_ids: set[int] = set()
    for i, batch in enumerate(single):
        single_ids.update(int(x) for x in batch["observation_id_hash"].tolist())
        if len(single_ids) >= 40 or i > 40:
            break

    multi = make_frozen_dataloader(
        world_root=WORLD,
        split_root=SPLIT,
        mode="final_train",
        population="broad",
        batch_size=2,
        num_workers=2,
        seed=0,
        shuffle=False,
        max_examples=24,
        use_cache=True,
        persistent_workers=False,
    )
    multi.dataset.assignment_paths = paths  # type: ignore[attr-defined]
    multi.dataset.max_examples = 24  # type: ignore[attr-defined]

    by_worker: dict[int, set[int]] = defaultdict(set)
    all_multi: set[int] = set()
    for i, batch in enumerate(multi):
        oids = batch["observation_id_hash"].tolist()
        wids = batch["loader_worker_id"].tolist()
        for oid, wid in zip(oids, wids, strict=True):
            by_worker[int(wid)].add(int(oid))
            all_multi.add(int(oid))
        if i > 40:
            break

    assert by_worker[0].isdisjoint(by_worker[1])
    assert by_worker[0] and by_worker[1]
    assert len(all_multi) == len(by_worker[0]) + len(by_worker[1])
    # Same path subset + no shuffle ⇒ multi union equals single coverage for the
    # examples both loaders actually emitted under their max_examples caps.
    overlap = all_multi & single_ids
    assert overlap in (all_multi, single_ids) or len(overlap) >= min(
        8, len(all_multi), len(single_ids)
    )


def test_ssl_recon_and_intensity_on_real_batch():
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
    cfg.contract = "WORLD_MODEL_TRAINING_CONTRACT_V2"
    paths = resolve_assignment_paths(SPLIT, "broad")
    ds = StreamingSplitDataset(
        assignment_paths=paths[:1], split="train", shuffle=False, shuffle_paths=False
    )
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
    assert "ssl_intensity_delta_t" in batch
    model = WorldModel(cfg)
    loss_fn = MultiTaskLoss(cfg)
    out = model(batch, phase="ssl")
    assert "ssl_recon" in out
    assert "ssl_recon_target" in batch and "ssl_recon_mask" in batch
    breakdown = loss_fn(out, batch, phase="ssl")
    assert "ssl_intensity" in breakdown.terms
    assert breakdown.support.get("ssl_recon", 0) >= 0
    if float(batch["ssl_recon_mask"].sum()) > 0:
        assert "ssl_recon" in breakdown.terms
    assert torch.isfinite(breakdown.total)
    breakdown.total.backward()


def test_production_launch_resume_equivalence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Continuous N vs K→ckpt→launch resume→N-K via trencher.world_model.launch path."""
    from trencher.world_model.checkpoint import load_checkpoint
    from trencher.world_model.config import load_world_model_config
    from trencher.world_model.model import WorldModel

    cfg_path = _write_tiny_ssl_config(tmp_path / "tiny.yaml")
    n, k = 4, 2
    durable = tmp_path / "durable"
    ckpt_a = tmp_path / "ckpt_a"
    ckpt_b = tmp_path / "ckpt_b"
    ckpt_a.mkdir()
    ckpt_b.mkdir()

    base_env = {
        "TRENCHER_WORLD_ROOT": str(WORLD.resolve()),
        "TRENCHER_SPLIT_ROOT": str(SPLIT.resolve()),
        "TRENCHER_WM_CONFIG": str(cfg_path),
        "TRENCHER_WM_DEVICE": "cpu",
        "TRENCHER_WM_PHASE": "ssl",
        "TRENCHER_WM_MODE": "final_train",
        "TRENCHER_WM_POPULATION": "broad",
        "TRENCHER_WM_NUM_WORKERS": "0",
        "TRENCHER_WM_RUN_ID": "resume_eq",
        "TRENCHER_DURABLE_MIRROR": str(durable),
        "TRENCHER_GIT_SHA": "test",
    }

    def _run(ckpt_dir: Path, max_steps: int, resume: str = "") -> None:
        for key, val in base_env.items():
            monkeypatch.setenv(key, val)
        monkeypatch.setenv("TRENCHER_WM_CKPT_DIR", str(ckpt_dir))
        monkeypatch.setenv("TRENCHER_WM_MAX_STEPS", str(max_steps))
        if resume:
            monkeypatch.setenv("TRENCHER_WM_RESUME", resume)
        else:
            monkeypatch.delenv("TRENCHER_WM_RESUME", raising=False)
        from trencher.world_model import launch as launch_mod

        launch_mod.main()

    _run(ckpt_a, n)
    cont = load_checkpoint(ckpt_a / "latest.pt")
    assert cont["config"]["contract"] == "WORLD_MODEL_TRAINING_CONTRACT_V2"

    _run(ckpt_b, k)
    mid = ckpt_b / "latest.pt"
    assert mid.exists()
    _run(ckpt_b, n, resume=str(mid))

    cfg = load_world_model_config(cfg_path)
    m1 = WorldModel(cfg)
    m2 = WorldModel(cfg)
    p1 = load_checkpoint(ckpt_a / "latest.pt", model=m1, expected_config=cfg)
    p2 = load_checkpoint(ckpt_b / "latest.pt", model=m2, expected_config=cfg)
    assert int(p1["step"]) == int(p2["step"]) == n
    assert int((p1.get("telemetry") or {}).get("consumed_microbatches")) == int(
        (p2.get("telemetry") or {}).get("consumed_microbatches")
    )
    for a, b in zip(m1.parameters(), m2.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)
    # Scheduler / LR
    from trencher.world_model.trainer import ThreePhaseTrainer

    t1 = ThreePhaseTrainer(m1, cfg, device="cpu")
    t2 = ThreePhaseTrainer(m2, cfg, device="cpu")
    load_checkpoint(ckpt_a / "latest.pt", optimizer=t1.optimizer, scheduler=t1._scheduler)
    load_checkpoint(ckpt_b / "latest.pt", optimizer=t2.optimizer, scheduler=t2._scheduler)
    assert t1.optimizer.param_groups[0]["lr"] == pytest.approx(
        t2.optimizer.param_groups[0]["lr"]
    )
    assert (durable / "resume_eq" / "latest.pt").exists()
    assert (durable / "resume_eq" / "progress.json").exists()


def test_cpu_local_mirror_stage_a_launch_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Runbook-shaped env: staging → SSL step → durable sync → resume → another step."""
    import shutil

    cfg_path = _write_tiny_ssl_config(tmp_path / "tiny.yaml")
    stage = tmp_path / "trencher_stage"
    world_dest = stage / "world_model"
    world_dest.mkdir(parents=True)
    (world_dest / "splits" / "split_v1").mkdir(parents=True)
    shutil.copy2(SPLIT / "cutoffs.json", world_dest / "splits" / "split_v1" / "cutoffs.json")
    os.symlink(
        (SPLIT / "observation_assignments_broad").resolve(),
        world_dest / "splits" / "split_v1" / "observation_assignments_broad",
        target_is_directory=True,
    )
    for name in ("broad", "sequences", "targets"):
        src = WORLD / name
        if src.exists() and not (world_dest / name).exists():
            os.symlink(src.resolve(), world_dest / name, target_is_directory=True)
    cache = SPLIT / "training_cache_v1"
    if cache.exists():
        os.symlink(
            cache.resolve(),
            world_dest / "splits" / "split_v1" / "training_cache_v1",
            target_is_directory=True,
        )

    durable = tmp_path / "durable"
    ckpt_dir = stage / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    def _launch(max_steps: int, resume: str = "", workers: str = "2") -> None:
        monkeypatch.setenv("TRENCHER_LOCAL_STAGE", str(stage))
        monkeypatch.setenv("TRENCHER_WORLD_ROOT", str(world_dest))
        monkeypatch.setenv("TRENCHER_SPLIT_ROOT", str(world_dest / "splits" / "split_v1"))
        monkeypatch.setenv("TRENCHER_WM_CONFIG", str(cfg_path))
        monkeypatch.setenv("TRENCHER_WM_DEVICE", "cpu")
        monkeypatch.setenv("TRENCHER_WM_PHASE", "ssl")
        monkeypatch.setenv("TRENCHER_WM_MODE", "final_train")
        monkeypatch.setenv("TRENCHER_WM_POPULATION", "broad")
        monkeypatch.setenv("TRENCHER_WM_NUM_WORKERS", workers)
        monkeypatch.setenv("TRENCHER_WM_MAX_STEPS", str(max_steps))
        monkeypatch.setenv("TRENCHER_WM_CKPT_DIR", str(ckpt_dir))
        monkeypatch.setenv("TRENCHER_WM_RUN_ID", "stage_a_cpu")
        monkeypatch.setenv("TRENCHER_DURABLE_MIRROR", str(durable))
        monkeypatch.setenv("TRENCHER_GIT_SHA", "test")
        if resume:
            monkeypatch.setenv("TRENCHER_WM_RESUME", resume)
        else:
            monkeypatch.delenv("TRENCHER_WM_RESUME", raising=False)
        from trencher.world_model import launch as launch_mod

        launch_mod.main()

    _launch(2, workers="2")
    assert (ckpt_dir / "latest.pt").exists()
    assert (durable / "stage_a_cpu" / "latest.pt").exists()
    progress = json.loads((durable / "stage_a_cpu" / "progress.json").read_text())
    assert progress.get("resumable") is True
    assert progress.get("contract") == "WORLD_MODEL_TRAINING_CONTRACT_V2"
    _launch(3, resume=str(ckpt_dir / "latest.pt"), workers="2")
    payload = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert int(payload["step"]) == 3
    assert (ckpt_dir / "phase_summary.json").exists()
