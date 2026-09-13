"""CLI entry: run a real frozen-data training phase (Stage A/B/C).

Invoked by ``scripts/colab/launch_training.sh``. Never unlocks permanent TEST.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trencher.world_model.checkpoint import load_checkpoint
from trencher.world_model.config import load_world_model_config
from trencher.world_model.durable import (
    DurableSyncError,
    LocalMirrorStore,
    build_durable_store_from_env,
)
from trencher.world_model.model import WorldModel, count_parameters
from trencher.world_model.progress import ProgressEmitter
from trencher.world_model.resumable_loader import (
    LoaderCursorError,
    assert_phase_global_consistent,
    wrap_frozen_dataloader,
)
from trencher.world_model.trainer import ResumeIncompatibilityError, ThreePhaseTrainer
from trencher.world_model.v4_loader import assert_dataset_mode, make_frozen_dataloader


def _assert_bf16_accelerator(*, device: str, bf16: bool) -> None:
    """Fail closed when BF16 training is configured on an incapable accelerator."""
    if not bf16:
        return
    if device != "cuda":
        raise SystemExit(
            f"BF16 training requires CUDA; refusing device={device!r} fallback"
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            "BF16 training requires CUDA; CUDA is unavailable"
        )
    props = torch.cuda.get_device_properties(0)
    major, minor = int(props.major), int(props.minor)
    bf16_ok = bool(torch.cuda.is_bf16_supported())
    print(f"gpu_name={props.name}")
    print(f"compute_capability={major}.{minor}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"bf16_supported={bf16_ok}")
    if not bf16_ok:
        raise SystemExit(
            f"BF16 training configured but accelerator {props.name!r} "
            f"(compute {major}.{minor}) does not support bfloat16; refusing fallback"
        )


def _git_sha() -> str:
    env = os.environ.get("TRENCHER_GIT_SHA")
    if env:
        return env
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _repo_root() -> Path:
    import trencher

    # src/trencher/__init__.py → parents[2] = repository root (editable install).
    return Path(trencher.__file__).resolve().parents[2]


def resolve_config_path() -> Path:
    """Prefer explicit env; else checked-out repo SHA config (not staged copy)."""
    env = os.environ.get("TRENCHER_WM_CONFIG")
    if env:
        return Path(env)
    return _repo_root() / "configs" / "world_model" / "world_model_candidate_v2.yaml"


def resolve_world_root() -> Path:
    world = os.environ.get("TRENCHER_WORLD_ROOT")
    if world:
        return Path(world)
    stage = os.environ.get("TRENCHER_LOCAL_STAGE")
    if stage:
        return Path(stage) / "world_model"
    return Path("data/external/world_model")


def _check_loader_identity(
    ckpt_identity: dict[str, Any],
    current: dict[str, Any],
    *,
    override: bool,
) -> None:
    keys = (
        "seed",
        "num_workers",
        "batch_size",
        "mode",
        "population",
        "phase",
        "shuffle",
        "accum_steps",
    )
    mismatches = {
        k: {"ckpt": ckpt_identity.get(k), "current": current.get(k)}
        for k in keys
        if ckpt_identity.get(k) != current.get(k)
    }
    if mismatches and not override:
        raise ResumeIncompatibilityError(
            f"refuse resume due to loader identity mismatch: {mismatches}. "
            "Set TRENCHER_WM_RESUME_OVERRIDE=1 to force."
        )


def _maybe_fetch_durable_resume(resume: str, ckpt_dir: Path) -> str:
    """Resolve durable ``latest`` via local-mirror or GCS; never silent-fail if requested."""
    resume = (resume or "").strip()
    if resume and Path(resume).is_file():
        return resume

    explicit_latest = resume.lower() in {"latest", "durable", "durable:latest"}
    if not explicit_latest:
        # Empty / non-latest paths are left for the caller (fresh start or direct path).
        return resume

    run_id = os.environ.get("TRENCHER_WM_RUN_ID", "default_run")
    mirror = os.environ.get("TRENCHER_DURABLE_MIRROR") or os.environ.get(
        "TRENCHER_WM_DURABLE_MIRROR"
    )
    gcs = os.environ.get("TRENCHER_GCS_URI") or ""
    store = None
    if mirror:
        store = LocalMirrorStore(Path(mirror) / run_id)
    elif gcs:
        from trencher.world_model.durable import GCSStore

        store = GCSStore(gcs, run_id)
    else:
        raise DurableSyncError(
            "TRENCHER_WM_RESUME=latest requested but neither "
            "TRENCHER_DURABLE_MIRROR nor TRENCHER_GCS_URI is set"
        )

    dest = ckpt_dir / "latest.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    try:
        store.download_file("latest.pt", dest)
    except DurableSyncError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DurableSyncError(
            f"failed to download durable latest.pt from {store.uri_for('latest.pt')}: {exc}"
        ) from exc
    if not dest.is_file():
        raise DurableSyncError(f"durable latest.pt missing after download: {dest}")
    print(f"durable resume downloaded from {store.uri_for('latest.pt')} -> {dest}")
    return str(dest)


def main() -> None:
    config_path = resolve_config_path()
    world_root = resolve_world_root()
    split_root = Path(
        os.environ.get("TRENCHER_SPLIT_ROOT", world_root / "splits" / "split_v1")
    )
    phase = os.environ.get("TRENCHER_WM_PHASE", "ssl")
    device = os.environ.get("TRENCHER_WM_DEVICE", "cuda")
    mode = assert_dataset_mode(os.environ.get("TRENCHER_WM_MODE", "final_train"))
    population = os.environ.get("TRENCHER_WM_POPULATION", "broad")
    if phase == "ssl":
        population = "broad"
    ckpt_dir = Path(os.environ.get("TRENCHER_WM_CKPT_DIR", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume = os.environ.get("TRENCHER_WM_RESUME") or ""
    resume = _maybe_fetch_durable_resume(resume, ckpt_dir)
    max_steps_env = os.environ.get("TRENCHER_WM_MAX_STEPS") or ""
    max_steps = int(max_steps_env) if max_steps_env.strip() else None
    num_workers = int(os.environ.get("TRENCHER_WM_NUM_WORKERS", "2"))
    run_id = os.environ.get("TRENCHER_WM_RUN_ID", "default_run")
    override = os.environ.get("TRENCHER_WM_RESUME_OVERRIDE", "") in {
        "1",
        "true",
        "True",
    }

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested but unavailable; refusing silent CPU fallback for production launch."
        )

    cfg = load_world_model_config(config_path)
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"accelerator={props.name} total_mem_gb={props.total_memory / 1e9:.1f}")
    _assert_bf16_accelerator(device=device, bf16=bool(cfg.train.bf16))

    torch.manual_seed(int(cfg.train.seed))
    np.random.seed(int(cfg.train.seed))
    model = WorldModel(cfg)
    omit_raw = os.environ.get("TRENCHER_WM_OMIT_MODALITIES", "").strip()
    if omit_raw:
        omit_set = {x.strip() for x in omit_raw.split(",") if x.strip()}
        model.omit_modalities = omit_set  # type: ignore[attr-defined]
        print(f"omit_modalities={sorted(omit_set)}")
    print(f"parameters={count_parameters(model):,}")
    print(f"config_hash={cfg.config_hash()}")
    print(f"git_sha={_git_sha()}")
    print(f"contract={cfg.contract}")
    print(f"config_path={config_path}")
    print(f"world_root={world_root}")

    loader_identity = {
        "seed": int(cfg.train.seed),
        "num_workers": num_workers,
        "batch_size": int(cfg.train.batch_size),
        "mode": mode,
        "population": population,
        "phase": phase,
        "shuffle": True,
        "accum_steps": int(cfg.train.accum_steps),
        "world_root": str(world_root),
        "split_root": str(split_root),
        "contract": cfg.contract,
    }

    loader = make_frozen_dataloader(
        world_root=world_root,
        split_root=split_root,
        mode=mode,
        population=population,
        batch_size=int(cfg.train.batch_size),
        num_workers=num_workers,
        seed=int(cfg.train.seed),
        shuffle=True,
        pin_memory=(device == "cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        # Honor frozen encoder sequence caps. Wallet-prior is the Stage-C OOM
        # blocker: RichSequenceReader can concatenate many wallets × last_n rows.
        seq_max_len={"wallet_prior": int(cfg.wallet_prior.max_len)},
    )

    progress_path = ckpt_dir / "progress.json"
    progress = ProgressEmitter(
        "world_model_train",
        status_path=progress_path,
        phase_weights={
            "ssl": cfg.train.ssl_steps
            / max(cfg.train.ssl_steps + cfg.train.broad_steps + cfg.train.rich_steps, 1),
            "broad": cfg.train.broad_steps
            / max(cfg.train.ssl_steps + cfg.train.broad_steps + cfg.train.rich_steps, 1),
            "rich": cfg.train.rich_steps
            / max(cfg.train.ssl_steps + cfg.train.broad_steps + cfg.train.rich_steps, 1),
        },
    )
    durable = build_durable_store_from_env(run_id=run_id)
    # Prefer explicit local mirror when set (tests / dry-run).
    mirror = os.environ.get("TRENCHER_DURABLE_MIRROR") or os.environ.get(
        "TRENCHER_WM_DURABLE_MIRROR"
    )
    if mirror:
        durable = LocalMirrorStore(Path(mirror) / run_id)

    trainer = ThreePhaseTrainer(
        model,
        cfg,
        device=device,
        progress=progress,
        ckpt_dir=ckpt_dir,
        git_sha=_git_sha(),
        durable_store=durable,
        loader_identity=loader_identity,
        freeze_ids={
            "freeze_id": cfg.freeze_id,
            "contract": cfg.contract,
            "split": "split_v1",
            "sequence_dataset": "sequence_dataset_v2",
            "targets": "target_dataset_v3/broad_targets_v3",
        },
    )

    resume_step = 0
    consumed_batches = 0
    checkpoint_payload: dict[str, Any] | None = None
    if resume:
        checkpoint_payload = load_checkpoint(
            resume,
            model=model,
            optimizer=trainer.optimizer,
            scheduler=trainer._scheduler,
            expected_config=cfg,
        )
        ckpt_identity = (checkpoint_payload.get("extra") or {}).get("loader_identity") or {}
        if ckpt_identity:
            _check_loader_identity(ckpt_identity, loader_identity, override=override)
        trainer.restore_runtime_state(checkpoint_payload)
        resume_step = trainer.phase_step
        consumed_batches = trainer.consumed_microbatches
        try:
            assert_phase_global_consistent(
                phase=str(checkpoint_payload.get("phase") or phase),
                global_step=trainer.global_step,
                phase_step=trainer.phase_step,
                ssl_steps=int(cfg.train.ssl_steps),
                broad_steps=int(cfg.train.broad_steps),
            )
        except LoaderCursorError as exc:
            raise ResumeIncompatibilityError(str(exc)) from exc
        print(
            f"resumed from {resume} phase_step={resume_step} "
            f"global={trainer.global_step} consumed={consumed_batches}"
        )

    try:
        train_stream = wrap_frozen_dataloader(
            loader,
            consumed_batches=consumed_batches,
            checkpoint_payload=checkpoint_payload,
        )
    except LoaderCursorError as exc:
        raise SystemExit(f"resume stream refused: {exc}") from exc

    trainer.loader_epoch_batches = int(train_stream.epoch_batches)

    print(f"starting phase={phase} mode={mode} population={population}")
    # Cursor seek lives in train_stream; trainer must not replay batches.
    logs = trainer.train_phase(
        phase,
        train_stream,
        max_steps=max_steps,
        resume_step=resume_step,
        skip_microbatches=0,
    )
    summary = {
        "phase": phase,
        "steps": len(logs),
        "final_loss": logs[-1].loss if logs else None,
        "git_sha": _git_sha(),
        "config_hash": cfg.config_hash(),
        "contract": cfg.contract,
        "consumed_microbatches": trainer.consumed_microbatches,
        "durable_uri": trainer.last_durable_uri,
        "loader_identity": loader_identity,
    }
    summary_path = ckpt_dir / "phase_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if durable is not None:
        durable.upload_file(summary_path, "phase_summary.json")
        latest = ckpt_dir / "latest.pt"
        if latest.exists():
            durable.upload_file(latest, "latest.pt")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
