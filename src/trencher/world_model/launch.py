"""CLI entry: run a real frozen-data training phase (Stage A/B/C).

Invoked by ``scripts/colab/launch_training.sh``. Never unlocks permanent TEST.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import torch

from trencher.world_model.checkpoint import load_checkpoint
from trencher.world_model.config import load_world_model_config
from trencher.world_model.model import WorldModel, count_parameters
from trencher.world_model.progress import ProgressEmitter
from trencher.world_model.trainer import ThreePhaseTrainer
from trencher.world_model.v4_loader import assert_dataset_mode, make_frozen_dataloader


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


def main() -> None:
    config_path = Path(
        os.environ.get("TRENCHER_WM_CONFIG")
        or "configs/world_model/world_model_candidate_v2.yaml"
    )
    world_root = Path(os.environ.get("TRENCHER_WORLD_ROOT", "data/external/world_model"))
    split_root = Path(
        os.environ.get("TRENCHER_SPLIT_ROOT", world_root / "splits" / "split_v1")
    )
    phase = os.environ.get("TRENCHER_WM_PHASE", "ssl")
    device = os.environ.get("TRENCHER_WM_DEVICE", "cuda")
    mode = assert_dataset_mode(os.environ.get("TRENCHER_WM_MODE", "final_train"))
    population = os.environ.get("TRENCHER_WM_POPULATION", "broad")
    ckpt_dir = Path(os.environ.get("TRENCHER_WM_CKPT_DIR", "checkpoints"))
    resume = os.environ.get("TRENCHER_WM_RESUME") or ""
    max_steps_env = os.environ.get("TRENCHER_WM_MAX_STEPS") or ""
    max_steps = int(max_steps_env) if max_steps_env.strip() else None

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested but unavailable; refusing silent CPU fallback for production launch."
        )

    # Accelerator sanity
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"accelerator={props.name} total_mem_gb={props.total_memory / 1e9:.1f}")

    cfg = load_world_model_config(config_path)
    model = WorldModel(cfg)
    print(f"parameters={count_parameters(model):,}")
    print(f"config_hash={cfg.config_hash()}")
    print(f"git_sha={_git_sha()}")

    loader = make_frozen_dataloader(
        world_root=world_root,
        split_root=split_root,
        mode=mode,
        population=population if phase != "ssl" else "broad",
        batch_size=int(cfg.train.batch_size),
        num_workers=int(os.environ.get("TRENCHER_WM_NUM_WORKERS", "2")),
        seed=int(cfg.train.seed),
        shuffle=True,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
        prefetch_factor=2,
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
    trainer = ThreePhaseTrainer(
        model,
        cfg,
        device=device,
        progress=progress,
        ckpt_dir=ckpt_dir,
        git_sha=_git_sha(),
        freeze_ids={
            "freeze_id": cfg.freeze_id,
            "contract": cfg.contract,
            "split": "split_v1",
            "sequence_dataset": "sequence_dataset_v2",
            "targets": "target_dataset_v3/broad_targets_v3",
        },
    )

    resume_step = 0
    if resume:
        payload = load_checkpoint(
            resume, model=model, optimizer=trainer.optimizer, expected_config=cfg
        )
        resume_step = int((payload.get("telemetry") or {}).get("phase_step", 0))
        trainer.global_step = int(payload.get("step", 0))
        print(f"resumed from {resume} phase_step={resume_step} global={trainer.global_step}")

    print(f"starting phase={phase} mode={mode} population={population}")
    logs = trainer.train_phase(
        phase, loader, max_steps=max_steps, resume_step=resume_step
    )
    summary = {
        "phase": phase,
        "steps": len(logs),
        "final_loss": logs[-1].loss if logs else None,
        "git_sha": _git_sha(),
        "config_hash": cfg.config_hash(),
    }
    (ckpt_dir / "phase_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
