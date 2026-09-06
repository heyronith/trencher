"""Atomic checkpoint save / load / resume with RNG + config-hash guards."""

from __future__ import annotations

import contextlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trencher.world_model.config import WorldModelConfig, world_model_config_from_dict


class ConfigMismatchError(RuntimeError):
    """Raised when checkpoint config hash does not match the requested config."""


@dataclass
class CheckpointMeta:
    step: int
    phase: str
    config_hash: str
    config: dict[str, Any]


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    step: int = 0,
    phase: str = "broad",
    config: WorldModelConfig | None = None,
    telemetry: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomically write a resumable checkpoint."""
    path = Path(path)
    cfg = config
    if cfg is None and hasattr(model, "cfg"):
        cfg = model.cfg  # type: ignore[assignment]
    if cfg is None:
        raise ValueError("config required for checkpointing")
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "step": int(step),
        "phase": str(phase),
        "config": cfg.to_dict(),
        "config_hash": cfg.config_hash(),
        "rng": capture_rng_state(),
        "telemetry": telemetry or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()
    _atomic_torch_save(payload, path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "step": payload["step"],
                "phase": payload["phase"],
                "config_hash": payload["config_hash"],
                "path": str(path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    expected_config: WorldModelConfig | None = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load checkpoint; refuse config-hash mismatch when ``expected_config`` given."""
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    ckpt_hash = payload.get("config_hash")
    if expected_config is not None:
        want = expected_config.config_hash()
        if ckpt_hash != want:
            raise ConfigMismatchError(
                f"checkpoint config hash mismatch: ckpt={ckpt_hash} expected={want}"
            )
        # Also refuse if serialized config diverges in name/contract
        ckpt_cfg = payload.get("config") or {}
        if ckpt_cfg.get("name") and ckpt_cfg["name"] != expected_config.name:
            raise ConfigMismatchError(
                f"checkpoint name mismatch: {ckpt_cfg.get('name')} vs {expected_config.name}"
            )

    if model is not None and "model" in payload:
        model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng and "rng" in payload:
        restore_rng_state(payload["rng"])
    return payload


def resume_from_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    expected_config: WorldModelConfig | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointMeta:
    payload = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expected_config=expected_config,
        map_location=map_location,
    )
    cfg_dict = payload.get("config") or {}
    return CheckpointMeta(
        step=int(payload.get("step", 0)),
        phase=str(payload.get("phase", "broad")),
        config_hash=str(payload.get("config_hash", "")),
        config=cfg_dict if isinstance(cfg_dict, dict) else {},
    )


def config_from_checkpoint(path: str | Path) -> WorldModelConfig:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return world_model_config_from_dict(payload["config"])
