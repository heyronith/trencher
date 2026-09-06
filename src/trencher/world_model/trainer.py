"""Three-phase trainer: SSL → broad multi-task → rich fine-tune."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW

from trencher.world_model.checkpoint import save_checkpoint
from trencher.world_model.config import WorldModelConfig
from trencher.world_model.losses import MultiTaskLoss
from trencher.world_model.model import WorldModel
from trencher.world_model.progress import ProgressEmitter
from trencher.world_model.split_guard import assert_allowed_phase


@dataclass
class StepTelemetry:
    step: int
    phase: str
    loss: float
    terms: dict[str, float] = field(default_factory=dict)
    lr: float = 0.0
    grad_norm: float = 0.0
    samples_per_sec: float = 0.0
    peak_mem_mb: float = 0.0
    amp_dtype: str = "fp32"


class ThreePhaseTrainer:
    """CPU/GPU trainer with BF16 autocast hooks, clip, set_to_none zero_grad.

    Designed to run tiny CPU smoke steps in unit tests; production uses the
    YAML A100-sized config without changing the control flow.
    """

    def __init__(
        self,
        model: WorldModel,
        cfg: WorldModelConfig | None = None,
        *,
        device: str | torch.device | None = None,
        progress: ProgressEmitter | None = None,
        ckpt_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or model.cfg
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        self.loss_fn = MultiTaskLoss(self.cfg)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.progress = progress or ProgressEmitter("world_model_train")
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.global_step = 0
        self.telemetry: list[StepTelemetry] = []
        self._scaler: torch.cuda.amp.GradScaler | None = None
        # BF16 autocast is a no-op on CPU; kept as a hook for CUDA A100 runs.
        self.use_bf16 = bool(self.cfg.train.bf16) and self.device.type == "cuda"
        if self.use_bf16 and hasattr(torch.cuda.amp, "GradScaler"):
            # BF16 typically does not need GradScaler; keep None.
            self._scaler = None

    def _autocast(self):
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        # Null context
        from contextlib import nullcontext

        return nullcontext()

    def _clip_grads(self) -> float:
        return float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train.max_grad_norm
            )
        )

    def train_phase(
        self,
        phase: str,
        batches: Iterable[dict[str, Any]],
        *,
        max_steps: int | None = None,
    ) -> list[StepTelemetry]:
        phase = assert_allowed_phase(phase)
        if phase == "ssl":
            planned = max_steps if max_steps is not None else self.cfg.train.ssl_steps
        elif phase == "broad":
            planned = max_steps if max_steps is not None else self.cfg.train.broad_steps
        else:
            planned = max_steps if max_steps is not None else self.cfg.train.rich_steps
        planned = max(int(planned), 1)

        self.model.train()
        logs: list[StepTelemetry] = []
        for step_local, batch in enumerate(batches, start=1):
            if step_local > planned:
                break
            t0 = time.time()
            batch = {
                k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            self.optimizer.zero_grad(set_to_none=self.cfg.train.set_to_none)
            with self._autocast():
                outputs = self.model(batch, phase=phase)
                breakdown = self.loss_fn(outputs, batch, phase=phase)
                loss = breakdown.total
            loss.backward()
            grad_norm = self._clip_grads()
            self.optimizer.step()

            self.global_step += 1
            dt = max(time.time() - t0, 1e-9)
            n = int(next(iter(batch.values())).shape[0]) if batch else 0
            tel = StepTelemetry(
                step=self.global_step,
                phase=phase,
                loss=float(loss.detach().cpu()),
                terms={k: float(v.detach().cpu()) for k, v in breakdown.terms.items()},
                lr=float(self.optimizer.param_groups[0]["lr"]),
                grad_norm=grad_norm,
                samples_per_sec=n / dt,
                peak_mem_mb=_peak_mem_mb(self.device),
                amp_dtype="bf16" if self.use_bf16 else "fp32",
            )
            logs.append(tel)
            self.telemetry.append(tel)
            if step_local % max(self.cfg.train.log_every, 1) == 0:
                self.progress.phase_progress(
                    phase,
                    step_local,
                    planned,
                    message=f"loss={tel.loss:.4f}",
                    global_step=tel.step,
                    grad_norm=tel.grad_norm,
                    samples_per_sec=tel.samples_per_sec,
                )
            if (
                self.ckpt_dir is not None
                and self.cfg.train.ckpt_every > 0
                and self.global_step % self.cfg.train.ckpt_every == 0
            ):
                save_checkpoint(
                    self.ckpt_dir / f"ckpt_step{self.global_step}.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    step=self.global_step,
                    phase=phase,
                    config=self.cfg,
                    telemetry={"last_loss": tel.loss},
                )
        self.progress.phase_progress(
            phase, planned, planned, message=f"{phase}_done", force=True
        )
        return logs

    def run_all_phases(
        self,
        *,
        ssl_batches: Iterable[dict[str, Any]],
        broad_batches: Iterable[dict[str, Any]],
        rich_batches: Iterable[dict[str, Any]],
        ssl_steps: int | None = None,
        broad_steps: int | None = None,
        rich_steps: int | None = None,
    ) -> list[StepTelemetry]:
        out: list[StepTelemetry] = []
        out.extend(self.train_phase("ssl", ssl_batches, max_steps=ssl_steps))
        out.extend(self.train_phase("broad", broad_batches, max_steps=broad_steps))
        out.extend(self.train_phase("rich", rich_batches, max_steps=rich_steps))
        self.progress.complete(message="three_phase_complete")
        return out


def _peak_mem_mb(device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    return 0.0
