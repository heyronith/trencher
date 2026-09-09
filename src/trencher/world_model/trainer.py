"""Production three-phase trainer with real telemetry, health gates, and resume."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from trencher.world_model.checkpoint import save_checkpoint
from trencher.world_model.config import WorldModelConfig
from trencher.world_model.durable import DurableStore, DurableSyncError, write_progress_sidecar
from trencher.world_model.input_sanitize import assert_finite_model_inputs
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
    event_tokens_per_sec: float = 0.0
    graph_edges_per_sec: float = 0.0
    data_wait_sec: float = 0.0
    h2d_sec: float = 0.0
    forward_sec: float = 0.0
    backward_sec: float = 0.0
    optimizer_sec: float = 0.0
    peak_mem_mb: float = 0.0
    allocated_mem_mb: float = 0.0
    reserved_mem_mb: float = 0.0
    amp_dtype: str = "fp32"
    batch_event_len_mean: float = 0.0
    healthy: bool = True
    health_message: str = ""


def _cuda_sync_if(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


class ResumeIncompatibilityError(RuntimeError):
    """Raised when checkpoint loader identity does not match the current launch."""


class ThreePhaseTrainer:
    """Production trainer for ssl → broad → rich with BF16, accum, health gates."""

    def __init__(
        self,
        model: WorldModel,
        cfg: WorldModelConfig | None = None,
        *,
        device: str | torch.device | None = None,
        progress: ProgressEmitter | None = None,
        ckpt_dir: Path | None = None,
        freeze_ids: dict[str, str] | None = None,
        git_sha: str | None = None,
        class_weights: dict[str, torch.Tensor] | None = None,
        durable_store: DurableStore | None = None,
        loader_identity: dict[str, Any] | None = None,
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
        planned = (
            int(self.cfg.train.ssl_steps)
            + int(self.cfg.train.broad_steps)
            + int(self.cfg.train.rich_steps)
        )
        self.planned_total_steps = max(planned, 1)
        self.phase_weights = {
            "ssl": self.cfg.train.ssl_steps / self.planned_total_steps,
            "broad": self.cfg.train.broad_steps / self.planned_total_steps,
            "rich": self.cfg.train.rich_steps / self.planned_total_steps,
        }
        self.progress = progress or ProgressEmitter(
            "world_model_train", phase_weights=self.phase_weights
        )
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.global_step = 0
        self.phase_step = 0
        self.epoch = 0
        self.microstep = 0
        self.consumed_microbatches = 0
        self.telemetry: list[StepTelemetry] = []
        self.freeze_ids = freeze_ids or {
            "freeze_id": self.cfg.freeze_id,
            "loader_contract": self.cfg.contract,
        }
        self.git_sha = git_sha
        self.class_weights = class_weights or {}
        self.use_bf16 = bool(self.cfg.train.bf16) and self.device.type == "cuda"
        self._scheduler: LambdaLR | None = None
        self.durable_store = durable_store
        self.loader_identity = dict(loader_identity or {})
        self.last_durable_uri: str | None = None
        self._build_scheduler()
        if bool(getattr(self.cfg.train, "compile", False)) and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[assignment]

    def _build_scheduler(self) -> None:
        warm = max(int(self.cfg.train.warmup_steps), 0)
        total = self.planned_total_steps

        def lr_lambda(step: int) -> float:
            if warm > 0 and step < warm:
                return float(step + 1) / float(warm)
            progress = (step - warm) / max(total - warm, 1)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        self._scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _mem_stats(self) -> tuple[float, float, float]:
        if self.device.type != "cuda":
            return 0.0, 0.0, 0.0
        alloc = torch.cuda.memory_allocated(self.device) / (1024**2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
        peak = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        return alloc, reserved, peak

    def set_encoder_trainable(self, name: str, trainable: bool) -> None:
        mod = getattr(self.model, name, None)
        if mod is None:
            raise KeyError(name)
        for p in mod.parameters():
            p.requires_grad = bool(trainable)

    def _move_batch(
        self, batch: dict[str, Any], *, profile: bool
    ) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        non_blocking = self.device.type == "cuda"
        out = {
            k: (v.to(self.device, non_blocking=non_blocking) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        _cuda_sync_if(self.device, profile)
        return out, time.perf_counter() - t0

    def _health_check(
        self, loss: torch.Tensor, grad_norm: float, data_wait: float
    ) -> tuple[bool, str]:
        if not torch.isfinite(loss):
            return False, "nonfinite_loss"
        if not math.isfinite(grad_norm) or grad_norm == 0.0:
            return False, "invalid_or_zero_grad_norm"
        if data_wait > 30.0:
            return False, "data_starvation"
        return True, "ok"

    def restore_runtime_state(self, payload: dict[str, Any]) -> None:
        """Restore counters from a checkpoint payload (model/optim/sched already loaded)."""
        tel = payload.get("telemetry") or {}
        extra = payload.get("extra") or {}
        self.global_step = int(payload.get("step", 0))
        self.phase_step = int(tel.get("phase_step", 0))
        self.epoch = int(tel.get("epoch", 0))
        self.microstep = int(tel.get("microstep", 0))
        self.consumed_microbatches = int(
            tel.get("consumed_microbatches", extra.get("consumed_microbatches", 0))
        )

    def train_phase(
        self,
        phase: str,
        batches: Iterable[dict[str, Any]],
        *,
        max_steps: int | None = None,
        resume_step: int | None = None,
        skip_microbatches: int | None = None,
    ) -> list[StepTelemetry]:
        phase = assert_allowed_phase(phase)
        if phase == "ssl":
            planned = max_steps if max_steps is not None else self.cfg.train.ssl_steps
        elif phase == "broad":
            planned = max_steps if max_steps is not None else self.cfg.train.broad_steps
        else:
            planned = max_steps if max_steps is not None else self.cfg.train.rich_steps
        planned = max(int(planned), 1)
        profile_steps = int(getattr(self.cfg.train, "profile_steps", 200) or 0)

        self.model.train()
        logs: list[StepTelemetry] = []
        it: Iterator[dict[str, Any]] = iter(batches)
        accum = max(int(self.cfg.train.accum_steps), 1)
        if resume_step is not None:
            self.phase_step = int(resume_step)
        skip = (
            int(skip_microbatches)
            if skip_microbatches is not None
            else int(self.consumed_microbatches)
        )
        for _ in range(max(skip, 0)):
            try:
                next(it)
            except StopIteration as exc:
                raise RuntimeError(
                    f"resume skip_microbatches={skip} exhausted dataloader before training"
                ) from exc
        # Cursor after skip becomes the baseline for newly consumed batches this run.
        # consumed_microbatches already accounts for skipped batches when restored.
        exhausted = False

        while self.phase_step < planned:
            profile = self.phase_step < profile_steps
            t_wait0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                exhausted = True
                break
            data_wait = time.perf_counter() - t_wait0
            self.consumed_microbatches += 1

            batch, h2d = self._move_batch(batch, profile=profile)
            for k, w in self.class_weights.items():
                batch[k] = w.to(self.device)
            assert_finite_model_inputs(batch)

            if self.microstep == 0:
                self.optimizer.zero_grad(set_to_none=self.cfg.train.set_to_none)

            _cuda_sync_if(self.device, profile)
            t_fwd = time.perf_counter()
            with self._autocast():
                outputs = self.model(batch, phase=phase)
                breakdown = self.loss_fn(outputs, batch, phase=phase)
                loss = breakdown.total / accum
            if not loss.requires_grad:
                # Empty-support / degenerate batches: keep optimizer step well-defined.
                for p in self.model.parameters():
                    if p.requires_grad:
                        loss = loss + p.sum() * 0.0
                        break
            if bool(self.cfg.train.refuse_nan_loss) and not torch.isfinite(breakdown.total):
                raise RuntimeError(
                    f"non-finite loss at global_step={self.global_step} phase={phase} "
                    f"terms={{{', '.join(f'{k}={float(v)}' for k, v in breakdown.terms.items())}}}"
                )
            _cuda_sync_if(self.device, profile)
            fwd = time.perf_counter() - t_fwd

            t_bwd = time.perf_counter()
            loss.backward()
            _cuda_sync_if(self.device, profile)
            bwd = time.perf_counter() - t_bwd

            self.microstep += 1
            if self.microstep < accum:
                continue
            self.microstep = 0

            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.train.max_grad_norm
                )
            )
            healthy, health_msg = self._health_check(
                breakdown.total, grad_norm, data_wait
            )
            health_gate_step = self.phase_step + 1
            if (
                not healthy
                and self.cfg.train.health_gates_enabled
                and health_gate_step >= max(profile_steps // 4, 1)
            ):
                raise RuntimeError(f"health gate failed: {health_msg}")

            t_opt = time.perf_counter()
            self.optimizer.step()
            if self._scheduler is not None:
                self._scheduler.step()
            _cuda_sync_if(self.device, profile)
            opt = time.perf_counter() - t_opt

            self.phase_step += 1
            self.global_step += 1
            bsz = int(batch.get("sample_mask", torch.ones(1)).shape[0]) if batch else 1
            event_tokens = (
                int(batch["event_mask"].sum().item()) if "event_mask" in batch else 0
            )
            graph_edges = (
                int(batch["graph_mask"].sum().item()) if "graph_mask" in batch else 0
            )
            step_time = max(fwd + bwd + opt + h2d, 1e-9)
            alloc, reserved, peak = self._mem_stats()
            tel = StepTelemetry(
                step=self.global_step,
                phase=phase,
                loss=float(breakdown.total.detach().cpu()),
                terms={k: float(v.detach().cpu()) for k, v in breakdown.terms.items()},
                lr=float(self.optimizer.param_groups[0]["lr"]),
                grad_norm=grad_norm,
                samples_per_sec=bsz / step_time,
                event_tokens_per_sec=event_tokens / step_time,
                graph_edges_per_sec=graph_edges / step_time,
                data_wait_sec=data_wait,
                h2d_sec=h2d,
                forward_sec=fwd,
                backward_sec=bwd,
                optimizer_sec=opt,
                peak_mem_mb=peak,
                allocated_mem_mb=alloc,
                reserved_mem_mb=reserved,
                amp_dtype="bf16" if self.use_bf16 else "fp32",
                batch_event_len_mean=(
                    float(batch["event_mask"].float().sum(dim=-1).mean().item())
                    if "event_mask" in batch
                    else 0.0
                ),
                healthy=healthy,
                health_message=health_msg,
            )
            logs.append(tel)
            self.telemetry.append(tel)
            self.progress.phase_progress(
                phase,
                self.phase_step,
                planned,
                message=f"loss={tel.loss:.4f} ex/s={tel.samples_per_sec:.1f}",
                global_step=self.global_step,
                planned_total=self.planned_total_steps,
                lr=tel.lr,
                grad_norm=tel.grad_norm,
                force=True,
            )

            if self.ckpt_dir and self.global_step % max(self.cfg.train.ckpt_every, 1) == 0:
                self._save("periodic", phase=phase)

        if exhausted and self.phase_step < planned:
            raise RuntimeError(
                f"training data exhausted at phase_step={self.phase_step}/{planned} "
                f"for phase={phase}; refuse silent 100% completion"
            )

        if self.ckpt_dir:
            self._save("latest", phase=phase)
        return logs

    def _autocast(self):
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        from contextlib import nullcontext

        return nullcontext()

    def _save(self, tag: str, *, phase: str) -> Path | None:
        if self.ckpt_dir is None:
            return None
        path = self.ckpt_dir / f"{tag}.pt"
        next_oid = None
        if self.telemetry:
            # Best-effort: last batch id not retained; cursor is consumed_microbatches.
            next_oid = None
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self._scheduler,
            step=self.global_step,
            phase=phase,
            config=self.cfg,
            telemetry={
                "phase_step": self.phase_step,
                "epoch": self.epoch,
                "microstep": self.microstep,
                "planned_total_steps": self.planned_total_steps,
                "consumed_microbatches": self.consumed_microbatches,
                "contract": self.cfg.contract,
            },
            extra={
                "git_sha": self.git_sha,
                "freeze_ids": self.freeze_ids,
                "class_weights_keys": list(self.class_weights),
                "checkpoint_format_version": 3,
                "loader_identity": self.loader_identity,
                "consumed_microbatches": self.consumed_microbatches,
                "contract": self.cfg.contract,
                "next_observation_id_hash": next_oid,
            },
        )
        progress_payload = {
            "global_step": self.global_step,
            "phase_step": self.phase_step,
            "phase": phase,
            "consumed_microbatches": self.consumed_microbatches,
            "contract": self.cfg.contract,
            "git_sha": self.git_sha,
            "loader_identity": self.loader_identity,
            "checkpoint": str(path),
            "durable_uri": None,
            "resumable": True,
        }
        if self.durable_store is not None:
            try:
                uri = self.durable_store.upload_file(path, f"{tag}.pt")
                progress_payload["durable_uri"] = uri
                self.last_durable_uri = uri
                if self.ckpt_dir:
                    prog_path = self.ckpt_dir / "progress.json"
                    write_progress_sidecar(prog_path, progress_payload)
                    self.durable_store.upload_file(prog_path, "progress.json")
                    meta_path = path.with_suffix(path.suffix + ".meta.json")
                    if meta_path.exists():
                        self.durable_store.upload_file(meta_path, f"{tag}.pt.meta.json")
            except DurableSyncError:
                progress_payload["resumable"] = False
                progress_payload["durable_error"] = "upload_failed"
                if self.ckpt_dir:
                    write_progress_sidecar(self.ckpt_dir / "progress.json", progress_payload)
                raise
        elif self.ckpt_dir is not None:
            write_progress_sidecar(self.ckpt_dir / "progress.json", progress_payload)
        return path
