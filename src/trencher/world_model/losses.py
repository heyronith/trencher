"""Multi-task losses with masks, censoring, pinball, and BCE."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from trencher.world_model.config import LossConfig, WorldModelConfig


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    m = mask.to(dtype=x.dtype)
    while m.ndim < x.ndim:
        m = m.unsqueeze(-1)
    m = m.expand_as(x)
    denom = m.sum().clamp_min(1.0)
    return (x * m).sum() / denom


def binary_ce_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits, target.to(dtype=logits.dtype), reduction="none"
    )
    m = _broadcast_mask(loss, mask, sample_mask)
    return masked_mean(loss, m)


def _broadcast_mask(
    ref: torch.Tensor,
    mask: torch.Tensor | None,
    sample_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Combine optional target mask + sample mask to ref's shape."""
    out: torch.Tensor | None = None
    if mask is not None:
        m = mask.to(dtype=ref.dtype)
        while m.ndim < ref.ndim:
            m = m.unsqueeze(-1)
        out = m.expand_as(ref)
    if sample_mask is not None:
        s = sample_mask.to(dtype=ref.dtype).view(-1, *([1] * (ref.ndim - 1)))
        s = s.expand_as(ref)
        out = s if out is None else out * s
    return out


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    taus: torch.Tensor,
    mask: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantile / pinball loss. pred [B, Q], target [B] or [B, Q], taus [Q]."""
    err = target.unsqueeze(-1) - pred if target.ndim == 1 else target - pred
    t = taus.view(1, -1).to(dtype=pred.dtype)
    loss = torch.maximum(t * err, (t - 1.0) * err)
    return masked_mean(loss, _broadcast_mask(loss, mask, sample_mask))


def censor_aware_survival_loss(
    hazard_logits: torch.Tensor,
    event: torch.Tensor,
    time_idx: torch.Tensor,
    mask: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Discrete-time survival with right-censoring.

    Args:
        hazard_logits: [B, H] per-bin logits
        event: [B] 1 if event observed, 0 if censored
        time_idx: [B] long — event/censor bin index in [0, H)
        mask: [B] sample mask
    """
    hazards = torch.sigmoid(hazard_logits).clamp(1e-6, 1.0 - 1e-6)
    b, h = hazards.shape
    device = hazards.device
    arange = torch.arange(h, device=device).view(1, -1)
    t = time_idx.view(-1, 1).clamp(0, h - 1)
    surv_mask = arange < t
    log_surv = torch.log1p(-hazards).masked_fill(~surv_mask, 0.0).sum(dim=-1)
    gather_h = hazards.gather(1, t).squeeze(1)
    log_event = torch.log(gather_h)
    nll = -(log_surv + event.to(dtype=hazards.dtype) * log_event)
    return masked_mean(nll, _broadcast_mask(nll, mask, sample_mask))


@dataclass
class LossBreakdown:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]


class MultiTaskLoss(nn.Module):
    """Combine SSL + supervised heads with per-term masks."""

    def __init__(self, cfg: WorldModelConfig | LossConfig | None = None) -> None:
        super().__init__()
        if isinstance(cfg, WorldModelConfig):
            self.cfg = cfg.loss
            self.quantile_taus = torch.tensor(cfg.heads.quantile_taus, dtype=torch.float32)
        elif isinstance(cfg, LossConfig):
            self.cfg = cfg
            self.quantile_taus = torch.tensor((0.1, 0.25, 0.5, 0.75, 0.9), dtype=torch.float32)
        else:
            self.cfg = LossConfig()
            self.quantile_taus = torch.tensor((0.1, 0.25, 0.5, 0.75, 0.9), dtype=torch.float32)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *,
        phase: str = "broad",
    ) -> LossBreakdown:
        c = self.cfg
        terms: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=_device_of(outputs, batch))

        if phase == "ssl" or "ssl_next_side_logits" in outputs:
            side_logits = outputs.get("ssl_next_side_logits")
            if side_logits is not None and "ssl_next_side" in batch:
                tgt = batch["ssl_next_side"].long()
                m = batch.get("ssl_mask")
                loss = F.cross_entropy(
                    side_logits.reshape(-1, side_logits.size(-1)),
                    tgt.reshape(-1),
                    reduction="none",
                )
                loss = (
                    masked_mean(loss.view_as(tgt), m) if m is not None else loss.mean()
                )
                terms["ssl_side"] = loss
                total = total + c.ssl_side_weight * loss

            if "ssl_next_log_dt" in outputs and "ssl_next_log_dt" in batch:
                pred = outputs["ssl_next_log_dt"]
                tgt = batch["ssl_next_log_dt"]
                m = batch.get("ssl_mask")
                loss = masked_mean(F.smooth_l1_loss(pred, tgt, reduction="none"), m)
                terms["ssl_log_dt"] = loss
                total = total + c.ssl_log_dt_weight * loss

            if "ssl_next_size" in outputs and "ssl_next_size" in batch:
                pred = outputs["ssl_next_size"]
                tgt = batch["ssl_next_size"]
                m = batch.get("ssl_mask")
                loss = masked_mean(F.smooth_l1_loss(pred, tgt, reduction="none"), m)
                terms["ssl_size"] = loss
                total = total + c.ssl_size_weight * loss

            if phase == "ssl":
                return LossBreakdown(total=total, terms=terms)

        sample_mask = batch.get("sample_mask")

        if "explosion_logits" in outputs and "explosion_target" in batch:
            loss = binary_ce_logits(
                outputs["explosion_logits"],
                batch["explosion_target"],
                batch.get("explosion_mask"),
                sample_mask,
            )
            terms["explosion"] = loss
            total = total + c.explosion_weight * loss

        if "survival_hazard" in outputs and "survival_event" in batch:
            loss = censor_aware_survival_loss(
                outputs["survival_hazard"],
                batch["survival_event"],
                batch["survival_time_idx"].long(),
                batch.get("survival_mask"),
                sample_mask,
            )
            terms["survival"] = loss
            total = total + c.survival_weight * loss

        if "barrier_logits" in outputs and "barrier_target" in batch:
            loss = binary_ce_logits(
                outputs["barrier_logits"],
                batch["barrier_target"],
                batch.get("barrier_mask"),
                sample_mask,
            )
            terms["barrier"] = loss
            total = total + c.barrier_weight * loss

        if "quantile_returns" in outputs and "return_target" in batch:
            taus = self.quantile_taus.to(device=outputs["quantile_returns"].device)
            loss = pinball_loss(
                outputs["quantile_returns"],
                batch["return_target"],
                taus,
                batch.get("return_mask"),
                sample_mask,
            )
            terms["quantile"] = loss
            total = total + c.quantile_weight * loss

        if "mfe" in outputs and "mfe_target" in batch:
            loss_mfe = masked_mean(
                F.smooth_l1_loss(outputs["mfe"], batch["mfe_target"], reduction="none"),
                _broadcast_mask(outputs["mfe"], batch.get("mfe_mask"), sample_mask),
            )
            loss_mae = masked_mean(
                F.smooth_l1_loss(outputs["mae"], batch["mae_target"], reduction="none"),
                _broadcast_mask(outputs["mae"], batch.get("mae_mask"), sample_mask),
            )
            loss = loss_mfe + loss_mae
            terms["mfe_mae"] = loss
            total = total + c.mfe_mae_weight * loss

        if "persistence_logits" in outputs and "persistence_target" in batch:
            logits = outputs["persistence_logits"]
            tgt = batch["persistence_target"].long()
            loss = F.cross_entropy(logits, tgt, reduction="none")
            loss = masked_mean(
                loss, _broadcast_mask(loss, batch.get("persistence_mask"), sample_mask)
            )
            terms["persistence"] = loss
            total = total + c.persistence_weight * loss

        if "path_pred" in outputs and "path_target" in batch:
            loss = F.smooth_l1_loss(outputs["path_pred"], batch["path_target"], reduction="none")
            loss = masked_mean(
                loss, _broadcast_mask(loss, batch.get("path_mask"), sample_mask)
            )
            terms["path"] = loss
            total = total + c.path_weight * loss

        if "execution" in outputs and "execution_target" in batch:
            loss = F.smooth_l1_loss(
                outputs["execution"], batch["execution_target"], reduction="none"
            )
            loss = masked_mean(
                loss, _broadcast_mask(loss, batch.get("execution_mask"), sample_mask)
            )
            terms["execution"] = loss
            total = total + c.execution_weight * loss

        if not terms:
            total = total + sum(v.sum() * 0.0 for v in outputs.values() if torch.is_tensor(v))

        return LossBreakdown(total=total, terms=terms)


def _device_of(outputs: dict, batch: dict) -> torch.device:
    for d in (outputs, batch):
        for v in d.values():
            if torch.is_tensor(v):
                return v.device
    return torch.device("cpu")
