"""Multi-task losses with masks, censoring, pinball, and Student-t NLL (V4)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from trencher.world_model.config import LossConfig, WorldModelConfig


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    # Always preserve autograd connectivity when ``x`` requires grad.
    if x.numel() == 0:
        return x.sum() * 0.0
    if mask is None:
        return x.mean()
    m = mask.to(dtype=x.dtype)
    while m.ndim < x.ndim:
        m = m.unsqueeze(-1)
    m = m.expand_as(x)
    denom = m.sum()
    if float(denom.detach().item()) <= 0.0:
        return (x * 0.0).sum()
    return (x * m).sum() / denom.clamp_min(1.0)


def support_count(mask: torch.Tensor | None, ref: torch.Tensor) -> int:
    if mask is None:
        return int(ref.numel())
    m = mask.to(dtype=ref.dtype)
    while m.ndim < ref.ndim:
        m = m.unsqueeze(-1)
    return int(m.expand_as(ref).sum().item())


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


def binary_ce_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
    *,
    pos_weight: torch.Tensor | float | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    tgt = target.to(dtype=logits.dtype)
    if pos_weight is not None:
        pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
        # Expand scalar / [H] pos_weight to logits shape
        while pw.ndim < logits.ndim:
            pw = pw.unsqueeze(0)
        weight = torch.where(tgt > 0.5, pw.expand_as(logits), torch.ones_like(logits))
    else:
        weight = None
    loss = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none", weight=weight)
    if focal_gamma and focal_gamma > 0.0:
        p = torch.sigmoid(logits)
        p_t = torch.where(tgt > 0.5, p, 1.0 - p)
        loss = loss * (1.0 - p_t).clamp_min(0.0).pow(focal_gamma)
    m = _broadcast_mask(loss, mask, sample_mask)
    return masked_mean(loss, m)


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    taus: torch.Tensor,
    mask: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantile / pinball loss.

    pred: [B, Q] or [B, H, Q]
    target: [B], [B, H], or same rank as pred
    taus: [Q]
    """
    if pred.ndim == 2:
        err = target.unsqueeze(-1) - pred if target.ndim == 1 else target - pred
        t = taus.view(1, -1).to(dtype=pred.dtype, device=pred.device)
    elif pred.ndim == 3:
        err = target.unsqueeze(-1) - pred if target.ndim == 2 else target - pred
        t = taus.view(1, 1, -1).to(dtype=pred.dtype, device=pred.device)
    else:
        raise ValueError(f"pinball pred ndim must be 2 or 3, got {pred.ndim}")
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
        mask: [B] sample / at-risk mask
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


def student_t_nll(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    df: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Student-t negative log-likelihood for path predictions."""
    scale = log_scale.exp().clamp_min(1e-6)
    nu = df.unsqueeze(-1).expand_as(mean).clamp_min(2.0 + 1e-4)
    z = (target - mean) / scale
    # log Γ((ν+1)/2) − log Γ(ν/2) − ½ log(νπ) − log σ − (ν+1)/2 log(1 + z²/ν)
    log_prob = (
        torch.lgamma((nu + 1.0) * 0.5)
        - torch.lgamma(nu * 0.5)
        - 0.5 * (torch.log(nu) + torch.log(torch.tensor(torch.pi, device=mean.device, dtype=mean.dtype)))  # noqa: E501
        - log_scale
        - 0.5 * (nu + 1.0) * torch.log1p((z * z) / nu)
    )
    nll = -log_prob
    return masked_mean(nll, _broadcast_mask(nll, mask, sample_mask))


def masked_ce_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy over last dim with optional mask (censored → ignore)."""
    # logits [..., C], target [...]
    n_classes = logits.size(-1)
    flat_logits = logits.reshape(-1, n_classes)
    flat_tgt = target.reshape(-1).long()
    if mask is not None:
        m = mask.to(dtype=torch.bool)
        while m.ndim < target.ndim:
            m = m.unsqueeze(-1)
        m = m.expand_as(target).reshape(-1)
        flat_tgt = flat_tgt.clone()
        flat_tgt = torch.where(m, flat_tgt, torch.full_like(flat_tgt, ignore_index))
    loss = F.cross_entropy(flat_logits, flat_tgt, reduction="none", ignore_index=ignore_index)
    loss = loss.view_as(target)
    return masked_mean(loss, _broadcast_mask(loss, mask, sample_mask))


@dataclass
class LossBreakdown:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]
    support: dict[str, int] = field(default_factory=dict)


class MultiTaskLoss(nn.Module):
    """Combine SSL + supervised heads with per-term masks (frozen V4)."""

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

    def _add(
        self,
        terms: dict[str, torch.Tensor],
        support: dict[str, int],
        total: torch.Tensor,
        name: str,
        loss: torch.Tensor,
        weight: float,
        n_support: int,
    ) -> torch.Tensor:
        if n_support <= 0:
            # Keep a graph-connected zero so empty-support batches still backward.
            zero = (loss * 0.0).sum() if torch.is_tensor(loss) else loss
            terms[name] = zero
            support[name] = 0
            return total + zero
        if not torch.isfinite(loss).all():
            zero = (loss * 0.0).sum() if torch.is_tensor(loss) else loss
            terms[name] = zero.detach()
            support[name] = 0
            return total
        terms[name] = loss
        support[name] = n_support
        return total + weight * loss

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *,
        phase: str = "broad",
    ) -> LossBreakdown:
        c = self.cfg
        terms: dict[str, torch.Tensor] = {}
        support: dict[str, int] = {}
        total = torch.zeros((), device=_device_of(outputs, batch))

        if phase == "ssl" or "ssl_next_side_logits" in outputs:
            side_logits = outputs.get("ssl_next_side_logits")
            if side_logits is not None and "ssl_next_side" in batch:
                tgt = batch["ssl_next_side"].long()
                m = batch.get("ssl_side_mask", batch.get("ssl_mask"))
                loss = F.cross_entropy(
                    side_logits.reshape(-1, side_logits.size(-1)),
                    tgt.reshape(-1),
                    reduction="none",
                )
                loss_v = masked_mean(loss.view_as(tgt), m) if m is not None else loss.mean()
                total = self._add(
                    terms, support, total, "ssl_side", loss_v, c.ssl_side_weight,
                    support_count(m, tgt),
                )

            if "ssl_next_log_dt" in outputs and "ssl_next_log_dt" in batch:
                pred = outputs["ssl_next_log_dt"]
                tgt = batch["ssl_next_log_dt"]
                m = batch.get("ssl_log_dt_mask", batch.get("ssl_mask"))
                loss = masked_mean(F.smooth_l1_loss(pred, tgt, reduction="none"), m)
                total = self._add(
                    terms, support, total, "ssl_log_dt", loss, c.ssl_log_dt_weight,
                    support_count(m, tgt),
                )

            if "ssl_next_size" in outputs and "ssl_next_size" in batch:
                pred = outputs["ssl_next_size"]
                tgt = batch["ssl_next_size"]
                m = batch.get("ssl_size_mask", batch.get("ssl_mask"))
                loss = masked_mean(F.smooth_l1_loss(pred, tgt, reduction="none"), m)
                total = self._add(
                    terms, support, total, "ssl_size", loss, c.ssl_size_weight,
                    support_count(m, tgt),
                )

            if "ssl_intensity_log" in outputs and "ssl_intensity_delta_t" in batch:
                log_lam = outputs["ssl_intensity_log"].clamp(-20.0, 20.0)
                delta = batch["ssl_intensity_delta_t"].clamp_min(1e-6)
                m = batch.get("ssl_intensity_mask", batch.get("ssl_mask"))
                # Conditional exponential waiting-time NLL: λΔt − log λ
                # with λ = exp(log λ).
                nll = log_lam.exp() * delta - log_lam
                loss = masked_mean(nll, m)
                total = self._add(
                    terms, support, total, "ssl_intensity", loss, c.ssl_intensity_weight,
                    support_count(m, delta),
                )

            if "ssl_reserve_delta" in outputs and "ssl_reserve_delta" in batch:
                pred = outputs["ssl_reserve_delta"]
                tgt = batch["ssl_reserve_delta"]
                m = batch.get("ssl_reserve_mask", batch.get("ssl_mask"))
                loss = masked_mean(F.smooth_l1_loss(pred, tgt, reduction="none"), m)
                total = self._add(
                    terms, support, total, "ssl_reserve", loss, c.ssl_reserve_weight,
                    support_count(m, tgt),
                )

            if "ssl_recon" in outputs and "ssl_recon_target" in batch:
                pred = outputs["ssl_recon"]
                tgt = batch["ssl_recon_target"]
                m = batch.get("ssl_recon_mask", batch.get("ssl_mask"))
                loss = masked_mean(F.smooth_l1_loss(pred, tgt, reduction="none"), m)
                total = self._add(
                    terms, support, total, "ssl_recon", loss, c.ssl_recon_weight,
                    support_count(m, tgt),
                )

            if phase == "ssl":
                return LossBreakdown(total=total, terms=terms, support=support)

        sample_mask = batch.get("sample_mask")

        if "explosion_logits" in outputs and "explosion_target" in batch:
            pos_w = batch.get("explosion_pos_weight", c.explosion_pos_weight)
            if torch.is_tensor(pos_w) and pos_w.numel() == 1:
                pos_w = float(pos_w.item())
            loss = binary_ce_logits(
                outputs["explosion_logits"],
                batch["explosion_target"],
                batch.get("explosion_mask"),
                sample_mask,
                pos_weight=pos_w,
                focal_gamma=c.explosion_focal_gamma,
            )
            total = self._add(
                terms, support, total, "explosion", loss, c.explosion_weight,
                support_count(
                    _broadcast_mask(
                        outputs["explosion_logits"],
                        batch.get("explosion_mask"),
                        sample_mask,
                    ),
                    outputs["explosion_logits"],
                ),
            )

        if "explosion_tte_hazard" in outputs and "explosion_tte_event" in batch:
            loss = censor_aware_survival_loss(
                outputs["explosion_tte_hazard"],
                batch["explosion_tte_event"],
                batch["explosion_tte_time_idx"].long(),
                batch.get("explosion_tte_mask"),
                sample_mask,
            )
            total = self._add(
                terms, support, total, "explosion_tte", loss, c.explosion_tte_weight,
                support_count(batch.get("explosion_tte_mask"), batch["explosion_tte_event"]),
            )

        if "explosion_phase_logits" in outputs and "explosion_phase_target" in batch:
            logits = outputs["explosion_phase_logits"]
            tgt = batch["explosion_phase_target"]
            if tgt.dtype in (torch.long, torch.int64, torch.int32):
                loss = F.cross_entropy(logits, tgt.long(), reduction="none")
                loss = masked_mean(
                    loss, _broadcast_mask(loss, batch.get("explosion_phase_mask"), sample_mask)
                )
            else:
                loss = masked_mean(
                    F.smooth_l1_loss(logits, tgt, reduction="none"),
                    _broadcast_mask(logits, batch.get("explosion_phase_mask"), sample_mask),
                )
            total = self._add(
                terms, support, total, "explosion_phase", loss, c.explosion_phase_weight,
                support_count(batch.get("explosion_phase_mask"), tgt),
            )

        if "graduation_tte_hazard" in outputs and "graduation_tte_event" in batch:
            loss = censor_aware_survival_loss(
                outputs["graduation_tte_hazard"],
                batch["graduation_tte_event"],
                batch["graduation_tte_time_idx"].long(),
                batch.get("graduation_tte_mask"),
                sample_mask,
            )
            total = self._add(
                terms, support, total, "graduation_tte", loss, c.graduation_tte_weight,
                support_count(batch.get("graduation_tte_mask"), batch["graduation_tte_event"]),
            )

        if "survival_horizon_logits" in outputs and "survival_horizon_target" in batch:
            loss = binary_ce_logits(
                outputs["survival_horizon_logits"],
                batch["survival_horizon_target"],
                batch.get("survival_horizon_mask"),
                sample_mask,
            )
            total = self._add(
                terms, support, total, "survival", loss, c.survival_weight,
                support_count(
                    _broadcast_mask(
                        outputs["survival_horizon_logits"],
                        batch.get("survival_horizon_mask"),
                        sample_mask,
                    ),
                    outputs["survival_horizon_logits"],
                ),
            )

        if "survival_tte_hazard" in outputs and "survival_tte_event" in batch:
            loss = censor_aware_survival_loss(
                outputs["survival_tte_hazard"],
                batch["survival_tte_event"],
                batch["survival_tte_time_idx"].long(),
                batch.get("survival_tte_mask"),
                sample_mask,
            )
            total = self._add(
                terms, support, total, "survival_tte", loss, c.survival_tte_weight,
                support_count(batch.get("survival_tte_mask"), batch["survival_tte_event"]),
            )

        if "barrier_logits" in outputs and "barrier_target" in batch:
            logits = outputs["barrier_logits"]  # [B, P, H, C]
            tgt = batch["barrier_target"]  # [B, P, H]
            loss = masked_ce_logits(
                logits, tgt, batch.get("barrier_mask"), sample_mask
            )
            total = self._add(
                terms, support, total, "barrier", loss, c.barrier_weight,
                support_count(batch.get("barrier_mask"), tgt),
            )

        if "return_quantiles" in outputs and "return_target" in batch:
            taus = self.quantile_taus.to(device=outputs["return_quantiles"].device)
            loss = pinball_loss(
                outputs["return_quantiles"],
                batch["return_target"],
                taus,
                batch.get("return_mask"),
                sample_mask,
            )
            total = self._add(
                terms, support, total, "quantile", loss, c.quantile_weight,
                support_count(batch.get("return_mask"), batch["return_target"]),
            )

        if "mfe_quantiles" in outputs and "mfe_target" in batch:
            taus = self.quantile_taus.to(device=outputs["mfe_quantiles"].device)
            loss_mfe = pinball_loss(
                outputs["mfe_quantiles"],
                batch["mfe_target"],
                taus,
                batch.get("mfe_mask"),
                sample_mask,
            )
            loss_mae = pinball_loss(
                outputs["mae_quantiles"],
                batch["mae_target"],
                taus,
                batch.get("mae_mask"),
                sample_mask,
            )
            loss = loss_mfe + loss_mae
            total = self._add(
                terms, support, total, "mfe_mae", loss, c.mfe_mae_weight,
                support_count(batch.get("mfe_mask"), batch["mfe_target"]),
            )

        if "time_to_mfe" in outputs and "time_to_mfe_target" in batch:
            loss_tm = masked_mean(
                F.smooth_l1_loss(
                    outputs["time_to_mfe"], batch["time_to_mfe_target"], reduction="none"
                ),
                _broadcast_mask(
                    outputs["time_to_mfe"], batch.get("time_to_mfe_mask"), sample_mask
                ),
            )
            loss_ta = masked_mean(
                F.smooth_l1_loss(
                    outputs["time_to_mae"], batch["time_to_mae_target"], reduction="none"
                ),
                _broadcast_mask(
                    outputs["time_to_mae"], batch.get("time_to_mae_mask"), sample_mask
                ),
            )
            loss = loss_tm + loss_ta
            total = self._add(
                terms, support, total, "time_to_extreme", loss, c.time_to_extreme_weight,
                support_count(batch.get("time_to_mfe_mask"), batch["time_to_mfe_target"]),
            )

        if "persistence_pred" in outputs and "persistence_target" in batch:
            loss = F.smooth_l1_loss(
                outputs["persistence_pred"], batch["persistence_target"], reduction="none"
            )
            loss = masked_mean(
                loss,
                _broadcast_mask(
                    outputs["persistence_pred"], batch.get("persistence_mask"), sample_mask
                ),
            )
            total = self._add(
                terms, support, total, "persistence", loss, c.persistence_weight,
                support_count(batch.get("persistence_mask"), batch["persistence_target"]),
            )

        if (
            "path_mean" in outputs
            and "path_log_scale" in outputs
            and "path_df" in outputs
            and "path_target" in batch
        ):
            loss = student_t_nll(
                outputs["path_mean"],
                outputs["path_log_scale"],
                outputs["path_df"],
                batch["path_target"],
                batch.get("path_mask"),
                sample_mask,
            )
            total = self._add(
                terms, support, total, "path", loss, c.path_weight,
                support_count(batch.get("path_mask"), batch["path_target"]),
            )
        elif "path_pred" in outputs and "path_target" in batch:
            # Fallback if only means are present
            loss = F.smooth_l1_loss(
                outputs["path_pred"], batch["path_target"], reduction="none"
            )
            loss = masked_mean(
                loss, _broadcast_mask(loss, batch.get("path_mask"), sample_mask)
            )
            total = self._add(
                terms, support, total, "path", loss, c.path_weight,
                support_count(batch.get("path_mask"), batch["path_target"]),
            )

        if "execution" in outputs and "execution_target" in batch:
            loss = F.smooth_l1_loss(
                outputs["execution"], batch["execution_target"], reduction="none"
            )
            loss = masked_mean(
                loss,
                _broadcast_mask(
                    outputs["execution"], batch.get("execution_mask"), sample_mask
                ),
            )
            total = self._add(
                terms, support, total, "execution", loss, c.execution_weight,
                support_count(batch.get("execution_mask"), batch["execution_target"]),
            )

        if not terms:
            total = total + sum(v.sum() * 0.0 for v in outputs.values() if torch.is_tensor(v))

        return LossBreakdown(total=total, terms=terms, support=support)


def _device_of(outputs: dict, batch: dict) -> torch.device:
    for d in (outputs, batch):
        for v in d.values():
            if torch.is_tensor(v):
                return v.device
    return torch.device("cpu")
