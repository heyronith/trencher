"""Ownership / transfer-event encoder with quality & availability."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from trencher.world_model.config import SequenceEncoderConfig
from trencher.world_model.encoders._common import SequenceBackbone, last_valid_token


@dataclass
class OwnershipEncoderOutput:
    hidden: torch.Tensor
    pooled: torch.Tensor
    available: torch.Tensor
    quality: torch.Tensor


class OwnershipEncoder(nn.Module):
    """Event-local ownership evidence (no invented holder-state embeddings)."""

    def __init__(self, cfg: SequenceEncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SequenceEncoderConfig(n_channels=9)
        c = self.cfg
        self.backbone = SequenceBackbone(
            n_channels=c.n_channels,
            d_model=c.d_model,
            n_layers=c.n_layers,
            n_heads=c.n_heads,
            ffn_mult=c.ffn_mult,
            dropout=c.dropout,
            n_time_freqs=16,
            use_time=True,
        )
        self.quality_proj = nn.Linear(1, c.d_model)
        self.null_vec = nn.Parameter(torch.zeros(c.d_model))

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None = None,
        available: torch.Tensor | None = None,
        quality: torch.Tensor | None = None,
    ) -> OwnershipEncoderOutput:
        """
        Args:
            available: [B]
            quality: [B] in [0, 1] source-quality / evidence strength
        """
        h = self.backbone(x, times=times, mask=mask)
        pooled = last_valid_token(h, mask)
        b = x.size(0)
        if available is None:
            avail = torch.ones(b, device=x.device, dtype=pooled.dtype)
        else:
            avail = available.to(dtype=pooled.dtype).view(-1)
        qual = (
            avail
            if quality is None
            else quality.to(dtype=pooled.dtype).view(-1).clamp(0.0, 1.0)
        )
        pooled = pooled + self.quality_proj(qual.unsqueeze(-1))
        pooled = pooled * avail.unsqueeze(-1) + self.null_vec * (1.0 - avail).unsqueeze(-1)
        return OwnershipEncoderOutput(
            hidden=h, pooled=pooled, available=avail, quality=qual
        )
