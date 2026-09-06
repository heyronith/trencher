"""Global market / network temporal encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from trencher.world_model.config import GlobalTemporalConfig
from trencher.world_model.encoders._common import SequenceBackbone, last_valid_token


@dataclass
class GlobalTemporalOutput:
    hidden: torch.Tensor
    pooled: torch.Tensor


class GlobalTemporalEncoder(nn.Module):
    """Last-N second buckets + daily/global scalars (last_completed_second policy)."""

    def __init__(self, cfg: GlobalTemporalConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GlobalTemporalConfig()
        c = self.cfg
        self.backbone = SequenceBackbone(
            n_channels=c.n_bucket_channels,
            d_model=c.d_model,
            n_layers=c.n_layers,
            n_heads=c.n_heads,
            ffn_mult=c.ffn_mult,
            dropout=c.dropout,
            n_time_freqs=16,
            use_time=True,
        )
        self.scalar_proj = nn.Sequential(
            nn.Linear(c.n_scalars, c.d_model),
            nn.GELU(),
            nn.Linear(c.d_model, c.d_model),
        )
        self.out_norm = nn.LayerNorm(c.d_model)

    def forward(
        self,
        buckets: torch.Tensor,
        times: torch.Tensor,
        scalars: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> GlobalTemporalOutput:
        """
        Args:
            buckets: [B, G, C]
            times: [B, G]
            scalars: [B, S]
            mask: [B, G]
        """
        h = self.backbone(buckets, times=times, mask=mask)
        pooled = last_valid_token(h, mask) + self.scalar_proj(scalars)
        pooled = self.out_norm(pooled)
        return GlobalTemporalOutput(hidden=h, pooled=pooled)
