"""Bonding-curve / reserve state sequence encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from trencher.world_model.config import SequenceEncoderConfig
from trencher.world_model.encoders._common import SequenceBackbone, last_valid_token


@dataclass
class CurveEncoderOutput:
    hidden: torch.Tensor
    pooled: torch.Tensor


class CurveStateEncoder(nn.Module):
    """Encode causal curve/reserve snapshots (price, v_sol, v_tok, curve_pct, …)."""

    def __init__(self, cfg: SequenceEncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SequenceEncoderConfig(n_channels=7)
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

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> CurveEncoderOutput:
        h = self.backbone(x, times=times, mask=mask)
        return CurveEncoderOutput(hidden=h, pooled=last_valid_token(h, mask))
