"""Wallet behavioral prior encoder — no wallet/mint/creator ID tables."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from trencher.world_model.config import SequenceEncoderConfig
from trencher.world_model.encoders._common import SequenceBackbone, last_valid_token


@dataclass
class WalletPriorEncoderOutput:
    hidden: torch.Tensor
    pooled: torch.Tensor
    available: torch.Tensor


class WalletPriorEncoder(nn.Module):
    """Encode focal-wallet behavioral edge sequences (features only, no ID embeds).

    Node indices may appear upstream for graph joining, but this module never
    allocates a learned mint/wallet/creator embedding table.
    """

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
        self.null_vec = nn.Parameter(torch.zeros(c.d_model))
        # Explicit guard attribute for audits / unit tests
        self.forbid_id_embedding_tables = True

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None = None,
        available: torch.Tensor | None = None,
    ) -> WalletPriorEncoderOutput:
        h = self.backbone(x, times=times, mask=mask)
        pooled = last_valid_token(h, mask)
        if available is None:
            avail = torch.ones(x.size(0), device=x.device, dtype=pooled.dtype)
        else:
            avail = available.to(dtype=pooled.dtype).view(-1)
        pooled = pooled * avail.unsqueeze(-1) + self.null_vec * (1.0 - avail).unsqueeze(-1)
        return WalletPriorEncoderOutput(hidden=h, pooled=pooled, available=avail)
