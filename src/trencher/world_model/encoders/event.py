"""Causal event transformer with continuous time and SSL heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from trencher.world_model.config import EventEncoderConfig
from trencher.world_model.encoders._common import SequenceBackbone, last_valid_token


@dataclass
class EventEncoderOutput:
    hidden: torch.Tensor  # [B, T, D]
    pooled: torch.Tensor  # [B, D]
    ssl: dict[str, torch.Tensor]


class EventTransformerEncoder(nn.Module):
    """Token-event sequence encoder (pure on-chain; no ID embedding tables).

    SSL heads predict next-side, next log-Δt, next size, intensity, reserve
    delta, and optional masked reconstruction from causal states.
    """

    def __init__(self, cfg: EventEncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EventEncoderConfig()
        c = self.cfg
        self.backbone = SequenceBackbone(
            n_channels=c.n_channels,
            d_model=c.d_model,
            n_layers=c.n_layers,
            n_heads=c.n_heads,
            ffn_mult=c.ffn_mult,
            dropout=c.dropout,
            n_time_freqs=c.n_time_freqs,
            use_time=True,
        )
        self.ssl_side = nn.Linear(c.d_model, c.n_sides)
        self.ssl_log_dt = nn.Linear(c.d_model, 1)
        self.ssl_size = nn.Linear(c.d_model, 1)
        # Hawkes-inspired intensity auxiliary (log-intensity; not a literal classical Hawkes).
        self.ssl_intensity = nn.Linear(c.d_model, 1)
        self.ssl_reserve_delta = nn.Linear(c.d_model, 1)
        self.ssl_recon = nn.Linear(c.d_model, c.n_channels)

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_ssl: bool = True,
    ) -> EventEncoderOutput:
        """
        Args:
            x: [B, T, C] event channels
            times: [B, T] continuous relative times (seconds)
            mask: [B, T] bool valid positions
        """
        h = self.backbone(x, times=times, mask=mask)
        pooled = last_valid_token(h, mask)
        ssl: dict[str, torch.Tensor] = {}
        if return_ssl:
            # Predict next token attributes from positions 0..T-2
            ssl_h = h[:, :-1]
            ssl = {
                "next_side_logits": self.ssl_side(ssl_h),
                "next_log_dt": self.ssl_log_dt(ssl_h).squeeze(-1),
                "next_size": self.ssl_size(ssl_h).squeeze(-1),
                "intensity_log": self.ssl_intensity(ssl_h).squeeze(-1),
                "reserve_delta": self.ssl_reserve_delta(ssl_h).squeeze(-1),
                "recon": self.ssl_recon(ssl_h),
            }
        return EventEncoderOutput(hidden=h, pooled=pooled, ssl=ssl)
