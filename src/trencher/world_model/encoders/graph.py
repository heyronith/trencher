"""TGN-style temporal neighborhood aggregation (feature edges, no ID tables)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from trencher.world_model.config import GraphEncoderConfig
from trencher.world_model.encoders._common import ContinuousTimeEncoding


@dataclass
class TemporalGraphEncoderOutput:
    pooled: torch.Tensor
    neighbor_states: torch.Tensor
    available: torch.Tensor


class TemporalGraphEncoder(nn.Module):
    """Neighborhood aggregator over time-stamped edge features.

    Uses continuous-time encoding + attention over neighbors (TGN-style message
    passing readout). Does **not** learn mint/wallet/creator ID embedding tables;
    messages are built only from edge attribute channels.
    """

    def __init__(self, cfg: GraphEncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GraphEncoderConfig()
        c = self.cfg
        self.edge_proj = nn.Linear(c.n_edge_channels, c.d_model)
        self.time_enc = ContinuousTimeEncoding(c.d_model, n_freqs=c.n_time_freqs)
        self.msg_mlp = nn.Sequential(
            nn.Linear(c.d_model * 2, c.d_model),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.d_model, c.d_model),
        )
        self.layers = nn.ModuleList(
            [_NeighborAttention(c.d_model, c.n_heads, c.dropout) for _ in range(c.n_layers)]
        )
        self.norm = nn.LayerNorm(c.d_model)
        self.null_vec = nn.Parameter(torch.zeros(c.d_model))
        self.forbid_id_embedding_tables = True

    def forward(
        self,
        edge_feat: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None = None,
        available: torch.Tensor | None = None,
    ) -> TemporalGraphEncoderOutput:
        """
        Args:
            edge_feat: [B, E, C] neighbor edge attributes (sorted causally)
            times: [B, E] edge times relative to observation T
            mask: [B, E] valid neighbors
            available: [B]
        """
        h = self.edge_proj(edge_feat) + self.time_enc(times)
        # Message from edge features alone
        msg = self.msg_mlp(torch.cat([h, h], dim=-1))
        for layer in self.layers:
            msg = layer(msg, mask=mask)
        msg = self.norm(msg)
        if mask is None:
            pooled = msg.mean(dim=1)
        else:
            m = mask.to(dtype=msg.dtype).unsqueeze(-1)
            pooled = (msg * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        if available is None:
            avail = torch.ones(edge_feat.size(0), device=edge_feat.device, dtype=pooled.dtype)
        else:
            avail = available.to(dtype=pooled.dtype).view(-1)
        pooled = pooled * avail.unsqueeze(-1) + self.null_vec * (1.0 - avail).unsqueeze(-1)
        return TemporalGraphEncoderOutput(
            pooled=pooled, neighbor_states=msg, available=avail
        )


class _NeighborAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = float(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, e, d = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(b, e, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_mask = None
        if mask is not None:
            pad = ~mask
            attn_mask = pad[:, None, None, :].to(dtype=q.dtype) * torch.finfo(q.dtype).min
        drop = self.dropout if self.training else 0.0
        # Non-causal over neighbors (already causally truncated upstream)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=False
        )
        y = y.transpose(1, 2).contiguous().reshape(b, e, d)
        return residual + self.out(y)
