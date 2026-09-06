"""Causal temporal-neighborhood graph encoder (TGN/DyGFormer-inspired, stateless).

Reconstructs token-local interaction state from past neighborhood histories and
optional wallet-prior sequence features visible at T only. No global streaming
TGN memory (avoids order leakage under shuffled DataLoaders). No learned
wallet/mint ID embedding tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from trencher.world_model.config import GraphEncoderConfig
from trencher.world_model.encoders._common import ContinuousTimeEncoding


@dataclass
class CausalTemporalNeighborhoodOutput:
    pooled: torch.Tensor
    neighbor_states: torch.Tensor
    available: torch.Tensor


class CausalTemporalNeighborhoodEncoder(nn.Module):
    """Stateless causal temporal-neighborhood network.

    Messages combine edge attributes, relative event time, and (optional)
    wallet-prior pooled state. Only edges already truncated to ``event_time <= T``
    should be supplied by the Dataset.
    """

    def __init__(self, cfg: GraphEncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GraphEncoderConfig()
        c = self.cfg
        self.edge_proj = nn.Linear(c.n_edge_channels, c.d_model)
        self.wallet_proj = nn.Linear(c.d_model, c.d_model)
        self.time_enc = ContinuousTimeEncoding(c.d_model, n_freqs=c.n_time_freqs)
        self.msg_mlp = nn.Sequential(
            nn.Linear(c.d_model * 3, c.d_model),
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
        wallet_prior_state: torch.Tensor | None = None,
    ) -> CausalTemporalNeighborhoodOutput:
        """
        Args:
            edge_feat: [B, E, C] causal neighborhood edge attrs
            times: [B, E] relative times
            mask: [B, E]
            available: [B]
            wallet_prior_state: [B, D] pooled wallet-prior encoding (optional)
        """
        h = self.edge_proj(edge_feat) + self.time_enc(times)
        if wallet_prior_state is None:
            w = torch.zeros(
                edge_feat.size(0), self.cfg.d_model, device=edge_feat.device, dtype=h.dtype
            )
        else:
            w = self.wallet_proj(wallet_prior_state)
        w_exp = w.unsqueeze(1).expand(-1, h.size(1), -1)
        msg = self.msg_mlp(torch.cat([h, h, w_exp], dim=-1))
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
        return CausalTemporalNeighborhoodOutput(
            pooled=pooled, neighbor_states=msg, available=avail
        )


# Back-compat alias (accurate name preferred in new code)
TemporalGraphEncoder = CausalTemporalNeighborhoodEncoder
TemporalGraphEncoderOutput = CausalTemporalNeighborhoodOutput


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
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=False
        )
        y = y.transpose(1, 2).contiguous().reshape(b, e, d)
        return residual + self.out(y)
