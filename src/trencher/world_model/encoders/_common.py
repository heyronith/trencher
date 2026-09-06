"""Shared encoder primitives (continuous time, causal SDPA blocks)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousTimeEncoding(nn.Module):
    """Fourier features of absolute / relative event times (no learned ID tables)."""

    def __init__(self, d_model: int, n_freqs: int = 32) -> None:
        super().__init__()
        self.n_freqs = int(n_freqs)
        self.proj = nn.Linear(2 * self.n_freqs, d_model)
        # Fixed geometric frequencies (not identity embeddings).
        freqs = torch.exp(torch.linspace(math.log(1e-2), math.log(10.0), self.n_freqs))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        # times: [B, T] seconds (relative or absolute)
        x = times.unsqueeze(-1) * self.freqs  # [B, T, F]
        enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.proj(enc)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = d_model * ffn_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention via scaled_dot_product_attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = float(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            key_padding_mask: [B, T] True = valid token, False = pad
        """
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, Hd]
        q, k, v = qkv[0], qkv[1], qkv[2]

        dropout_p = self.dropout if self.training else 0.0
        # SDPA forbids combining is_causal=True with an explicit attn_mask.
        # When padding is present, build a causal+pad additive mask instead.
        if key_padding_mask is None:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            causal = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
            # True = allow attend
            allow = causal.unsqueeze(0) & key_padding_mask.unsqueeze(1)  # [B, T, T]
            # Also block attending FROM pad queries (optional cleanliness)
            allow = allow & key_padding_mask.unsqueeze(2)
            attn_mask = torch.zeros(b, 1, t, t, device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~allow.unsqueeze(1), torch.finfo(q.dtype).min)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
            )
        y = y.transpose(1, 2).contiguous().reshape(b, t, d)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ffn_mult=ffn_mult, dropout=dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.ff(self.norm2(x))
        return x


class SequenceBackbone(nn.Module):
    """Channel projection + continuous time + causal transformer stack."""

    def __init__(
        self,
        *,
        n_channels: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        n_time_freqs: int = 16,
        use_time: bool = True,
    ) -> None:
        super().__init__()
        self.use_time = use_time
        self.input_proj = nn.Linear(n_channels, d_model)
        self.time_enc = ContinuousTimeEncoding(d_model, n_freqs=n_time_freqs) if use_time else None
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, ffn_mult=ffn_mult, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
            times: [B, T]
            mask: [B, T] bool valid
        Returns:
            [B, T, D] hidden states
        """
        h = self.input_proj(x)
        if self.time_enc is not None and times is not None:
            h = h + self.time_enc(times)
        h = self.dropout(h)
        for block in self.blocks:
            h = block(h, key_padding_mask=mask)
        return self.norm(h)


def masked_mean_pool(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Pool [B, T, D] with optional bool mask [B, T]."""
    if mask is None:
        return h.mean(dim=1)
    m = mask.to(dtype=h.dtype).unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (h * m).sum(dim=1) / denom


def last_valid_token(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Take the last valid token per row (causal readout)."""
    if mask is None:
        return h[:, -1]
    # mask: True = valid
    lengths = mask.long().sum(dim=1).clamp_min(1) - 1
    idx = lengths.view(-1, 1, 1).expand(-1, 1, h.size(-1))
    return h.gather(1, idx).squeeze(1)
