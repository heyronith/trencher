"""Gated multimodal fusion with per-modality availability masks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from trencher.world_model.config import FusionConfig

MODALITY_ORDER: tuple[str, ...] = (
    "events",
    "curve",
    "global",
    "microstructure",
    "ownership",
    "wallet_prior",
    "graph",
)


@dataclass
class FusionOutput:
    fused: torch.Tensor
    gates: torch.Tensor
    availability: torch.Tensor


class GatedMultimodalFusion(nn.Module):
    """Gate and sum modality embeddings; unavailable modalities get gate≈0."""

    def __init__(self, cfg: FusionConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FusionConfig()
        c = self.cfg
        self.n_modalities = c.n_modalities
        self.pre_norm = nn.ModuleList([nn.LayerNorm(c.d_model) for _ in range(c.n_modalities)])
        self.gate_mlp = nn.Sequential(
            nn.Linear(c.d_model * c.n_modalities, c.gate_hidden),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.gate_hidden, c.n_modalities),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(c.d_model, c.d_model),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.LayerNorm(c.d_model),
        )

    def forward(
        self,
        modality_vecs: dict[str, torch.Tensor],
        availability: dict[str, torch.Tensor] | None = None,
    ) -> FusionOutput:
        """
        Args:
            modality_vecs: name → [B, D] (missing names treated as zeros)
            availability: name → [B] in {0,1}
        """
        device = next(iter(modality_vecs.values())).device
        dtype = next(iter(modality_vecs.values())).dtype
        b = next(iter(modality_vecs.values())).size(0)
        d = self.cfg.d_model

        stacked: list[torch.Tensor] = []
        avail_list: list[torch.Tensor] = []
        names = MODALITY_ORDER[: self.n_modalities]
        for i, name in enumerate(names):
            if name in modality_vecs:
                v = modality_vecs[name]
            else:
                v = torch.zeros(b, d, device=device, dtype=dtype)
            stacked.append(self.pre_norm[i](v))
            if availability is not None and name in availability:
                a = availability[name].to(dtype=dtype).view(b)
            else:
                a = torch.ones(b, device=device, dtype=dtype)
            avail_list.append(a)

        stack = torch.stack(stacked, dim=1)  # [B, M, D]
        avail = torch.stack(avail_list, dim=1)  # [B, M]
        flat = stack.reshape(b, -1)
        logits = self.gate_mlp(flat)
        # Mask unavailable before softmax
        logits = logits.masked_fill(avail < 0.5, torch.finfo(logits.dtype).min)
        gates = F.softmax(logits, dim=-1) * avail
        # Renormalize if all available (avoid NaN when none available)
        gate_sum = gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        gates = gates / gate_sum
        fused = (stack * gates.unsqueeze(-1)).sum(dim=1)
        fused = self.out_proj(fused)
        # If nothing available, return zeros
        any_avail = (avail.sum(dim=-1, keepdim=True) > 0).to(dtype=fused.dtype)
        fused = fused * any_avail
        return FusionOutput(fused=fused, gates=gates, availability=avail)
