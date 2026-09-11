"""Simple temporal deep control: GRU over causal event sequences."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from trencher.world_model.config import WorldModelConfig
from trencher.world_model.evaluation import STEP17_DURABLE_PREFIX
from trencher.world_model.evaluation.ablations import STAGE_B_CKPT_SHA256
from trencher.world_model.heads import WorldModelHeads


class EventGRUControl(nn.Module):
    """Economical temporal control: event GRU + small structured-state projection.

    No graph / wallet-prior / ownership Transformer / gated multimodal fusion.
    Uses the same frozen head geometry as the world model for comparable targets.
    """

    def __init__(self, cfg: WorldModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or WorldModelConfig()
        d = int(self.cfg.d_model)
        n_in = int(self.cfg.event.n_channels)
        self.input_proj = nn.Linear(n_in, d)
        self.gru = nn.GRU(
            input_size=d,
            hidden_size=d,
            num_layers=2,
            batch_first=True,
            dropout=float(self.cfg.event.dropout),
        )
        # Structured curve/global projection (mean-pool over available channels).
        self.curve_proj = nn.Linear(int(self.cfg.curve.n_channels), d)
        self.global_proj = nn.Linear(
            int(self.cfg.global_temporal.n_bucket_channels)
            + int(self.cfg.global_temporal.n_scalars),
            d,
        )
        self.combine = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.heads = WorldModelHeads(self.cfg.heads)

    def forward(self, batch: dict[str, Any], *, phase: str = "rich") -> dict[str, Any]:
        del phase  # heads always supervised for control scoring
        x = batch["event_x"]
        mask = batch.get("event_mask")
        h = self.input_proj(x)
        packed, _ = self.gru(h)
        if mask is not None:
            lengths = mask.sum(dim=1).clamp(min=1).long() - 1
            idx = lengths.view(-1, 1, 1).expand(-1, 1, packed.size(-1))
            event_vec = packed.gather(1, idx).squeeze(1)
        else:
            event_vec = packed[:, -1, :]

        if "curve_x" in batch and "curve_mask" in batch:
            cm = batch["curve_mask"].unsqueeze(-1).clamp(min=0)
            curve = (batch["curve_x"] * cm).sum(dim=1) / cm.sum(dim=1).clamp(min=1.0)
            curve_vec = self.curve_proj(curve)
        else:
            curve_vec = torch.zeros_like(event_vec)

        if "global_buckets" in batch:
            gb = batch["global_buckets"]
            gs = batch.get("global_scalars")
            if gb.dim() == 3:
                gmask = batch.get("global_mask")
                if gmask is not None:
                    gm = gmask.unsqueeze(-1).clamp(min=0)
                    gb = (gb * gm).sum(dim=1) / gm.sum(dim=1).clamp(min=1.0)
                else:
                    gb = gb.mean(dim=1)
            if gs is None:
                gs = torch.zeros(gb.size(0), self.cfg.global_temporal.n_scalars, device=gb.device)
            global_vec = self.global_proj(torch.cat([gb, gs], dim=-1))
        else:
            global_vec = torch.zeros_like(event_vec)

        z = self.combine(torch.cat([event_vec, curve_vec, global_vec], dim=-1))
        heads = self.heads(z)
        return {
            "fused": z,
            "explosion_logits": heads.explosion_logits,
            "explosion_tte_hazard": heads.explosion_tte_hazard,
            "explosion_phase_logits": heads.explosion_phase_logits,
            "graduation_tte_hazard": heads.graduation_tte_hazard,
            "survival_horizon_logits": heads.survival_horizon_logits,
            "survival_tte_hazard": heads.survival_tte_hazard,
            "barrier_logits": heads.barrier_logits,
            "return_quantiles": heads.return_quantiles,
            "mfe_quantiles": heads.mfe_quantiles,
            "mae_quantiles": heads.mae_quantiles,
            "time_to_mfe": heads.time_to_mfe,
            "time_to_mae": heads.time_to_mae,
            "persistence_pred": heads.persistence_pred,
            "path_pred": heads.path_pred,
            "path_mean": heads.path_mean,
            "path_log_scale": heads.path_log_scale,
            "path_df": heads.path_df,
            "execution": heads.execution,
        }


def count_control_parameters(cfg: WorldModelConfig | None = None) -> int:
    model = EventGRUControl(cfg)
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def gru_control_plan(*, world_model_params: int = 15_679_133) -> dict[str, Any]:
    from trencher.world_model.config import load_world_model_config

    cfg = load_world_model_config(
        "configs/world_model/world_model_candidate_v2.yaml"
    )
    n = count_control_parameters(cfg)
    return {
        "run_id": "simple_temporal_gru_control",
        "seed": 42,
        "prespecified": True,
        "hyperparameter_search": False,
        "parameters": n,
        "world_model_parameters": world_model_params,
        "parameter_ratio": n / max(world_model_params, 1),
        "modalities": ["events", "curve", "global"],
        "forbidden_modules": [
            "graph",
            "wallet_prior",
            "ownership_transformer",
            "gated_multimodal_fusion",
        ],
        "train_steps_rich_or_equivalent": 25000,
        "expected_gpu": "1x A100 40GB",
        "expected_wallclock_hours": 1.0,
        "q85_test_accessed": False,
        "durable_prefix": f"{STEP17_DURABLE_PREFIX}/simple_temporal_gru_control/",
        "note": (
            "Scientific control only. Do not launch during the editor task. "
            "Same frozen target semantics and permanent_val_eval lock."
        ),
        "launch_command": (
            "uv run python -m trencher.world_model.evaluation.step17 gru-train "
            "--world-root /content/trencher_stage/world_model "
            "--split-root /content/trencher_stage/world_model/splits/split_v1 "
            "--device cuda --max-steps 25000 --seed 42"
        ),
        "lineage_note": (
            f"Does not resume Stage-B weights (architecture differs); "
            f"Stage-B sha {STAGE_B_CKPT_SHA256} remains the rich-finetune reference "
            f"for modality ablations only."
        ),
    }
