"""WorldModel — multimodal on-chain world model (no social / no ID tables)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from trencher.world_model.config import WorldModelConfig, tiny_smoke_config
from trencher.world_model.encoders.curve import CurveStateEncoder
from trencher.world_model.encoders.event import EventTransformerEncoder
from trencher.world_model.encoders.global_temporal import GlobalTemporalEncoder
from trencher.world_model.encoders.graph import TemporalGraphEncoder
from trencher.world_model.encoders.microstructure import MicrostructureEncoder
from trencher.world_model.encoders.ownership import OwnershipEncoder
from trencher.world_model.encoders.wallet_prior import WalletPriorEncoder
from trencher.world_model.fusion import GatedMultimodalFusion
from trencher.world_model.heads import WorldModelHeads


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    if trainable_only:
        return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    return int(sum(p.numel() for p in module.parameters()))


class WorldModel(nn.Module):
    """Compose encoders → gated fusion → multi-task heads.

    Constraints enforced by construction:
    - Pure on-chain modalities (no social/text/image towers)
    - No learned mint/wallet/creator ID embedding tables
    """

    def __init__(self, cfg: WorldModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or WorldModelConfig()
        if not self.cfg.pure_on_chain:
            raise ValueError("WorldModel requires pure_on_chain=True")
        if not self.cfg.forbid_id_embedding_tables:
            raise ValueError("WorldModel forbids ID embedding tables")

        self.event_encoder = EventTransformerEncoder(self.cfg.event)
        self.curve_encoder = CurveStateEncoder(self.cfg.curve)
        self.global_encoder = GlobalTemporalEncoder(self.cfg.global_temporal)
        self.micro_encoder = MicrostructureEncoder(self.cfg.microstructure)
        self.ownership_encoder = OwnershipEncoder(self.cfg.ownership)
        self.wallet_prior_encoder = WalletPriorEncoder(self.cfg.wallet_prior)
        self.graph_encoder = TemporalGraphEncoder(self.cfg.graph)
        self.fusion = GatedMultimodalFusion(self.cfg.fusion)
        self.heads = WorldModelHeads(self.cfg.heads)
        # Lightweight creator prior from curve/event-adjacent scalars (no ID table)
        self.creator_proj = nn.Sequential(
            nn.Linear(4, self.cfg.d_model),
            nn.GELU(),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
        )

    @classmethod
    def from_config(cls, cfg: WorldModelConfig) -> WorldModel:
        return cls(cfg)

    @classmethod
    def tiny_for_tests(cls) -> WorldModel:
        return cls(tiny_smoke_config())

    def encode_modalities(
        self,
        batch: dict[str, Any],
        *,
        active: set[str] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return (pooled_vecs, availability, ssl_outputs)."""
        vecs: dict[str, torch.Tensor] = {}
        avail: dict[str, torch.Tensor] = {}
        ssl: dict[str, torch.Tensor] = {}
        active = active or set(self.cfg.modalities_rich)

        if "events" in active and "event_x" in batch:
            out = self.event_encoder(
                batch["event_x"],
                batch["event_times"],
                batch.get("event_mask"),
                return_ssl=True,
            )
            vecs["events"] = out.pooled
            avail["events"] = batch.get(
                "event_available",
                torch.ones(out.pooled.size(0), device=out.pooled.device),
            )
            if out.ssl:
                ssl["ssl_next_side_logits"] = out.ssl["next_side_logits"]
                ssl["ssl_next_log_dt"] = out.ssl["next_log_dt"]
                ssl["ssl_next_size"] = out.ssl["next_size"]
                if "intensity_log" in out.ssl:
                    ssl["ssl_intensity_log"] = out.ssl["intensity_log"]
                if "reserve_delta" in out.ssl:
                    ssl["ssl_reserve_delta"] = out.ssl["reserve_delta"]
                if "recon" in out.ssl:
                    ssl["ssl_recon"] = out.ssl["recon"]

        if "curve" in active and "curve_x" in batch:
            out = self.curve_encoder(
                batch["curve_x"], batch["curve_times"], batch.get("curve_mask")
            )
            vecs["curve"] = out.pooled
            avail["curve"] = batch.get(
                "curve_available",
                torch.ones(out.pooled.size(0), device=out.pooled.device),
            )

        if "global" in active and "global_buckets" in batch:
            out = self.global_encoder(
                batch["global_buckets"],
                batch["global_times"],
                batch["global_scalars"],
                batch.get("global_mask"),
            )
            vecs["global"] = out.pooled
            avail["global"] = batch.get(
                "global_available",
                torch.ones(out.pooled.size(0), device=out.pooled.device),
            )

        if "creator" in active and "creator_feat" in batch:
            # creator_feat: [B, 4] behavioral/rank features — NOT an ID embedding lookup
            cvec = self.creator_proj(batch["creator_feat"])
            # Creator has no fusion slot; fold behavioral features into events.
            if "events" in vecs:
                vecs["events"] = vecs["events"] + cvec
            else:
                vecs["events"] = cvec
                avail["events"] = batch.get(
                    "creator_available",
                    torch.ones(cvec.size(0), device=cvec.device),
                )

        if "microstructure" in active and "micro_x" in batch:
            out = self.micro_encoder(
                batch["micro_x"],
                batch["micro_times"],
                batch.get("micro_mask"),
                available=batch.get("micro_available"),
            )
            vecs["microstructure"] = out.pooled
            avail["microstructure"] = out.available

        if "ownership" in active and "own_x" in batch:
            out = self.ownership_encoder(
                batch["own_x"],
                batch["own_times"],
                batch.get("own_mask"),
                available=batch.get("own_available"),
                quality=batch.get("own_quality"),
            )
            vecs["ownership"] = out.pooled
            avail["ownership"] = out.available

        wallet_state = None
        if "wallet_prior" in active and "wallet_x" in batch:
            out = self.wallet_prior_encoder(
                batch["wallet_x"],
                batch["wallet_times"],
                batch.get("wallet_mask"),
                available=batch.get("wallet_available"),
            )
            vecs["wallet_prior"] = out.pooled
            avail["wallet_prior"] = out.available
            wallet_state = out.pooled

        if "graph" in active and "graph_edge_feat" in batch:
            out = self.graph_encoder(
                batch["graph_edge_feat"],
                batch["graph_times"],
                batch.get("graph_mask"),
                available=batch.get("graph_available"),
                wallet_prior_state=wallet_state,
            )
            vecs["graph"] = out.pooled
            avail["graph"] = out.available

        return vecs, avail, ssl

    def forward(
        self,
        batch: dict[str, Any],
        *,
        phase: str = "broad",
        active: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            phase: ``ssl`` | ``broad`` | ``rich``
        """
        if active is None:
            if phase == "ssl":
                active = {"events"}
            elif phase == "broad":
                active = set(self.cfg.modalities_broad)
            else:
                active = set(self.cfg.modalities_rich)
            omit = getattr(self, "omit_modalities", None)
            if omit:
                active -= {str(x) for x in omit}

        if phase == "ssl" and "event_x" in batch:
            from trencher.world_model.ssl_masking import apply_ssl_event_masking

            ratio = float(getattr(self.cfg.loss, "ssl_recon_mask_ratio", 0.15))
            masked = apply_ssl_event_masking(
                batch, seed=int(self.cfg.train.seed), mask_ratio=ratio
            )
            # Mutate caller batch so MultiTaskLoss sees recon target/mask.
            batch["event_x"] = masked["event_x"]
            batch["ssl_recon_target"] = masked["ssl_recon_target"]
            batch["ssl_recon_mask"] = masked["ssl_recon_mask"]

        vecs, avail, ssl = self.encode_modalities(batch, active=active)
        outputs: dict[str, torch.Tensor] = dict(ssl)

        if phase == "ssl":
            return outputs

        # Ensure at least a zero events vector for fusion shape stability
        if not vecs:
            raise ValueError("batch produced no modality vectors")

        fuse = self.fusion(vecs, avail)
        outputs["fused"] = fuse.fused
        outputs["fusion_gates"] = fuse.gates
        heads = self.heads(fuse.fused)
        outputs.update(
            {
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
        )
        return outputs
