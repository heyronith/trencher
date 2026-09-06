"""Multi-task prediction heads for the on-chain world model (frozen V4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from trencher.world_model.config import HeadsConfig


def _mlp(d_in: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_in),
        nn.Linear(d_in, d_in),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_in, d_out),
    )


@dataclass
class HeadsOutput:
    explosion_logits: torch.Tensor
    explosion_tte_hazard: torch.Tensor
    explosion_phase_logits: torch.Tensor
    graduation_tte_hazard: torch.Tensor
    survival_horizon_logits: torch.Tensor
    survival_tte_hazard: torch.Tensor
    barrier_logits: torch.Tensor
    return_quantiles: torch.Tensor
    mfe_quantiles: torch.Tensor
    mae_quantiles: torch.Tensor
    time_to_mfe: torch.Tensor
    time_to_mae: torch.Tensor
    persistence_pred: torch.Tensor
    path_pred: torch.Tensor
    path_mean: torch.Tensor
    path_log_scale: torch.Tensor
    path_df: torch.Tensor
    execution: torch.Tensor


class ExplosionHead(nn.Module):
    """Multi-horizon explosion imminence logits [B, n_horizons]."""

    def __init__(self, d_model: int, n_horizons: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = _mlp(d_model, n_horizons, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ExplosionTTEHead(nn.Module):
    """Discrete-time explosion TTE hazard logits [B, bins]."""

    def __init__(self, d_model: int, n_bins: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = _mlp(d_model, n_bins, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ExplosionPhaseHead(nn.Module):
    """Small MLP for imminence / phase auxiliary logits [B, phase_dim]."""

    def __init__(self, d_model: int, phase_dim: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = _mlp(d_model, phase_dim, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GraduationTTEHead(nn.Module):
    """Discrete-time graduation TTE hazard logits [B, bins]."""

    def __init__(self, d_model: int, n_bins: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = _mlp(d_model, n_bins, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SurvivalHorizonHead(nn.Module):
    """Binary active@H logits per survival horizon [B, n_horizons]."""

    def __init__(self, d_model: int, n_horizons: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = _mlp(d_model, n_horizons, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SurvivalTTEHead(nn.Module):
    """Discrete-time survival / inactivity TTE hazard logits [B, bins]."""

    def __init__(self, d_model: int, n_bins: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = _mlp(d_model, n_bins, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BarrierCompetingHead(nn.Module):
    """Competing barrier outcomes [B, n_pairs, n_horizons, n_classes]."""

    def __init__(
        self,
        d_model: int,
        n_pairs: int,
        n_horizons: int,
        n_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_pairs = n_pairs
        self.n_horizons = n_horizons
        self.n_classes = n_classes
        self.net = _mlp(d_model, n_pairs * n_horizons * n_classes, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        return self.net(z).view(b, self.n_pairs, self.n_horizons, self.n_classes)


class QuantileReturnHead(nn.Module):
    """Pinball quantile predictions for forward returns [B, H, Q]."""

    def __init__(
        self, d_model: int, n_horizons: int, n_quantiles: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
        self.net = _mlp(d_model, n_horizons * n_quantiles, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        return self.net(z).view(b, self.n_horizons, self.n_quantiles)


class QuantileMFEMAEHead(nn.Module):
    """Quantile MFE/MAE [B, H, Q] plus time-to-extreme [B, H]."""

    def __init__(
        self, d_model: int, n_horizons: int, n_quantiles: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
        self.mfe = _mlp(d_model, n_horizons * n_quantiles, dropout)
        self.mae = _mlp(d_model, n_horizons * n_quantiles, dropout)
        self.time_to_mfe = _mlp(d_model, n_horizons, dropout)
        self.time_to_mae = _mlp(d_model, n_horizons, dropout)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = z.size(0)
        mfe = self.mfe(z).view(b, self.n_horizons, self.n_quantiles)
        mae = self.mae(z).view(b, self.n_horizons, self.n_quantiles)
        return mfe, mae, self.time_to_mfe(z), self.time_to_mae(z)


class PersistenceHead(nn.Module):
    """Regress trade_count / activity_rate / accel per horizon [B, H, 3]."""

    def __init__(self, d_model: int, n_horizons: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_horizons = n_horizons
        self.net = _mlp(d_model, n_horizons * 3, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        return self.net(z).view(b, self.n_horizons, 3)


class PathHead(nn.Module):
    """Conditional autoregressive Student-t path decoder over ordered future knots.

    Emits coherent mean trajectories plus per-step log-scale / df parameters so
    samples are ordered-path-consistent (replaceable later by diffusion without
    changing the fusion/world-state interface).
    """

    def __init__(
        self, d_model: int, path_steps: int, path_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.path_steps = path_steps
        self.path_dim = path_dim
        self.cell = nn.GRUCell(path_dim + d_model, d_model)
        self.proj_mean = nn.Linear(d_model, path_dim)
        self.proj_log_scale = nn.Linear(d_model, path_dim)
        self.proj_df = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return mean path [B, steps, dim] (training target alignment)."""
        means, _, _ = self.forward_dist(z)
        return means

    def forward_dist(
        self, z: torch.Tensor, *, teacher: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = z.size(0)
        h = z
        prev = z.new_zeros(b, self.path_dim)
        means = []
        log_scales = []
        dfs = []
        for t in range(self.path_steps):
            inp = torch.cat([prev, self.dropout(z)], dim=-1)
            h = self.cell(inp, h)
            mean = self.proj_mean(h)
            log_scale = self.proj_log_scale(h).clamp(-5.0, 5.0)
            df = F.softplus(self.proj_df(h)) + 2.0
            means.append(mean)
            log_scales.append(log_scale)
            dfs.append(df)
            prev = teacher[:, t] if teacher is not None else mean
        return (
            torch.stack(means, dim=1),
            torch.stack(log_scales, dim=1),
            torch.stack(dfs, dim=1).squeeze(-1),
        )

    def sample(self, z: torch.Tensor, *, n_samples: int = 1) -> torch.Tensor:
        """Draw coherent paths [n_samples, B, steps, dim]."""
        means, log_scales, dfs = self.forward_dist(z)
        scale = log_scales.exp()
        outs = []
        for _ in range(n_samples):
            eps = torch.randn_like(means)
            chi2 = torch.distributions.Chi2(dfs.unsqueeze(-1).expand_as(means)).rsample()
            outs.append(means + scale * eps * (dfs.unsqueeze(-1) / chi2.clamp_min(1e-4)).sqrt())
        return torch.stack(outs, dim=0)


class ExecutionHead(nn.Module):
    """Size × kind execution scores [B, n_sizes, n_kinds]."""

    def __init__(
        self, d_model: int, n_sizes: int, n_kinds: int = 3, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.n_sizes = n_sizes
        self.n_kinds = n_kinds
        self.net = _mlp(d_model, n_sizes * n_kinds, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        return self.net(z).view(b, self.n_sizes, self.n_kinds)


class WorldModelHeads(nn.Module):
    """Bundle of MT heads used in broad / rich phases."""

    def __init__(self, cfg: HeadsConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or HeadsConfig()
        c = self.cfg
        n_ret = len(c.return_horizons)
        n_q = len(c.quantile_taus)
        n_pairs = len(c.barrier_pairs)
        self.explosion = ExplosionHead(c.d_model, len(c.explosion_horizons), c.dropout)
        self.explosion_tte = ExplosionTTEHead(c.d_model, c.explosion_tte_bins, c.dropout)
        self.explosion_phase = ExplosionPhaseHead(
            c.d_model, c.explosion_phase_dim, c.dropout
        )
        self.graduation_tte = GraduationTTEHead(
            c.d_model, c.graduation_tte_bins, c.dropout
        )
        self.survival_horizon = SurvivalHorizonHead(
            c.d_model, len(c.survival_horizons), c.dropout
        )
        self.survival_tte = SurvivalTTEHead(c.d_model, c.survival_tte_bins, c.dropout)
        self.barrier = BarrierCompetingHead(
            c.d_model, n_pairs, n_ret, c.barrier_classes, c.dropout
        )
        self.quantile = QuantileReturnHead(c.d_model, n_ret, n_q, c.dropout)
        self.mfe_mae = QuantileMFEMAEHead(c.d_model, n_ret, n_q, c.dropout)
        self.persistence = PersistenceHead(c.d_model, n_ret, c.dropout)
        self.path = PathHead(c.d_model, c.path_steps, c.path_dim, c.dropout)
        self.execution = ExecutionHead(
            c.d_model, len(c.execution_sizes), c.execution_kinds, c.dropout
        )

    def forward(self, z: torch.Tensor) -> HeadsOutput:
        mfe, mae, ttmfe, ttmae = self.mfe_mae(z)
        path_mean, path_log_scale, path_df = self.path.forward_dist(z)
        return HeadsOutput(
            explosion_logits=self.explosion(z),
            explosion_tte_hazard=self.explosion_tte(z),
            explosion_phase_logits=self.explosion_phase(z),
            graduation_tte_hazard=self.graduation_tte(z),
            survival_horizon_logits=self.survival_horizon(z),
            survival_tte_hazard=self.survival_tte(z),
            barrier_logits=self.barrier(z),
            return_quantiles=self.quantile(z),
            mfe_quantiles=mfe,
            mae_quantiles=mae,
            time_to_mfe=ttmfe,
            time_to_mae=ttmae,
            persistence_pred=self.persistence(z),
            path_pred=path_mean,
            path_mean=path_mean,
            path_log_scale=path_log_scale,
            path_df=path_df,
            execution=self.execution(z),
        )
