"""Multi-task prediction heads for the on-chain world model."""

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
    survival_hazard: torch.Tensor
    barrier_logits: torch.Tensor
    quantile_returns: torch.Tensor
    mfe: torch.Tensor
    mae: torch.Tensor
    persistence_logits: torch.Tensor
    path_pred: torch.Tensor
    execution: torch.Tensor


class ExplosionHead(nn.Module):
    """Multi-horizon explosion imminence logits."""

    def __init__(self, d_model: int, n_horizons: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = _mlp(d_model, n_horizons, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SurvivalHead(nn.Module):
    """Discrete-time hazard rates per horizon (sigmoid applied in loss/metrics)."""

    def __init__(self, d_model: int, n_horizons: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = _mlp(d_model, n_horizons, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BarrierHead(nn.Module):
    """Probability of hitting return barriers."""

    def __init__(self, d_model: int, n_levels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = _mlp(d_model, n_levels, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class QuantileReturnHead(nn.Module):
    """Pinball quantile predictions for forward returns."""

    def __init__(self, d_model: int, n_quantiles: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = _mlp(d_model, n_quantiles, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MFEMAEHead(nn.Module):
    """Max favorable / adverse excursion regression."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mfe = _mlp(d_model, 1, dropout)
        self.mae = _mlp(d_model, 1, dropout)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mfe(z).squeeze(-1), self.mae(z).squeeze(-1)


class PersistenceHead(nn.Module):
    """Regime / persistence class logits."""

    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = _mlp(d_model, n_classes, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


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
        # Student-t via Normal / chi2 approximation for sampling.
        outs = []
        for _ in range(n_samples):
            eps = torch.randn_like(means)
            chi2 = torch.distributions.Chi2(dfs.unsqueeze(-1).expand_as(means)).rsample()
            outs.append(means + scale * eps * (dfs.unsqueeze(-1) / chi2.clamp_min(1e-4)).sqrt())
        return torch.stack(outs, dim=0)


class ExecutionHead(nn.Module):
    """Execution / slippage / fillability scores."""

    def __init__(self, d_model: int, exec_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = _mlp(d_model, exec_dim, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class WorldModelHeads(nn.Module):
    """Bundle of MT heads used in broad / rich phases."""

    def __init__(self, cfg: HeadsConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or HeadsConfig()
        c = self.cfg
        self.explosion = ExplosionHead(c.d_model, len(c.explosion_horizons), c.dropout)
        self.survival = SurvivalHead(c.d_model, len(c.survival_horizons), c.dropout)
        self.barrier = BarrierHead(c.d_model, len(c.barrier_levels), c.dropout)
        self.quantile = QuantileReturnHead(c.d_model, len(c.quantile_taus), c.dropout)
        self.mfe_mae = MFEMAEHead(c.d_model, c.dropout)
        self.persistence = PersistenceHead(c.d_model, c.persistence_classes, c.dropout)
        self.path = PathHead(c.d_model, c.path_steps, c.path_dim, c.dropout)
        self.execution = ExecutionHead(c.d_model, c.execution_dim, c.dropout)

    def forward(self, z: torch.Tensor) -> HeadsOutput:
        mfe, mae = self.mfe_mae(z)
        return HeadsOutput(
            explosion_logits=self.explosion(z),
            survival_hazard=self.survival(z),
            barrier_logits=self.barrier(z),
            quantile_returns=self.quantile(z),
            mfe=mfe,
            mae=mae,
            persistence_logits=self.persistence(z),
            path_pred=self.path(z),
            execution=self.execution(z),
        )
