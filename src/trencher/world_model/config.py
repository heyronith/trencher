"""World-model configuration dataclasses and YAML loader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# Frozen V4 horizon / barrier / size contracts (mirrors targets.status).
_HORIZONS_SEC: tuple[int, ...] = (
    10,
    30,
    60,
    120,
    300,
    600,
    1800,
    3600,
    21600,
    86400,
)
_SURVIVAL_HORIZONS_SEC: tuple[int, ...] = (
    30,
    60,
    120,
    300,
    600,
    1800,
    3600,
    21600,
    86400,
)
_EXPLOSION_HORIZONS_SEC: tuple[int, ...] = (10, 30, 60, 120, 300)
_BARRIER_PAIRS: tuple[tuple[float, float], ...] = (
    (10.0, 10.0),
    (25.0, 15.0),
    (50.0, 20.0),
    (100.0, 30.0),
)
_EXECUTION_SIZES: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0)
_QUANTILE_TAUS: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


@dataclass
class EventEncoderConfig:
    n_channels: int = 13
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    max_len: int = 512
    n_sides: int = 3
    time_encoding: str = "continuous_fourier"
    n_time_freqs: int = 32


@dataclass
class SequenceEncoderConfig:
    n_channels: int = 8
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    max_len: int = 256


@dataclass
class GlobalTemporalConfig:
    n_bucket_channels: int = 8
    n_scalars: int = 9
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    max_buckets: int = 128


@dataclass
class GraphEncoderConfig:
    n_edge_channels: int = 7
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    max_neighbors: int = 64
    time_encoding: str = "continuous_fourier"
    n_time_freqs: int = 16


@dataclass
class FusionConfig:
    d_model: int = 256
    n_modalities: int = 7
    dropout: float = 0.1
    gate_hidden: int = 256


@dataclass
class HeadsConfig:
    d_model: int = 256
    explosion_horizons: tuple[int, ...] = _EXPLOSION_HORIZONS_SEC
    survival_horizons: tuple[int, ...] = _SURVIVAL_HORIZONS_SEC
    return_horizons: tuple[int, ...] = _HORIZONS_SEC
    barrier_pairs: tuple[tuple[float, float], ...] = _BARRIER_PAIRS
    barrier_classes: int = 3  # upper_first, lower_first, neither
    quantile_taus: tuple[float, ...] = _QUANTILE_TAUS
    path_steps: int = 8
    path_dim: int = 4
    explosion_tte_bins: int = 8
    graduation_tte_bins: int = 8
    survival_tte_bins: int = 8
    explosion_phase_dim: int = 4
    execution_sizes: tuple[float, ...] = _EXECUTION_SIZES
    execution_kinds: int = 3  # gross, entry_impact, curve_impact
    dropout: float = 0.1


@dataclass
class LossConfig:
    ssl_side_weight: float = 1.0
    ssl_log_dt_weight: float = 1.0
    ssl_size_weight: float = 1.0
    ssl_intensity_weight: float = 0.5
    ssl_reserve_weight: float = 0.5
    ssl_recon_weight: float = 0.25
    ssl_recon_mask_ratio: float = 0.15
    explosion_weight: float = 1.0
    explosion_focal_gamma: float = 2.0
    explosion_pos_weight: float | None = None
    explosion_tte_weight: float = 1.0
    explosion_phase_weight: float = 0.25
    graduation_tte_weight: float = 1.0
    survival_weight: float = 1.0
    survival_tte_weight: float = 1.0
    barrier_weight: float = 1.0
    quantile_weight: float = 1.0
    mfe_mae_weight: float = 1.0
    time_to_extreme_weight: float = 0.5
    persistence_weight: float = 0.5
    path_weight: float = 0.5
    execution_weight: float = 0.5
    label_smoothing: float = 0.0


@dataclass
class TrainConfig:
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    ssl_steps: int = 50_000
    broad_steps: int = 100_000
    rich_steps: int = 25_000
    batch_size: int = 64
    accum_steps: int = 1
    bf16: bool = True
    compile: bool = False
    profile_steps: int = 200
    health_gates_enabled: bool = True
    refuse_nan_loss: bool = True
    refuse_empty_support: bool = True
    seed: int = 42
    log_every: int = 50
    ckpt_every: int = 2000
    set_to_none: bool = True


_TUPLE_FIELDS = frozenset(
    {
        "explosion_horizons",
        "survival_horizons",
        "return_horizons",
        "barrier_pairs",
        "quantile_taus",
        "execution_sizes",
        "modalities_broad",
        "modalities_rich",
    }
)

_NESTED_TUPLE_FIELDS = frozenset({"barrier_pairs"})


@dataclass
class WorldModelConfig:
    """Full production world-model config (A100-oriented defaults)."""

    name: str = "world_model_candidate_v2"
    d_model: int = 256
    dropout: float = 0.1
    event: EventEncoderConfig = field(default_factory=EventEncoderConfig)
    curve: SequenceEncoderConfig = field(
        default_factory=lambda: SequenceEncoderConfig(n_channels=7, max_len=256)
    )
    global_temporal: GlobalTemporalConfig = field(default_factory=GlobalTemporalConfig)
    microstructure: SequenceEncoderConfig = field(
        default_factory=lambda: SequenceEncoderConfig(n_channels=11, max_len=128)
    )
    ownership: SequenceEncoderConfig = field(
        default_factory=lambda: SequenceEncoderConfig(n_channels=9, max_len=128)
    )
    wallet_prior: SequenceEncoderConfig = field(
        default_factory=lambda: SequenceEncoderConfig(n_channels=7, max_len=256)
    )
    graph: GraphEncoderConfig = field(default_factory=GraphEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    modalities_broad: tuple[str, ...] = (
        "events",
        "curve",
        "global",
        "creator",
    )
    modalities_rich: tuple[str, ...] = (
        "events",
        "curve",
        "global",
        "creator",
        "microstructure",
        "ownership",
        "wallet_prior",
        "graph",
    )
    pure_on_chain: bool = True
    forbid_id_embedding_tables: bool = True
    freeze_id: str = "world_model_freeze_v1"
    contract: str = "WORLD_MODEL_TRAINING_CONTRACT_V2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_tuple_field(name: str, val: Any) -> Any:
    if not isinstance(val, list):
        return val
    if name in _NESTED_TUPLE_FIELDS:
        return tuple(tuple(x) for x in val)
    if name in _TUPLE_FIELDS:
        return tuple(val)
    return val


def _build_nested(cls: type, data: dict[str, Any] | None) -> Any:
    """Construct a dataclass tree from a (possibly partial) dict."""
    if data is None:
        return cls()
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        if f.default_factory is not MISSING and isinstance(val, dict):
            sample = f.default_factory()
            if is_dataclass(sample):
                kwargs[f.name] = _build_nested(type(sample), val)
                continue
        kwargs[f.name] = _as_tuple_field(f.name, val)
    return cls(**kwargs)


def load_world_model_config(path: str | Path) -> WorldModelConfig:
    """Load ``WorldModelConfig`` from a YAML file."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {p}")
    return _build_nested(WorldModelConfig, raw)


def world_model_config_from_dict(data: dict[str, Any]) -> WorldModelConfig:
    return _build_nested(WorldModelConfig, data)


def tiny_smoke_config(**overrides: Any) -> WorldModelConfig:
    """CPU-friendly tiny config for unit tests (not for production training)."""
    cfg = WorldModelConfig(
        name="world_model_smoke_tiny",
        d_model=32,
        dropout=0.0,
        event=EventEncoderConfig(
            n_channels=13,
            d_model=32,
            n_layers=2,
            n_heads=4,
            ffn_mult=2,
            dropout=0.0,
            max_len=32,
            n_time_freqs=8,
        ),
        curve=SequenceEncoderConfig(
            n_channels=7, d_model=32, n_layers=1, n_heads=4, ffn_mult=2, dropout=0.0, max_len=16
        ),
        global_temporal=GlobalTemporalConfig(
            n_bucket_channels=8,
            n_scalars=9,
            d_model=32,
            n_layers=1,
            n_heads=4,
            ffn_mult=2,
            dropout=0.0,
            max_buckets=16,
        ),
        microstructure=SequenceEncoderConfig(
            n_channels=11, d_model=32, n_layers=1, n_heads=4, ffn_mult=2, dropout=0.0, max_len=16
        ),
        ownership=SequenceEncoderConfig(
            n_channels=9, d_model=32, n_layers=1, n_heads=4, ffn_mult=2, dropout=0.0, max_len=16
        ),
        wallet_prior=SequenceEncoderConfig(
            n_channels=7, d_model=32, n_layers=1, n_heads=4, ffn_mult=2, dropout=0.0, max_len=16
        ),
        graph=GraphEncoderConfig(
            n_edge_channels=7,
            d_model=32,
            n_layers=1,
            n_heads=4,
            dropout=0.0,
            max_neighbors=16,
            n_time_freqs=8,
        ),
        fusion=FusionConfig(d_model=32, n_modalities=7, dropout=0.0, gate_hidden=32),
        heads=HeadsConfig(
            d_model=32,
            explosion_horizons=(30, 60),
            survival_horizons=(30, 60),
            return_horizons=(30, 60),
            barrier_pairs=((10.0, 10.0), (25.0, 15.0)),
            barrier_classes=3,
            quantile_taus=(0.25, 0.5, 0.75),
            path_steps=4,
            path_dim=2,
            explosion_tte_bins=4,
            graduation_tte_bins=4,
            survival_tte_bins=4,
            explosion_phase_dim=2,
            execution_sizes=(0.1, 1.0),
            execution_kinds=3,
            dropout=0.0,
        ),
        train=TrainConfig(
            lr=1e-3,
            weight_decay=0.0,
            max_grad_norm=1.0,
            warmup_steps=0,
            ssl_steps=5,
            broad_steps=5,
            rich_steps=5,
            batch_size=4,
            bf16=False,
            compile=False,
            profile_steps=2,
            health_gates_enabled=False,
            seed=0,
            log_every=1,
            ckpt_every=100,
        ),
    )
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(k)
        setattr(cfg, k, v)
    if "d_model" in overrides:
        d = int(overrides["d_model"])
        for attr in (
            "event",
            "curve",
            "global_temporal",
            "microstructure",
            "ownership",
            "wallet_prior",
            "graph",
            "fusion",
            "heads",
        ):
            getattr(cfg, attr).d_model = d
    return cfg
