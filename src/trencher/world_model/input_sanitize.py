"""Deep-model input finite sanitization (tensorization boundary only).

Does not mutate frozen raw/canonical stores. Transforms for finite values are
applied upstream in ``sequences_batch``; this module records original finiteness,
replaces remaining non-finite cells with transformed-space 0.0, and exposes
finite masks for SSL / reconstruction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from trencher.data.world_model.training.schemas import (
    CURVE_CHANNELS,
    EVENT_CHANNELS,
    GLOBAL_BUCKET_CHANNELS,
    GRAPH_CHANNELS,
    MICRO_CHANNELS,
    OWN_CHANNELS,
    WALLET_PRIOR_CHANNELS,
)

DEEP_INPUT_SCHEMA_VERSION = "deep_input_finite_v1"

# Event channel indices (EVENT_CHANNELS layout)
_EVT_REL_DT = 0
_EVT_INTERARRIVAL = 1
_EVT_SIDE = 3
_EVT_SOL = 4
_EVT_V_SOL = 7

NEURAL_SEQ_INPUTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("event_x", "event_mask", EVENT_CHANNELS),
    ("curve_x", "curve_mask", CURVE_CHANNELS),
    ("global_buckets", "global_mask", GLOBAL_BUCKET_CHANNELS),
    ("micro_x", "micro_mask", MICRO_CHANNELS),
    ("own_x", "own_mask", OWN_CHANNELS),
    ("wallet_x", "wallet_mask", WALLET_PRIOR_CHANNELS),
    ("graph_edge_feat", "graph_mask", GRAPH_CHANNELS),
)

NEURAL_FIXED_INPUTS: tuple[str, ...] = ("creator_feat", "global_scalars")


class NonFiniteModelInputError(RuntimeError):
    """Raised when a non-finite value reaches the collated model-input boundary."""


def sanitize_feature_matrix(
    arr: np.ndarray,
    *,
    neutral: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (finite_array, orig_finite_mask, sanitized_count).

    ``orig_finite_mask`` is True where the *pre-sanitization* value was finite.
    Non-finite cells become ``neutral`` (transformed-space zero).
    """
    x = np.asarray(arr, dtype=np.float32)
    if x.size == 0:
        return x.copy(), np.zeros(x.shape, dtype=bool), 0
    finite = np.isfinite(x)
    sanitized = int((~finite).sum())
    out = x.copy()
    if sanitized:
        out[~finite] = np.float32(neutral)
    return out, finite, sanitized


def sanitize_vector(
    arr: np.ndarray, *, neutral: float = 0.0
) -> tuple[np.ndarray, np.ndarray, int]:
    return sanitize_feature_matrix(arr, neutral=neutral)


def build_event_ssl_from_raw(events_raw: np.ndarray) -> dict[str, torch.Tensor]:
    """Build SSL targets + per-objective masks from *pre-sanitization* events."""
    n = int(events_raw.shape[0])
    if n < 2:
        return {}
    raw = np.asarray(events_raw, dtype=np.float64)
    t_len = n - 1

    side = raw[1:, _EVT_SIDE]
    side_ok = np.isfinite(side)
    # Unknown side id 0 is still a valid categorical target when finite.
    ssl_next_side = np.where(side_ok, side, 0.0).astype(np.int64)

    inter = raw[1:, _EVT_INTERARRIVAL]
    rel = raw[:, _EVT_REL_DT]
    dt_diff = np.diff(rel)
    # Prefer stored interarrival when finite+positive; else finite positive Δrel_dt.
    use_inter = np.isfinite(inter) & (inter > 0)
    use_diff = (~use_inter) & np.isfinite(dt_diff) & (dt_diff > 0)
    delta = np.zeros(t_len, dtype=np.float64)
    delta[use_inter] = inter[use_inter]
    delta[use_diff] = dt_diff[use_diff]
    log_dt_ok = use_inter | use_diff
    ssl_next_log_dt = np.log1p(np.maximum(delta, 0.0)).astype(np.float32)
    ssl_next_log_dt[~log_dt_ok] = 0.0

    sol = raw[1:, _EVT_SOL]
    size_ok = np.isfinite(sol)
    ssl_next_size = np.where(size_ok, sol, 0.0).astype(np.float32)

    intensity_ok = np.isfinite(delta) & (delta > 0)
    ssl_intensity_delta_t = np.where(intensity_ok, delta, 1e-6).astype(np.float32)

    v0 = raw[:-1, _EVT_V_SOL]
    v1 = raw[1:, _EVT_V_SOL]
    reserve_ok = np.isfinite(v0) & np.isfinite(v1)
    ssl_reserve_delta = np.where(reserve_ok, v1 - v0, 0.0).astype(np.float32)

    # Legacy union mask (any objective support) for callers that still expect it.
    ssl_mask = side_ok | log_dt_ok | size_ok | intensity_ok | reserve_ok

    return {
        "ssl_next_side": torch.tensor(ssl_next_side, dtype=torch.long),
        "ssl_next_log_dt": torch.tensor(ssl_next_log_dt, dtype=torch.float32),
        "ssl_next_size": torch.tensor(ssl_next_size, dtype=torch.float32),
        "ssl_intensity_delta_t": torch.tensor(ssl_intensity_delta_t, dtype=torch.float32),
        "ssl_reserve_delta": torch.tensor(ssl_reserve_delta, dtype=torch.float32),
        "ssl_side_mask": torch.tensor(side_ok, dtype=torch.bool),
        "ssl_log_dt_mask": torch.tensor(log_dt_ok, dtype=torch.bool),
        "ssl_size_mask": torch.tensor(size_ok, dtype=torch.bool),
        "ssl_intensity_mask": torch.tensor(intensity_ok, dtype=torch.bool),
        "ssl_reserve_mask": torch.tensor(reserve_ok, dtype=torch.bool),
        "ssl_mask": torch.tensor(ssl_mask, dtype=torch.bool),
    }


def assert_finite_model_inputs(batch: dict[str, Any]) -> None:
    """Fail closed if any neural numerical input has non-finite values on valid positions."""
    for key, mask_key, channels in NEURAL_SEQ_INPUTS:
        if key not in batch:
            continue
        x = batch[key]
        if not torch.is_tensor(x) or x.numel() == 0:
            continue
        mask = batch.get(mask_key)
        bad = ~torch.isfinite(x)
        if mask is not None and torch.is_tensor(mask):
            # mask: [B, T] → broadcast over channels
            m = mask
            while m.ndim < x.ndim:
                m = m.unsqueeze(-1)
            bad = bad & m.expand_as(x).to(dtype=torch.bool)
        n_bad = int(bad.sum().item())
        if n_bad == 0:
            continue
        # Locate first offending channel
        ch_idx = None
        if x.ndim >= 3:
            # sum over batch,time
            per_ch = bad.reshape(-1, x.shape[-1]).sum(dim=0)
            ch_idx = int(torch.argmax(per_ch).item())
        ch_name = (
            channels[ch_idx]
            if ch_idx is not None and ch_idx < len(channels)
            else str(ch_idx)
        )
        raise NonFiniteModelInputError(
            f"non-finite model input tensor={key} channel={ch_name!r} "
            f"count={n_bad} schema={DEEP_INPUT_SCHEMA_VERSION}"
        )

    for key in NEURAL_FIXED_INPUTS:
        if key not in batch:
            continue
        x = batch[key]
        if not torch.is_tensor(x) or x.numel() == 0:
            continue
        bad = ~torch.isfinite(x)
        n_bad = int(bad.sum().item())
        if n_bad:
            raise NonFiniteModelInputError(
                f"non-finite model input tensor={key} count={n_bad} "
                f"schema={DEEP_INPUT_SCHEMA_VERSION}"
            )
