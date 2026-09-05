"""Batch collation with padding / modality / target-status masks."""

from __future__ import annotations

from typing import Any

import numpy as np

from trencher.data.world_model.sequences import apply_training_window
from trencher.data.world_model.training.forbid import assert_no_forbidden


def pad_sequences(
    sequences: list[np.ndarray], *, pad_value: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Pad variable-length [L, D] sequences → [B, Lmax, D] + padding mask [B, Lmax]."""
    if not sequences:
        return np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0, 0), dtype=bool)
    dims = sequences[0].shape[-1] if sequences[0].ndim == 2 else 1
    lengths = [int(s.shape[0]) for s in sequences]
    lmax = max(lengths) if lengths else 0
    b = len(sequences)
    out = np.full((b, lmax, dims), pad_value, dtype=np.float32)
    mask = np.zeros((b, lmax), dtype=bool)
    for i, seq in enumerate(sequences):
        arr = np.asarray(seq, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        n = arr.shape[0]
        out[i, :n] = arr
        mask[i, :n] = True
    return out, mask


def causal_window_slice(
    times: list[float],
    *,
    observation_time_t: float,
    preset: dict[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    """Apply documented causal window; never silently truncate without metadata."""
    len(times)
    # end_idx is exclusive upper bound of events with time <= T
    from bisect import bisect_right

    end_idx = bisect_right(times, observation_time_t)
    lo, hi = apply_training_window(
        times=times,
        start_idx=0,
        end_idx=end_idx,
        observation_time_t=observation_time_t,
        preset=preset,
    )
    meta = {
        "window_preset": preset.get("name"),
        "window_kind": preset.get("kind"),
        "raw_len": end_idx,
        "window_lo": lo,
        "window_hi": hi,
        "capped": hi - lo < end_idx,
        "policy": "most_recent_N_or_last_seconds_causal",
    }
    return lo, hi, meta


def collate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate feature/target dicts into batch tensors + masks."""
    for ex in examples:
        assert_no_forbidden(ex["features"])

    feature_keys = sorted({k for ex in examples for k in ex["features"]})
    b = len(examples)
    features = {}
    for k in feature_keys:
        col = []
        for ex in examples:
            v = ex["features"].get(k)
            col.append(np.nan if v is None else float(v))
        features[k] = np.asarray(col, dtype=np.float32)

    target_keys = sorted({k for ex in examples for k in ex["targets"]})
    targets: dict[str, Any] = {}
    for k in target_keys:
        vals = []
        for ex in examples:
            v = ex["targets"].get(k)
            if isinstance(v, (bool, np.bool_)) or v is None or isinstance(v, str):
                vals.append(v)
            else:
                vals.append(v)
        # Keep list for mixed None/str; numeric → array with NaN for missing
        if all(v is None or isinstance(v, (int, float, np.floating)) for v in vals):
            arr = np.array(
                [np.nan if v is None else float(v) for v in vals], dtype=np.float32
            )
            targets[k] = arr
            targets[f"{k}_is_missing"] = np.isnan(arr)
        else:
            targets[k] = vals

    modality = {
        k: features[k]
        for k in features
        if k.startswith("modality_mask_")
    }
    return {
        "batch_size": b,
        "features": features,
        "targets": targets,
        "modality_masks": modality,
        "meta": [ex.get("meta") for ex in examples],
    }


def make_numeric_sequence_features(
    rows: list[dict[str, Any]],
    *,
    time_key: str = "event_time_unix",
    value_keys: list[str],
    observation_time_t: float,
    preset: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build [L, D] features with time-deltas; identities already excluded."""
    times = [float(r[time_key]) for r in rows]
    lo, hi, wmeta = causal_window_slice(
        times, observation_time_t=observation_time_t, preset=preset
    )
    sliced = rows[lo:hi]
    if not sliced:
        return np.zeros((0, len(value_keys) + 1), dtype=np.float32), np.zeros(
            (0,), dtype=bool
        ), wmeta
    t0 = float(sliced[0][time_key])
    feats = []
    for r in sliced:
        t = float(r[time_key])
        vec = [t - t0]  # time delta from window start
        for k in value_keys:
            v = r.get(k)
            vec.append(float(v) if v is not None else np.nan)
        feats.append(vec)
    arr = np.asarray(feats, dtype=np.float32)
    mask = np.ones((arr.shape[0],), dtype=bool)
    return arr, mask, wmeta
