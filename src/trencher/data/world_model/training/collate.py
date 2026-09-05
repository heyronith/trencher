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
    dims = max(
        (int(s.shape[-1]) for s in sequences if getattr(s, "ndim", 0) == 2 and s.size),
        default=1,
    )
    lengths = [int(s.shape[0]) if getattr(s, "ndim", 0) >= 1 else 0 for s in sequences]
    lmax = max(lengths) if lengths else 0
    b = len(sequences)
    out = np.full((b, lmax, dims), pad_value, dtype=np.float32)
    mask = np.zeros((b, lmax), dtype=bool)
    for i, seq in enumerate(sequences):
        arr = np.asarray(seq, dtype=np.float32)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr[:, None]
        n = arr.shape[0]
        d = min(arr.shape[1], dims)
        out[i, :n, :d] = arr[:, :d]
        mask[i, :n] = True
    return out, mask


def causal_window_slice(
    times: list[float],
    *,
    observation_time_t: float,
    preset: dict[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    """Apply documented causal window; never silently truncate without metadata."""
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
    """Collate feature/target/sequence dicts into batch tensors + masks."""
    for ex in examples:
        assert_no_forbidden(ex["features"])

    feature_keys = sorted({k for ex in examples for k in ex["features"]})
    b = len(examples)
    features = {
        k: np.asarray(
            [
                np.nan if ex["features"].get(k) is None else float(ex["features"][k])
                for ex in examples
            ],
            dtype=np.float32,
        )
        for k in feature_keys
    }

    # Sequences (production)
    seq_batch: dict[str, Any] = {}
    length_stats: dict[str, Any] = {}
    trunc_rates: dict[str, float] = {}
    seq_keys = sorted({k for ex in examples for k in (ex.get("sequences") or {})})
    for sk in seq_keys:
        seqs = [
            np.asarray((ex.get("sequences") or {}).get(sk, np.zeros((0, 1))), np.float32)
            for ex in examples
        ]
        padded, mask = pad_sequences(seqs)
        seq_batch[sk] = padded
        seq_batch[f"{sk}_padding_mask"] = mask
        lengths = [int(s.shape[0]) if s.ndim else 0 for s in seqs]
        length_stats[sk] = {
            "mean": float(np.mean(lengths)) if lengths else 0.0,
            "max": int(max(lengths) if lengths else 0),
            "min": int(min(lengths) if lengths else 0),
        }
    for sk in ("events", "curve", "creator", "micro", "ownership", "graph"):
        rates = [1.0 if (ex.get("truncation") or {}).get(sk) else 0.0 for ex in examples]
        trunc_rates[sk] = float(np.mean(rates)) if rates else 0.0

    # Targets: support nested production packs OR flat synthetic dicts
    nested_targets = [ex.get("targets") or {} for ex in examples]
    flat_targets: dict[str, Any] = {}
    sample = nested_targets[0] if nested_targets else {}
    is_nested = any(
        isinstance(sample.get(k), dict)
        and (
            "value" in sample[k]
            or "is_missing" in sample[k]
            or "effective_event_observed" in sample[k]
        )
        for k in sample
    )
    if is_nested:
        for head in ("returns_h60", "explosion_imminence_h60", "survival_h60"):
            vals: list[float] = []
            miss: list[bool] = []
            for t in nested_targets:
                pack = t.get(head) or {}
                v = pack.get("value_or_nan", pack.get("value"))
                if v is None or isinstance(v, (str, bool)):
                    vals.append(np.nan)
                else:
                    vals.append(float(v))
                miss.append(bool(pack.get("is_missing", v is None)))
            flat_targets[head] = np.asarray(vals, dtype=np.float32)
            flat_targets[f"{head}_is_missing"] = np.asarray(miss, dtype=bool)
        targets_out: Any = nested_targets
    else:
        # Flat synthetic path (unit tests)
        target_keys = sorted({k for t in nested_targets for k in t})
        targets_out = {}
        for k in target_keys:
            vals = [t.get(k) for t in nested_targets]
            if all(v is None or isinstance(v, (int, float, np.floating)) for v in vals):
                arr = np.array(
                    [np.nan if v is None else float(v) for v in vals], dtype=np.float32
                )
                targets_out[k] = arr
                targets_out[f"{k}_is_missing"] = np.isnan(arr)
            else:
                targets_out[k] = vals

    modality = {k: features[k] for k in features if k.startswith("modality_mask_")}
    return {
        "batch_size": b,
        "features": features,
        "sequences": seq_batch,
        "sequence_length_stats": length_stats,
        "truncation_rates": trunc_rates,
        "targets": targets_out,
        "flat_targets": flat_targets,
        "modality_masks": modality,
        "causal_ok": all(bool(ex.get("causal_ok", True)) for ex in examples),
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
        return (
            np.zeros((0, len(value_keys) + 1), dtype=np.float32),
            np.zeros((0,), dtype=bool),
            wmeta,
        )
    t0 = float(sliced[0][time_key])
    feats = []
    for r in sliced:
        t = float(r[time_key])
        vec = [t - t0]
        for k in value_keys:
            v = r.get(k)
            vec.append(float(v) if v is not None else np.nan)
        feats.append(vec)
    arr = np.asarray(feats, dtype=np.float32)
    mask = np.ones((arr.shape[0],), dtype=bool)
    return arr, mask, wmeta
