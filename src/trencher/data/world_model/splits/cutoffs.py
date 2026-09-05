"""Shared chronological split cutoffs (launch-time quantiles)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np

# Permanent holdout + walk-forward interior cuts (fractions of full event_curve mass).
QUANTILE_KEYS: tuple[tuple[str, float], ...] = (
    ("q40", 0.40),
    ("q55", 0.55),
    ("q70", 0.70),
    ("q85", 0.85),
)

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
BASE_SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)


def _iso(unix: float) -> str:
    return datetime.fromtimestamp(float(unix), tz=UTC).isoformat()


def compute_launch_cutoffs(detected_at_unix: np.ndarray) -> dict[str, Any]:
    """Deterministic cutoffs from launch-time distribution (not prevalence-tuned).

    Uses exact quantiles of the provided launch times. Tie policy is applied later
    at assignment time; this function only records the quantile timestamps.
    """
    arr = np.asarray(detected_at_unix, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("empty launch-time array")
    arr.sort()
    quantiles: dict[str, float] = {}
    for name, q in QUANTILE_KEYS:
        quantiles[name] = float(np.quantile(arr, q, method="linear"))
    return {
        "method": "global_quantiles_of_event_curve_detected_at_unix",
        "n_tokens_used_for_quantiles": int(arr.size),
        "launch_min_unix": float(arr[0]),
        "launch_max_unix": float(arr[-1]),
        "launch_min_iso": _iso(arr[0]),
        "launch_max_iso": _iso(arr[-1]),
        "quantiles_unix": quantiles,
        "quantiles_iso": {k: _iso(v) for k, v in quantiles.items()},
        "permanent_split_intervals": {
            SPLIT_TRAIN: {
                "start_unix": float(arr[0]),
                "end_unix_exclusive": quantiles["q70"],
                "rule": "[launch_min, q70)",
            },
            SPLIT_VAL: {
                "start_unix": quantiles["q70"],
                "end_unix_exclusive": quantiles["q85"],
                "rule": "[q70, q85)",
            },
            SPLIT_TEST: {
                "start_unix": quantiles["q85"],
                "end_unix_exclusive": None,
                "rule": "[q85, +inf)",
            },
        },
        "tie_policy": (
            "tokens sharing an exact detected_at_unix stay on the same side of "
            "each boundary; report actual resulting percentages"
        ),
    }


def assign_base_split(detected_at_unix: float, *, q70: float, q85: float) -> str:
    t = float(detected_at_unix)
    if t < q70:
        return SPLIT_TRAIN
    if t < q85:
        return SPLIT_VAL
    return SPLIT_TEST


def split_end_exclusive(
    split: str, *, q70: float, q85: float, test_open_end: float
) -> float:
    """Exclusive end of the split's usable time interval for purge checks."""
    if split == SPLIT_TRAIN:
        return float(q70)
    if split == SPLIT_VAL:
        return float(q85)
    return float(test_open_end)


def split_start_inclusive(split: str, *, launch_min: float, q70: float, q85: float) -> float:
    if split == SPLIT_TRAIN:
        return float(launch_min)
    if split == SPLIT_VAL:
        return float(q70)
    return float(q85)
