"""Clustered paired bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def clustered_indices(
    cluster_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample observation indices by sampling clusters with replacement."""
    clusters = np.asarray(cluster_ids)
    uniq = np.unique(clusters)
    chosen = rng.choice(uniq, size=uniq.size, replace=True)
    parts: list[np.ndarray] = []
    for c in chosen:
        parts.append(np.flatnonzero(clusters == c))
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(parts)


def paired_delta_bootstrap(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    *,
    cluster_ids: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    aggregator: Callable[[np.ndarray], float] | None = None,
) -> dict[str, Any]:
    """Bootstrap CI for mean(metric_b - metric_a) with optional cluster resampling.

    ``metric_a`` / ``metric_b`` are per-observation contributions already aligned.
    For AUROC-style global metrics, pass precomputed per-bootstrap values via a
    custom path; this helper covers additive metrics (Brier, MAE, accuracy, etc.).
    """
    a = np.asarray(metric_a, dtype=np.float64).reshape(-1)
    b = np.asarray(metric_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"paired shapes differ: {a.shape} vs {b.shape}")
    n = int(a.size)
    if n == 0:
        return {
            "n_paired": 0,
            "point_estimate_a": None,
            "point_estimate_b": None,
            "delta": None,
            "relative_delta": None,
            "ci95": [None, None],
            "n_boot": n_boot,
            "seed": seed,
            "clustered": cluster_ids is not None,
            "n_clusters": 0,
        }
    agg = aggregator or (lambda x: float(np.mean(x)))
    point_a = agg(a)
    point_b = agg(b)
    delta = float(point_b - point_a)
    rel = None if point_a == 0 else float(delta / abs(point_a))
    rng = np.random.default_rng(seed)
    if cluster_ids is None:
        clusters = np.arange(n)
    else:
        clusters = np.asarray(cluster_ids)
        if clusters.shape[0] != n:
            raise ValueError("cluster_ids length must match paired observations")
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = clustered_indices(clusters, rng)
        deltas[i] = agg(b[idx]) - agg(a[idx])
    lo = float(np.quantile(deltas, alpha / 2.0))
    hi = float(np.quantile(deltas, 1.0 - alpha / 2.0))
    return {
        "n_paired": n,
        "point_estimate_a": point_a,
        "point_estimate_b": point_b,
        "delta": delta,
        "relative_delta": rel,
        "ci95": [lo, hi],
        "n_boot": n_boot,
        "seed": seed,
        "clustered": True,
        "n_clusters": int(np.unique(clusters).size),
    }


def bootstrap_metric_delta(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    mask: np.ndarray,
    *,
    metric_fn: Callable[[np.ndarray, np.ndarray], float | None],
    cluster_ids: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap CI for metric(score_b) - metric(score_a) on masked rows."""
    m = np.asarray(mask).reshape(-1) > 0.5
    y_m = np.asarray(y).reshape(-1)[m]
    a_m = np.asarray(score_a, dtype=np.float64).reshape(-1)[m]
    b_m = np.asarray(score_b, dtype=np.float64).reshape(-1)[m]
    n = int(y_m.size)
    clusters = (
        np.arange(n) if cluster_ids is None else np.asarray(cluster_ids).reshape(-1)[m]
    )
    point_a = metric_fn(y_m, a_m)
    point_b = metric_fn(y_m, b_m)
    if point_a is None or point_b is None or n == 0:
        return {
            "n_paired": n,
            "point_estimate_a": point_a,
            "point_estimate_b": point_b,
            "delta": None,
            "relative_delta": None,
            "ci95": [None, None],
            "n_boot": n_boot,
            "seed": seed,
            "clustered": True,
            "n_clusters": int(np.unique(clusters).size) if n else 0,
            "identifiable": False,
        }
    delta = float(point_b - point_a)
    rel = None if point_a == 0 else float(delta / abs(point_a))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = clustered_indices(clusters, rng)
        va = metric_fn(y_m[idx], a_m[idx])
        vb = metric_fn(y_m[idx], b_m[idx])
        if va is None or vb is None:
            deltas[i] = np.nan
        else:
            deltas[i] = float(vb - va)
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        lo = hi = None
    else:
        lo = float(np.quantile(finite, alpha / 2.0))
        hi = float(np.quantile(finite, 1.0 - alpha / 2.0))
    return {
        "n_paired": n,
        "point_estimate_a": float(point_a),
        "point_estimate_b": float(point_b),
        "delta": delta,
        "relative_delta": rel,
        "ci95": [lo, hi],
        "n_boot": n_boot,
        "seed": seed,
        "clustered": True,
        "n_clusters": int(np.unique(clusters).size),
        "identifiable": True,
    }
