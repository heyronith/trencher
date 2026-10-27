"""Parallel mint-cluster bootstrap for Phase-6 scale (~1.2M rows)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def mint_bootstrap_delta(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    mints: np.ndarray,
    *,
    metric: str = "roc_auc",
    n_boot: int = 2000,
    seed: int = 0,
    n_workers: int = 8,
) -> dict[str, Any]:
    """Paired Δmetric (b − a); mint-cluster bootstrap with threaded reps."""
    y = np.asarray(y).astype(np.int8)
    a = np.asarray(score_a, dtype=np.float64)
    b = np.asarray(score_b, dtype=np.float64)
    mints = np.asarray(mints)

    def _m(yy: np.ndarray, ss: np.ndarray) -> float | None:
        if yy.size == 0 or yy.min() == yy.max():
            return None
        if metric == "roc_auc":
            return float(roc_auc_score(yy, ss))
        return float(average_precision_score(yy, ss))

    point_a = _m(y, a)
    point_b = _m(y, b)
    n_mints = int(np.unique(mints).size)
    base: dict[str, Any] = {
        "metric": metric,
        "point_a": point_a,
        "point_b": point_b,
        "delta": None if point_a is None or point_b is None else float(point_b - point_a),
        "ci95": [None, None],
        "n": int(y.size),
        "n_mints": n_mints,
        "n_boot": n_boot,
        "seed": seed,
    }
    if point_a is None or point_b is None:
        return base

    uniq, inv = np.unique(mints, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(uniq.size)]
    # Pre-generate all resample indices with a single RNG for reproducibility
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, uniq.size, size=(n_boot, uniq.size))

    def _one(i: int) -> float:
        idx = np.concatenate([groups[j] for j in choices[i]])
        pa = _m(y[idx], a[idx])
        pb = _m(y[idx], b[idx])
        if pa is None or pb is None:
            return float("nan")
        return float(pb - pa)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        deltas = list(pool.map(_one, range(n_boot), chunksize=max(1, n_boot // (n_workers * 4))))
    arr = np.asarray(deltas, dtype=np.float64)
    ok = arr[np.isfinite(arr)]
    if ok.size:
        base["ci95"] = [float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))]
    base["n_boot_finite"] = int(ok.size)
    return base
