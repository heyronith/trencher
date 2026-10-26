"""Faster mint-clustered paired ΔAUROC / ΔAP bootstrap for Phase-4 scale."""

from __future__ import annotations

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
) -> dict[str, Any]:
    """Paired Δmetric (b − a) with mint-level cluster bootstrap."""
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
    base = {
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

    # Pre-index rows by mint for O(n_mints) resample instead of O(n) scans.
    uniq, inv = np.unique(mints, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(uniq.size)]
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    finite = 0
    for i in range(n_boot):
        chosen = rng.integers(0, uniq.size, size=uniq.size)
        idx = np.concatenate([groups[j] for j in chosen])
        pa = _m(y[idx], a[idx])
        pb = _m(y[idx], b[idx])
        if pa is None or pb is None:
            deltas[i] = np.nan
            continue
        deltas[i] = pb - pa
        finite += 1
    ok = deltas[np.isfinite(deltas)]
    if ok.size:
        base["ci95"] = [float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))]
    base["n_boot_finite"] = int(finite)
    return base
