"""Paired Stage-B vs Stage-C (or ablation) comparisons on shared observations."""

from __future__ import annotations

from typing import Any

import numpy as np

from trencher.baselines.metrics import average_precision, brier_score
from trencher.world_model.evaluation.bootstrap import bootstrap_metric_delta, paired_delta_bootstrap
from trencher.world_model.evaluation.metrics import (
    _safe_auc,
    barrier_metrics,
    binary_masked_metrics,
    return_quantile_metrics,
)


def align_by_observation_hash(
    hashes_a: np.ndarray,
    hashes_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return index arrays into A and B for the intersection, sorted by hash."""
    a = np.asarray(hashes_a).reshape(-1)
    b = np.asarray(hashes_b).reshape(-1)
    map_b = {int(h): i for i, h in enumerate(b)}
    idx_a: list[int] = []
    idx_b: list[int] = []
    for i, h in enumerate(a):
        j = map_b.get(int(h))
        if j is not None:
            idx_a.append(i)
            idx_b.append(j)
    return np.asarray(idx_a, dtype=np.int64), np.asarray(idx_b, dtype=np.int64)


def population_compatibility_report(
    *,
    population_a: str,
    population_b: str,
    n_a: int,
    n_b: int,
    n_paired: int,
) -> dict[str, Any]:
    comparable = population_a == population_b and n_paired > 0
    return {
        "population_a": population_a,
        "population_b": population_b,
        "n_a": n_a,
        "n_b": n_b,
        "n_paired": n_paired,
        "comparable": comparable,
        "mismatch_note": None
        if comparable
        else (
            "Observation populations differ or intersection is empty; "
            "do not interpret aggregate deltas as paired scientific evidence."
        ),
    }


def compare_binary_head(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    mask: np.ndarray,
    *,
    cluster_ids: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    base_a = binary_masked_metrics(y, score_a, mask)
    base_b = binary_masked_metrics(y, score_b, mask)
    out: dict[str, Any] = {
        "metrics_a": base_a,
        "metrics_b": base_b,
        "deltas": {},
    }
    if not base_a.get("identifiable") or not base_b.get("identifiable"):
        out["deltas"] = {
            "auroc": None,
            "ap": None,
            "brier": None,
            "note": "non_identifiable_head",
        }
        return out

    def auc_fn(yy: np.ndarray, ss: np.ndarray) -> float | None:
        return _safe_auc(yy, ss)

    def ap_fn(yy: np.ndarray, ss: np.ndarray) -> float | None:
        return average_precision(yy, ss)

    def brier_fn(yy: np.ndarray, ss: np.ndarray) -> float | None:
        if yy.size == 0:
            return None
        return brier_score(yy, ss)

    out["deltas"]["auroc"] = bootstrap_metric_delta(
        y,
        score_a,
        score_b,
        mask,
        metric_fn=auc_fn,
        cluster_ids=cluster_ids,
        n_boot=n_boot,
        seed=seed,
    )
    out["deltas"]["ap"] = bootstrap_metric_delta(
        y,
        score_a,
        score_b,
        mask,
        metric_fn=ap_fn,
        cluster_ids=cluster_ids,
        n_boot=n_boot,
        seed=seed + 1,
    )
    out["deltas"]["brier"] = bootstrap_metric_delta(
        y,
        score_a,
        score_b,
        mask,
        metric_fn=brier_fn,
        cluster_ids=cluster_ids,
        n_boot=n_boot,
        seed=seed + 2,
    )
    return out


def compare_barrier_head(
    y: np.ndarray,
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    mask: np.ndarray,
    *,
    cluster_ids: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    ma = barrier_metrics(y, logits_a, mask)
    mb = barrier_metrics(y, logits_b, mask)
    m = np.asarray(mask).reshape(-1) > 0.5
    ya = np.asarray(y).reshape(-1)[m]
    pa = np.asarray(logits_a)
    pb = np.asarray(logits_b)
    if pa.ndim >= 2 and pa.shape[-1] == 3:
        pred_a = pa.reshape(-1, pa.shape[-1]).argmax(axis=-1)[m]
        pred_b = pb.reshape(-1, pb.shape[-1]).argmax(axis=-1)[m]
    else:
        pred_a = pa.reshape(-1)[m]
        pred_b = pb.reshape(-1)[m]
    correct_a = (pred_a == ya).astype(np.float64)
    correct_b = (pred_b == ya).astype(np.float64)
    clusters = None if cluster_ids is None else np.asarray(cluster_ids).reshape(-1)[m]
    delta = paired_delta_bootstrap(
        correct_a, correct_b, cluster_ids=clusters, n_boot=n_boot, seed=seed
    )
    return {"metrics_a": ma, "metrics_b": mb, "deltas": {"accuracy": delta}}


def compare_return_head(
    y: np.ndarray,
    quantiles_a: np.ndarray,
    quantiles_b: np.ndarray,
    mask: np.ndarray,
    *,
    taus: tuple[float, ...] | list[float],
    cluster_ids: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    ma = return_quantile_metrics(y, quantiles_a, mask, taus=taus)
    mb = return_quantile_metrics(y, quantiles_b, mask, taus=taus)
    m = np.asarray(mask).reshape(-1) > 0.5
    yy = np.asarray(y, dtype=np.float64).reshape(-1)[m]
    qa = np.asarray(quantiles_a, dtype=np.float64)[m]
    qb = np.asarray(quantiles_b, dtype=np.float64)[m]
    taus_l = [float(t) for t in taus]
    median_idx = min(range(len(taus_l)), key=lambda i: abs(taus_l[i] - 0.5))
    err_a = np.abs(qa[:, median_idx] - yy)
    err_b = np.abs(qb[:, median_idx] - yy)
    clusters = None if cluster_ids is None else np.asarray(cluster_ids).reshape(-1)[m]
    # Lower MAE is better: delta = B - A (negative means B improved).
    delta = paired_delta_bootstrap(
        err_a, err_b, cluster_ids=clusters, n_boot=n_boot, seed=seed
    )
    return {"metrics_a": ma, "metrics_b": mb, "deltas": {"median_mae": delta}}
