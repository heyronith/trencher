"""Head-level metrics for world-model permanent-validation predictions."""

from __future__ import annotations

from typing import Any

import numpy as np

from trencher.baselines.metrics import (
    average_precision,
    pinball_loss,
)


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0:
        return None
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    y_ord = y[order]
    ranks = np.empty(y.size, dtype=np.float64)
    i = 0
    while i < y.size:
        j = i + 1
        while j < y.size and s[order[j]] == s[order[i]]:
            j += 1
        avg = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg
        i = j
    sum_pos = float(ranks[y_ord == 1].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_masked_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Binary metrics with explicit non-identifiable marking for zero positives."""
    y = np.asarray(y_true).reshape(-1)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    m = np.asarray(mask).reshape(-1) > 0.5
    y = y[m].astype(np.int64)
    p = p[m]
    n = int(y.size)
    positives = int(y.sum()) if n else 0
    negatives = n - positives
    if n == 0:
        return {
            "support": 0,
            "positives": 0,
            "negatives": 0,
            "prevalence": None,
            "auroc": None,
            "ap": None,
            "average_precision": None,
            "brier": None,
            "identifiable": False,
            "non_identifiable_reason": "zero_support",
        }
    brier = float(np.mean((p - y.astype(np.float64)) ** 2))
    prevalence = float(positives / n)
    if positives == 0 or negatives == 0:
        return {
            "support": n,
            "positives": positives,
            "negatives": negatives,
            "prevalence": prevalence,
            "auroc": None,
            "ap": None,
            "average_precision": None,
            "brier": brier,
            "identifiable": False,
            "non_identifiable_reason": (
                "zero_positives" if positives == 0 else "zero_negatives"
            ),
        }
    auroc = _safe_auc(y, p)
    ap = average_precision(y, p)
    return {
        "support": n,
        "positives": positives,
        "negatives": negatives,
        "prevalence": prevalence,
        "auroc": auroc,
        "ap": ap,
        "average_precision": ap,
        "brier": brier,
        "identifiable": True,
        "non_identifiable_reason": None,
    }


def barrier_metrics(
    y_true: np.ndarray,
    logits_or_pred: np.ndarray,
    mask: np.ndarray,
    *,
    n_classes: int = 3,
) -> dict[str, Any]:
    y = np.asarray(y_true).reshape(-1)
    m = np.asarray(mask).reshape(-1) > 0.5
    pred_in = np.asarray(logits_or_pred)
    if pred_in.ndim >= 2 and pred_in.shape[-1] == n_classes:
        logits = pred_in.reshape(-1, pred_in.shape[-1])
        pred = logits.argmax(axis=-1)
        probs = _softmax(logits)
    else:
        pred = pred_in.reshape(-1)
        probs = None
    y = y[m].astype(np.int64)
    pred = pred[m].astype(np.int64)
    n = int(y.size)
    if n == 0:
        return {
            "support": 0,
            "accuracy": None,
            "macro_f1": None,
            "log_loss": None,
            "per_class_support": {},
            "identifiable": False,
            "non_identifiable_reason": "zero_support",
        }
    accuracy = float(np.mean(y == pred))
    per_class: dict[str, dict[str, Any]] = {}
    f1s: list[float] = []
    for c in range(n_classes):
        yt = y == c
        yp = pred == c
        support = int(yt.sum())
        tp = int((yt & yp).sum())
        fp = int((~yt & yp).sum())
        fn = int((yt & ~yp).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[str(c)] = {
            "support": support,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
        if support > 0:
            f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else None
    logloss = None
    if probs is not None:
        probs_m = probs[m]
        clipped = np.clip(probs_m[np.arange(n), y], 1e-12, 1.0)
        logloss = float(-np.mean(np.log(clipped)))
    return {
        "support": n,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "log_loss": logloss,
        "per_class_support": per_class,
        "identifiable": True,
        "non_identifiable_reason": None,
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)


def return_quantile_metrics(
    y_true: np.ndarray,
    quantiles: np.ndarray,
    mask: np.ndarray,
    *,
    taus: tuple[float, ...] | list[float],
) -> dict[str, Any]:
    """``quantiles`` shaped [N, Q]; ``y_true`` / ``mask`` shaped [N]."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    m = np.asarray(mask).reshape(-1) > 0.5
    q = np.asarray(quantiles, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"quantiles must be [N,Q], got shape={q.shape}")
    y = y[m]
    q = q[m]
    n = int(y.size)
    if n == 0:
        return {
            "support": 0,
            "median_mae": None,
            "pinball": {},
            "identifiable": False,
            "non_identifiable_reason": "zero_support",
        }
    taus_l = [float(t) for t in taus]
    median_idx = min(range(len(taus_l)), key=lambda i: abs(taus_l[i] - 0.5))
    median_mae = float(np.mean(np.abs(q[:, median_idx] - y)))
    pinball = {
        str(t): float(pinball_loss(y, q[:, i], t)) for i, t in enumerate(taus_l)
    }
    return {
        "support": n,
        "median_mae": median_mae,
        "pinball": pinball,
        "identifiable": True,
        "non_identifiable_reason": None,
    }


def mae_masked(y_true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    m = np.asarray(mask).reshape(-1) > 0.5
    y = y[m]
    p = p[m]
    if y.size == 0:
        return {
            "support": 0,
            "mae": None,
            "identifiable": False,
            "non_identifiable_reason": "zero_support",
        }
    return {
        "support": int(y.size),
        "mae": float(np.mean(np.abs(p - y))),
        "identifiable": True,
        "non_identifiable_reason": None,
    }


def summarize_prediction_artifact(
    outputs: dict[str, Any],
    targets: dict[str, Any],
    *,
    explosion_horizons: tuple[int, ...] | list[int],
    survival_horizons: tuple[int, ...] | list[int],
    return_horizons: tuple[int, ...] | list[int],
    quantile_taus: tuple[float, ...] | list[float],
    execution_sizes: tuple[float, ...] | list[float] | None = None,
) -> dict[str, Any]:
    """Compute multi-head metrics from concatenated prediction tensors/arrays."""

    def _np(x: Any) -> np.ndarray:
        if hasattr(x, "detach"):
            return x.detach().float().cpu().numpy()
        return np.asarray(x)

    metrics: dict[str, Any] = {
        "explosion": {},
        "survival": {},
        "returns": {},
        "barriers": {},
        "execution": {},
        "path": {},
        "persistence": {},
    }

    if "explosion_logits" in outputs and "explosion_target" in targets:
        probs = 1.0 / (1.0 + np.exp(-_np(outputs["explosion_logits"])))
        y = _np(targets["explosion_target"])
        m = _np(targets["explosion_mask"])
        for i, h in enumerate(explosion_horizons):
            metrics["explosion"][str(h)] = binary_masked_metrics(
                y[:, i], probs[:, i], m[:, i]
            )

    if "survival_horizon_logits" in outputs and "survival_horizon_target" in targets:
        probs = 1.0 / (1.0 + np.exp(-_np(outputs["survival_horizon_logits"])))
        y = _np(targets["survival_horizon_target"])
        m = _np(targets["survival_horizon_mask"])
        for i, h in enumerate(survival_horizons):
            metrics["survival"][str(h)] = binary_masked_metrics(
                y[:, i], probs[:, i], m[:, i]
            )

    if "barrier_logits" in outputs and "barrier_target" in targets:
        metrics["barriers"] = barrier_metrics(
            _np(targets["barrier_target"]),
            _np(outputs["barrier_logits"]),
            _np(targets["barrier_mask"]),
        )

    if "return_quantiles" in outputs and "return_target" in targets:
        rq = _np(outputs["return_quantiles"])
        y = _np(targets["return_target"])
        m = _np(targets["return_mask"])
        for i, h in enumerate(return_horizons):
            metrics["returns"][str(h)] = return_quantile_metrics(
                y[:, i], rq[:, i, :], m[:, i], taus=quantile_taus
            )

    if (
        execution_sizes
        and "execution" in outputs
        and "execution_target" in targets
        and "execution_mask" in targets
    ):
        pred = _np(outputs["execution"])
        y = _np(targets["execution_target"])
        m = _np(targets["execution_mask"])
        kinds = ["gross", "entry_impact", "curve_impact"]
        for si, size in enumerate(execution_sizes):
            metrics["execution"][str(size)] = {
                kinds[ki]: mae_masked(y[:, si, ki], pred[:, si, ki], m[:, si, ki])
                for ki in range(len(kinds))
            }

    if "persistence_pred" in outputs and "persistence_target" in targets:
        metrics["persistence"] = mae_masked(
            _np(targets["persistence_target"]),
            _np(outputs["persistence_pred"]),
            _np(
                targets.get(
                    "persistence_mask",
                    np.ones_like(_np(targets["persistence_target"])),
                )
            ),
        )

    if "path_mean" in outputs and "path_target" in targets:
        metrics["path"] = mae_masked(
            _np(targets["path_target"]),
            _np(outputs["path_mean"]),
            _np(targets.get("path_mask", np.ones_like(_np(targets["path_target"])))),
        )

    return metrics
