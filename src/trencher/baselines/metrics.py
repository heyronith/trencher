"""Baseline metrics: ranking, calibration, regression, TTE, trading tables."""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or len(np.unique(y)) < 2:
        return None
    order = np.argsort(s)
    y = y[order]
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = np.arange(1, y.size + 1, dtype=np.float64)
    # average ranks for ties
    # Mann–Whitney
    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or y.sum() == 0:
        return None
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.clip(tp + fp, 1, None)
    # AP as sum of precision at each positive
    return float(np.sum(precision * y) / float(y.sum()))


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, *, n_bins: int = 15
) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(probs, dtype=np.float64), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if not np.any(m):
            continue
        ece += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def precision_recall_at_threshold(
    y_true: np.ndarray, probs: np.ndarray, thr: float
) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(probs) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "threshold": thr}


def topk_precision_recall(
    y_true: np.ndarray, scores: np.ndarray, fracs: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05)
) -> dict[str, dict[str, float]]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    n = y.size
    order = np.argsort(-s)
    y_sorted = y[order]
    out: dict[str, dict[str, float]] = {}
    for f in fracs:
        k = max(1, int(round(n * f)))
        top = y_sorted[:k]
        prec = float(top.mean()) if k else 0.0
        rec = float(top.sum() / y.sum()) if y.sum() else 0.0
        out[f"top_{f:g}"] = {"k": float(k), "precision": prec, "recall": rec}
    return out


def binary_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probs, dtype=np.float64)
    prev = float(y.mean()) if y.size else 0.0
    return {
        "n": int(y.size),
        "prevalence": prev,
        "pr_auc": average_precision(y, p),
        "roc_auc": _safe_auc(y, p),
        "brier": brier_score(y, p),
        "ece": expected_calibration_error(y, p),
        "at_0.5": precision_recall_at_threshold(y, p, 0.5),
        "topk": topk_precision_recall(y, p),
    }


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    # Spearman via rank correlation
    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a), dtype=np.float64)
        return ranks

    if y.size >= 2:
        rs = np.corrcoef(_rank(y), _rank(p))[0, 1]
        spearman = float(rs) if np.isfinite(rs) else None
    else:
        spearman = None
    dir_acc = float(np.mean(np.sign(p) == np.sign(y))) if y.size else 0.0
    return {
        "n": int(y.size),
        "mae": mae,
        "rmse": rmse,
        "spearman": spearman,
        "directional_accuracy": dir_acc,
    }


def pinball_loss(y_true: np.ndarray, pred: np.ndarray, quantile: float) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    u = y - p
    return float(np.mean(np.maximum(quantile * u, (quantile - 1) * u)))


def concordance_index(
    durations: np.ndarray, events: np.ndarray, scores: np.ndarray
) -> float | None:
    """Harrell C-index; higher score = higher risk (shorter expected time)."""
    d = np.asarray(durations, dtype=np.float64)
    e = np.asarray(events).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    n = d.size
    num = den = 0.0
    for i in range(n):
        if e[i] != 1:
            continue
        for j in range(n):
            if d[i] >= d[j]:
                continue
            # i fails before j's time
            den += 1.0
            if s[i] > s[j]:
                num += 1.0
            elif s[i] == s[j]:
                num += 0.5
    if den == 0:
        return None
    return float(num / den)


def decile_trading_table(
    scores: np.ndarray,
    *,
    gross_return: np.ndarray | None = None,
    entry_impact_adjusted_return: np.ndarray | None = None,
    curve_impact_adjusted_return: np.ndarray | None = None,
    impact_adj_return: np.ndarray | None = None,  # legacy alias → entry
    mfe: np.ndarray | None = None,
    mae: np.ndarray | None = None,
    explosion: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    s = np.asarray(scores, dtype=np.float64)
    n = s.size
    if n == 0:
        return []
    if entry_impact_adjusted_return is None and impact_adj_return is not None:
        entry_impact_adjusted_return = impact_adj_return
    order = np.argsort(-s)
    edges = [int(round(i * n / 10)) for i in range(11)]
    rows = []
    for d in range(10):
        idx = order[edges[d] : edges[d + 1]]
        if idx.size == 0:
            continue
        rec: dict[str, Any] = {"decile": d + 1, "n": int(idx.size)}
        if gross_return is not None:
            rec["gross_return_mean"] = float(np.nanmean(gross_return[idx]))
        if entry_impact_adjusted_return is not None:
            rec["entry_impact_adjusted_return_mean"] = float(
                np.nanmean(entry_impact_adjusted_return[idx])
            )
        if curve_impact_adjusted_return is not None:
            rec["curve_impact_adjusted_return_mean"] = float(
                np.nanmean(curve_impact_adjusted_return[idx])
            )
        if mfe is not None:
            rec["mfe_mean"] = float(np.nanmean(mfe[idx]))
        if mae is not None:
            rec["mae_mean"] = float(np.nanmean(mae[idx]))
        if explosion is not None:
            rec["explosion_rate"] = float(np.nanmean(explosion[idx]))
        rec["net_return"] = "unavailable_no_row_level_fees"
        rows.append(rec)
    return rows
