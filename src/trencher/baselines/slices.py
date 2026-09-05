"""Validation-only slice reports for baseline heads."""

from __future__ import annotations

from typing import Any

import numpy as np

from trencher.baselines.metrics import binary_metrics, regression_metrics


def _age_bucket(age: float | None) -> str:
    if age is None or not np.isfinite(age):
        return "age_unknown"
    a = float(age)
    if a < 30:
        return "age_<30s"
    if a < 120:
        return "age_30s_2m"
    if a < 600:
        return "age_2m_10m"
    if a < 3600:
        return "age_10m_1h"
    return "age_>=1h"


def _regime_bucket(t: float | None) -> str:
    if t is None or not np.isfinite(t):
        return "regime_unknown"
    # Corpus spans Jun–Jul 2026; bucket by UTC day-of-month-ish via unix
    # Keep coarse chronological thirds of train+val window via quantiles later;
    # here use week-of-corpus proxy from unix.
    day = int(float(t) // 86400)
    return f"day_{day}"


def slice_binary_metrics(
    y: np.ndarray,
    probs: np.ndarray,
    meta: list[dict[str, Any]],
    *,
    creator_flags: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """PR-AUC / prevalence by requested validation slices."""
    y = np.asarray(y).astype(int)
    p = np.asarray(probs, dtype=np.float64)
    n = y.size
    out: dict[str, Any] = {}

    def _group(label: str, mask: np.ndarray) -> None:
        if mask.sum() < 20:
            out[label] = {"n": int(mask.sum()), "note": "too_few"}
            return
        out[label] = binary_metrics(y[mask], p[mask])

    # token age
    age_keys = [_age_bucket(m.get("token_age_seconds")) for m in meta]
    for key in sorted(set(age_keys)):
        _group(f"token_age/{key}", np.array([a == key for a in age_keys], dtype=bool))

    # jul3 proximity
    jul = np.array(
        [bool(m.get("observation_during_jul3_outage")) for m in meta], dtype=bool
    )
    prox = np.array(
        [
            (
                float(m["outage_proximity_sec"]) < 86400
                if m.get("outage_proximity_sec") is not None
                else False
            )
            for m in meta
        ],
        dtype=bool,
    )
    _group("jul3/during_outage", jul)
    _group("jul3/within_1d", prox & ~jul)
    _group("jul3/far", ~prox & ~jul)

    # rich vs broad
    rich = np.array([bool(m.get("in_rich_subset")) for m in meta], dtype=bool)
    _group("modality/rich_subset", rich)
    _group("modality/broad_only", ~rich)

    # modality masks from feature row if present
    for mask_name in ("mask_events", "mask_global", "mask_creator", "mask_micro"):
        vals = []
        for m in meta:
            v = m.get(mask_name)
            vals.append(float(v) if v is not None else np.nan)
        arr = np.asarray(vals, dtype=np.float64)
        if np.isfinite(arr).any():
            _group(f"modality/{mask_name}=1", np.isfinite(arr) & (arr >= 0.5))
            _group(f"modality/{mask_name}=0", np.isfinite(arr) & (arr < 0.5))

    # creator overlap (mint-level flags)
    if creator_flags:
        unseen = []
        overlap = []
        for m in meta:
            mint = str(m.get("mint") or "")
            fl = creator_flags.get(mint) or {}
            unseen.append(bool(fl.get("eval_unseen_creator_vs_train")))
            overlap.append(bool(fl.get("eval_creator_overlap_vs_train")))
        _group("creator/unseen_vs_train", np.asarray(unseen, dtype=bool))
        _group("creator/overlap_vs_train", np.asarray(overlap, dtype=bool))

    # date/protocol regime (coarse day buckets — keep top populated)
    regimes = [_regime_bucket(m.get("observation_time_t")) for m in meta]
    counts: dict[str, int] = {}
    for r in regimes:
        counts[r] = counts.get(r, 0) + 1
    top_regimes = sorted(counts, key=counts.get, reverse=True)[:6]  # type: ignore[arg-type]
    for key in top_regimes:
        _group(f"regime/{key}", np.array([r == key for r in regimes], dtype=bool))

    # explosion prevalence regime via age×score not needed; report global prevalence
    out["prevalence_overall"] = float(y.mean()) if n else 0.0
    return out


def slice_regression_metrics(
    y: np.ndarray, pred: np.ndarray, meta: list[dict[str, Any]]
) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    out: dict[str, Any] = {}
    age_keys = [_age_bucket(m.get("token_age_seconds")) for m in meta]
    for key in sorted(set(age_keys)):
        mask = np.array([a == key for a in age_keys], dtype=bool)
        if mask.sum() < 20:
            out[f"token_age/{key}"] = {"n": int(mask.sum()), "note": "too_few"}
        else:
            out[f"token_age/{key}"] = regression_metrics(y[mask], p[mask])
    rich = np.array([bool(m.get("in_rich_subset")) for m in meta], dtype=bool)
    if rich.sum() >= 20:
        out["modality/rich_subset"] = regression_metrics(y[rich], p[rich])
    if (~rich).sum() >= 20:
        out["modality/broad_only"] = regression_metrics(y[~rich], p[~rich])
    return out


def load_creator_flags(path) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq

    if not path.exists():
        return {}
    table = pq.read_table(
        path,
        columns=[
            "mint",
            "eval_unseen_creator_vs_train",
            "eval_creator_overlap_vs_train",
        ],
    )
    out: dict[str, dict[str, Any]] = {}
    for r in table.to_pylist():
        mint = str(r.get("mint") or "")
        if mint:
            out[mint] = r
    return out
