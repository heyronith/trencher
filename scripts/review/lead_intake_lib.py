"""Shared helpers for lead-intake read-only analysis (train + permanent_val only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRATCH = REPO / "scratch" / "lead_review_v1"
OUT = REPO / "manifests" / "world_model_data" / "evaluation" / "lead_review_v1"
WORLD = REPO / "data" / "external" / "world_model"
SPLIT = WORLD / "splits" / "split_v1"

WANT_PRED = "2d97c92e3bf6a9b721355d9ea2537b7a54787594ba0ee659105643e8200def13"
WANT_METRICS = "c4f0edcb35fd6e9cb9fcf465f4a601e3d93756c75dd0c8c2a13b7b983c2be741"
WANT_STAGE_B = "55b305fe602ac7cfeceab436ab87e960717e432229942c26e5a4c203505dfe75"
N_BOOT = 2000
BOOT_SEED = 0
EXPLOSION_HORIZONS = [10, 30, 60, 120, 300]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def oid_hash(oid: str) -> int:
    return int(hashlib.sha256(oid.encode("utf-8")).hexdigest()[:16], 16) % (2**63)


def softmax_pos_prob(logits: np.ndarray) -> np.ndarray:
    """Binary-ish: Stage-C explosion head is multi-horizon independent logits.

    Permanent-val metrics use per-horizon sigmoid of the horizon logit
    (see world_model evaluation). Use sigmoid of the selected horizon logit.
    """
    x = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def binary_auc_ap(y: np.ndarray, s: np.ndarray) -> dict[str, Any]:
    from trencher.baselines.metrics import binary_metrics

    return binary_metrics(y, s)


def topk_precision_recall(
    y: np.ndarray, s: np.ndarray, fracs: list[float]
) -> dict[str, dict[str, float]]:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    n = int(y.size)
    n_pos = int(y.sum())
    order = np.argsort(-s)
    out: dict[str, dict[str, float]] = {}
    for f in fracs:
        k = max(1, int(round(n * f)))
        top = order[:k]
        tp = int(y[top].sum())
        out[f"top_{f:g}"] = {
            "k": float(k),
            "precision": float(tp / k) if k else float("nan"),
            "recall": float(tp / n_pos) if n_pos else float("nan"),
        }
    return out


def mint_bootstrap_metric_delta(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    mints: np.ndarray,
    *,
    metric: str,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Paired Δmetric (b - a) with mint-level cluster bootstrap."""
    from trencher.baselines.metrics import binary_metrics
    from trencher.world_model.evaluation.bootstrap import clustered_indices

    y = np.asarray(y).astype(int)
    a = np.asarray(score_a, dtype=np.float64)
    b = np.asarray(score_b, dtype=np.float64)
    mints = np.asarray(mints)

    def _m(yy: np.ndarray, ss: np.ndarray) -> float | None:
        if yy.size == 0 or len(np.unique(yy)) < 2:
            return None
        met = binary_metrics(yy, ss)
        return float(met["roc_auc"] if metric == "roc_auc" else met["pr_auc"])

    point_a = _m(y, a)
    point_b = _m(y, b)
    if point_a is None or point_b is None:
        return {
            "metric": metric,
            "point_a": point_a,
            "point_b": point_b,
            "delta": None,
            "ci95": [None, None],
            "n": int(y.size),
            "n_mints": int(np.unique(mints).size),
            "n_boot": n_boot,
            "seed": seed,
        }
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = clustered_indices(mints, rng)
        pa = _m(y[idx], a[idx])
        pb = _m(y[idx], b[idx])
        if pa is None or pb is None:
            deltas[i] = np.nan
        else:
            deltas[i] = pb - pa
    deltas = deltas[np.isfinite(deltas)]
    lo = float(np.quantile(deltas, 0.025)) if deltas.size else float("nan")
    hi = float(np.quantile(deltas, 0.975)) if deltas.size else float("nan")
    return {
        "metric": metric,
        "point_a": point_a,
        "point_b": point_b,
        "delta": float(point_b - point_a),
        "ci95": [lo, hi],
        "n": int(y.size),
        "n_mints": int(np.unique(mints).size),
        "n_boot": n_boot,
        "seed": seed,
        "n_boot_finite": int(deltas.size),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(text, encoding="utf-8")
