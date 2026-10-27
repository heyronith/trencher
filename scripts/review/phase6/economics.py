"""Section 5: descriptive economic context (no strategy claims)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scripts.review.phase6.common import PHASE4_PREDS, Q85, SCRATCH, write_json


def _summ(x: np.ndarray) -> dict[str, Any]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": None, "median": None}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
    }


def section_economics() -> dict[str, Any]:
    preds = pq.read_table(PHASE4_PREDS)
    side = np.load(SCRATCH / "cache" / "val_side_fields.npz", allow_pickle=True)
    scores = preds["xgb_explosion_300s"].to_numpy(zero_copy_only=False).astype(np.float64)
    elig = preds["elig_explosion_300s"].to_numpy(zero_copy_only=False).astype(bool)
    thr = float(np.quantile(scores[elig], 0.99))
    alert = elig & (scores >= thr)

    ret = side["h300_simple_return"]
    mfe = side["h300_mfe"]
    mae = side["h300_mae"]
    ref = side["reference_price"]
    fmark = side["h300_future_mark"]
    fmark_t = side["h300_future_mark_time"]
    obs_t = side["observation_time_t"]

    out = {
        **Q85,
        "top1_threshold": thr,
        "n_alert": int(alert.sum()),
        "n_all_eligible": int(elig.sum()),
        "all_eligible": {
            "h300_simple_return": _summ(ret[elig]),
            "h300_mfe": _summ(mfe[elig]),
            "h300_mae": _summ(mae[elig]),
        },
        "top1_alerts": {
            "h300_simple_return": _summ(ret[alert]),
            "h300_mfe": _summ(mfe[alert]),
            "h300_mae": _summ(mae[alert]),
        },
        "entry_delay": {
            "status": "not_available",
            "reason": (
                "No next-trade-after-delay price columns exist in training_cache. "
                "Closest fields are reference_price (last trade ≤ T) and "
                "h300_future_mark / h300_future_mark_time (last mark in the 300s "
                "horizon, not T+2s/T+5s entry). Delayed entry at T+2s / T+5s "
                "cannot be computed from available freeze artifacts."
            ),
            "available_proxies": {
                "reference_price_summary_alerts": _summ(ref[alert]),
                "h300_future_mark_summary_alerts": _summ(fmark[alert]),
                "frac_alerts_with_future_mark_time_after_T": float(
                    np.mean(
                        np.isfinite(fmark_t[alert])
                        & np.isfinite(obs_t[alert])
                        & (fmark_t[alert] > obs_t[alert])
                    )
                )
                if alert.any()
                else None,
            },
        },
        "disclaimer": (
            "Descriptive only. Not a trading strategy, PnL, or capacity claim."
        ),
    }
    write_json(SCRATCH / "cache" / "section_economics.json", out)
    return out
