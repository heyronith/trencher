"""Join already_started / tte / returns overlays for broad permanent val."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scripts.review.phase6.common import PHASE4_PREDS, Q85, SCRATCH, SPLIT, write_json


def load_champion_preds() -> dict[str, Any]:
    preds = pq.read_table(PHASE4_PREDS)
    return {
        "observation_id": np.array(preds["observation_id"].to_pylist(), dtype=object),
        "mint": np.array(preds["mint"].to_pylist(), dtype=object),
        "table": preds,
        **Q85,
    }


def load_val_side_fields(oids: np.ndarray) -> dict[str, Any]:
    """Stream training_cache for already_started, tte, obs_t, returns/MFE/MAE, upside."""
    want = set(map(str, oids.tolist()))
    cols = [
        "observation_id",
        "explosion_already_started_at_t",
        "time_to_explosion_sec",
        "observation_time_t",
        "remaining_upside_to_explosion_peak",
        "h300_simple_return",
        "h300_mfe",
        "h300_mae",
        "h300_future_mark",
        "h300_future_mark_time",
        "reference_price",
        "reference_price_time",
    ]
    out: dict[str, dict[str, Any]] = {}
    cache = SPLIT / "training_cache_v1" / "broad" / "by_mint_partition"
    for part in sorted(cache.glob("*/joined.parquet")):
        schema = set(pq.read_schema(part).names)
        use = [c for c in cols if c in schema]
        if "observation_id" not in use:
            continue
        t = pq.read_table(part, columns=use)
        for row in t.to_pylist():
            oid = str(row["observation_id"])
            if oid not in want:
                continue
            out[oid] = row
    already = np.zeros(len(oids), dtype=bool)
    tte = np.full(len(oids), np.nan)
    obs_t = np.full(len(oids), np.nan)
    upside = np.full(len(oids), np.nan)
    h300_ret = np.full(len(oids), np.nan)
    h300_mfe = np.full(len(oids), np.nan)
    h300_mae = np.full(len(oids), np.nan)
    ref_px = np.full(len(oids), np.nan)
    future_mark = np.full(len(oids), np.nan)
    future_mark_t = np.full(len(oids), np.nan)
    for i, oid in enumerate(oids):
        row = out.get(str(oid))
        if not row:
            continue
        already[i] = bool(row.get("explosion_already_started_at_t") or False)
        if row.get("time_to_explosion_sec") is not None:
            tte[i] = float(row["time_to_explosion_sec"])
        if row.get("observation_time_t") is not None:
            obs_t[i] = float(row["observation_time_t"])
        if row.get("remaining_upside_to_explosion_peak") is not None:
            upside[i] = float(row["remaining_upside_to_explosion_peak"])
        if row.get("h300_simple_return") is not None:
            h300_ret[i] = float(row["h300_simple_return"])
        if row.get("h300_mfe") is not None:
            h300_mfe[i] = float(row["h300_mfe"])
        if row.get("h300_mae") is not None:
            h300_mae[i] = float(row["h300_mae"])
        if row.get("reference_price") is not None:
            ref_px[i] = float(row["reference_price"])
        if row.get("h300_future_mark") is not None:
            future_mark[i] = float(row["h300_future_mark"])
        if row.get("h300_future_mark_time") is not None:
            future_mark_t[i] = float(row["h300_future_mark_time"])
    result = {
        "already_started": already,
        "tte_sec": tte,
        "observation_time_t": obs_t,
        "remaining_upside": upside,
        "h300_simple_return": h300_ret,
        "h300_mfe": h300_mfe,
        "h300_mae": h300_mae,
        "reference_price": ref_px,
        "h300_future_mark": future_mark,
        "h300_future_mark_time": future_mark_t,
        "n_matched": int(sum(1 for o in oids if str(o) in out)),
        **Q85,
    }
    write_json(
        SCRATCH / "cache" / "val_side_fields_meta.json",
        {"n_oids": len(oids), "n_matched": result["n_matched"], **Q85},
    )
    return result


def variant_keep(
    elig: np.ndarray,
    y: np.ndarray,
    already: np.ndarray,
    tte: np.ndarray,
    mode: str,
) -> np.ndarray:
    keep = elig.astype(bool).copy()
    if mode == "all":
        return keep
    keep &= ~already
    if mode == "a_exclude_already_started":
        return keep
    pos = y.astype(bool)
    pos_bad = pos & (~np.isfinite(tte) | (tte < 5.0))
    return keep & (~pos_bad)
