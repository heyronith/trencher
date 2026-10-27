"""Section 2: pump-level / lead-time / alert-feed eval of frozen V2 tree champion."""

from __future__ import annotations

import resource
import time
from collections import defaultdict
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scripts.review.phase5.common import (
    PHASE4_PREDS,
    Q85,
    SCRATCH,
    SPLIT,
    WORLD,
    metrics_block,
    write_json,
)


def _rss_mb() -> float:
    # ru_maxrss is bytes on Linux, KiB on macOS.
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if rss > 1e9:  # already bytes (unlikely for maxrss units)
        return rss / (1024 * 1024)
    return rss / 1024.0  # KiB -> MiB on Darwin


def join_tte_overlays(val_oids: set[str]) -> dict[str, Any]:
    """Predicate-pushdown join of broad TTE overlays onto val observation_ids."""
    t0 = time.time()
    rss0 = _rss_mb()
    cols = [
        "observation_id",
        "mint",
        "observation_time_t",
        "base_split",
        "explosion_tte_effective_tte_sec",
        "explosion_tte_effective_event_observed",
        "explosion_tte_raw_tte_sec",
        "explosion_tte_raw_event_observed",
    ]
    overlay_dir = SPLIT / "tte_overlays_broad"
    chunks: list[pa.Table] = []
    n_scanned = 0
    for part in sorted(overlay_dir.glob("part-*.parquet")):
        # Read only needed columns; filter val split then oid set in-memory
        # (pyarrow dataset filter on base_split is pushdown-friendly).
        t = pq.read_table(part, columns=cols, filters=[("base_split", "==", "val")])
        n_scanned += t.num_rows
        if t.num_rows == 0:
            continue
        # Keep only oids present in preds
        oid_arr = t["observation_id"]
        # Build boolean mask via set membership — chunked for memory
        keep = [str(o) in val_oids for o in oid_arr.to_pylist()]
        if not any(keep):
            continue
        chunks.append(t.filter(pa.array(keep)))
    table = pa.concat_tables(chunks) if chunks else pa.table({c: [] for c in cols})
    elapsed = time.time() - t0
    rss1 = _rss_mb()
    return {
        "table": table,
        "stats": {
            "elapsed_sec": float(elapsed),
            "rss_mb_before": rss0,
            "rss_mb_after": rss1,
            "rss_delta_mb": float(rss1 - rss0),
            "n_overlay_val_rows_scanned": int(n_scanned),
            "n_joined": int(table.num_rows),
            "n_pred_oids": int(len(val_oids)),
            "columns_read": cols,
            "predicate": "base_split == 'val' then observation_id in pred set",
            **Q85,
        },
    }


def load_already_started(val_oids: set[str]) -> dict[str, dict[str, Any]]:
    """Stream training_cache broad partitions for already_started + tte."""
    t0 = time.time()
    rss0 = _rss_mb()
    out: dict[str, dict[str, Any]] = {}
    cache = SPLIT / "training_cache_v1" / "broad" / "by_mint_partition"
    cols = [
        "observation_id",
        "explosion_already_started_at_t",
        "time_to_explosion_sec",
        "observation_time_t",
    ]
    n_parts = 0
    for part in sorted(cache.glob("*/joined.parquet")):
        n_parts += 1
        schema = set(pq.read_schema(part).names)
        use = [c for c in cols if c in schema]
        # Always need observation_id
        if "observation_id" not in use:
            continue
        t = pq.read_table(part, columns=use)
        for row in t.to_pylist():
            oid = str(row["observation_id"])
            if oid not in val_oids:
                continue
            out[oid] = {
                "already_started": bool(row.get("explosion_already_started_at_t") or False),
                "tte_sec": (
                    float(row["time_to_explosion_sec"])
                    if row.get("time_to_explosion_sec") is not None
                    else float("nan")
                ),
                "observation_time_t": (
                    float(row["observation_time_t"])
                    if row.get("observation_time_t") is not None
                    else float("nan")
                ),
            }
    return {
        "by_oid": out,
        "stats": {
            "elapsed_sec": float(time.time() - t0),
            "rss_mb_before": rss0,
            "rss_mb_after": _rss_mb(),
            "n_parts": n_parts,
            "n_matched": len(out),
            **Q85,
        },
    }


def _variant_keep(
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
    # c_onset_ge_5s
    pos = y.astype(bool)
    pos_bad = pos & (~np.isfinite(tte) | (tte < 5.0))
    return keep & (~pos_bad)


def pump_level_recall(
    *,
    oids: np.ndarray,
    mints: np.ndarray,
    scores: np.ndarray,
    obs_t: np.ndarray,
    top_fracs: tuple[float, ...] = (0.01, 0.02),
) -> dict[str, Any]:
    """Fraction of primary-definition val pumps caught at top-k, with lead times."""
    tokens = pq.read_table(SPLIT / "token_assignments.parquet")
    mint_split = {
        m: s
        for m, s in zip(tokens["mint"].to_pylist(), tokens["base_split"].to_pylist(), strict=True)
    }
    ev = pq.read_table(WORLD / "targets/target_dataset_v3/explosion_events_v1.parquet")
    prim = ev.filter(
        (pc.field("definition_id") == "explosion_primary_v1")
        & (pc.field("explosion_exists") == True)  # noqa: E712
    )
    ev_mints = prim["mint"].to_pylist()
    onset = prim["onset_unix"].to_pylist()
    val_events = [
        {"mint": ev_mints[i], "onset_unix": float(onset[i]) if onset[i] is not None else None}
        for i in range(len(ev_mints))
        if mint_split.get(ev_mints[i]) == "val" and onset[i] is not None
    ]
    by_mint: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(mints):
        by_mint[str(m)].append(i)

    out: dict[str, Any] = {"n_primary_val_events": len(val_events), "by_top_frac": {}}
    n_events = len(val_events)
    for frac in top_fracs:
        thr = float(np.quantile(scores, 1.0 - frac))
        caught = 0
        leads: list[float] = []
        # False alerts: top-frac observations whose mint is not among caught pumps
        # before their onset — count alerts that never precede a real val pump.
        alert_idx = np.where(scores >= thr)[0]
        alert_mints = {str(mints[i]) for i in alert_idx}
        caught_mints: set[str] = set()
        for evn in val_events:
            mint = evn["mint"]
            onset_u = evn["onset_unix"]
            idxs = by_mint.get(mint, [])
            cand = []
            for i in idxs:
                T = obs_t[i]
                if np.isfinite(T) and onset_u > T and scores[i] >= thr:
                    cand.append((T, i))
            if not cand:
                continue
            cand.sort()
            T, _ = cand[0]
            caught += 1
            caught_mints.add(mint)
            leads.append(onset_u - T)
        # False-alert count: top-frac alerts on mints that never get caught as a pump
        # OR alerts after we already "caught" — use mint-deduped alerts not in caught set
        false_alert_mints = alert_mints - caught_mints
        # Also count alert volume on non-event mints
        n_false_alert_obs = int(
            sum(1 for i in alert_idx if str(mints[i]) not in caught_mints)
        )
        arr = np.asarray(leads, dtype=np.float64)
        out["by_top_frac"][f"top_{frac:g}"] = {
            "threshold": thr,
            "n_events": n_events,
            "n_caught": caught,
            "fraction_caught": float(caught / n_events) if n_events else None,
            "lead_time_sec_percentiles": (
                {f"p{q}": float(np.percentile(arr, q)) for q in (10, 25, 50, 75, 90, 99)}
                if arr.size
                else None
            ),
            "n_alert_observations": int(alert_idx.size),
            "n_false_alert_observations": n_false_alert_obs,
            "n_false_alert_mints": int(len(false_alert_mints)),
            "false_alerts_per_caught_pump": (
                float(n_false_alert_obs / caught) if caught else None
            ),
        }
    return out


def alert_feed_hourly(
    *,
    scores: np.ndarray,
    y300: np.ndarray,
    elig300: np.ndarray,
    mints: np.ndarray,
    obs_t: np.ndarray,
    top_n: int = 10,
) -> dict[str, Any]:
    """Per-hour: top-10 scored obs deduped by mint; fraction that explode in 300s."""
    # Bucket by UTC hour of observation_time_t
    hours = np.floor(obs_t / 3600.0).astype(np.int64)
    keep = elig300.astype(bool) & np.isfinite(obs_t)
    by_hour: dict[int, list[int]] = defaultdict(list)
    for i in np.where(keep)[0]:
        by_hour[int(hours[i])].append(int(i))

    hour_stats = []
    fracs = []
    for h, idxs in sorted(by_hour.items()):
        idxs = np.asarray(idxs, dtype=np.int64)
        order = idxs[np.argsort(-scores[idxs])]
        seen: set[str] = set()
        picked: list[int] = []
        for i in order:
            m = str(mints[i])
            if m in seen:
                continue
            seen.add(m)
            picked.append(int(i))
            if len(picked) >= top_n:
                break
        if not picked:
            continue
        yy = y300[picked]
        frac = float(np.mean(yy)) if len(picked) else float("nan")
        fracs.append(frac)
        hour_stats.append(
            {
                "hour_unix": int(h),
                "n_eligible": int(idxs.size),
                "n_picked": len(picked),
                "frac_explode_300s": frac,
                "n_pos": int(yy.sum()),
            }
        )
    base = float(y300[keep].mean()) if keep.any() else float("nan")
    return {
        "top_n_per_hour": top_n,
        "dedupe": "mint",
        "n_hours": len(hour_stats),
        "mean_frac_explode_300s": float(np.mean(fracs)) if fracs else None,
        "median_frac_explode_300s": float(np.median(fracs)) if fracs else None,
        "base_rate_eligible": base,
        "lift_vs_base": (
            float(np.mean(fracs) / base) if fracs and base > 0 else None
        ),
        "per_hour": hour_stats,
        **Q85,
    }


def section_pump_eval() -> dict[str, Any]:
    print("loading phase4 broad tree preds...", flush=True)
    preds = pq.read_table(PHASE4_PREDS)
    oids = np.array(preds["observation_id"].to_pylist(), dtype=object)
    mints = np.array(preds["mint"].to_pylist(), dtype=object)
    val_oid_set = set(map(str, oids.tolist()))

    print("joining TTE overlays (val predicate)...", flush=True)
    tte_join = join_tte_overlays(val_oid_set)
    print(f"  overlay join stats: {tte_join['stats']}", flush=True)

    print("loading already_started from training_cache...", flush=True)
    started = load_already_started(val_oid_set)
    print(f"  already_started stats: {started['stats']}", flush=True)

    by_oid = started["by_oid"]
    already = np.array(
        [by_oid.get(str(o), {}).get("already_started", False) for o in oids], dtype=bool
    )
    tte = np.array(
        [by_oid.get(str(o), {}).get("tte_sec", float("nan")) for o in oids],
        dtype=np.float64,
    )
    obs_t = np.array(
        [by_oid.get(str(o), {}).get("observation_time_t", float("nan")) for o in oids],
        dtype=np.float64,
    )
    # Fill missing obs_t from overlay if available
    overlay = tte_join["table"]
    if overlay.num_rows:
        oid_to_ot = {
            str(o): float(t) if t is not None else float("nan")
            for o, t in zip(
                overlay["observation_id"].to_pylist(),
                overlay["observation_time_t"].to_pylist(),
                strict=True,
            )
        }
        for i, o in enumerate(oids):
            if not np.isfinite(obs_t[i]) and str(o) in oid_to_ot:
                obs_t[i] = oid_to_ot[str(o)]

    champions = ("lgb", "xgb")
    heads = ("explosion_60s", "explosion_300s")
    variants = ("all", "a_exclude_already_started", "c_onset_ge_5s")

    metrics_out: dict[str, Any] = {}
    for head in heads:
        y = preds[f"y_{head}"].to_numpy(zero_copy_only=False).astype(np.float64)
        elig = preds[f"elig_{head}"].to_numpy(zero_copy_only=False).astype(bool)
        # y may be float with nan for ineligible — coerce
        y_bin = np.nan_to_num(y, nan=0.0).astype(int)
        head_block: dict[str, Any] = {}
        for fam in champions:
            scores = preds[f"{fam}_{head}"].to_numpy(zero_copy_only=False).astype(np.float64)
            fam_block: dict[str, Any] = {}
            for mode in variants:
                keep = _variant_keep(elig, y_bin, already, tte, mode)
                yy = y_bin[keep]
                ss = scores[keep]
                fam_block[mode] = metrics_block(yy, ss)
            head_block[fam] = fam_block
        metrics_out[head] = head_block

    # Pump-level + alert feed on explosion_300s / 60s using XGB (stronger) and LGBM
    pump_out: dict[str, Any] = {}
    alert_out: dict[str, Any] = {}
    y300 = np.nan_to_num(
        preds["y_explosion_300s"].to_numpy(zero_copy_only=False), nan=0.0
    ).astype(int)
    elig300 = preds["elig_explosion_300s"].to_numpy(zero_copy_only=False).astype(bool)
    for fam in champions:
        for head in heads:
            scores = preds[f"{fam}_{head}"].to_numpy(zero_copy_only=False).astype(np.float64)
            # Restrict pump eval to eligible rows for that head
            keep = elig300 if head.endswith("300s") else preds[
                f"elig_{head}"
            ].to_numpy(zero_copy_only=False).astype(bool)
            pump_out[f"{fam}_{head}"] = pump_level_recall(
                oids=oids[keep],
                mints=mints[keep],
                scores=scores[keep],
                obs_t=obs_t[keep],
            )
        alert_out[fam] = alert_feed_hourly(
            scores=preds[f"{fam}_explosion_300s"].to_numpy(zero_copy_only=False).astype(
                np.float64
            ),
            y300=y300,
            elig300=elig300,
            mints=mints,
            obs_t=obs_t,
        )

    result = {
        **Q85,
        "champion_source": str(PHASE4_PREDS),
        "tte_overlay_join": tte_join["stats"],
        "already_started_join": started["stats"],
        "already_started_fraction": float(already.mean()),
        "metrics": metrics_out,
        "pump_level_recall": pump_out,
        "alert_feed_hourly": alert_out,
        "permanent_val_configs_scored": [
            "frozen_v2_lgbm_calibrated_from_phase4",
            "frozen_v2_xgb_calibrated_from_phase4",
        ],
    }
    write_json(SCRATCH / "cache" / "section_pump_eval.json", result)
    return result
