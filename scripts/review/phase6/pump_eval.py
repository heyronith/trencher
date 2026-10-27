"""Section 1: fixed broad pump-level recall + full-hour alert feed."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scripts.review.phase6.common import (
    BROAD_EVENTS,
    DEFINITIONS,
    PHASE4_PREDS,
    Q85,
    SCRATCH,
    val_window,
    write_json,
)
from scripts.review.phase6.overlays import load_champion_preds, load_val_side_fields


def _events_in_val_window(definition_id: str) -> list[dict[str, Any]]:
    q70, q85 = val_window()
    ev = pq.read_table(BROAD_EVENTS)
    prim = ev.filter(
        (pc.field("definition_id") == definition_id)
        & (pc.field("explosion_exists") == True)  # noqa: E712
    )
    mints = prim["mint"].to_pylist()
    onset = prim["onset_unix"].to_pylist()
    out = []
    for m, o in zip(mints, onset, strict=True):
        if o is None:
            continue
        ou = float(o)
        if q70 <= ou < q85:
            out.append({"mint": str(m), "onset_unix": ou})
    return out


def pump_level_recall(
    *,
    scores: np.ndarray,
    mints: np.ndarray,
    obs_t: np.ndarray,
    events: list[dict[str, Any]],
    top_fracs: tuple[float, ...] = (0.01, 0.02),
) -> dict[str, Any]:
    by_mint: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(mints):
        by_mint[str(m)].append(i)

    # Precompute onset sets for false-alert definition: onset within 300s after T
    mint_onsets: dict[str, np.ndarray] = defaultdict(list)
    for ev in events:
        mint_onsets[ev["mint"]].append(ev["onset_unix"])
    mint_onsets_arr = {m: np.asarray(v, dtype=np.float64) for m, v in mint_onsets.items()}

    out: dict[str, Any] = {
        "n_events": len(events),
        "event_filter": "onset_unix in broad permanent_val [q70,q85)",
        "events_source": str(BROAD_EVENTS),
        "by_top_frac": {},
        **Q85,
    }
    for frac in top_fracs:
        thr = float(np.quantile(scores, 1.0 - frac))
        alert_idx = np.where(scores >= thr)[0]
        caught = 0
        leads: list[float] = []
        caught_event_keys: set[tuple[str, float]] = set()
        for ev in events:
            mint = ev["mint"]
            onset_u = ev["onset_unix"]
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
            caught_event_keys.add((mint, onset_u))
            leads.append(onset_u - T)

        # False alert: alerted obs on a mint with no onset in (T, T+300]
        n_false = 0
        for i in alert_idx:
            T = obs_t[i]
            mint = str(mints[i])
            if not np.isfinite(T):
                n_false += 1
                continue
            onsets = mint_onsets_arr.get(mint)
            if onsets is None or onsets.size == 0:
                n_false += 1
                continue
            hit = np.any((onsets > T) & (onsets <= T + 300.0))
            if not hit:
                n_false += 1

        arr = np.asarray(leads, dtype=np.float64)
        out["by_top_frac"][f"top_{frac:g}"] = {
            "threshold": thr,
            "n_caught": caught,
            "fraction_caught": float(caught / len(events)) if events else None,
            "lead_time_sec_percentiles": (
                {
                    f"p{q}": float(np.percentile(arr, q))
                    for q in (10, 25, 50, 75, 90)
                }
                if arr.size
                else None
            ),
            "n_alert_observations": int(alert_idx.size),
            "n_false_alert_observations": int(n_false),
            "false_alerts_per_caught_event": (
                float(n_false / caught) if caught else None
            ),
            "false_alert_definition": (
                "alerted observation whose mint has no onset in (T, T+300]"
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
    window_lo: float | None = None,
    window_hi: float | None = None,
) -> dict[str, Any]:
    """Cover every hour in the observation window; report empty hours separately."""
    if window_lo is None or window_hi is None:
        window_lo, window_hi = val_window()
    # Hour bins spanning the launch/obs window using obs times
    finite = np.isfinite(obs_t) & elig300.astype(bool)
    if not finite.any():
        return {"error": "no finite obs times", **Q85}
    t_min = float(np.nanmin(obs_t[finite]))
    t_max = float(np.nanmax(obs_t[finite]))
    # Also span full cutoff window hours
    h0 = int(np.floor(min(t_min, float(window_lo)) / 3600.0))
    h1 = int(np.floor(max(t_max, float(window_hi) - 1e-9) / 3600.0))
    all_hours = list(range(h0, h1 + 1))

    hours = np.floor(obs_t / 3600.0).astype(np.int64)
    by_hour: dict[int, list[int]] = defaultdict(list)
    for i in np.where(finite)[0]:
        by_hour[int(hours[i])].append(int(i))

    hour_stats = []
    empty_hours = []
    fracs = []
    for h in all_hours:
        idxs = by_hour.get(h, [])
        if not idxs:
            empty_hours.append(h)
            continue
        idxs_a = np.asarray(idxs, dtype=np.int64)
        order = idxs_a[np.argsort(-scores[idxs_a])]
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
        yy = y300[picked]
        frac = float(np.mean(yy)) if picked else float("nan")
        fracs.append(frac)
        hour_stats.append(
            {
                "hour_unix": int(h),
                "n_eligible": int(idxs_a.size),
                "n_picked": len(picked),
                "frac_explode_300s": frac,
                "n_pos": int(yy.sum()) if picked else 0,
            }
        )
    base = float(y300[finite].mean()) if finite.any() else float("nan")
    return {
        "top_n_per_hour": top_n,
        "dedupe": "mint",
        "n_hours_in_window": len(all_hours),
        "n_hours_with_observations": len(hour_stats),
        "n_hours_empty": len(empty_hours),
        "empty_hours_unix": empty_hours,
        "mean_frac_explode_300s": float(np.mean(fracs)) if fracs else None,
        "median_frac_explode_300s": float(np.median(fracs)) if fracs else None,
        "base_rate_eligible": base,
        "lift_vs_base": (float(np.mean(fracs) / base) if fracs and base > 0 else None),
        "per_hour": hour_stats,
        "window": {
            "obs_t_min": t_min,
            "obs_t_max": t_max,
            "window_lo": float(window_lo),
            "window_hi": float(window_hi),
        },
        **Q85,
    }


def section_pump_eval() -> dict[str, Any]:
    print("loading champion preds + side fields...", flush=True)
    champ = load_champion_preds()
    oids = champ["observation_id"]
    mints = champ["mint"]
    preds = champ["table"]
    side = load_val_side_fields(oids)
    obs_t = side["observation_time_t"]

    y300 = np.nan_to_num(
        preds["y_explosion_300s"].to_numpy(zero_copy_only=False), nan=0.0
    ).astype(int)
    elig300 = preds["elig_explosion_300s"].to_numpy(zero_copy_only=False).astype(bool)

    event_counts = {}
    pump_out: dict[str, Any] = {}
    for label, defn in DEFINITIONS.items():
        events = _events_in_val_window(defn)
        event_counts[label] = len(events)
        print(f"  {label} events in val onset window: {len(events)}", flush=True)
        for fam in ("xgb", "lgb"):
            for head in ("explosion_60s", "explosion_300s"):
                scores = preds[f"{fam}_{head}"].to_numpy(zero_copy_only=False).astype(
                    np.float64
                )
                elig = preds[f"elig_{head}"].to_numpy(zero_copy_only=False).astype(bool)
                keep = elig & np.isfinite(obs_t)
                key = f"{fam}_{head}_{label}"
                pump_out[key] = pump_level_recall(
                    scores=scores[keep],
                    mints=mints[keep],
                    obs_t=obs_t[keep],
                    events=events,
                )

    alert_out = {}
    for fam in ("xgb", "lgb"):
        scores = preds[f"{fam}_explosion_300s"].to_numpy(zero_copy_only=False).astype(
            np.float64
        )
        alert_out[fam] = alert_feed_hourly(
            scores=scores,
            y300=y300,
            elig300=elig300,
            mints=mints,
            obs_t=obs_t,
        )

    # Phase5 bug note
    note = {
        "phase5_bug": (
            "Phase5 filtered explosion_events via token_assignments base_split==val "
            "on the RICH target_dataset_v3 table (50 primary events). Phase6 uses "
            "broad_targets_v1/explosion_events_v1 with onset_unix in [q70,q85)."
        ),
        "phase5_n_primary": 50,
        "phase6_n_primary_onset_in_window": event_counts.get("primary"),
        "champion_preds": str(PHASE4_PREDS),
    }
    result = {
        **Q85,
        "note": note,
        "event_counts_onset_in_val_window": event_counts,
        "pump_level_recall": pump_out,
        "alert_feed_hourly": alert_out,
        "already_started_fraction": float(side["already_started"].mean()),
    }
    write_json(SCRATCH / "cache" / "section_pump_eval.json", result)
    # persist side fields as npz for other sections
    np.savez_compressed(
        SCRATCH / "cache" / "val_side_fields.npz",
        already_started=side["already_started"],
        tte_sec=side["tte_sec"],
        observation_time_t=side["observation_time_t"],
        remaining_upside=side["remaining_upside"],
        h300_simple_return=side["h300_simple_return"],
        h300_mfe=side["h300_mfe"],
        h300_mae=side["h300_mae"],
        reference_price=side["reference_price"],
        h300_future_mark=side["h300_future_mark"],
        h300_future_mark_time=side["h300_future_mark_time"],
        observation_id=oids,
        mint=mints,
    )
    return result
