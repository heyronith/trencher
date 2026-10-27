"""Compute every TEST_PROTOCOL_V2 metric on a scored split (val or test)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scripts.review.phase5.common import metrics_block
from scripts.review.phase6.boot import mint_bootstrap_delta
from scripts.review.phase6.common import BROAD_EVENTS, DEFINITIONS
from scripts.review.phase6.overlays import load_val_side_fields, variant_keep
from scripts.review.phase6.pump_eval import alert_feed_hourly, pump_level_recall
from scripts.review.phase7.common import AGE_CURVE_FEATURES
from scripts.review.phase7.strata import (
    _age_bucket,
    _approx_trades_last_60s,
    _load_triggers,
    _stratum_metrics,
)
from scripts.review.phase7.twox import _build_twox_labels, _load_reach_and_upside_maps
from scripts.review.phase8.common import Q85
from scripts.review.phase8.windows import event_onset_window


def _align_stage_b(
    oids: np.ndarray, sb: dict[str, Any], head_id: str
) -> np.ndarray:
    col = f"p_{head_id}"
    oid_to_i = {str(o): i for i, o in enumerate(sb["observation_id"].tolist())}
    arr = sb[col]
    return np.array(
        [arr[oid_to_i[str(o)]] if str(o) in oid_to_i else np.nan for o in oids],
        dtype=np.float64,
    )


def score_head_bundle(
    *,
    head_id: str,
    y: np.ndarray,
    elig: np.ndarray,
    scores: dict[str, np.ndarray],
    mints: np.ndarray,
    times: np.ndarray,
    oids: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    already: np.ndarray,
    tte: np.ndarray,
    run_bootstrap: bool = True,
    split: str = "val",
) -> dict[str, Any]:
    """Variants × models metrics + strata + Δ CIs for one explosion head."""
    name_to_i = {n: i for i, n in enumerate(feature_names)}
    win_lo, win_hi = event_onset_window(split)  # type: ignore[arg-type]
    base_models = ("xgb_champion", "lgbm_tuned", "logistic_l2", "age_curve_xgb")
    finite = elig.astype(bool).copy()
    for m in base_models:
        finite &= np.isfinite(scores[m])

    variants: dict[str, Any] = {}
    for mode in ("all", "a_exclude_already_started", "c_onset_ge_5s"):
        keep = variant_keep(finite, y, already, tte, mode)
        block: dict[str, Any] = {"n": int(keep.sum()), "models": {}}
        yy = y[keep].astype(int)
        block["n_pos"] = int(yy.sum()) if yy.size else 0
        block["prevalence"] = float(yy.mean()) if yy.size else None
        for name, s in scores.items():
            if name == "stage_b":
                m = keep & np.isfinite(s)
                if m.sum() == 0 or len(np.unique(y[m].astype(int))) < 2:
                    block["models"][name] = {"roc_auc": None, "pr_auc": None}
                    continue
                block["models"][name] = metrics_block(y[m].astype(int), s[m])
            else:
                if keep.sum() == 0 or len(np.unique(yy)) < 2:
                    block["models"][name] = {"roc_auc": None, "pr_auc": None}
                    continue
                block["models"][name] = metrics_block(yy, s[keep])
        if mode == "all" and run_bootstrap and keep.any():
            yy = y[keep].astype(int)
            mm = mints[keep]
            champ = scores["xgb_champion"][keep]
            deltas = {}
            for other in ("logistic_l2", "stage_b", "age_curve_xgb"):
                if other not in scores:
                    continue
                so = scores[other][keep]
                ok = np.isfinite(champ) & np.isfinite(so)
                if ok.sum() < 100 or len(np.unique(yy[ok])) < 2:
                    continue
                deltas[f"champion_minus_{other}"] = {
                    "roc_auc": mint_bootstrap_delta(
                        yy[ok], so[ok], champ[ok], mm[ok], metric="roc_auc"
                    ),
                    "pr_auc": mint_bootstrap_delta(
                        yy[ok], so[ok], champ[ok], mm[ok], metric="pr_auc"
                    ),
                }
            block["deltas_vs_champion"] = deltas
        variants[mode] = block

    triggers = _load_triggers(oids)
    age = np.asarray(X[:, name_to_i["token_age_seconds"]], dtype=np.float64)
    buckets = _age_bucket(age)
    c30 = np.asarray(X[:, name_to_i["event_count_w30"]], dtype=np.float64)
    c120 = np.asarray(X[:, name_to_i["event_count_w120"]], dtype=np.float64)
    trades_60 = _approx_trades_last_60s(c30, c120)
    score_for_strata = {
        k: scores[k]
        for k in ("xgb_champion", "logistic_l2", "stage_b", "age_curve_xgb")
        if k in scores
    }
    if "lgbm_tuned" in scores:
        score_for_strata["lgbm_tuned"] = scores["lgbm_tuned"]

    elig_s = finite & np.isfinite(scores.get("stage_b", np.ones(len(oids))))
    strata: dict[str, Any] = {
        "by_trigger": {},
        "by_age_bucket": {},
        "active_only": {},
        "trade_count_proxy_note": (
            "Last-60s count ≈ event_count_w30 + (event_count_w120-event_count_w30)*(30/90)."
        ),
    }
    for trig in ("fixed_age", "event_count"):
        strata["by_trigger"][trig] = _stratum_metrics(
            y, score_for_strata, elig_s & (triggers == trig)
        )
    for b in ("[0,30s)", "[30s,2m)", "[2m,10m)", "[10m,1h)", ">=1h"):
        strata["by_age_bucket"][b] = _stratum_metrics(
            y, score_for_strata, elig_s & (buckets == b)
        )
    strata["active_only"]["ge1_trade_last_60s_approx"] = _stratum_metrics(
        y, score_for_strata, elig_s & (trades_60 >= 1)
    )
    strata["active_only"]["ge5_trades_last_60s_approx"] = _stratum_metrics(
        y, score_for_strata, elig_s & (trades_60 >= 5)
    )

    alert = None
    if head_id == "explosion_300s":
        alert = alert_feed_hourly(
            scores=scores["xgb_champion"],
            y300=y.astype(int),
            elig300=finite,
            mints=mints,
            obs_t=times,
            window_lo=win_lo,
            window_hi=win_hi,
        )

    return {
        **Q85,
        **({"q85_test_accessed": True} if split == "test" else {}),
        "head_id": head_id,
        "split": split,
        "event_window": [win_lo, win_hi],
        "variants": variants,
        "strata": strata,
        "alert_feed": alert,
        "age_curve_features": AGE_CURVE_FEATURES,
    }


def score_twox(
    *,
    scores: np.ndarray,
    oids: np.ndarray,
    y300: np.ndarray,
    elig300: np.ndarray,
    mints: np.ndarray,
    times: np.ndarray,
    split: str = "val",
) -> dict[str, Any]:
    maps = _load_reach_and_upside_maps()
    y, elig = _build_twox_labels(oids, y300, elig300, maps["reach"])
    keep = elig.astype(bool) & np.isfinite(scores)
    met = metrics_block(y[keep].astype(int), scores[keep])
    win_lo, win_hi = event_onset_window(split)  # type: ignore[arg-type]
    alert = alert_feed_hourly(
        scores=scores,
        y300=y.astype(int),
        elig300=elig,
        mints=mints,
        obs_t=times,
        window_lo=win_lo,
        window_hi=win_hi,
    )
    ev = pq.read_table(BROAD_EVENTS)
    prim = ev.filter(
        (pc.field("definition_id") == DEFINITIONS["primary"])
        & (pc.field("explosion_exists") == True)  # noqa: E712
    )
    events = []
    for m, o, g in zip(
        prim["mint"].to_pylist(),
        prim["onset_unix"].to_pylist(),
        prim["explosion_magnitude"].to_pylist(),
        strict=True,
    ):
        if o is None or g is None:
            continue
        if not (win_lo <= float(o) < win_hi):
            continue
        if float(g) < 1.0:
            continue
        events.append({"mint": str(m), "onset_unix": float(o)})
    pump = pump_level_recall(
        scores=scores[keep],
        mints=mints[keep],
        obs_t=times[keep],
        events=events,
    )
    return {
        **Q85,
        **({"q85_test_accessed": True} if split == "test" else {}),
        "definition": "within_300s AND reaches_plus_100pct",
        "split": split,
        "event_window": [win_lo, win_hi],
        "n_elig": int(keep.sum()),
        "n_pos": int(y[keep].sum()),
        "prevalence": float(y[keep].mean()) if keep.any() else None,
        "metrics": met,
        "alert_feed": alert,
        "pump_level_recall_magnitude_ge_1": pump,
        "n_primary_events_mag_ge_1": len(events),
    }


def load_side_fields_for_oids(oids: np.ndarray) -> dict[str, Any]:
    return load_val_side_fields(oids)


def pump_recall_primary(
    *,
    scores: np.ndarray,
    mints: np.ndarray,
    times: np.ndarray,
    elig: np.ndarray,
    split: str = "val",
) -> dict[str, Any]:
    win_lo, win_hi = event_onset_window(split)  # type: ignore[arg-type]
    ev = pq.read_table(BROAD_EVENTS)
    prim = ev.filter(
        (pc.field("definition_id") == DEFINITIONS["primary"])
        & (pc.field("explosion_exists") == True)  # noqa: E712
    )
    events = []
    for m, o in zip(prim["mint"].to_pylist(), prim["onset_unix"].to_pylist(), strict=True):
        if o is None:
            continue
        if win_lo <= float(o) < win_hi:
            events.append({"mint": str(m), "onset_unix": float(o)})
    keep = elig.astype(bool) & np.isfinite(scores)
    return pump_level_recall(
        scores=scores[keep], mints=mints[keep], obs_t=times[keep], events=events
    )
