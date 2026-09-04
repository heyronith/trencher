"""Deterministic LABEL-ONLY explosion / ignition onset detector family.

Thresholds are prespecified before any model training. Do not tune against
predictive model scores. Primary definition chosen for economic interpretability,
stability, sample size, and consistency with explosive trajectories.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from trencher.data.world_model.targets.status import (
    CORPUS_END_UNIX,
    OUTAGE_END_UNIX,
    OUTAGE_START_UNIX,
    crosses_outage,
)

DETECTOR_VERSION = "explosion_detector_v1"
PRIMARY_DEFINITION_ID = "explosion_primary_v1"


@dataclass(frozen=True)
class ExplosionDefinition:
    definition_id: str
    label: str  # conservative | primary | aggressive
    window_sec: float
    baseline_sec: float
    confirm_sec: float
    min_log_price_disp: float
    min_intensity_mult: float
    min_unique_buyers: int
    min_buy_sol_mult: float
    min_subsequent_mfe: float
    min_trades_in_window: int
    peak_search_sec: float
    exhaustion_drawdown: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Prespecified sensitivity family — fixed before training.
EXPLOSION_DEFINITIONS: tuple[ExplosionDefinition, ...] = (
    ExplosionDefinition(
        definition_id="explosion_conservative_v1",
        label="conservative",
        window_sec=45.0,
        baseline_sec=180.0,
        confirm_sec=90.0,
        min_log_price_disp=math.log(1.25),
        min_intensity_mult=4.0,
        min_unique_buyers=5,
        min_buy_sol_mult=3.0,
        min_subsequent_mfe=0.50,
        min_trades_in_window=8,
        peak_search_sec=1800.0,
        exhaustion_drawdown=0.25,
    ),
    ExplosionDefinition(
        definition_id=PRIMARY_DEFINITION_ID,
        label="primary",
        window_sec=30.0,
        baseline_sec=120.0,
        confirm_sec=60.0,
        min_log_price_disp=math.log(1.15),
        min_intensity_mult=3.0,
        min_unique_buyers=3,
        min_buy_sol_mult=2.5,
        min_subsequent_mfe=0.25,
        min_trades_in_window=5,
        peak_search_sec=1800.0,
        exhaustion_drawdown=0.20,
    ),
    ExplosionDefinition(
        definition_id="explosion_aggressive_v1",
        label="aggressive",
        window_sec=20.0,
        baseline_sec=90.0,
        confirm_sec=45.0,
        min_log_price_disp=math.log(1.08),
        min_intensity_mult=2.0,
        min_unique_buyers=2,
        min_buy_sol_mult=2.0,
        min_subsequent_mfe=0.15,
        min_trades_in_window=3,
        peak_search_sec=1800.0,
        exhaustion_drawdown=0.15,
    ),
)


def detector_config_manifest() -> dict[str, Any]:
    return {
        "detector_version": DETECTOR_VERSION,
        "primary_definition_id": PRIMARY_DEFINITION_ID,
        "selection_basis": [
            "economic_interpretability",
            "stability_across_tokens",
            "sufficient_sample_size",
            "consistency_with_visible_explosive_trajectories",
        ],
        "not_selected_by": ["model_predictive_score", "post_hoc_outcome_tuning"],
        "definitions": [d.to_dict() for d in EXPLOSION_DEFINITIONS],
        "onset_meaning": (
            "Earliest future time where a short window shows sustained positive "
            "price displacement + activity/buy-flow acceleration vs local baseline, "
            "confirmed by subsequent economically meaningful MFE. Onset is the "
            "beginning of the accelerating transition, not the eventual peak."
        ),
        "outage_policy": (
            "Detector windows overlapping the Jul 3 collection outage are "
            "censored/unavailable; never invent onset around source gaps."
        ),
    }


def _pregrad_mask(arr: dict[str, np.ndarray]) -> np.ndarray:
    return (arr["regime"] == 0) & arr["usable_price"]


def _window_slice(
    t: np.ndarray, start: float, end: float
) -> tuple[int, int]:
    i0 = int(np.searchsorted(t, start, side="left"))
    i1 = int(np.searchsorted(t, end, side="right"))
    return i0, i1


def _detect_for_definition(
    arr: dict[str, np.ndarray],
    definition: ExplosionDefinition,
    *,
    mint: str,
) -> dict[str, Any]:
    """Detect first qualifying explosion on a mint path for one definition."""
    mask = _pregrad_mask(arr)
    t = arr["t"]
    price = arr["price"]
    is_buy = arr["is_buy"]
    sol = arr["sol"]
    wallets = arr["wallet"]

    base: dict[str, Any] = {
        "mint": mint,
        "definition_id": definition.definition_id,
        "definition_label": definition.label,
        "explosion_exists": False,
        "censoring_status": "no_event",
        "onset_unix": None,
        "peak_unix": None,
        "end_unix": None,
        "onset_to_peak_sec": None,
        "explosion_magnitude": None,
        "peak_mfe_from_onset": None,
        "max_drawdown_after_peak": None,
        "pre_explosion_activity_rate": None,
        "explosion_activity_intensity": None,
        "onset_path_idx": None,
        "peak_path_idx": None,
        "end_path_idx": None,
        "precursor": None,
    }

    idxs = np.flatnonzero(mask)
    if idxs.size < definition.min_trades_in_window + 2:
        base["censoring_status"] = "insufficient_pregrad_path"
        return base

    # Candidate onset anchors: each usable pre-grad trade index.
    last_t = float(t[idxs[-1]])
    coverage_end = min(last_t, CORPUS_END_UNIX)

    for k in idxs:
        t0 = float(t[k])
        w_end = t0 + definition.window_sec
        c_end = t0 + definition.confirm_sec
        # Need confirmation horizon within coverage.
        if c_end > coverage_end:
            continue
        if crosses_outage(t0 - definition.baseline_sec, c_end):
            continue
        # Skip candidates whose detect windows hit the known outage.
        if crosses_outage(t0, c_end):
            continue

        i_w0, i_w1 = _window_slice(t, t0, w_end)
        if i_w1 - i_w0 < definition.min_trades_in_window:
            continue
        # Restrict to usable pre-grad within window.
        w_sel = np.arange(i_w0, i_w1)
        w_sel = w_sel[mask[w_sel]]
        if w_sel.size < definition.min_trades_in_window:
            continue

        p0 = float(price[k])
        if not math.isfinite(p0) or p0 <= 0:
            continue
        p1 = float(price[w_sel[-1]])
        log_disp = math.log(p1 / p0)
        if log_disp < definition.min_log_price_disp:
            continue

        # Intensity vs baseline.
        b0, b1 = _window_slice(t, t0 - definition.baseline_sec, t0)
        base_sel = np.arange(b0, b1)
        base_sel = base_sel[mask[base_sel]] if base_sel.size else base_sel
        base_rate = (
            float(base_sel.size) / definition.baseline_sec
            if definition.baseline_sec > 0
            else 0.0
        )
        win_rate = float(w_sel.size) / definition.window_sec
        intensity_ok = win_rate >= definition.min_intensity_mult * max(base_rate, 1e-6)
        # Also allow absolute intensity when baseline is near-zero (cold start).
        if base_sel.size == 0:
            intensity_ok = w_sel.size >= definition.min_trades_in_window

        buys = w_sel[is_buy[w_sel] == 1]
        unique_buyers = len({str(wallets[i]) for i in buys if wallets[i]})
        buyers_ok = unique_buyers >= definition.min_unique_buyers

        buy_sol_win = float(np.nansum(sol[buys])) if buys.size else 0.0
        base_buys = base_sel[is_buy[base_sel] == 1] if base_sel.size else base_sel
        buy_sol_base = float(np.nansum(sol[base_buys])) if base_buys.size else 0.0
        base_buy_rate = buy_sol_base / definition.baseline_sec
        win_buy_rate = buy_sol_win / definition.window_sec
        buyflow_ok = win_buy_rate >= definition.min_buy_sol_mult * max(
            base_buy_rate, 1e-9
        ) or (base_buys.size == 0 and buy_sol_win > 0)

        if not (intensity_ok and buyers_ok and buyflow_ok):
            continue

        # Subsequent MFE confirmation within confirm_sec (and up to peak_search).
        conf0, conf1 = _window_slice(t, t0, t0 + definition.confirm_sec)
        conf_sel = np.arange(conf0, conf1)
        conf_sel = conf_sel[mask[conf_sel]]
        if conf_sel.size == 0:
            continue
        peak_px = float(np.max(price[conf_sel]))
        mfe = (peak_px / p0) - 1.0
        if mfe < definition.min_subsequent_mfe:
            # Extend confirmation slightly using peak_search but require early accel.
            p0_i, p1_i = _window_slice(t, t0, t0 + min(definition.peak_search_sec, 300.0))
            ext = np.arange(p0_i, p1_i)
            ext = ext[mask[ext]]
            if ext.size == 0:
                continue
            peak_px = float(np.max(price[ext]))
            mfe = (peak_px / p0) - 1.0
            if mfe < definition.min_subsequent_mfe:
                continue
            conf_sel = ext

        # Qualifying onset found at t0 (start of accelerating window).
        # Peak within peak_search_sec after onset.
        pk0, pk1 = _window_slice(t, t0, t0 + definition.peak_search_sec)
        pk_sel = np.arange(pk0, pk1)
        pk_sel = pk_sel[mask[pk_sel]]
        if pk_sel.size == 0:
            continue
        peak_i = int(pk_sel[int(np.argmax(price[pk_sel]))])
        peak_t = float(t[peak_i])
        peak_price = float(price[peak_i])
        magnitude = (peak_price / p0) - 1.0
        peak_mfe = magnitude

        # Exhaustion: first time after peak price drops by exhaustion_drawdown.
        end_i = None
        end_t = None
        max_dd = None
        after = np.arange(peak_i, len(t))
        after = after[mask[after]]
        if after.size > 1:
            post_prices = price[after]
            running_max = np.maximum.accumulate(post_prices)
            dd = 1.0 - (post_prices / np.maximum(running_max, 1e-18))
            max_dd = float(np.max(dd))
            hit = np.flatnonzero(post_prices <= peak_price * (1.0 - definition.exhaustion_drawdown))
            if hit.size:
                end_i = int(after[int(hit[0])])
                end_t = float(t[end_i])

        # Precursor lead times (activity / buy-flow / curve acceleration).
        precursor = _measure_precursors(
            arr,
            onset_unix=t0,
            onset_idx=int(k),
            definition=definition,
            mask=mask,
        )

        pre_rate = base_rate
        expl_intensity = win_rate

        return {
            "mint": mint,
            "definition_id": definition.definition_id,
            "definition_label": definition.label,
            "explosion_exists": True,
            "censoring_status": "observed",
            "onset_unix": t0,
            "peak_unix": peak_t,
            "end_unix": end_t,
            "onset_to_peak_sec": peak_t - t0,
            "explosion_magnitude": magnitude,
            "peak_mfe_from_onset": peak_mfe,
            "max_drawdown_after_peak": max_dd,
            "pre_explosion_activity_rate": pre_rate,
            "explosion_activity_intensity": expl_intensity,
            "onset_path_idx": int(arr["path_idx"][k]),
            "peak_path_idx": int(arr["path_idx"][peak_i]),
            "end_path_idx": int(arr["path_idx"][end_i]) if end_i is not None else None,
            "onset_price": p0,
            "peak_price": peak_price,
            "unique_buyers_at_onset_window": unique_buyers,
            "buy_sol_at_onset_window": buy_sol_win,
            "precursor": precursor,
        }

    # Right-censor if path ends before we could confirm late candidates.
    if last_t < CORPUS_END_UNIX - 60:
        base["censoring_status"] = "no_event"
    else:
        base["censoring_status"] = "right_censored_path_end"
    # Mark outage-affected tokens separately when all activity abuts outage.
    if OUTAGE_START_UNIX <= last_t <= OUTAGE_END_UNIX:
        base["censoring_status"] = "known_source_outage"
    return base


def _measure_precursors(
    arr: dict[str, np.ndarray],
    *,
    onset_unix: float,
    onset_idx: int,
    definition: ExplosionDefinition,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Causal lead-up structure before onset (does not prove predictability)."""
    t = arr["t"]
    is_buy = arr["is_buy"]
    sol = arr["sol"]
    wallets = arr["wallet"]
    curve = arr["curve"]

    lead_horizons = (5, 10, 30, 60, 120)
    trades_before: dict[str, int] = {}
    wallets_before: dict[str, int] = {}
    for h in lead_horizons:
        i0, i1 = _window_slice(t, onset_unix - h, onset_unix)
        sel = np.arange(i0, i1)
        sel = sel[mask[sel]] if sel.size else sel
        trades_before[f"trades_{h}s_before"] = int(sel.size)
        wallets_before[f"unique_wallets_{h}s_before"] = len(
            {str(wallets[i]) for i in sel if wallets[i]}
        )

    # First measurable accelerations before onset.
    def _first_accel(kind: str) -> float | None:
        # Scan backward looking for first time short-window rate exceeds 2x prior.
        probe = 15.0
        base_w = 60.0
        # Walk candidate times before onset on usable trades.
        pre = np.flatnonzero(mask & (t < onset_unix) & (t >= onset_unix - 600.0))
        if pre.size == 0:
            return None
        for idx in pre:
            tt = float(t[idx])
            if crosses_outage(tt - base_w, tt + probe):
                continue
            a0, a1 = _window_slice(t, tt, tt + probe)
            b0, b1 = _window_slice(t, tt - base_w, tt)
            a_sel = np.arange(a0, a1)
            b_sel = np.arange(b0, b1)
            a_sel = a_sel[mask[a_sel]] if a_sel.size else a_sel
            b_sel = b_sel[mask[b_sel]] if b_sel.size else b_sel
            if kind == "activity":
                ar = a_sel.size / probe
                br = b_sel.size / base_w if base_w else 0.0
                if a_sel.size >= 3 and ar >= 2.0 * max(br, 1e-6):
                    return onset_unix - tt
            elif kind == "buy_flow":
                a_buys = a_sel[is_buy[a_sel] == 1] if a_sel.size else a_sel
                b_buys = b_sel[is_buy[b_sel] == 1] if b_sel.size else b_sel
                ar = float(np.nansum(sol[a_buys])) / probe if a_buys.size else 0.0
                br = float(np.nansum(sol[b_buys])) / base_w if b_buys.size else 0.0
                if a_buys.size >= 2 and ar >= 2.0 * max(br, 1e-9):
                    return onset_unix - tt
            elif kind == "curve":
                if a_sel.size < 2 or b_sel.size < 2:
                    continue
                a_curve = curve[a_sel]
                b_curve = curve[b_sel]
                if np.isnan(a_curve).all() or np.isnan(b_curve).all():
                    continue
                a_delta = float(np.nanmax(a_curve) - np.nanmin(a_curve))
                b_delta = float(np.nanmax(b_curve) - np.nanmin(b_curve))
                if a_delta >= max(2.0 * max(b_delta, 1e-6), 1.0):
                    return onset_unix - tt
        return None

    act_lead = _first_accel("activity")
    buy_lead = _first_accel("buy_flow")
    curve_lead = _first_accel("curve")

    trades_30 = trades_before.get("trades_30s_before", 0)
    zero_precursor = (
        trades_30 <= 1
        and act_lead is None
        and buy_lead is None
        and curve_lead is None
    )

    return {
        "time_activity_accel_to_onset_sec": act_lead,
        "time_buyflow_accel_to_onset_sec": buy_lead,
        "time_curve_accel_to_onset_sec": curve_lead,
        "trades_before": trades_before,
        "wallets_before": wallets_before,
        "essentially_zero_measurable_precursor": zero_precursor,
        "meaningful_preonset_warning": (
            (act_lead is not None and act_lead >= 10)
            or (buy_lead is not None and buy_lead >= 10)
            or trades_before.get("trades_30s_before", 0) >= 3
        ),
        "definition_window_sec": definition.window_sec,
        "onset_idx": onset_idx,
    }


def detect_explosions_for_mint(
    mint: str, arr: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    return [_detect_for_definition(arr, d, mint=mint) for d in EXPLOSION_DEFINITIONS]


def primary_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for e in events:
        if e.get("definition_id") == PRIMARY_DEFINITION_ID:
            return e
    return events[0] if events else None
