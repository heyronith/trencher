"""Observation-level LABEL-ONLY targets: returns, barriers, MFE/MAE, survival, etc."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from trencher.data.world_model.targets.explosion import PRIMARY_DEFINITION_ID
from trencher.data.world_model.targets.status import (
    BARRIER_CENSORED,
    BARRIER_LOWER_FIRST,
    BARRIER_NEITHER,
    BARRIER_PAIRS,
    BARRIER_UNAVAILABLE,
    BARRIER_UPPER_FIRST,
    CORPUS_END_UNIX,
    EXPLOSION_IMMINENCE_HORIZONS_SEC,
    HORIZONS_SEC,
    INACTIVITY_CONFIRMATION_GAP_SEC,
    OUTAGE_END_UNIX,
    POSITION_SIZES_SOL,
    STATUS_KNOWN_SOURCE_OUTAGE,
    STATUS_MIGRATION_REGIME_UNAVAILABLE,
    STATUS_MODALITY_UNAVAILABLE,
    STATUS_REFERENCE_PRICE_UNAVAILABLE,
    STATUS_RIGHT_CENSORED,
    STATUS_SOURCE_BOUNDARY_UNAVAILABLE,
    STATUS_VALID_NEGATIVE,
    STATUS_VALID_POSITIVE,
    SURVIVAL_HORIZONS_SEC,
    crosses_outage,
)


def _search_le(t: np.ndarray, T: float) -> int:
    """Last index with t[i] <= T; -1 if none."""
    return int(np.searchsorted(t, T, side="right") - 1)


def _search_gt(t: np.ndarray, T: float) -> int:
    """First index with t[i] > T; len if none."""
    return int(np.searchsorted(t, T, side="right"))


def reference_price_at_t(arr: dict[str, np.ndarray], T: float) -> dict[str, Any]:
    t = arr["t"]
    usable = arr["usable_price"]
    i = _search_le(t, T)
    while i >= 0 and not usable[i]:
        i -= 1
    if i < 0:
        return {
            "reference_price": None,
            "reference_price_time": None,
            "reference_price_source": None,
            "reference_price_age_sec": None,
            "price_target_available": False,
            "reference_regime": None,
        }
    return {
        "reference_price": float(arr["price"][i]),
        "reference_price_time": float(t[i]),
        "reference_price_source": str(arr["price_source"][i]),
        "reference_price_age_sec": float(T - t[i]),
        "price_target_available": True,
        "reference_regime": (
            "post_graduation" if int(arr["regime"][i]) == 1 else "pre_graduation"
        ),
        "reference_idx": i,
        "reference_v_sol": (
            float(arr["v_sol"][i]) if math.isfinite(float(arr["v_sol"][i])) else None
        ),
        "reference_v_tok": (
            float(arr["v_tok"][i]) if math.isfinite(float(arr["v_tok"][i])) else None
        ),
    }


def _future_window(
    arr: dict[str, np.ndarray], T: float, horizon: float
) -> tuple[np.ndarray, str | None]:
    """Indices with T < t <= T+horizon on usable continuous regime path.

    Censors when migration breaks continuous coverage inside the window.
    """
    t = arr["t"]
    usable = arr["usable_price"]
    regime = arr["regime"]
    i0 = _search_gt(t, T)
    i1 = _search_le(t, T + horizon) + 1
    if i0 >= len(t) or i1 <= i0:
        return np.asarray([], dtype=np.int64), None

    # Start from reference regime at/before T.
    ref = reference_price_at_t(arr, T)
    start_regime = 0
    if ref.get("reference_idx") is not None:
        start_regime = int(regime[ref["reference_idx"]])

    idxs = []
    for i in range(i0, min(i1, len(t))):
        if not usable[i]:
            continue
        if int(regime[i]) != start_regime:
            # Migration / regime break — do not fabricate continuity.
            return np.asarray(idxs, dtype=np.int64), STATUS_MIGRATION_REGIME_UNAVAILABLE
        idxs.append(i)
    return np.asarray(idxs, dtype=np.int64), None


def _horizon_coverage_status(
    *,
    T: float,
    horizon: float,
    last_path_t: float | None,
    path_has_rows: bool,
) -> str | None:
    end = T + horizon
    if crosses_outage(T, end):
        # Outage may still allow labels if we have actual trades; only force
        # outage status when we lack confirming activity through the gap.
        pass
    if not path_has_rows:
        return STATUS_MODALITY_UNAVAILABLE
    if last_path_t is None:
        return STATUS_SOURCE_BOUNDARY_UNAVAILABLE
    # Right censor when horizon extends past last observed path + corpus end.
    if end > min(last_path_t + 1e-9, CORPUS_END_UNIX) and last_path_t < end:
        return STATUS_RIGHT_CENSORED
    return None


def compute_return_barrier_mfe(
    arr: dict[str, np.ndarray],
    *,
    T: float,
    ref: dict[str, Any],
    last_path_t: float | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    p0 = ref.get("reference_price")
    price_ok = bool(ref.get("price_target_available")) and p0 is not None and p0 > 0

    for h in HORIZONS_SEC:
        prefix = f"h{h}"
        if not price_ok:
            out[f"{prefix}_status"] = STATUS_REFERENCE_PRICE_UNAVAILABLE
            out[f"{prefix}_simple_return"] = None
            out[f"{prefix}_log_return"] = None
            out[f"{prefix}_future_mark"] = None
            out[f"{prefix}_future_mark_time"] = None
            out[f"{prefix}_mark_staleness_sec"] = None
            out[f"{prefix}_mfe"] = None
            out[f"{prefix}_mae"] = None
            out[f"{prefix}_log_mfe"] = None
            out[f"{prefix}_log_mae"] = None
            out[f"{prefix}_time_to_mfe_sec"] = None
            out[f"{prefix}_time_to_mae_sec"] = None
            out[f"{prefix}_max_drawdown"] = None
            for _, _, bid in BARRIER_PAIRS:
                out[f"{prefix}_barrier_{bid}_outcome"] = BARRIER_UNAVAILABLE
                out[f"{prefix}_barrier_{bid}_time_to_upper"] = None
                out[f"{prefix}_barrier_{bid}_time_to_lower"] = None
                out[f"{prefix}_barrier_{bid}_time_to_first"] = None
                out[f"{prefix}_barrier_{bid}_first_type"] = None
            continue

        idxs, regime_break = _future_window(arr, T, float(h))
        cov = _horizon_coverage_status(
            T=T,
            horizon=float(h),
            last_path_t=last_path_t,
            path_has_rows=len(arr["t"]) > 0,
        )

        if regime_break and len(idxs) == 0:
            status = regime_break
        elif len(idxs) == 0:
            status = cov or STATUS_RIGHT_CENSORED
        else:
            # Valid path marks exist inside horizon.
            last_i = int(idxs[-1])
            mark_t = float(arr["t"][last_i])
            # If horizon end is beyond last mark and beyond coverage, censored
            # unless we reached near the horizon with a mark.
            if mark_t < T + h - 1e-9 and (cov == STATUS_RIGHT_CENSORED):
                status = STATUS_RIGHT_CENSORED
            else:
                status = STATUS_VALID_POSITIVE  # placeholder; refine by return sign later

        if len(idxs) == 0:
            out[f"{prefix}_status"] = status
            out[f"{prefix}_simple_return"] = None
            out[f"{prefix}_log_return"] = None
            out[f"{prefix}_future_mark"] = None
            out[f"{prefix}_future_mark_time"] = None
            out[f"{prefix}_mark_staleness_sec"] = None
            out[f"{prefix}_mfe"] = None
            out[f"{prefix}_mae"] = None
            out[f"{prefix}_log_mfe"] = None
            out[f"{prefix}_log_mae"] = None
            out[f"{prefix}_time_to_mfe_sec"] = None
            out[f"{prefix}_time_to_mae_sec"] = None
            out[f"{prefix}_max_drawdown"] = None
            for _, _, bid in BARRIER_PAIRS:
                out[f"{prefix}_barrier_{bid}_outcome"] = (
                    BARRIER_CENSORED
                    if status == STATUS_RIGHT_CENSORED
                    else BARRIER_UNAVAILABLE
                )
                out[f"{prefix}_barrier_{bid}_time_to_upper"] = None
                out[f"{prefix}_barrier_{bid}_time_to_lower"] = None
                out[f"{prefix}_barrier_{bid}_time_to_first"] = None
                out[f"{prefix}_barrier_{bid}_first_type"] = None
            continue

        prices = arr["price"][idxs]
        times = arr["t"][idxs]
        last_price = float(prices[-1])
        last_time = float(times[-1])
        simple = (last_price / float(p0)) - 1.0
        log_ret = math.log(last_price / float(p0))
        # Terminal status by sign for auditing; availability is separate.
        if status == STATUS_RIGHT_CENSORED and last_time < T + h - 1e-6:
            term_status = STATUS_RIGHT_CENSORED
        else:
            term_status = (
                STATUS_VALID_POSITIVE if simple >= 0 else STATUS_VALID_NEGATIVE
            )

        rel = prices / float(p0) - 1.0
        mfe_i = int(np.argmax(rel))
        mae_i = int(np.argmin(rel))
        mfe = float(rel[mfe_i])
        mae = float(rel[mae_i])
        log_mfe = math.log(float(prices[mfe_i]) / float(p0))
        log_mae = math.log(float(prices[mae_i]) / float(p0))
        # Max drawdown from running peak within window.
        running_max = np.maximum.accumulate(prices)
        dd = 1.0 - prices / np.maximum(running_max, 1e-18)
        max_dd = float(np.max(dd))

        out[f"{prefix}_status"] = term_status
        out[f"{prefix}_simple_return"] = simple
        out[f"{prefix}_log_return"] = log_ret
        out[f"{prefix}_future_mark"] = last_price
        out[f"{prefix}_future_mark_time"] = last_time
        out[f"{prefix}_mark_staleness_sec"] = (T + h) - last_time
        out[f"{prefix}_mfe"] = mfe
        out[f"{prefix}_mae"] = mae
        out[f"{prefix}_log_mfe"] = log_mfe
        out[f"{prefix}_log_mae"] = log_mae
        out[f"{prefix}_time_to_mfe_sec"] = float(times[mfe_i] - T)
        out[f"{prefix}_time_to_mae_sec"] = float(times[mae_i] - T)
        out[f"{prefix}_max_drawdown"] = max_dd

        # Path descriptors
        if len(prices) >= 2:
            dt = float(times[-1] - times[0]) or 1.0
            slope = (math.log(last_price) - math.log(float(prices[0]))) / dt
            out[f"{prefix}_path_slope"] = slope
            mid = len(prices) // 2
            if mid >= 1 and len(prices) - mid >= 1:
                s1 = (
                    math.log(float(prices[mid])) - math.log(float(prices[0]))
                ) / max(float(times[mid] - times[0]), 1e-9)
                s2 = (
                    math.log(last_price) - math.log(float(prices[mid]))
                ) / max(float(times[-1] - times[mid]), 1e-9)
                out[f"{prefix}_slope_acceleration"] = s2 - s1
            else:
                out[f"{prefix}_slope_acceleration"] = None
            logp = np.log(np.maximum(prices, 1e-18))
            if len(logp) >= 3:
                out[f"{prefix}_realized_vol"] = float(np.std(np.diff(logp)))
            else:
                out[f"{prefix}_realized_vol"] = None
        else:
            out[f"{prefix}_path_slope"] = None
            out[f"{prefix}_slope_acceleration"] = None
            out[f"{prefix}_realized_vol"] = None

        # Barriers — walk in event order.
        for up, lo, bid in BARRIER_PAIRS:
            up_hit = None
            lo_hit = None
            for px, tt in zip(prices, times, strict=True):
                r = float(px) / float(p0) - 1.0
                if up_hit is None and r >= up:
                    up_hit = float(tt - T)
                if lo_hit is None and r <= lo:
                    lo_hit = float(tt - T)
                if up_hit is not None and lo_hit is not None:
                    break
            if up_hit is None and lo_hit is None:
                if term_status == STATUS_RIGHT_CENSORED:
                    outcome = BARRIER_CENSORED
                    first_type = None
                    first_t = None
                else:
                    outcome = BARRIER_NEITHER
                    first_type = "neither"
                    first_t = None
            elif up_hit is not None and (lo_hit is None or up_hit <= lo_hit):
                outcome = BARRIER_UPPER_FIRST
                first_type = "upper"
                first_t = up_hit
            else:
                outcome = BARRIER_LOWER_FIRST
                first_type = "lower"
                first_t = lo_hit
            out[f"{prefix}_barrier_{bid}_outcome"] = outcome
            out[f"{prefix}_barrier_{bid}_time_to_upper"] = up_hit
            out[f"{prefix}_barrier_{bid}_time_to_lower"] = lo_hit
            out[f"{prefix}_barrier_{bid}_time_to_first"] = first_t
            out[f"{prefix}_barrier_{bid}_first_type"] = first_type

            # Consistency vs MFE/MAE
            inconsistent = (
                (outcome == BARRIER_UPPER_FIRST and mfe + 1e-12 < up)
                or (outcome == BARRIER_LOWER_FIRST and mae - 1e-12 > lo)
            )
            out[f"{prefix}_barrier_{bid}_consistency_ok"] = not inconsistent

        # MFE/MAE vs terminal consistency flags
        out[f"{prefix}_mfe_terminal_consistent"] = not (simple > 0 and mfe + 1e-12 < simple)
        out[f"{prefix}_mae_terminal_consistent"] = not (simple < 0 and mae - 1e-12 > simple)

    return out


def compute_activity_survival(
    arr: dict[str, np.ndarray],
    *,
    T: float,
    coverage: Any,
    migration_unix: float | None = None,
) -> dict[str, Any]:
    """Survival-to-H: last genuine future activity >= T+H.

    Positive only when activity is observed at or after the horizon.
    Negatives require pump-trade *source* coverage through T+H — never the
    mint's own last_path_t. Jul 3 outage cannot create a death label.
    """
    del migration_unix  # unused; kept for call-site compatibility
    t = arr["t"]
    trade_mask = arr["regime"] == 0
    out: dict[str, Any] = {}
    future_trade_times = t[(t > T) & trade_mask]
    last_valid_trade = float(future_trade_times[-1]) if future_trade_times.size else None
    last_any_before = t[(t <= T) & trade_mask]
    out["last_valid_trade_time_at_or_before_t"] = (
        float(last_any_before[-1]) if last_any_before.size else None
    )
    out["last_valid_future_trade_time"] = last_valid_trade
    out["survival_definition"] = "last_genuine_future_activity_ge_T_plus_H"

    trade_cov = coverage.pump_trades
    for h in SURVIVAL_HORIZONS_SEC:
        horizon_t = T + float(h)
        # Positive: demonstrably active at/beyond horizon.
        if last_valid_trade is not None and last_valid_trade >= horizon_t:
            out[f"surv_h{h}_status"] = STATUS_VALID_POSITIVE
            out[f"surv_h{h}_active"] = True
            continue

        # Absence of activity at/after horizon — needs source coverage.
        resolved_later = (
            last_valid_trade is not None and last_valid_trade >= OUTAGE_END_UNIX
        )
        absence = trade_cov.absence_status(
            T,
            horizon_t,
            resolved_by_later_observation=resolved_later
            and trade_cov.covers_through(horizon_t),
        )
        # If outage intersects confirmation but post-outage coverage reaches
        # horizon_t with no activity >= horizon_t, still need covers_through.
        if absence == STATUS_KNOWN_SOURCE_OUTAGE:
            # Resolve when source covers horizon after outage and we can see
            # there is no activity >= horizon_t.
            if trade_cov.covers_through(horizon_t) and (
                last_valid_trade is None or last_valid_trade < horizon_t
            ):
                # covers_through alone isn't enough if gap sits inside (T, H].
                if not trade_cov.covers_window(T, horizon_t):
                    out[f"surv_h{h}_status"] = STATUS_KNOWN_SOURCE_OUTAGE
                    out[f"surv_h{h}_active"] = None
                    continue
            else:
                out[f"surv_h{h}_status"] = STATUS_KNOWN_SOURCE_OUTAGE
                out[f"surv_h{h}_active"] = None
                continue

        if absence == STATUS_VALID_NEGATIVE or trade_cov.covers_window(T, horizon_t):
            out[f"surv_h{h}_status"] = STATUS_VALID_NEGATIVE
            out[f"surv_h{h}_active"] = False
        else:
            out[f"surv_h{h}_status"] = absence
            out[f"surv_h{h}_active"] = None

    # Time-to-inactivity with confirmation gap using source coverage.
    out["inactivity_confirmation_gap_sec"] = INACTIVITY_CONFIRMATION_GAP_SEC
    if last_valid_trade is None:
        confirm_end = T + INACTIVITY_CONFIRMATION_GAP_SEC
        st = trade_cov.absence_status(T, confirm_end)
        if st == STATUS_VALID_NEGATIVE:
            out["time_to_inactivity_sec"] = 0.0
            out["time_to_inactivity_status"] = STATUS_VALID_POSITIVE
        else:
            out["time_to_inactivity_sec"] = None
            out["time_to_inactivity_status"] = st
    else:
        last_ft = float(last_valid_trade)
        confirm_end = last_ft + INACTIVITY_CONFIRMATION_GAP_SEC
        st = trade_cov.absence_status(
            last_ft,
            confirm_end,
            resolved_by_later_observation=False,
        )
        if st == STATUS_VALID_NEGATIVE:
            out["time_to_inactivity_sec"] = last_ft - T
            out["time_to_inactivity_status"] = STATUS_VALID_POSITIVE
        else:
            out["time_to_inactivity_sec"] = None
            out["time_to_inactivity_status"] = st

    return out


def compute_graduation(
    *,
    T: float,
    migration_unix: float | None,
    coverage: Any,
) -> dict[str, Any]:
    """Graduation labels from migration events + migration-source coverage.

    Valid negatives for no-migration do NOT depend on the token's trade path.
    """
    out: dict[str, Any] = {
        "migration_unix": migration_unix,
        "graduated_at_or_before_t": (
            migration_unix is not None and migration_unix <= T
        ),
    }
    mig_cov = coverage.migrations
    for h in HORIZONS_SEC:
        end = T + float(h)
        if migration_unix is not None and migration_unix <= T:
            out[f"grad_h{h}_status"] = STATUS_VALID_NEGATIVE
            out[f"grad_h{h}_graduated"] = False
            out[f"grad_h{h}_already_graduated"] = True
            continue
        if migration_unix is not None and T < migration_unix <= end:
            out[f"grad_h{h}_status"] = STATUS_VALID_POSITIVE
            out[f"grad_h{h}_graduated"] = True
            out[f"grad_h{h}_already_graduated"] = False
            out[f"grad_h{h}_time_to_grad_sec"] = migration_unix - T
            continue
        if migration_unix is not None and migration_unix > end:
            # Later migration proves not-yet-graduated within horizon.
            out[f"grad_h{h}_status"] = STATUS_VALID_NEGATIVE
            out[f"grad_h{h}_graduated"] = False
            out[f"grad_h{h}_already_graduated"] = False
            out[f"grad_h{h}_time_to_grad_sec"] = migration_unix - T
            continue
        # No migration observed — use migration-source coverage only.
        absence = mig_cov.absence_status(T, end)
        if absence == STATUS_VALID_NEGATIVE:
            out[f"grad_h{h}_status"] = STATUS_VALID_NEGATIVE
            out[f"grad_h{h}_graduated"] = False
            out[f"grad_h{h}_already_graduated"] = False
            out[f"grad_h{h}_time_to_grad_sec"] = None
        else:
            out[f"grad_h{h}_status"] = absence
            out[f"grad_h{h}_graduated"] = None
            out[f"grad_h{h}_already_graduated"] = False
            out[f"grad_h{h}_time_to_grad_sec"] = None

    if migration_unix is not None and migration_unix > T:
        out["time_to_graduation_sec"] = migration_unix - T
        out["time_to_graduation_status"] = STATUS_VALID_POSITIVE
    elif migration_unix is not None:
        out["time_to_graduation_sec"] = None
        out["time_to_graduation_status"] = STATUS_VALID_NEGATIVE
    else:
        # Right-censor time-to-event unless migration source covers to corpus end.
        st = mig_cov.absence_status(T, CORPUS_END_UNIX)
        out["time_to_graduation_sec"] = None
        out["time_to_graduation_status"] = (
            STATUS_VALID_NEGATIVE if st == STATUS_VALID_NEGATIVE else st
        )
    return out


def _persistence_nulls(out: dict[str, Any], h: int) -> None:
    for k in (
        "trade_count",
        "buy_count",
        "sell_count",
        "unique_wallets",
        "unique_buyers",
        "unique_sellers",
        "sol_buy_flow",
        "sol_sell_flow",
        "net_flow",
        "activity_rate",
        "mean_interarrival",
        "activity_acceleration",
        "curve_progression_change",
        "curve_velocity",
    ):
        out[f"act_h{h}_{k}"] = None


def compute_persistence(
    arr: dict[str, np.ndarray],
    *,
    T: float,
    coverage: Any,
) -> dict[str, Any]:
    """Activity/persistence with first-class horizon status.

    Numeric zeros are valid only when the source window (T, T+H] is covered.
    Uncovered windows get status + null numerics (never fake zeros).
    """
    t = arr["t"]
    trade_mask = arr["regime"] == 0
    out: dict[str, Any] = {}
    trade_cov = coverage.pump_trades
    for h in HORIZONS_SEC:
        end = T + float(h)
        win_status = None
        if trade_cov.covers_window(T, end):
            win_status = STATUS_VALID_NEGATIVE  # may upgrade if activity exists
        else:
            win_status = trade_cov.absence_status(T, end)
            if win_status == STATUS_VALID_NEGATIVE:
                # absence_status said covered — treat as covered empty window
                pass
            else:
                out[f"act_h{h}_status"] = win_status
                _persistence_nulls(out, h)
                continue

        i0 = _search_gt(t, T)
        i1 = _search_le(t, end) + 1
        sel = np.arange(i0, max(i0, i1))
        sel = sel[(sel < len(t)) & trade_mask[sel]] if len(t) else sel

        # Covered window: zeros are legitimate.
        out[f"act_h{h}_status"] = (
            STATUS_VALID_POSITIVE if sel.size > 0 else STATUS_VALID_NEGATIVE
        )
        if sel.size == 0:
            out[f"act_h{h}_trade_count"] = 0
            out[f"act_h{h}_buy_count"] = 0
            out[f"act_h{h}_sell_count"] = 0
            out[f"act_h{h}_unique_wallets"] = 0
            out[f"act_h{h}_unique_buyers"] = 0
            out[f"act_h{h}_unique_sellers"] = 0
            out[f"act_h{h}_sol_buy_flow"] = 0.0
            out[f"act_h{h}_sol_sell_flow"] = 0.0
            out[f"act_h{h}_net_flow"] = 0.0
            out[f"act_h{h}_activity_rate"] = 0.0
            out[f"act_h{h}_mean_interarrival"] = None
            out[f"act_h{h}_activity_acceleration"] = None
            out[f"act_h{h}_curve_progression_change"] = None
            out[f"act_h{h}_curve_velocity"] = None
            continue

        is_buy = arr["is_buy"][sel]
        buys = sel[is_buy == 1]
        sells = sel[is_buy == 0]
        wallets = arr["wallet"][sel]
        buy_w = arr["wallet"][buys] if buys.size else np.asarray([], dtype=object)
        sell_w = arr["wallet"][sells] if sells.size else np.asarray([], dtype=object)
        sol_buy = float(np.nansum(arr["sol"][buys])) if buys.size else 0.0
        sol_sell = float(np.nansum(arr["sol"][sells])) if sells.size else 0.0
        times = t[sel]
        inter = np.diff(times) if times.size >= 2 else np.asarray([])
        mid_t = T + h / 2.0
        r1 = float(np.sum(times <= mid_t)) / max(h / 2.0, 1e-9)
        r2 = float(np.sum(times > mid_t)) / max(h / 2.0, 1e-9)
        curve = arr["curve"][sel]
        curve_change = None
        curve_vel = None
        if np.isfinite(curve).any():
            finite = curve[np.isfinite(curve)]
            if finite.size:
                curve_change = float(finite[-1] - finite[0])
                curve_vel = curve_change / max(h, 1e-9)

        out[f"act_h{h}_trade_count"] = int(sel.size)
        out[f"act_h{h}_buy_count"] = int(buys.size)
        out[f"act_h{h}_sell_count"] = int(sells.size)
        out[f"act_h{h}_unique_wallets"] = len({str(w) for w in wallets if w})
        out[f"act_h{h}_unique_buyers"] = len({str(w) for w in buy_w if w})
        out[f"act_h{h}_unique_sellers"] = len({str(w) for w in sell_w if w})
        out[f"act_h{h}_sol_buy_flow"] = sol_buy
        out[f"act_h{h}_sol_sell_flow"] = sol_sell
        out[f"act_h{h}_net_flow"] = sol_buy - sol_sell
        out[f"act_h{h}_activity_rate"] = float(sel.size) / float(h)
        out[f"act_h{h}_mean_interarrival"] = (
            float(np.mean(inter)) if inter.size else None
        )
        out[f"act_h{h}_activity_acceleration"] = r2 - r1
        out[f"act_h{h}_curve_progression_change"] = curve_change
        out[f"act_h{h}_curve_velocity"] = curve_vel
    return out


def compute_explosion_obs(
    *,
    T: float,
    primary: dict[str, Any] | None,
    all_defs: list[dict[str, Any]],
    coverage: Any,
    ref: dict[str, Any],
    last_path_t: float | None = None,
) -> dict[str, Any]:
    del last_path_t  # unused; absence uses source coverage
    out: dict[str, Any] = {
        "explosion_definition_primary": PRIMARY_DEFINITION_ID,
        "explosion_event_observed": False,
        "explosion_right_censored": False,
        "explosion_already_started_at_t": False,
        "time_to_explosion_sec": None,
        "time_since_explosion_onset_sec": None,
        "explosion_peak_future_return": None,
        "time_from_t_to_explosion_peak_sec": None,
        "explosion_phase_pseudolabel": "no_detected_explosion",
        "explosion_phase_target_type": "auxiliary_pseudolabel",
    }
    # Agreement flags across definitions
    for ev in all_defs:
        lid = ev.get("definition_label")
        out[f"explosion_{lid}_exists"] = bool(ev.get("explosion_exists"))
        out[f"explosion_{lid}_onset_unix"] = ev.get("onset_unix")
        out[f"explosion_{lid}_magnitude"] = ev.get("explosion_magnitude")

    trade_cov = coverage.pump_trades
    if primary is None:
        for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
            out[f"explosion_within_{h}s"] = None
            out[f"explosion_within_{h}s_status"] = STATUS_MODALITY_UNAVAILABLE
        return out

    exists = bool(primary.get("explosion_exists"))
    onset = primary.get("onset_unix")
    peak = primary.get("peak_unix")
    end = primary.get("end_unix")
    cens = primary.get("censoring_status")

    if not exists:
        out["explosion_right_censored"] = cens == "right_censored_path_end"
        out["explosion_phase_pseudolabel"] = (
            "censored_unknown"
            if out["explosion_right_censored"] or cens == "known_source_outage"
            else "no_detected_explosion"
        )
        for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
            if cens == "known_source_outage":
                out[f"explosion_within_{h}s"] = None
                out[f"explosion_within_{h}s_status"] = STATUS_KNOWN_SOURCE_OUTAGE
                continue
            absence = trade_cov.absence_status(T, T + float(h))
            if absence == STATUS_VALID_NEGATIVE or trade_cov.covers_window(
                T, T + float(h)
            ):
                out[f"explosion_within_{h}s"] = False
                out[f"explosion_within_{h}s_status"] = STATUS_VALID_NEGATIVE
            else:
                out[f"explosion_within_{h}s"] = None
                out[f"explosion_within_{h}s_status"] = absence
        return out

    assert onset is not None
    out["explosion_event_observed"] = True
    already = onset <= T
    out["explosion_already_started_at_t"] = already
    if already:
        out["time_since_explosion_onset_sec"] = T - onset
        out["time_to_explosion_sec"] = None
    else:
        out["time_to_explosion_sec"] = onset - T
        out["time_since_explosion_onset_sec"] = None

    if peak is not None and peak > T and ref.get("price_target_available"):
        p0 = ref["reference_price"]
        peak_price = primary.get("peak_price")
        if p0 and peak_price:
            out["explosion_peak_future_return"] = (float(peak_price) / float(p0)) - 1.0
        out["time_from_t_to_explosion_peak_sec"] = peak - T

    for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
        if already:
            out[f"explosion_within_{h}s"] = False
            out[f"explosion_within_{h}s_status"] = STATUS_VALID_NEGATIVE
            continue
        if onset <= T + h:
            out[f"explosion_within_{h}s"] = True
            out[f"explosion_within_{h}s_status"] = STATUS_VALID_POSITIVE
        else:
            out[f"explosion_within_{h}s"] = False
            out[f"explosion_within_{h}s_status"] = STATUS_VALID_NEGATIVE

    # Magnitude / opportunity (pre-explosion only)
    if not already and ref.get("price_target_available"):
        p0 = float(ref["reference_price"])
        peak_price = primary.get("peak_price")
        if peak_price and p0 > 0:
            out["remaining_upside_to_explosion_peak"] = (float(peak_price) / p0) - 1.0
        out["time_to_explosion_peak_sec"] = (peak - T) if peak else None
        out["explosion_duration_onset_to_peak_sec"] = primary.get("onset_to_peak_sec")
        out["peak_to_post_peak_drawdown"] = primary.get("max_drawdown_after_peak")
        mag = primary.get("explosion_magnitude")
        if mag is not None:
            for thr, name in ((0.10, "10"), (0.25, "25"), (0.50, "50"), (1.0, "100")):
                out[f"explosion_reaches_plus_{name}pct"] = bool(mag >= thr)
        out["gross_opportunity_if_enter_at_t"] = out.get(
            "remaining_upside_to_explosion_peak"
        )
    else:
        out["remaining_upside_to_explosion_peak"] = None
        out["gross_opportunity_if_enter_at_t"] = None

    # Auxiliary phase pseudolabel
    if already:
        if (end is not None and end <= T) or (peak is not None and peak <= T):
            phase = "post_peak_exhaustion"
        else:
            phase = "exploding"
    else:
        tte = onset - T
        phase = "ignition_imminent" if tte <= 60 else "pre_ignition_dormant"
    out["explosion_phase_pseudolabel"] = phase
    return out


def _amm_buy_tokens(v_sol: float, v_tok: float, sol_in: float) -> float | None:
    if v_sol <= 0 or v_tok <= 0 or sol_in <= 0:
        return None
    k = v_sol * v_tok
    new_sol = v_sol + sol_in
    new_tok = k / new_sol
    tokens_out = v_tok - new_tok
    return tokens_out if tokens_out > 0 else None


def _amm_sell_sol(v_sol: float, v_tok: float, tokens_in: float) -> float | None:
    if v_sol <= 0 or v_tok <= 0 or tokens_in <= 0:
        return None
    k = v_sol * v_tok
    new_tok = v_tok + tokens_in
    new_sol = k / new_tok
    sol_out = v_sol - new_sol
    return sol_out if sol_out > 0 else None


def _future_pregrad_reserves(
    arr: dict[str, np.ndarray], idxs: np.ndarray
) -> tuple[float | None, float | None, float | None]:
    """Latest future pre-grad mark with usable reserves inside idxs."""
    if len(idxs) == 0:
        return None, None, None
    for i in reversed(list(idxs)):
        if int(arr["regime"][i]) != 0:
            continue
        vs = float(arr["v_sol"][i])
        vt = float(arr["v_tok"][i])
        px = float(arr["price"][i])
        if math.isfinite(vs) and math.isfinite(vt) and vs > 0 and vt > 0:
            return vs, vt, px if math.isfinite(px) and px > 0 else None
    return None, None, None


def compute_execution(
    arr: dict[str, np.ndarray],
    *,
    T: float,
    ref: dict[str, Any],
    horizon: float = 300.0,
) -> dict[str, Any]:
    """Execution-adjusted targets.

    Status encodes availability only — never conflated with return sign.
    Round-trip curve impact when entry+exit reserves exist; otherwise
    entry-impact-only. Fully net return unavailable without fee schedule.
    """
    out: dict[str, Any] = {
        "execution_fee_schedule_available": False,
        "execution_fee_note": (
            "Pump fee bps not row-level in freeze; fully net opportunity unavailable."
        ),
    }
    if not ref.get("price_target_available"):
        for sz in POSITION_SIZES_SOL:
            tag = str(sz).replace(".", "p")
            out[f"exec_{tag}sol_available"] = False
            out[f"exec_{tag}sol_status"] = STATUS_REFERENCE_PRICE_UNAVAILABLE
            out[f"exec_{tag}sol_gross_return"] = None
            out[f"exec_{tag}sol_entry_impact_adjusted_return"] = None
            out[f"exec_{tag}sol_curve_impact_adjusted_return"] = None
            out[f"exec_{tag}sol_net_return"] = None
            out[f"exec_{tag}sol_execution_target_available"] = 0
        return out

    p0 = float(ref["reference_price"])
    v_sol = ref.get("reference_v_sol")
    v_tok = ref.get("reference_v_tok")
    has_entry_reserves = (
        v_sol is not None and v_tok is not None and v_sol > 0 and v_tok > 0
    )

    idxs, _ = _future_window(arr, T, horizon)
    future_mark = float(arr["price"][idxs[-1]]) if len(idxs) else None
    fut_v_sol, fut_v_tok, _ = _future_pregrad_reserves(arr, idxs)

    for sz in POSITION_SIZES_SOL:
        tag = str(sz).replace(".", "p")
        out[f"exec_{tag}sol_position_size"] = sz
        out[f"exec_{tag}sol_gross_entry_price"] = p0
        out[f"exec_{tag}sol_net_return"] = None
        out[f"exec_{tag}sol_execution_target_available"] = 0  # fees unknown
        out[f"exec_{tag}sol_entry_impact_adjusted_return"] = None
        out[f"exec_{tag}sol_curve_impact_adjusted_return"] = None
        # Deprecated alias removed as primary name; keep null for clarity.
        out[f"exec_{tag}sol_slippage_adjusted_return"] = None

        if future_mark is None:
            out[f"exec_{tag}sol_available"] = False
            out[f"exec_{tag}sol_status"] = STATUS_RIGHT_CENSORED
            out[f"exec_{tag}sol_gross_return"] = None
            out[f"exec_{tag}sol_gross_available"] = False
            continue

        gross = (future_mark / p0) - 1.0
        out[f"exec_{tag}sol_gross_return"] = gross
        out[f"exec_{tag}sol_future_mark"] = future_mark
        out[f"exec_{tag}sol_gross_available"] = True
        out[f"exec_{tag}sol_available"] = True
        # Availability status only — not a claim that the return is positive.
        out[f"exec_{tag}sol_status"] = "available"

        if has_entry_reserves:
            tokens = _amm_buy_tokens(float(v_sol), float(v_tok), float(sz))
            if tokens is not None and tokens > 0:
                entry_exec = float(sz) / tokens
                impact = (entry_exec / p0) - 1.0
                out[f"exec_{tag}sol_executable_entry_price"] = entry_exec
                out[f"exec_{tag}sol_price_impact"] = impact
                # Entry-impact only vs future mark (no exit impact modeled here).
                entry_adj = (future_mark / entry_exec) - 1.0
                out[f"exec_{tag}sol_entry_impact_adjusted_return"] = entry_adj
                out[f"exec_{tag}sol_entry_impact_available"] = True

                if (
                    fut_v_sol is not None
                    and fut_v_tok is not None
                    and fut_v_sol > 0
                    and fut_v_tok > 0
                ):
                    sol_out = _amm_sell_sol(fut_v_sol, fut_v_tok, tokens)
                    if sol_out is not None:
                        curve_rt = (sol_out / float(sz)) - 1.0
                        out[f"exec_{tag}sol_curve_impact_adjusted_return"] = curve_rt
                        out[f"exec_{tag}sol_curve_impact_available"] = True
                        out[f"exec_{tag}sol_executable_exit_sol"] = sol_out
                    else:
                        out[f"exec_{tag}sol_curve_impact_available"] = False
                else:
                    out[f"exec_{tag}sol_curve_impact_available"] = False
            else:
                out[f"exec_{tag}sol_executable_entry_price"] = None
                out[f"exec_{tag}sol_price_impact"] = None
                out[f"exec_{tag}sol_entry_impact_available"] = False
                out[f"exec_{tag}sol_curve_impact_available"] = False
        else:
            out[f"exec_{tag}sol_executable_entry_price"] = None
            out[f"exec_{tag}sol_price_impact"] = None
            out[f"exec_{tag}sol_entry_impact_available"] = False
            out[f"exec_{tag}sol_curve_impact_available"] = False
    return out


def compute_ownership_targets(
    *,
    mint: str,
    T: float,
    ownership_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    """Sparse ownership targets — only where frozen evidence justifies."""
    out: dict[str, Any] = {
        "ownership_target_available": False,
        "ownership_target_status": STATUS_MODALITY_UNAVAILABLE,
        "holder_count_change": None,
        "concentration_change": None,
        "accumulation_distribution": None,
        "churn_persistence": None,
    }
    if not ownership_meta:
        return out
    recon = bool(ownership_meta.get("holder_state_reconstructable"))
    level = ownership_meta.get("ownership_evidence_level")
    out["ownership_evidence_level"] = level
    out["holder_state_reconstructable"] = recon
    if recon:
        out["ownership_target_status"] = STATUS_MODALITY_UNAVAILABLE
        out["ownership_target_note"] = (
            "holder_state_reconstructable mint but no frozen time-indexed holder "
            "ledger path for post-T deltas; dynamic holder targets left sparse."
        )
        return out
    if ownership_meta.get("availability_mask_transfer_ledger_partial") or ownership_meta.get(
        "availability_mask_rpc_balance"
    ):
        out["ownership_target_status"] = STATUS_MODALITY_UNAVAILABLE
        out["ownership_target_note"] = (
            "partial BQ/RPC ownership only supports directly observed edges; "
            "no manufactured full holder trajectory targets."
        )
        out["ownership_observed_edges_only"] = True
        return out
    return out


def future_path_refs_for_obs(
    arr: dict[str, np.ndarray], *, T: float
) -> dict[str, Any]:
    """Compact LABEL-ONLY refs: T < event_time <= T+horizon."""
    t = arr["t"]
    path_idx = arr["path_idx"]
    out: dict[str, Any] = {}
    for h in HORIZONS_SEC:
        i0 = _search_gt(t, T)
        i1 = _search_le(t, T + h) + 1
        if i0 >= i1 or i0 >= len(t):
            out[f"path_h{h}_start_idx"] = None
            out[f"path_h{h}_end_idx"] = None
            out[f"path_h{h}_count"] = 0
        else:
            out[f"path_h{h}_start_idx"] = int(path_idx[i0])
            out[f"path_h{h}_end_idx"] = int(path_idx[i1 - 1]) + 1
            out[f"path_h{h}_count"] = int(i1 - i0)
    return out
