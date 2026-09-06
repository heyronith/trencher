"""Causal tabular feature extraction from frozen sequence stores (<= T)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from trencher.baselines.feature_spec import WINDOWS_SEC, all_feature_names


def _safe_div(a: float, b: float) -> float:
    if b == 0.0 or not np.isfinite(b):
        return 0.0
    return float(a / b)


def _nan_stats(x: np.ndarray) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 0.0, 0.0
    return float(x.mean()), float(x.std()) if x.size > 1 else 0.0, float(x.max())


def _window_slice(
    times: np.ndarray, T: float, window: int | None
) -> slice:
    """Return slice into times for events with t <= T (and optionally > T-window)."""
    hi = int(np.searchsorted(times, T, side="right"))
    if hi <= 0:
        return slice(0, 0)
    if window is None:
        return slice(0, hi)
    lo = int(np.searchsorted(times, float(T) - float(window), side="right"))
    return slice(lo, hi)


def event_features_from_arrays(
    *,
    times: np.ndarray,
    is_buy: np.ndarray,
    sol: np.ndarray,
    price: np.ndarray,
    v_sol: np.ndarray,
    curve: np.ndarray,
    T: float,
) -> dict[str, float]:
    """Multi-resolution event/curve summaries from causal arrays (already <= store)."""
    out: dict[str, float] = {}
    times = np.asarray(times, dtype=np.float64)
    is_buy = np.asarray(is_buy, dtype=np.float64)
    sol = np.asarray(sol, dtype=np.float64)
    price = np.asarray(price, dtype=np.float64)
    v_sol = np.asarray(v_sol, dtype=np.float64)
    curve = np.asarray(curve, dtype=np.float64)

    # Only keep <= T
    hi_all = int(np.searchsorted(times, T, side="right"))
    times = times[:hi_all]
    is_buy = is_buy[:hi_all]
    sol = sol[:hi_all]
    price = price[:hi_all]
    v_sol = v_sol[:hi_all]
    curve = curve[:hi_all]

    has_events = times.size > 0
    out["mask_events"] = 1.0 if has_events else 0.0
    out["mask_curve"] = 1.0 if has_events else 0.0

    windows: list[tuple[str, int | None]] = [(str(w), w) for w in WINDOWS_SEC]
    windows.append(("full", None))

    for label, win in windows:
        sl = _window_slice(times, T, win)
        t = times[sl]
        b = is_buy[sl]
        s = sol[sl]
        p = price[sl]
        vs = v_sol[sl]
        c = curve[sl]
        n = int(t.size)
        buys = float(np.nansum(b > 0.5)) if n else 0.0
        sells = float(n - buys) if n else 0.0
        buy_sol = float(np.nansum(s[b > 0.5])) if n else 0.0
        sell_sol = float(np.nansum(s[b <= 0.5])) if n else 0.0
        net = buy_sol - sell_sol
        denom = buy_sol + sell_sol
        sol_mean, sol_std, sol_max = _nan_stats(s) if n else (0.0, 0.0, 0.0)

        price_return = 0.0
        price_slope = 0.0
        realized_vol = 0.0
        drawdown = 0.0
        price_accel = 0.0
        if n >= 2 and np.isfinite(p[0]) and np.isfinite(p[-1]) and p[0] > 0:
            price_return = float(p[-1] / p[0] - 1.0)
            dt = float(t[-1] - t[0])
            price_slope = _safe_div(price_return, max(dt, 1.0))
            rets = np.diff(p) / np.clip(p[:-1], 1e-18, None)
            rets = rets[np.isfinite(rets)]
            if rets.size:
                realized_vol = float(rets.std())
            peak = np.maximum.accumulate(p)
            dd = (p - peak) / np.clip(peak, 1e-18, None)
            drawdown = float(np.nanmin(dd)) if dd.size else 0.0
            if n >= 3:
                mid = n // 2
                r1 = float(p[mid] / p[0] - 1.0) if p[0] > 0 else 0.0
                r2 = float(p[-1] / p[mid] - 1.0) if p[mid] > 0 else 0.0
                price_accel = r2 - r1

        ia_mean = ia_std = 0.0
        burst = 0.0
        intens_accel = 0.0
        if n >= 2:
            gaps = np.diff(t)
            gaps = gaps[np.isfinite(gaps) & (gaps >= 0)]
            if gaps.size:
                ia_mean = float(gaps.mean())
                ia_std = float(gaps.std()) if gaps.size > 1 else 0.0
            span = max(float(t[-1] - t[0]), 1.0)
            burst = float(n) / span
            mid_t = t[0] + span / 2.0
            n1 = int(np.searchsorted(t, mid_t, side="right"))
            n2 = n - n1
            intens_accel = (n2 / max(span / 2.0, 1.0)) - (n1 / max(span / 2.0, 1.0))

        reserve_delta = float(vs[-1] - vs[0]) if n >= 2 else 0.0
        curve_delta = float(c[-1] - c[0]) if n >= 2 else 0.0
        curve_vel = _safe_div(curve_delta, max(float(t[-1] - t[0]), 1.0) if n >= 2 else 1.0)
        curve_accel = 0.0
        if n >= 3:
            mid = n // 2
            d1 = float(c[mid] - c[0])
            d2 = float(c[-1] - c[mid])
            curve_accel = d2 - d1

        prefix = f"_w{label}"
        out[f"event_count{prefix}"] = float(n)
        out[f"buy_count{prefix}"] = buys
        out[f"sell_count{prefix}"] = sells
        out[f"buy_fraction{prefix}"] = _safe_div(buys, float(n)) if n else 0.0
        out[f"buy_sol{prefix}"] = buy_sol
        out[f"sell_sol{prefix}"] = sell_sol
        out[f"net_sol{prefix}"] = net
        out[f"flow_imbalance{prefix}"] = _safe_div(net, denom) if denom else 0.0
        out[f"sol_mean{prefix}"] = sol_mean
        out[f"sol_std{prefix}"] = sol_std
        out[f"sol_max{prefix}"] = sol_max
        out[f"price_return{prefix}"] = price_return
        out[f"price_slope{prefix}"] = price_slope
        out[f"realized_vol{prefix}"] = realized_vol
        out[f"drawdown{prefix}"] = drawdown
        out[f"price_accel{prefix}"] = price_accel
        out[f"interarrival_mean{prefix}"] = ia_mean
        out[f"interarrival_std{prefix}"] = ia_std
        out[f"burst_intensity{prefix}"] = burst
        out[f"intensity_accel{prefix}"] = float(intens_accel)
        out[f"reserve_delta_sol{prefix}"] = reserve_delta
        out[f"curve_delta{prefix}"] = curve_delta
        out[f"curve_velocity{prefix}"] = curve_vel
        out[f"curve_accel{prefix}"] = curve_accel

    if has_events:
        out["latest_price"] = float(price[-1]) if np.isfinite(price[-1]) else 0.0
        out["latest_v_sol"] = float(v_sol[-1]) if np.isfinite(v_sol[-1]) else 0.0
        out["latest_v_tok"] = 0.0  # filled by caller if available
        out["latest_curve_pct"] = float(curve[-1]) if np.isfinite(curve[-1]) else 0.0
        out["latest_is_buy"] = float(is_buy[-1] > 0.5)
    else:
        for k in (
            "latest_price",
            "latest_v_sol",
            "latest_v_tok",
            "latest_curve_pct",
            "latest_is_buy",
        ):
            out[k] = 0.0
    return out


def age_features(token_age_seconds: float | None) -> dict[str, float]:
    age = float(token_age_seconds or 0.0)
    return {
        "token_age_seconds": age,
        "token_age_log1p": float(math.log1p(max(age, 0.0))),
    }


def creator_features(
    *,
    prior_rank: float | None,
    prior_is_mayhem: float | None,
    history_n: float | None,
    available: bool,
) -> dict[str, float]:
    return {
        "mask_creator": 1.0 if available else 0.0,
        "creator_prior_rank": float(prior_rank or 0.0) if available else 0.0,
        "creator_prior_is_mayhem": float(prior_is_mayhem or 0.0) if available else 0.0,
        "creator_history_n": float(history_n or 0.0) if available else 0.0,
    }


def global_features_from_ctx(ctx: dict[str, Any] | None) -> dict[str, float]:
    out = {
        "mask_global": 0.0,
        "global_launches_1h": 0.0,
        "global_launches_24h": 0.0,
        "global_migrations_1h": 0.0,
        "global_migrations_24h": 0.0,
        "daily_network_launches": 0.0,
        "daily_network_graduations": 0.0,
        "daily_network_trades": 0.0,
        "daily_network_buy_volume_sol": 0.0,
        "daily_network_graduation_rate": 0.0,
        "global_bucket_trades_sum": 0.0,
        "global_bucket_buy_sol_sum": 0.0,
        "global_bucket_sell_sol_sum": 0.0,
        "global_bucket_imbalance": 0.0,
        "global_bucket_unique_mints_mean": 0.0,
    }
    if not ctx:
        return out
    out["mask_global"] = float(ctx.get("modality_mask_global") or 0.0)
    scalars = ctx.get("global_scalars") or {}
    for k in (
        "global_launches_1h",
        "global_launches_24h",
        "global_migrations_1h",
        "global_migrations_24h",
        "daily_network_launches",
        "daily_network_graduations",
        "daily_network_trades",
        "daily_network_buy_volume_sol",
        "daily_network_graduation_rate",
    ):
        out[k] = float(scalars.get(k) or 0.0)
    g = ctx.get("global")
    if g is not None and getattr(g, "size", 0):
        arr = np.asarray(g, dtype=np.float64)
        # channels: rel_dt, trades, buys, sells, buy_sol, sell_sol, unique_mints, unique_wallets
        if arr.ndim == 2 and arr.shape[1] >= 7:
            trades = arr[:, 1]
            buy_sol = arr[:, 4]
            sell_sol = arr[:, 5]
            um = arr[:, 6]
            out["global_bucket_trades_sum"] = float(np.nansum(trades))
            out["global_bucket_buy_sol_sum"] = float(np.nansum(buy_sol))
            out["global_bucket_sell_sol_sum"] = float(np.nansum(sell_sol))
            denom = out["global_bucket_buy_sol_sum"] + out["global_bucket_sell_sol_sum"]
            out["global_bucket_imbalance"] = _safe_div(
                out["global_bucket_buy_sol_sum"] - out["global_bucket_sell_sol_sum"],
                denom,
            )
            out["global_bucket_unique_mints_mean"] = float(np.nanmean(um))
    return out


def rich_features_from_arrays(
    *,
    micro: dict[str, np.ndarray] | None = None,
    ownership: dict[str, np.ndarray] | None = None,
    graph: dict[str, np.ndarray] | None = None,
    wallet_prior: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {
        "mask_micro": 0.0,
        "mask_ownership": 0.0,
        "mask_graph": 0.0,
        "mask_wallet_prior": 0.0,
        "micro_n": 0.0,
        "micro_fee_mean": 0.0,
        "micro_cu_mean": 0.0,
        "micro_priority_fee_mean": 0.0,
        "micro_success_rate": 0.0,
        "micro_account_key_mean": 0.0,
        "own_n": 0.0,
        "own_value_sum": 0.0,
        "own_delta_sum": 0.0,
        "own_delta_abs_mean": 0.0,
        "graph_n": 0.0,
        "graph_buy_edges": 0.0,
        "graph_sell_edges": 0.0,
        "graph_sol_sum": 0.0,
        "graph_token_sum": 0.0,
        "wallet_prior_n": 0.0,
        "wallet_prior_sol_sum": 0.0,
        "wallet_prior_unique_approx": 0.0,
    }
    if micro and len(micro.get("fee_lamports", [])) > 0:
        fee = np.asarray(micro["fee_lamports"], dtype=np.float64)
        out["mask_micro"] = 1.0
        out["micro_n"] = float(fee.size)
        out["micro_fee_mean"] = float(np.nanmean(fee))
        out["micro_cu_mean"] = float(
            np.nanmean(np.asarray(micro.get("compute_units_consumed", fee), dtype=np.float64))
        )
        pf = np.asarray(micro.get("priority_fee_lamports", np.zeros_like(fee)), dtype=np.float64)
        out["micro_priority_fee_mean"] = float(np.nanmean(pf)) if np.isfinite(pf).any() else 0.0
        succ = np.asarray(micro.get("success", np.ones_like(fee)), dtype=np.float64)
        out["micro_success_rate"] = float(np.nanmean(succ))
        ak = np.asarray(micro.get("account_key_count", np.zeros_like(fee)), dtype=np.float64)
        out["micro_account_key_mean"] = float(np.nanmean(ak))
    if ownership and len(ownership.get("value", [])) > 0:
        val = np.asarray(ownership["value"], dtype=np.float64)
        # Prefer explicit delta_ui from frozen sequence; never invent pre=0.
        if "delta_ui" in ownership:
            delta = np.asarray(ownership["delta_ui"], dtype=np.float64)
        else:
            pre = np.asarray(ownership.get("pre_ui_amount", []), dtype=np.float64)
            post = np.asarray(ownership.get("post_ui_amount", []), dtype=np.float64)
            delta = np.full(val.shape, np.nan, dtype=np.float64)
            if pre.size == val.size and post.size == val.size:
                both = np.isfinite(pre) & np.isfinite(post)
                delta[both] = post[both] - pre[both]
        out["mask_ownership"] = 1.0
        out["own_n"] = float(val.size)
        out["own_value_sum"] = float(np.nansum(val))
        finite_delta = delta[np.isfinite(delta)]
        out["own_delta_sum"] = float(np.nansum(finite_delta)) if finite_delta.size else 0.0
        out["own_delta_abs_mean"] = (
            float(np.nanmean(np.abs(finite_delta))) if finite_delta.size else 0.0
        )
    if graph and len(graph.get("sol_amount", [])) > 0:
        sol = np.asarray(graph["sol_amount"], dtype=np.float64)
        tok = np.asarray(graph.get("token_amount", np.zeros_like(sol)), dtype=np.float64)
        et = graph.get("edge_type")
        out["mask_graph"] = 1.0
        out["graph_n"] = float(sol.size)
        out["graph_sol_sum"] = float(np.nansum(sol))
        out["graph_token_sum"] = float(np.nansum(tok))
        if et is not None:
            et = [str(x) for x in et]
            out["graph_buy_edges"] = float(sum(1 for x in et if "buy" in x))
            out["graph_sell_edges"] = float(sum(1 for x in et if "sell" in x))
    if wallet_prior and len(wallet_prior.get("sol_amount", [])) > 0:
        sol = np.asarray(wallet_prior["sol_amount"], dtype=np.float64)
        out["mask_wallet_prior"] = 1.0
        out["wallet_prior_n"] = float(sol.size)
        out["wallet_prior_sol_sum"] = float(np.nansum(sol))
        wallets = wallet_prior.get("wallet")
        if wallets:
            out["wallet_prior_unique_approx"] = float(len({str(w) for w in wallets if w}))
        else:
            out["wallet_prior_unique_approx"] = float(min(sol.size, 10000))
    return out


def vectorize_features(
    feat: dict[str, float], *, include_rich: bool = True
) -> tuple[list[str], np.ndarray]:
    names = all_feature_names(include_rich=include_rich)
    arr = np.zeros((len(names),), dtype=np.float32)
    for i, n in enumerate(names):
        v = feat.get(n)
        if v is None or not np.isfinite(v):
            arr[i] = 0.0
        else:
            arr[i] = float(v)
    return names, arr
