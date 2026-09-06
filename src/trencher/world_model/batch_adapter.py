"""Convert frozen V4 production examples into WorldModel tensor batches."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

from trencher.data.world_model.targets.status import (
    BARRIER_LOWER_FIRST,
    BARRIER_NEITHER,
    BARRIER_PAIRS,
    BARRIER_UPPER_FIRST,
    EXPLOSION_IMMINENCE_HORIZONS_SEC,
    HORIZONS_SEC,
    POSITION_SIZES_SOL,
    SURVIVAL_HORIZONS_SEC,
)
from trencher.world_model.input_sanitize import (
    assert_finite_model_inputs,
    build_event_ssl_from_raw,
    sanitize_feature_matrix,
    sanitize_vector,
)
from trencher.world_model.split_guard import PermanentTestAccessError, refuse_test_split

_BARRIER_CLASS = {
    BARRIER_UPPER_FIRST: 0,
    "upper": 0,
    BARRIER_LOWER_FIRST: 1,
    "lower": 1,
    BARRIER_NEITHER: 2,
    "neither": 2,
}


def barrier_bids_from_pairs(pairs: tuple[tuple[float, float], ...]) -> tuple[str, ...]:
    """Map (+up,-down) pairs to frozen bid strings like u10_l10."""
    out: list[str] = []
    for up, down in pairs:
        # Config may store 10.0 meaning +10% or 0.10; normalize.
        if abs(up) <= 1.5:
            u = int(round(abs(up) * 100))
            d = int(round(abs(down) * 100))
        else:
            u = int(round(abs(up)))
            d = int(round(abs(down)))
        out.append(f"u{u}_l{d}")
    return tuple(out)


def _as_float_keep_nan(v: Any) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _as_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _pack_valid(pack: dict[str, Any] | None) -> tuple[float, float]:
    """Return (value, mask) — mask=1 only for valid observed (not missing/censored)."""
    if not pack:
        return 0.0, 0.0
    if pack.get("is_missing") or pack.get("is_censored"):
        return 0.0, 0.0
    st = str(pack.get("effective_status") or pack.get("frozen_status") or "")
    if st in {
        "right_censored",
        "modality_unavailable",
        "source_boundary_unavailable",
        "split_boundary_purged",
        "known_source_outage",
        "reference_price_unavailable",
        "no_event",
        "migration_regime_unavailable",
    }:
        return 0.0, 0.0
    v = pack.get("value_or_nan", pack.get("value"))
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 0.0, 0.0
    if isinstance(v, (bool, np.bool_)):
        return float(v), 1.0
    if isinstance(v, str):
        return 0.0, 0.0
    return float(v), 1.0


def _barrier_class(pack: dict[str, Any] | None) -> tuple[int, float]:
    if not pack or pack.get("is_missing") or pack.get("is_censored"):
        return 0, 0.0
    raw = pack.get("value")
    if raw is None:
        return 0, 0.0
    key = str(raw).strip().lower()
    if key in {"censored", "unavailable"}:
        return 0, 0.0
    if key not in _BARRIER_CLASS:
        return 0, 0.0
    return _BARRIER_CLASS[key], 1.0


def _tte_to_bin(duration: float | None, edges: np.ndarray) -> int:
    if duration is None or not np.isfinite(duration):
        return int(len(edges) - 2)
    # edges length bins+1
    idx = int(np.searchsorted(edges, float(duration), side="right") - 1)
    return int(np.clip(idx, 0, len(edges) - 2))


def _rel_times_from_feat(seq: np.ndarray) -> np.ndarray:
    """First channel is typically rel_dt; fall back to arange."""
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((arr.shape[0] if arr.ndim else 0,), dtype=np.float32)
    return np.asarray(arr[:, 0], dtype=np.float32)


def example_to_tensors(
    example: dict[str, Any],
    *,
    explosion_horizons: tuple[int, ...] = EXPLOSION_IMMINENCE_HORIZONS_SEC,
    survival_horizons: tuple[int, ...] = SURVIVAL_HORIZONS_SEC,
    return_horizons: tuple[int, ...] = HORIZONS_SEC,
    barrier_bids: tuple[str, ...] | None = None,
    execution_sizes: tuple[float, ...] | None = None,
    tte_bin_edges: np.ndarray | None = None,
    path_steps: int = 8,
    path_dim: int = 4,
    max_len: dict[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Materialize one observation into WorldModel-ready tensors."""
    meta = example.get("meta") or {}
    refuse_test_split(meta.get("base_split") or example.get("base_split"))
    if str(meta.get("base_split") or "").lower() == "test":
        raise PermanentTestAccessError("refused test observation")

    if barrier_bids is None:
        barrier_bids = tuple(bid for _, _, bid in BARRIER_PAIRS)
    if execution_sizes is None:
        execution_sizes = POSITION_SIZES_SOL

    seqs = example.get("sequences") or {}
    masks_mod = example.get("modality_masks") or {}
    targets = example.get("targets") or {}
    max_len = max_len or {}

    def _get_seq(name: str, n_ch: int) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(seqs.get(name, np.zeros((0, n_ch), np.float32)), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.size == 0:
            arr = np.zeros((0, n_ch), dtype=np.float32)
        cap = max_len.get(name)
        if cap is not None and arr.shape[0] > cap:
            arr = arr[-cap:]
        m = np.ones((arr.shape[0],), dtype=bool)
        return arr, m

    events_raw, em = _get_seq("events", 13)
    curve_raw, cm = _get_seq("curve", 7)
    micro_raw, mm = _get_seq("micro", 11)
    own_raw, om = _get_seq("ownership", 9)
    graph_raw, gm = _get_seq("graph_token", 7)
    wallet_raw, wm = _get_seq("wallet_prior", 7)
    glob_raw, gbm = _get_seq("global", 8)

    # SSL targets/masks from pre-sanitization event numerics
    ssl = build_event_ssl_from_raw(events_raw)

    events, events_finite, _ = sanitize_feature_matrix(events_raw)
    curve, curve_finite, _ = sanitize_feature_matrix(curve_raw)
    micro, micro_finite, _ = sanitize_feature_matrix(micro_raw)
    own, own_finite, _ = sanitize_feature_matrix(own_raw)
    graph, graph_finite, _ = sanitize_feature_matrix(graph_raw)
    wallet, wallet_finite, _ = sanitize_feature_matrix(wallet_raw)
    glob, glob_finite, _ = sanitize_feature_matrix(glob_raw)

    # Creator as 4-d behavioral vector from structured features when present
    feats = example.get("features") or {}
    creator_raw = np.asarray(
        [
            _as_float_keep_nan(feats.get("token_age_log1p")),
            _as_float_keep_nan(feats.get("token_age_day_frac")),
            _as_float_keep_nan(feats.get("creator_prior_rank")),
            _as_float_keep_nan(feats.get("mask_creator") if "mask_creator" in feats else 1.0),
        ],
        dtype=np.float32,
    )
    creator, creator_finite, _ = sanitize_vector(creator_raw)

    edges = tte_bin_edges
    if edges is None:
        edges = np.asarray(
            [0, 10, 30, 60, 120, 300, 600, 1800, 3600, 21600, 86400, 1e12],
            dtype=np.float64,
        )

    # Explosion multi-horizon
    exp_t = []
    exp_m = []
    for h in explosion_horizons:
        pack = targets.get(f"explosion_imminence_h{h}") or {}
        v, m = _pack_valid(pack)
        # Map truthy positives
        if m > 0 and isinstance(pack.get("value"), (bool, np.bool_, int, float)):
            v = float(bool(pack.get("value")) if not isinstance(pack.get("value"), float)
                      else (pack.get("value") > 0.5))
        exp_t.append(v)
        exp_m.append(m)

    # Left-truncation / already exploding
    expl_tte = targets.get("explosion_tte") or {}
    already = float(
        bool(
            expl_tte.get("already_started")
            or feats.get("explosion_already_started_at_t")
            or False
        )
    )
    left_trunc_mask = 1.0 - already  # do not supervise TTE as 0 when already started

    # Survival horizons (active)
    surv_t = []
    surv_m = []
    for h in survival_horizons:
        pack = targets.get(f"survival_h{h}") or {}
        v, m = _pack_valid(pack)
        if m > 0:
            v = float(bool(pack.get("value")))
        surv_t.append(v)
        surv_m.append(m)

    # Barriers [P, H] class + mask
    n_pairs = len(barrier_bids)
    n_h = len(return_horizons)
    bar_cls = np.zeros((n_pairs, n_h), dtype=np.int64)
    bar_m = np.zeros((n_pairs, n_h), dtype=np.float32)
    for pi, bid in enumerate(barrier_bids):
        for hi, h in enumerate(return_horizons):
            c, m = _barrier_class(targets.get(f"barrier_{bid}_h{h}"))
            bar_cls[pi, hi] = c
            bar_m[pi, hi] = m

    # Returns / MFE / MAE
    ret = np.zeros((n_h,), dtype=np.float32)
    ret_m = np.zeros((n_h,), dtype=np.float32)
    mfe = np.zeros((n_h,), dtype=np.float32)
    mfe_m = np.zeros((n_h,), dtype=np.float32)
    mae = np.zeros((n_h,), dtype=np.float32)
    mae_m = np.zeros((n_h,), dtype=np.float32)
    ttmfe = np.zeros((n_h,), dtype=np.float32)
    ttmfe_m = np.zeros((n_h,), dtype=np.float32)
    ttmae = np.zeros((n_h,), dtype=np.float32)
    ttmae_m = np.zeros((n_h,), dtype=np.float32)
    pers = np.zeros((n_h, 3), dtype=np.float32)
    pers_m = np.zeros((n_h, 3), dtype=np.float32)
    path_marks = np.full((path_steps, path_dim), np.nan, dtype=np.float32)

    for hi, h in enumerate(return_horizons):
        v, m = _pack_valid(targets.get(f"returns_h{h}"))
        ret[hi], ret_m[hi] = v, m
        v, m = _pack_valid(targets.get(f"mfe_h{h}"))
        mfe[hi], mfe_m[hi] = v, m
        v, m = _pack_valid(targets.get(f"mae_h{h}"))
        mae[hi], mae_m[hi] = v, m
        v, m = _pack_valid(targets.get(f"time_to_mfe_h{h}"))
        ttmfe[hi], ttmfe_m[hi] = v, m
        v, m = _pack_valid(targets.get(f"time_to_mae_h{h}"))
        ttmae[hi], ttmae_m[hi] = v, m
        for j, key in enumerate(
            (
                f"persistence_trade_count_h{h}",
                f"persistence_activity_rate_h{h}",
                f"persistence_activity_accel_h{h}",
            )
        ):
            v, m = _pack_valid(targets.get(key))
            pers[hi, j], pers_m[hi, j] = v, m
        # Path knots from future marks when available
        ref = targets.get(f"future_path_ref_h{h}") or {}
        val = ref.get("value") or {}
        mark = val.get("mark")
        if mark is not None and hi < path_steps:
            path_marks[hi, 0] = _as_float(mark, np.nan)

    # Fill path target: use available marks; mask nan
    path_target = np.nan_to_num(path_marks, nan=0.0).astype(np.float32)
    path_mask = np.isfinite(path_marks[:, 0]).astype(np.float32)

    # Execution [S, 3]
    n_s = len(execution_sizes)
    exec_t = np.zeros((n_s, 3), dtype=np.float32)
    exec_m = np.zeros((n_s, 3), dtype=np.float32)
    for si, sol in enumerate(execution_sizes):
        tag = str(sol).replace(".", "p")
        for ki, kind in enumerate(("gross", "entry_impact", "curve_impact")):
            if kind == "gross":
                key = f"execution_gross_{tag}sol"
            elif kind == "entry_impact":
                key = f"execution_entry_impact_{tag}sol"
            else:
                key = f"execution_curve_impact_{tag}sol"
            v, m = _pack_valid(targets.get(key))
            exec_t[si, ki], exec_m[si, ki] = v, m

    # TTE heads
    def _tte_fields(prefix: str) -> tuple[float, float, int, float]:
        pack = targets.get(prefix) or {}
        dur = pack.get("duration_sec", pack.get("effective_tte_sec"))
        event = float(bool(pack.get("event_observed") or pack.get("effective_event_observed")))
        at_risk = float(bool(pack.get("at_risk"))) and not bool(pack.get("is_missing"))
        if prefix == "explosion_tte":
            at_risk = at_risk * left_trunc_mask
        b = _tte_to_bin(float(dur) if dur is not None else None, edges)
        return float(dur or 0.0), event, b, float(at_risk)

    _, exp_event, exp_bin, exp_risk = _tte_fields("explosion_tte")
    _, grad_event, grad_bin, grad_risk = _tte_fields("graduation_tte")

    oid = str(meta.get("observation_id") or "")
    oid_hash = (
        int(hashlib.sha256(oid.encode("utf-8")).hexdigest()[:16], 16) % (2**63)
        if oid
        else 0
    )

    global_scalars_raw = np.zeros(9, dtype=np.float32)
    global_scalars, global_scalars_finite, _ = sanitize_vector(global_scalars_raw)

    out: dict[str, torch.Tensor] = {
        "observation_id_hash": torch.tensor(oid_hash, dtype=torch.long),
        "loader_worker_id": torch.tensor(0, dtype=torch.long),
        "deep_input_schema_version": torch.tensor(1, dtype=torch.long),  # deep_input_finite_v1
        "event_x": torch.from_numpy(events),
        "event_x_orig_finite": torch.from_numpy(events_finite),
        "event_times": torch.from_numpy(_rel_times_from_feat(events)),
        "event_mask": torch.from_numpy(em),
        "event_available": torch.tensor(
            float(masks_mod.get("mask_events", events.shape[0] > 0)), dtype=torch.float32
        ),
        "curve_x": torch.from_numpy(curve),
        "curve_x_orig_finite": torch.from_numpy(curve_finite),
        "curve_times": torch.from_numpy(_rel_times_from_feat(curve)),
        "curve_mask": torch.from_numpy(cm),
        "curve_available": torch.tensor(
            float(masks_mod.get("mask_curve", curve.shape[0] > 0)), dtype=torch.float32
        ),
        "global_buckets": torch.from_numpy(glob if glob.size else np.zeros((0, 8), np.float32)),
        "global_buckets_orig_finite": torch.from_numpy(
            glob_finite if glob_finite.size else np.zeros((0, 8), dtype=bool)
        ),
        "global_times": torch.from_numpy(_rel_times_from_feat(glob)),
        "global_scalars": torch.from_numpy(global_scalars),
        "global_scalars_orig_finite": torch.from_numpy(global_scalars_finite),
        "global_mask": torch.from_numpy(gbm),
        "global_available": torch.tensor(
            float(masks_mod.get("mask_global", glob.shape[0] > 0)), dtype=torch.float32
        ),
        "creator_feat": torch.from_numpy(creator),
        "creator_feat_orig_finite": torch.from_numpy(creator_finite),
        "micro_x": torch.from_numpy(micro),
        "micro_x_orig_finite": torch.from_numpy(micro_finite),
        "micro_times": torch.from_numpy(_rel_times_from_feat(micro)),
        "micro_mask": torch.from_numpy(mm),
        "micro_available": torch.tensor(
            float(masks_mod.get("mask_micro", micro.shape[0] > 0)), dtype=torch.float32
        ),
        "own_x": torch.from_numpy(own),
        "own_x_orig_finite": torch.from_numpy(own_finite),
        "own_times": torch.from_numpy(_rel_times_from_feat(own)),
        "own_mask": torch.from_numpy(om),
        "own_available": torch.tensor(
            float(masks_mod.get("mask_ownership", own.shape[0] > 0)), dtype=torch.float32
        ),
        "own_quality": torch.tensor(
            float(masks_mod.get("ownership_quality", 1.0 if own.shape[0] else 0.0)),
            dtype=torch.float32,
        ),
        "wallet_x": torch.from_numpy(wallet),
        "wallet_x_orig_finite": torch.from_numpy(wallet_finite),
        "wallet_times": torch.from_numpy(_rel_times_from_feat(wallet)),
        "wallet_mask": torch.from_numpy(wm),
        "wallet_available": torch.tensor(
            float(masks_mod.get("mask_wallet_prior", wallet.shape[0] > 0)),
            dtype=torch.float32,
        ),
        "graph_edge_feat": torch.from_numpy(graph),
        "graph_edge_feat_orig_finite": torch.from_numpy(graph_finite),
        "graph_times": torch.from_numpy(_rel_times_from_feat(graph)),
        "graph_mask": torch.from_numpy(gm),
        "graph_available": torch.tensor(
            float(masks_mod.get("mask_graph_token", graph.shape[0] > 0)),
            dtype=torch.float32,
        ),
        "explosion_target": torch.tensor(exp_t, dtype=torch.float32),
        "explosion_mask": torch.tensor(exp_m, dtype=torch.float32),
        "explosion_already_started": torch.tensor(already, dtype=torch.float32),
        "explosion_tte_event": torch.tensor(exp_event, dtype=torch.float32),
        "explosion_tte_time_idx": torch.tensor(exp_bin, dtype=torch.long),
        "explosion_tte_mask": torch.tensor(exp_risk, dtype=torch.float32),
        "graduation_tte_event": torch.tensor(grad_event, dtype=torch.float32),
        "graduation_tte_time_idx": torch.tensor(grad_bin, dtype=torch.long),
        "graduation_tte_mask": torch.tensor(grad_risk, dtype=torch.float32),
        "survival_horizon_target": torch.tensor(surv_t, dtype=torch.float32),
        "survival_horizon_mask": torch.tensor(surv_m, dtype=torch.float32),
        "barrier_target": torch.from_numpy(bar_cls),
        "barrier_mask": torch.from_numpy(bar_m),
        "return_target": torch.from_numpy(ret),
        "return_mask": torch.from_numpy(ret_m),
        "mfe_target": torch.from_numpy(mfe),
        "mfe_mask": torch.from_numpy(mfe_m),
        "mae_target": torch.from_numpy(mae),
        "mae_mask": torch.from_numpy(mae_m),
        "time_to_mfe_target": torch.from_numpy(ttmfe),
        "time_to_mfe_mask": torch.from_numpy(ttmfe_m),
        "time_to_mae_target": torch.from_numpy(ttmae),
        "time_to_mae_mask": torch.from_numpy(ttmae_m),
        "persistence_target": torch.from_numpy(pers),
        "persistence_mask": torch.from_numpy(pers_m),
        "path_target": torch.from_numpy(path_target),
        "path_mask": torch.from_numpy(path_mask),
        "execution_target": torch.from_numpy(exec_t),
        "execution_mask": torch.from_numpy(exec_m),
        "sample_mask": torch.tensor(1.0, dtype=torch.float32),
        "causal_ok": torch.tensor(
            float(bool(example.get("causal_ok", True))), dtype=torch.float32
        ),
        "had_original_nonfinite": torch.tensor(
            float(
                (events_finite.size and not bool(events_finite.all()))
                or (curve_finite.size and not bool(curve_finite.all()))
                or (micro_finite.size and not bool(micro_finite.all()))
                or (own_finite.size and not bool(own_finite.all()))
                or (graph_finite.size and not bool(graph_finite.all()))
                or (wallet_finite.size and not bool(wallet_finite.all()))
                or (glob_finite.size and not bool(glob_finite.all()))
                or (not bool(creator_finite.all()))
            ),
            dtype=torch.float32,
        ),
    }
    out.update(ssl)
    return out


def collate_tensor_examples(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length sequence fields and stack fixed fields."""
    if not rows:
        return {}
    for _r in rows:
        refuse_test_split(None)

    seq_keys = [
        ("event_x", "event_times", "event_mask", "event_x_orig_finite"),
        ("curve_x", "curve_times", "curve_mask", "curve_x_orig_finite"),
        ("global_buckets", "global_times", "global_mask", "global_buckets_orig_finite"),
        ("micro_x", "micro_times", "micro_mask", "micro_x_orig_finite"),
        ("own_x", "own_times", "own_mask", "own_x_orig_finite"),
        ("wallet_x", "wallet_times", "wallet_mask", "wallet_x_orig_finite"),
        ("graph_edge_feat", "graph_times", "graph_mask", "graph_edge_feat_orig_finite"),
    ]
    out: dict[str, torch.Tensor] = {}
    b = len(rows)

    for xk, tk, mk, fk in seq_keys:
        xs = [r[xk] for r in rows]
        ts = [r[tk] for r in rows]
        ms = [r[mk] for r in rows]
        fs = []
        for r in rows:
            if fk in r:
                fs.append(r[fk])
            elif r[xk].numel():
                fs.append(torch.ones_like(r[xk], dtype=torch.bool))
            else:
                fs.append(torch.zeros(0, dtype=torch.bool))
        lmax = max((int(x.shape[0]) for x in xs), default=0)
        d = int(xs[0].shape[-1]) if xs and xs[0].ndim == 2 else 1
        xpad = torch.zeros(b, lmax, d, dtype=torch.float32)
        tpad = torch.zeros(b, lmax, dtype=torch.float32)
        mpad = torch.zeros(b, lmax, dtype=torch.bool)
        fpad = torch.zeros(b, lmax, d, dtype=torch.bool)
        for i, (x, t, m, f) in enumerate(zip(xs, ts, ms, fs, strict=True)):
            n = int(x.shape[0])
            if n == 0:
                continue
            xpad[i, :n] = x[:n]
            tpad[i, :n] = t[:n]
            mpad[i, :n] = m[:n]
            if f.ndim == 2 and f.shape[0] >= n:
                fpad[i, :n] = f[:n]
        out[xk] = xpad
        out[tk] = tpad
        out[mk] = mpad
        out[fk] = fpad

    # SSL variable length — pad to max T-1
    ssl_mask_keys = (
        "ssl_side_mask",
        "ssl_log_dt_mask",
        "ssl_size_mask",
        "ssl_intensity_mask",
        "ssl_reserve_mask",
        "ssl_mask",
    )
    if any("ssl_next_side" in r for r in rows):
        sides = [r.get("ssl_next_side", torch.zeros(0, dtype=torch.long)) for r in rows]
        lmax = max((int(s.shape[0]) for s in sides), default=0)
        side = torch.zeros(b, lmax, dtype=torch.long)
        log_dt = torch.zeros(b, lmax)
        size = torch.zeros(b, lmax)
        intens = torch.zeros(b, lmax)
        rdel = torch.zeros(b, lmax)
        masks = {k: torch.zeros(b, lmax, dtype=torch.bool) for k in ssl_mask_keys}
        for i, r in enumerate(rows):
            if "ssl_next_side" not in r:
                continue
            n = int(r["ssl_next_side"].shape[0])
            side[i, :n] = r["ssl_next_side"]
            log_dt[i, :n] = r["ssl_next_log_dt"]
            size[i, :n] = r["ssl_next_size"]
            intens[i, :n] = r.get("ssl_intensity_delta_t", torch.zeros(n))
            rdel[i, :n] = r.get("ssl_reserve_delta", torch.zeros(n))
            for k in ssl_mask_keys:
                src = r.get(k)
                if src is None and k == "ssl_mask":
                    src = torch.ones(n, dtype=torch.bool)
                elif src is None:
                    src = torch.zeros(n, dtype=torch.bool)
                masks[k][i, :n] = src[:n]
        out["ssl_next_side"] = side
        out["ssl_next_log_dt"] = log_dt
        out["ssl_next_size"] = size
        out["ssl_intensity_delta_t"] = intens
        out["ssl_reserve_delta"] = rdel
        out.update(masks)

    skip = {xk for trip in seq_keys for xk in trip}
    skip |= {
        "ssl_next_side",
        "ssl_next_log_dt",
        "ssl_next_size",
        "ssl_intensity_delta_t",
        "ssl_reserve_delta",
        *ssl_mask_keys,
        "ssl_recon_target",
        "ssl_recon_mask",
    }
    keys = sorted({k for r in rows for k in r if k not in skip})
    for k in keys:
        vals = [r.get(k) for r in rows]
        if any(v is None for v in vals):
            continue
        if vals[0].ndim == 0:
            out[k] = torch.stack(vals)  # type: ignore[arg-type]
        else:
            out[k] = torch.stack(vals)  # type: ignore[arg-type]
    assert_finite_model_inputs(out)
    return out
