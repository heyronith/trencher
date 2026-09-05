"""Real causal sequence batching from frozen broad/rich stores.

Uses columnar NumPy views + bounded LRU caches (not full pylist materialization
of large stores). Graph / wallet IDs are indexing machinery only.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from trencher.data.world_model.broad.sequences import (
    CREATOR_STORE,
    CURVE_STORE,
    EVENTS_STORE,
)
from trencher.data.world_model.sequences import (
    WINDOW_PRESETS,
    apply_training_window,
)
from trencher.data.world_model.training.forbid import strip_forbidden
from trencher.data.world_model.training.global_batch import GlobalContextReader
from trencher.data.world_model.training.schemas import (
    CREATOR_CHANNELS,
    CREATOR_CONTINUOUS,
    CURVE_CHANNELS,
    CURVE_CONTINUOUS,
    EVENT_CHANNELS,
    EVENT_CONTINUOUS,
    GRAPH_CHANNELS,
    GRAPH_CONTINUOUS,
    MICRO_CHANNELS,
    MICRO_CONTINUOUS,
    OWN_CHANNELS,
    OWN_CONTINUOUS,
)
from trencher.data.world_model.training.transforms import apply_numeric_channel
from trencher.data.world_model.training.vocabs import (
    edge_type_id,
    event_kind_id,
    side_id,
    transfer_type_id,
)


def default_window_preset(name: str = "last_n_256") -> dict[str, Any]:
    for p in WINDOW_PRESETS:
        if p["name"] == name:
            return dict(p)
    return dict(WINDOW_PRESETS[1])


class _LRU:
    def __init__(self, capacity: int = 32) -> None:
        self.capacity = capacity
        self._data: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any) -> Any | None:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: Any, value: Any) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)


def _as_float_col(table: Any, name: str, n: int) -> np.ndarray:
    if name not in table.column_names:
        return np.full(n, np.nan, dtype=np.float64)
    col = table.column(name)
    try:
        return np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64)
    except (ValueError, TypeError):
        vals = col.to_pylist()
        out = np.full(n, np.nan, dtype=np.float64)
        for i, v in enumerate(vals):
            if v is None:
                continue
            if isinstance(v, (int, float, np.floating)):
                out[i] = float(v)
            elif isinstance(v, bool):
                out[i] = 1.0 if v else 0.0
            else:
                s = str(v).lower()
                if s in {"1", "true", "observed", "ok", "high", "good"}:
                    out[i] = 1.0
                elif s in {"0", "false", "missing", "bad", "low"}:
                    out[i] = 0.0
                else:
                    try:
                        out[i] = float(v)
                    except (TypeError, ValueError):
                        out[i] = 0.0
        return out


def _as_bool_col(table: Any, name: str, n: int) -> np.ndarray:
    if name not in table.column_names:
        return np.zeros(n, dtype=bool)
    col = table.column(name)
    try:
        return np.asarray(col.to_numpy(zero_copy_only=False), dtype=bool)
    except Exception:
        vals = col.to_pylist()
        return np.array([bool(v) if v is not None else False for v in vals], dtype=bool)


def _as_str_col(table: Any, name: str, n: int) -> list[Any]:
    if name not in table.column_names:
        return [None] * n
    return table.column(name).to_pylist()


def _window_indices(
    times: np.ndarray,
    *,
    observation_time_t: float,
    preset: dict[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    T = float(observation_time_t)
    if times.size == 0:
        return 0, 0, {
            "window_preset": preset.get("name"),
            "raw_len": 0,
            "window_lo": 0,
            "window_hi": 0,
            "truncated": False,
            "policy": "most_recent_N_or_last_seconds_causal",
            "max_event_time": None,
        }
    end = int(np.searchsorted(times, T, side="right"))
    tlist = times[:end].tolist()
    lo, hi = apply_training_window(
        times=tlist,
        start_idx=0,
        end_idx=end,
        observation_time_t=T,
        preset=preset,
    )
    meta = {
        "window_preset": preset.get("name"),
        "raw_len": end,
        "window_lo": lo,
        "window_hi": hi,
        "truncated": hi - lo < end,
        "policy": "most_recent_N_or_last_seconds_causal",
        "max_event_time": float(times[hi - 1]) if hi > lo else None,
    }
    return lo, hi, meta


def _apply_cont(
    arr: np.ndarray,
    channel_names: tuple[str, ...],
    continuous: tuple[str, ...],
    transforms: dict[str, Any] | None,
) -> np.ndarray:
    """In-place-ish transform of continuous columns only."""
    out = arr.astype(np.float32, copy=True)
    if not transforms:
        return out
    col_stats = transforms.get("columns") or {}
    for j, name in enumerate(channel_names):
        if name not in continuous:
            continue
        # rel_dt / interarrival: relative time — do not zscore absolute scale
        if name in ("rel_dt", "interarrival"):
            continue
        stats = col_stats.get(name)
        if not stats:
            continue
        col = out[:, j].astype(np.float64)
        finite = np.isfinite(col)
        if finite.any():
            col[finite] = apply_numeric_channel(col[finite], stats)
            out[:, j] = col.astype(np.float32)
    return out


def encode_events_table(
    table: Any,
    lo: int,
    hi: int,
    *,
    transforms: dict[str, Any] | None,
    v_sol_key: str = "v_sol",
    v_tok_key: str = "v_tok",
) -> np.ndarray:
    n = hi - lo
    if n <= 0:
        return np.zeros((0, len(EVENT_CHANNELS)), dtype=np.float32)
    sl = table.slice(lo, n)
    times = _as_float_col(sl, "event_time_unix", n)
    is_buy = _as_bool_col(sl, "is_buy", n).astype(np.float64)
    sides = _as_str_col(sl, "side", n)
    side_ids = np.array(
        [side_id(sides[i], bool(is_buy[i])) for i in range(n)], dtype=np.float64
    )
    sol = _as_float_col(sl, "sol_amount", n)
    tok = _as_float_col(sl, "token_amount", n)
    price = _as_float_col(sl, "price_sol", n)
    v_sol = _as_float_col(sl, v_sol_key if v_sol_key in sl.column_names else "v_sol", n)
    if v_sol_key not in sl.column_names and "v_sol_bonding_curve" in sl.column_names:
        v_sol = _as_float_col(sl, "v_sol_bonding_curve", n)
    v_tok = _as_float_col(sl, v_tok_key if v_tok_key in sl.column_names else "v_tok", n)
    if v_tok_key not in sl.column_names and "v_tokens_bonding_curve" in sl.column_names:
        v_tok = _as_float_col(sl, "v_tokens_bonding_curve", n)
    curve = _as_float_col(sl, "curve_pct_depleted", n)
    has_res = _as_bool_col(sl, "has_reserves", n).astype(np.float64)
    has_px = _as_bool_col(sl, "has_finite_price", n).astype(np.float64)
    quality = _as_float_col(sl, "source_quality_flag", n)
    quality = np.nan_to_num(quality, nan=0.0)

    out = np.zeros((n, len(EVENT_CHANNELS)), dtype=np.float64)
    t0 = float(times[0]) if n and np.isfinite(times[0]) else 0.0
    safe_t = np.where(np.isfinite(times), times, t0)
    out[:, 0] = safe_t - t0
    out[0, 1] = 0.0
    if n > 1:
        out[1:, 1] = safe_t[1:] - safe_t[:-1]
    out[:, 2] = is_buy
    out[:, 3] = side_ids
    out[:, 4] = sol
    out[:, 5] = tok
    out[:, 6] = price
    out[:, 7] = v_sol
    out[:, 8] = v_tok
    out[:, 9] = curve
    out[:, 10] = has_res
    out[:, 11] = has_px
    out[:, 12] = quality
    return _apply_cont(out, EVENT_CHANNELS, EVENT_CONTINUOUS, transforms)


def encode_curve_table(
    table: Any, lo: int, hi: int, *, transforms: dict[str, Any] | None
) -> np.ndarray:
    n = hi - lo
    if n <= 0:
        return np.zeros((0, len(CURVE_CHANNELS)), dtype=np.float32)
    sl = table.slice(lo, n)
    times = _as_float_col(sl, "event_time_unix", n)
    price = _as_float_col(sl, "price_sol", n)
    v_sol = _as_float_col(sl, "v_sol", n)
    if not np.isfinite(v_sol).any() and "v_sol_bonding_curve" in sl.column_names:
        v_sol = _as_float_col(sl, "v_sol_bonding_curve", n)
    v_tok = _as_float_col(sl, "v_tok", n)
    if not np.isfinite(v_tok).any() and "v_tokens_bonding_curve" in sl.column_names:
        v_tok = _as_float_col(sl, "v_tokens_bonding_curve", n)
    curve = _as_float_col(sl, "curve_pct_depleted", n)
    has_res = _as_bool_col(sl, "has_reserves", n)
    quality = _as_float_col(sl, "source_quality_flag", n)
    out = np.zeros((n, len(CURVE_CHANNELS)), dtype=np.float64)
    t0 = float(times[0]) if n else 0.0
    for i in range(n):
        out[i, 0] = float(times[i]) - t0
        out[i, 1] = price[i]
        out[i, 2] = v_sol[i]
        out[i, 3] = v_tok[i]
        out[i, 4] = curve[i]
        out[i, 5] = 1.0 if bool(has_res[i]) else 0.0
        out[i, 6] = quality[i] if np.isfinite(quality[i]) else 0.0
    return _apply_cont(out, CURVE_CHANNELS, CURVE_CONTINUOUS, transforms)


def encode_creator_table(
    table: Any, lo: int, hi: int, *, transforms: dict[str, Any] | None
) -> np.ndarray:
    n = hi - lo
    if n <= 0:
        return np.zeros((0, len(CREATOR_CHANNELS)), dtype=np.float32)
    sl = table.slice(lo, n)
    times = _as_float_col(sl, "event_time_unix", n)
    rank = _as_float_col(sl, "prior_rank", n)
    mayhem = _as_bool_col(sl, "prior_is_mayhem_mode", n)
    quality = _as_float_col(sl, "source_quality_flag", n)
    out = np.zeros((n, len(CREATOR_CHANNELS)), dtype=np.float64)
    t0 = float(times[0]) if n else 0.0
    for i in range(n):
        out[i, 0] = float(times[i]) - t0
        out[i, 1] = rank[i]
        out[i, 2] = 1.0 if bool(mayhem[i]) else 0.0
        out[i, 3] = quality[i] if np.isfinite(quality[i]) else 0.0
    return _apply_cont(out, CREATOR_CHANNELS, CREATOR_CONTINUOUS, transforms)


def encode_micro_table(
    table: Any, lo: int, hi: int, *, transforms: dict[str, Any] | None
) -> np.ndarray:
    n = hi - lo
    if n <= 0:
        return np.zeros((0, len(MICRO_CHANNELS)), dtype=np.float32)
    sl = table.slice(lo, n)
    times = _as_float_col(sl, "event_time_unix", n)
    success = _as_bool_col(sl, "success", n)
    fee = _as_float_col(sl, "fee_lamports", n)
    cu = _as_float_col(sl, "compute_units_consumed", n)
    prio = _as_float_col(sl, "priority_fee_lamports", n)
    ak = _as_float_col(sl, "account_key_count", n)
    bc = _as_float_col(sl, "balance_change_count", n)
    lm = _as_float_col(sl, "log_message_count", n)
    ii = _as_float_col(sl, "inner_instruction_groups", n)
    ap = _as_bool_col(sl, "availability_priority_fee", n)
    quality = _as_float_col(sl, "source_quality_flag", n)
    out = np.zeros((n, len(MICRO_CHANNELS)), dtype=np.float64)
    t0 = float(times[0]) if n else 0.0
    for i in range(n):
        out[i, 0] = float(times[i]) - t0
        out[i, 1] = 1.0 if bool(success[i]) else 0.0
        out[i, 2] = fee[i]
        out[i, 3] = cu[i]
        out[i, 4] = prio[i]
        out[i, 5] = ak[i]
        out[i, 6] = bc[i]
        out[i, 7] = lm[i]
        out[i, 8] = ii[i]
        out[i, 9] = 1.0 if bool(ap[i]) else 0.0
        out[i, 10] = quality[i] if np.isfinite(quality[i]) else 0.0
    return _apply_cont(out, MICRO_CHANNELS, MICRO_CONTINUOUS, transforms)


def encode_ownership_table(
    table: Any, lo: int, hi: int, *, transforms: dict[str, Any] | None
) -> np.ndarray:
    n = hi - lo
    if n <= 0:
        return np.zeros((0, len(OWN_CHANNELS)), dtype=np.float32)
    sl = table.slice(lo, n)
    times = _as_float_col(sl, "event_time_unix", n)
    kinds = _as_str_col(sl, "event_kind", n)
    ttypes = _as_str_col(sl, "transfer_type", n)
    value = _as_float_col(sl, "value", n)
    decimals = _as_float_col(sl, "decimals", n)
    pre = _as_float_col(sl, "pre_ui_amount", n)
    post = _as_float_col(sl, "post_ui_amount", n)
    quality = _as_float_col(sl, "source_quality_flag", n)
    out = np.zeros((n, len(OWN_CHANNELS)), dtype=np.float64)
    t0 = float(times[0]) if n else 0.0
    for i in range(n):
        out[i, 0] = float(times[i]) - t0
        out[i, 1] = event_kind_id(kinds[i])
        out[i, 2] = transfer_type_id(ttypes[i])
        out[i, 3] = value[i]
        out[i, 4] = decimals[i]
        out[i, 5] = pre[i]
        out[i, 6] = post[i]
        if np.isfinite(pre[i]) and np.isfinite(post[i]):
            out[i, 7] = post[i] - pre[i]
        else:
            out[i, 7] = np.nan
        out[i, 8] = quality[i] if np.isfinite(quality[i]) else 0.0
    return _apply_cont(out, OWN_CHANNELS, OWN_CONTINUOUS, transforms)


def encode_graph_table(
    table: Any,
    lo: int,
    hi: int,
    *,
    transforms: dict[str, Any] | None,
    node_vocab: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (features, src_idx, dst_idx). Node IDs are index tensors only."""
    n = hi - lo
    if n <= 0:
        return (
            np.zeros((0, len(GRAPH_CHANNELS)), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )
    sl = table.slice(lo, n)
    times = _as_float_col(sl, "event_time_unix", n)
    etypes = _as_str_col(sl, "edge_type", n)
    srcs = _as_str_col(sl, "src", n)
    dsts = _as_str_col(sl, "dst", n)
    sol = _as_float_col(sl, "sol_amount", n)
    tok = _as_float_col(sl, "token_amount", n)
    cohort = _as_bool_col(sl, "is_cohort_derived", n)
    quality = _as_float_col(sl, "source_quality_flag", n)
    vocab = node_vocab if node_vocab is not None else {}
    src_idx = np.zeros(n, dtype=np.int32)
    dst_idx = np.zeros(n, dtype=np.int32)
    out = np.zeros((n, len(GRAPH_CHANNELS)), dtype=np.float64)
    t0 = float(times[0]) if n else 0.0
    for i in range(n):
        out[i, 0] = float(times[i]) - t0
        out[i, 1] = edge_type_id(etypes[i])
        # direction relative to token mint when available is left for encoder;
        # here +1 if src present else 0
        out[i, 2] = 1.0 if srcs[i] else (-1.0 if dsts[i] else 0.0)
        out[i, 3] = sol[i]
        out[i, 4] = tok[i]
        out[i, 5] = 1.0 if bool(cohort[i]) else 0.0
        out[i, 6] = quality[i] if np.isfinite(quality[i]) else 0.0
        s = srcs[i]
        d = dsts[i]
        if s is not None:
            if s not in vocab:
                vocab[s] = len(vocab) + 1
            src_idx[i] = vocab[s]
        if d is not None:
            if d not in vocab:
                vocab[d] = len(vocab) + 1
            dst_idx[i] = vocab[d]
    feats = _apply_cont(out, GRAPH_CHANNELS, GRAPH_CONTINUOUS, transforms)
    return feats, src_idx, dst_idx


class BroadSequenceReader:
    def __init__(self, broad_root: Path, *, partition_lru: int = 48) -> None:
        self.broad_root = Path(broad_root)
        self.refs_root = self.broad_root / "broad_observation_sequence_refs_v1"
        self._ref_cache = _LRU(partition_lru)
        self._part_cache = _LRU(partition_lru)
        self._global: GlobalContextReader | None = None

    def set_world_root(self, world_root: Path) -> None:
        self._global = GlobalContextReader(world_root)

    def _load_refs(self, mint_partition: str) -> dict[str, dict[str, Any]]:
        hit = self._ref_cache.get(mint_partition)
        if hit is not None:
            return hit
        path = self.refs_root / mint_partition / "part.parquet"
        by_oid: dict[str, dict[str, Any]] = {}
        if path.exists():
            for row in pq.read_table(path).to_pylist():
                oid = row.get("observation_id")
                if oid:
                    by_oid[str(oid)] = row
        self._ref_cache.put(mint_partition, by_oid)
        return by_oid

    def _load_part(self, store: str, partition: str) -> Any:
        key = (store, partition)
        hit = self._part_cache.get(key)
        if hit is not None:
            return hit
        path = self.broad_root / store / partition / "part.parquet"
        table = pq.read_table(path) if path.exists() else None
        self._part_cache.put(key, table)
        return table

    def load_for_assignment(
        self,
        assignment: dict[str, Any],
        *,
        preset: dict[str, Any] | None = None,
        transforms: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        preset = preset or default_window_preset()
        oid = str(assignment["observation_id"])
        part = str(assignment.get("mint_partition") or "")
        T = float(assignment["observation_time_t"])
        ref = self._load_refs(part).get(oid)
        empty_e = np.zeros((0, len(EVENT_CHANNELS)), np.float32)
        empty_c = np.zeros((0, len(CURVE_CHANNELS)), np.float32)
        empty_cr = np.zeros((0, len(CREATOR_CHANNELS)), np.float32)
        if ref is None:
            return {
                "events": empty_e,
                "events_mask": np.zeros((0,), dtype=bool),
                "curve": empty_c,
                "curve_mask": np.zeros((0,), dtype=bool),
                "creator": empty_cr,
                "creator_mask": np.zeros((0,), dtype=bool),
                "global": np.zeros((0, 8), np.float32),
                "global_mask": np.zeros((0,), dtype=bool),
                "modality_masks": {
                    "modality_mask_events": 0.0,
                    "modality_mask_curve": 0.0,
                    "modality_mask_creator": 0.0,
                    "modality_mask_global": 0.0,
                },
                "window_meta": {},
                "truncation": {},
                "channel_schemas": {
                    "events": list(EVENT_CHANNELS),
                    "curve": list(CURVE_CHANNELS),
                    "creator": list(CREATOR_CHANNELS),
                },
                "causal_ok": True,
            }

        def _slice_window(store: str, start_key: str, end_key: str, part_key: str):
            p = str(ref.get(part_key) or part)
            table = self._load_part(store, p)
            if table is None or table.num_rows == 0:
                return None, 0, 0, {
                    "truncated": False,
                    "raw_len": 0,
                    "max_event_time": None,
                }
            lo0 = int(ref.get(start_key) or 0)
            hi0 = int(ref.get(end_key) or lo0)
            sl = table.slice(lo0, max(0, hi0 - lo0))
            times = _as_float_col(sl, "event_time_unix", sl.num_rows)
            wlo, whi, meta = _window_indices(times, observation_time_t=T, preset=preset)
            # Return slice relative to `sl` (0-based within observation span)
            return sl, wlo, whi, meta

        ev_sl, elo, ehi, emeta = _slice_window(
            EVENTS_STORE, "events_start_idx", "events_end_idx", "events_partition"
        )
        cu_sl, clo, chi, cmeta = _slice_window(
            CURVE_STORE, "curve_start_idx", "curve_end_idx", "curve_partition"
        )
        cr_sl, rlo, rhi, rmeta = _slice_window(
            CREATOR_STORE, "creator_start_idx", "creator_end_idx", "creator_partition"
        )

        events = (
            encode_events_table(ev_sl, elo, ehi, transforms=transforms)
            if ev_sl is not None
            else empty_e
        )
        curve = (
            encode_curve_table(cu_sl, clo, chi, transforms=transforms)
            if cu_sl is not None
            else empty_c
        )
        creator = (
            encode_creator_table(cr_sl, rlo, rhi, transforms=transforms)
            if cr_sl is not None
            else empty_cr
        )

        g = {
            "global": np.zeros((0, 8), np.float32),
            "global_mask": np.zeros((0,), dtype=bool),
            "global_scalars": {},
            "modality_mask_global": 0.0,
            "causal_ok": True,
        }
        if self._global is not None:
            g = self._global.load_for_t(T, transforms=transforms)

        return {
            "events": events,
            "events_mask": np.ones((events.shape[0],), dtype=bool),
            "curve": curve,
            "curve_mask": np.ones((curve.shape[0],), dtype=bool),
            "creator": creator,
            "creator_mask": np.ones((creator.shape[0],), dtype=bool),
            "global": g["global"],
            "global_mask": g["global_mask"],
            "modality_masks": {
                "modality_mask_events": 1.0 if events.shape[0] else 0.0,
                "modality_mask_curve": 1.0 if curve.shape[0] else 0.0,
                "modality_mask_creator": 1.0 if creator.shape[0] else 0.0,
                "modality_mask_global": float(g.get("modality_mask_global") or 0.0),
            },
            "window_meta": {"events": emeta, "curve": cmeta, "creator": rmeta},
            "truncation": {
                "events": bool(emeta.get("truncated")),
                "curve": bool(cmeta.get("truncated")),
                "creator": bool(rmeta.get("truncated")),
            },
            "channel_schemas": {
                "events": list(EVENT_CHANNELS),
                "curve": list(CURVE_CHANNELS),
                "creator": list(CREATOR_CHANNELS),
            },
            "structured_state": strip_forbidden(
                {
                    "token_age_seconds": float(assignment.get("token_age_seconds") or 0.0),
                    **(g.get("global_scalars") or {}),
                }
            ),
            "causal_ok": all(
                (m.get("max_event_time") is None or m["max_event_time"] <= T)
                for m in (emeta, cmeta, rmeta)
            )
            and bool(g.get("causal_ok", True)),
        }


class RichSequenceReader:
    def __init__(self, world_root: Path, *, store_lru: int = 8) -> None:
        self.world_root = Path(world_root)
        self.seq_dir = self.world_root / "sequences" / "sequence_dataset_v2"
        self._refs_by_oid: dict[str, dict[str, Any]] | None = None
        self._store_cache = _LRU(store_lru)
        self._entity_index: dict[str, dict[str, tuple[int, int]]] = {}
        self._global = GlobalContextReader(self.world_root)
        self._node_vocab: dict[str, int] = {}

    def _refs(self) -> dict[str, dict[str, Any]]:
        if self._refs_by_oid is None:
            path = self.seq_dir / "observation_sequence_refs_v2.parquet"
            self._refs_by_oid = {
                str(r["observation_id"]): r for r in pq.read_table(path).to_pylist()
            }
        return self._refs_by_oid

    def _store(self, name: str) -> Any:
        hit = self._store_cache.get(name)
        if hit is not None:
            return hit
        path = self.seq_dir / name
        table = pq.read_table(path) if path.exists() else None
        self._store_cache.put(name, table)
        return table

    def _entity(self, store_file: str) -> dict[str, tuple[int, int]]:
        if store_file not in self._entity_index:
            ipath = self.seq_dir / store_file.replace(".parquet", "_entity_index.parquet")
            idx: dict[str, tuple[int, int]] = {}
            if ipath.exists():
                for r in pq.read_table(ipath).to_pylist():
                    idx[str(r["entity"])] = (int(r["start_idx"]), int(r["end_idx"]))
            self._entity_index[store_file] = idx
        return self._entity_index[store_file]

    def load_for_assignment(
        self,
        assignment: dict[str, Any],
        *,
        preset: dict[str, Any] | None = None,
        transforms: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        preset = preset or default_window_preset()
        oid = str(assignment["observation_id"])
        T = float(assignment["observation_time_t"])
        ref = self._refs().get(oid)

        def _empty() -> dict[str, Any]:
            return {
                "events": np.zeros((0, len(EVENT_CHANNELS)), np.float32),
                "events_mask": np.zeros((0,), dtype=bool),
                "curve": np.zeros((0, len(CURVE_CHANNELS)), np.float32),
                "curve_mask": np.zeros((0,), dtype=bool),
                "micro": np.zeros((0, len(MICRO_CHANNELS)), np.float32),
                "micro_mask": np.zeros((0,), dtype=bool),
                "ownership": np.zeros((0, len(OWN_CHANNELS)), np.float32),
                "ownership_mask": np.zeros((0,), dtype=bool),
                "graph_token": np.zeros((0, len(GRAPH_CHANNELS)), np.float32),
                "graph_token_mask": np.zeros((0,), dtype=bool),
                "graph_src_idx": np.zeros((0,), dtype=np.int32),
                "graph_dst_idx": np.zeros((0,), dtype=np.int32),
                "wallet_prior": np.zeros((0, len(GRAPH_CHANNELS)), np.float32),
                "wallet_prior_mask": np.zeros((0,), dtype=bool),
                "wallet_node_idx": np.zeros((0,), dtype=np.int32),
                "global": np.zeros((0, 8), np.float32),
                "global_mask": np.zeros((0,), dtype=bool),
                "modality_masks": {
                    "modality_mask_events": 0.0,
                    "modality_mask_curve": 0.0,
                    "modality_mask_micro": 0.0,
                    "modality_mask_ownership": 0.0,
                    "modality_mask_graph": 0.0,
                    "modality_mask_wallet_prior": 0.0,
                    "modality_mask_global": 0.0,
                },
                "window_meta": {},
                "truncation": {},
                "channel_schemas": {
                    "events": list(EVENT_CHANNELS),
                    "curve": list(CURVE_CHANNELS),
                    "micro": list(MICRO_CHANNELS),
                    "ownership": list(OWN_CHANNELS),
                    "graph_token": list(GRAPH_CHANNELS),
                    "wallet_prior": list(GRAPH_CHANNELS),
                },
                "causal_ok": True,
            }

        if ref is None:
            return _empty()

        def _mod_window(store_file: str, start_k: str, end_k: str):
            table = self._store(store_file)
            if table is None:
                return None, 0, 0, {
                    "truncated": False,
                    "raw_len": 0,
                    "max_event_time": None,
                }
            lo0 = int(ref.get(start_k) or 0)
            hi0 = int(ref.get(end_k) or lo0)
            sl = table.slice(lo0, max(0, hi0 - lo0))
            times = _as_float_col(sl, "event_time_unix", sl.num_rows)
            wlo, whi, meta = _window_indices(times, observation_time_t=T, preset=preset)
            return sl, wlo, whi, meta

        ev_sl, elo, ehi, emeta = _mod_window(
            "sequence_events_v2.parquet", "events_start_idx", "events_end_idx"
        )
        cu_sl, clo, chi, cmeta = _mod_window(
            "sequence_curve_v2.parquet", "curve_start_idx", "curve_end_idx"
        )
        mi_sl, mlo, mhi, mmeta = _mod_window(
            "sequence_microstructure_v2.parquet", "micro_start_idx", "micro_end_idx"
        )
        ow_sl, olo, ohi, ometa = _mod_window(
            "sequence_ownership_v2.parquet",
            "ownership_start_idx",
            "ownership_end_idx",
        )
        gr_sl, glo, ghi, gmeta = _mod_window(
            "sequence_graph_token_v2.parquet",
            "graph_token_start_idx",
            "graph_token_end_idx",
        )

        events = (
            encode_events_table(
                ev_sl,
                elo,
                ehi,
                transforms=transforms,
                v_sol_key="v_sol_bonding_curve",
                v_tok_key="v_tokens_bonding_curve",
            )
            if ev_sl is not None
            else np.zeros((0, len(EVENT_CHANNELS)), np.float32)
        )
        curve = (
            encode_curve_table(cu_sl, clo, chi, transforms=transforms)
            if cu_sl is not None
            else np.zeros((0, len(CURVE_CHANNELS)), np.float32)
        )
        micro = (
            encode_micro_table(mi_sl, mlo, mhi, transforms=transforms)
            if mi_sl is not None and bool(ref.get("mask_micro"))
            else np.zeros((0, len(MICRO_CHANNELS)), np.float32)
        )
        ownership = (
            encode_ownership_table(ow_sl, olo, ohi, transforms=transforms)
            if ow_sl is not None and bool(ref.get("mask_ownership"))
            else np.zeros((0, len(OWN_CHANNELS)), np.float32)
        )
        if gr_sl is not None and bool(ref.get("mask_graph_token")):
            graph, src_idx, dst_idx = encode_graph_table(
                gr_sl, glo, ghi, transforms=transforms, node_vocab=self._node_vocab
            )
        else:
            graph = np.zeros((0, len(GRAPH_CHANNELS)), np.float32)
            src_idx = np.zeros((0,), dtype=np.int32)
            dst_idx = np.zeros((0,), dtype=np.int32)

        # Wallet prior: entity-index slices — never full 13M-row pylist
        prior_feats = np.zeros((0, len(GRAPH_CHANNELS)), np.float32)
        wallet_node_idx = np.zeros((0,), dtype=np.int32)
        prior_table = self._store("sequence_graph_wallet_prior_v2.parquet")
        prior_idx = self._entity("sequence_graph_wallet_prior_v2.parquet")
        # Wallets observable on this mint by T from graph slice
        wallets: list[str] = []
        if gr_sl is not None and ghi > glo:
            wcol = _as_str_col(gr_sl.slice(glo, ghi - glo), "wallet", ghi - glo)
            wallets = sorted({w for w in wcol if w})
        prior_rows_meta = []
        if prior_table is not None and wallets:
            chunks = []
            node_chunks = []
            last_n = int(preset.get("last_n") or 256)
            for w in wallets[:64]:  # bound fanout
                wlo, whi = prior_idx.get(w, (0, 0))
                if whi <= wlo:
                    continue
                # causal end via times
                wsl = prior_table.slice(wlo, whi - wlo)
                wtimes = _as_float_col(wsl, "event_time_unix", wsl.num_rows)
                end = int(np.searchsorted(wtimes, T, side="right")) if wtimes.size else 0
                start = max(0, end - last_n)
                if end <= start:
                    continue
                feats, _, _ = encode_graph_table(
                    wsl, start, end, transforms=transforms, node_vocab=self._node_vocab
                )
                if feats.shape[0] == 0:
                    continue
                chunks.append(feats)
                if w not in self._node_vocab:
                    self._node_vocab[w] = len(self._node_vocab) + 1
                node_chunks.append(
                    np.full((feats.shape[0],), self._node_vocab[w], dtype=np.int32)
                )
                prior_rows_meta.append({"wallet_node": self._node_vocab[w], "n": feats.shape[0]})
            if chunks:
                prior_feats = np.concatenate(chunks, axis=0)
                wallet_node_idx = np.concatenate(node_chunks, axis=0)

        gctx = self._global.load_for_t(T, transforms=transforms)

        modality_masks = {
            "modality_mask_events": 1.0
            if bool(ref.get("mask_events")) and events.shape[0]
            else 0.0,
            "modality_mask_curve": 1.0
            if bool(ref.get("mask_curve")) and curve.shape[0]
            else 0.0,
            "modality_mask_micro": 1.0
            if bool(ref.get("mask_micro")) and micro.shape[0]
            else 0.0,
            "modality_mask_ownership": 1.0
            if bool(ref.get("mask_ownership")) and ownership.shape[0]
            else 0.0,
            "modality_mask_graph": 1.0
            if bool(ref.get("mask_graph_token")) and graph.shape[0]
            else 0.0,
            "modality_mask_wallet_prior": 1.0 if prior_feats.shape[0] else 0.0,
            "modality_mask_global": float(gctx.get("modality_mask_global") or 0.0),
        }

        own_level = ref.get("ownership_level_at_t")
        structured = strip_forbidden(
            {
                "token_age_seconds": float(assignment.get("token_age_seconds") or 0.0),
                "ownership_any": 1.0
                if own_level not in (None, "", "none", "unavailable")
                else 0.0,
                **(gctx.get("global_scalars") or {}),
            }
        )

        return {
            "events": events,
            "events_mask": np.ones((events.shape[0],), dtype=bool),
            "curve": curve,
            "curve_mask": np.ones((curve.shape[0],), dtype=bool),
            "micro": micro,
            "micro_mask": np.ones((micro.shape[0],), dtype=bool),
            "ownership": ownership,
            "ownership_mask": np.ones((ownership.shape[0],), dtype=bool),
            "graph_token": graph,
            "graph_token_mask": np.ones((graph.shape[0],), dtype=bool),
            "graph_src_idx": src_idx,
            "graph_dst_idx": dst_idx,
            "wallet_prior": prior_feats,
            "wallet_prior_mask": np.ones((prior_feats.shape[0],), dtype=bool),
            "wallet_node_idx": wallet_node_idx,
            "global": gctx["global"],
            "global_mask": gctx["global_mask"],
            "modality_masks": modality_masks,
            "window_meta": {
                "events": emeta,
                "curve": cmeta,
                "micro": mmeta,
                "ownership": ometa,
                "graph": gmeta,
                "wallet_prior_n_wallets": len(prior_rows_meta),
            },
            "truncation": {
                "events": bool(emeta.get("truncated")),
                "curve": bool(cmeta.get("truncated")),
                "micro": bool(mmeta.get("truncated")),
                "ownership": bool(ometa.get("truncated")),
                "graph": bool(gmeta.get("truncated")),
            },
            "channel_schemas": {
                "events": list(EVENT_CHANNELS),
                "curve": list(CURVE_CHANNELS),
                "micro": list(MICRO_CHANNELS),
                "ownership": list(OWN_CHANNELS),
                "graph_token": list(GRAPH_CHANNELS),
                "wallet_prior": list(GRAPH_CHANNELS),
            },
            "structured_state": structured,
            "causal_ok": (emeta.get("max_event_time") is None or emeta["max_event_time"] <= T)
            and bool(gctx.get("causal_ok", True)),
        }


# Back-compat names used by older tests / smoke
BROAD_EVENT_NUMERIC = EVENT_CONTINUOUS
BROAD_CURVE_NUMERIC = CURVE_CONTINUOUS
BROAD_CREATOR_NUMERIC = CREATOR_CONTINUOUS
RICH_EVENT_NUMERIC = EVENT_CONTINUOUS
