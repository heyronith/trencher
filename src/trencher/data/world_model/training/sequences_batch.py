"""Real causal sequence batching from frozen broad/rich stores."""

from __future__ import annotations

from bisect import bisect_right
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
    load_observation_graph_neighborhood,
)
from trencher.data.world_model.training.forbid import strip_forbidden
from trencher.data.world_model.training.transforms import apply_clip, apply_zscore

# Numeric channels consumed by the model (not identities / binary flags).
BROAD_EVENT_NUMERIC = ("sol_amount", "token_amount", "price_sol", "v_sol", "v_tok")
BROAD_CURVE_NUMERIC = ("price_sol", "v_sol", "v_tok", "curve_pct_depleted")
BROAD_CREATOR_NUMERIC = ("prior_rank",)
RICH_EVENT_NUMERIC = (
    "sol_amount",
    "token_amount",
    "price_sol",
    "v_sol_bonding_curve",
    "v_tokens_bonding_curve",
)


def default_window_preset(name: str = "last_n_256") -> dict[str, Any]:
    for p in WINDOW_PRESETS:
        if p["name"] == name:
            return dict(p)
    return dict(WINDOW_PRESETS[1])


def _rows_to_numeric_array(
    rows: list[dict[str, Any]],
    *,
    time_key: str,
    value_keys: tuple[str, ...],
    transforms: dict[str, Any] | None,
) -> np.ndarray:
    """[L, 1+D] with relative time-delta from first event + numeric channels."""
    if not rows:
        return np.zeros((0, 1 + len(value_keys)), dtype=np.float32)
    t0 = float(rows[0][time_key])
    col_stats = (transforms or {}).get("columns") or {}
    out = np.zeros((len(rows), 1 + len(value_keys)), dtype=np.float32)
    for i, r in enumerate(rows):
        out[i, 0] = float(r[time_key]) - t0
        for j, k in enumerate(value_keys):
            v = r.get(k)
            x = np.nan if v is None else float(v)
            stats = col_stats.get(k)
            if stats and np.isfinite(x):
                x = float(apply_zscore(np.array([x]), stats)[0])
                x = float(apply_clip(np.array([x]), stats)[0])
            out[i, 1 + j] = x
    return out


def _window_rows(
    rows: list[dict[str, Any]],
    *,
    time_key: str,
    observation_time_t: float,
    preset: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    times = [float(r[time_key]) for r in rows]
    # Enforce causal: drop any > T (should already be absent).
    end = bisect_right(times, float(observation_time_t))
    times = times[:end]
    rows = rows[:end]
    lo, hi = apply_training_window(
        times=times,
        start_idx=0,
        end_idx=len(times),
        observation_time_t=float(observation_time_t),
        preset=preset,
    )
    meta = {
        "window_preset": preset.get("name"),
        "raw_len": end,
        "window_lo": lo,
        "window_hi": hi,
        "truncated": hi - lo < end,
        "policy": "most_recent_N_or_last_seconds_causal",
        "max_event_time": times[hi - 1] if hi > lo else None,
    }
    return rows[lo:hi], meta


class BroadSequenceReader:
    def __init__(self, broad_root: Path) -> None:
        self.broad_root = Path(broad_root)
        self.refs_root = self.broad_root / "broad_observation_sequence_refs_v1"
        self._ref_cache: dict[str, dict[str, dict[str, Any]]] = {}
        self._part_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def _load_refs_partition(self, mint_partition: str) -> dict[str, dict[str, Any]]:
        if mint_partition in self._ref_cache:
            return self._ref_cache[mint_partition]
        path = self.refs_root / mint_partition / "part.parquet"
        by_oid: dict[str, dict[str, Any]] = {}
        if path.exists():
            for row in pq.read_table(path).to_pylist():
                oid = row.get("observation_id")
                if oid:
                    by_oid[str(oid)] = row
        self._ref_cache[mint_partition] = by_oid
        return by_oid

    def _load_part_rows(self, store: str, partition: str) -> list[dict[str, Any]]:
        key = (store, partition)
        if key in self._part_cache:
            return self._part_cache[key]
        path = self.broad_root / store / partition / "part.parquet"
        rows = pq.read_table(path).to_pylist() if path.exists() else []
        self._part_cache[key] = rows
        return rows

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
        ref = self._load_refs_partition(part).get(oid)
        if ref is None:
            return {
                "events": np.zeros((0, 1 + len(BROAD_EVENT_NUMERIC)), np.float32),
                "events_mask": np.zeros((0,), dtype=bool),
                "curve": np.zeros((0, 1 + len(BROAD_CURVE_NUMERIC)), np.float32),
                "curve_mask": np.zeros((0,), dtype=bool),
                "creator": np.zeros((0, 1 + len(BROAD_CREATOR_NUMERIC)), np.float32),
                "creator_mask": np.zeros((0,), dtype=bool),
                "modality_masks": {
                    "modality_mask_events": 0.0,
                    "modality_mask_curve": 0.0,
                    "modality_mask_creator": 0.0,
                },
                "window_meta": {},
                "truncation": {},
            }

        def _slice(store: str, start_key: str, end_key: str, part_key: str) -> list[dict]:
            p = str(ref.get(part_key) or part)
            rows = self._load_part_rows(store, p)
            lo = int(ref.get(start_key) or 0)
            hi = int(ref.get(end_key) or lo)
            return rows[lo:hi]

        events_raw = _slice(
            EVENTS_STORE, "events_start_idx", "events_end_idx", "events_partition"
        )
        curve_raw = _slice(
            CURVE_STORE, "curve_start_idx", "curve_end_idx", "curve_partition"
        )
        creator_raw = _slice(
            CREATOR_STORE, "creator_start_idx", "creator_end_idx", "creator_partition"
        )

        events_w, emeta = _window_rows(
            events_raw, time_key="event_time_unix", observation_time_t=T, preset=preset
        )
        curve_w, cmeta = _window_rows(
            curve_raw, time_key="event_time_unix", observation_time_t=T, preset=preset
        )
        creator_w, crmeta = _window_rows(
            creator_raw, time_key="event_time_unix", observation_time_t=T, preset=preset
        )

        # Strip identity columns before numeric conversion
        for bucket in (events_w, curve_w, creator_w):
            for r in bucket:
                r.pop("mint", None)
                r.pop("wallet", None)
                r.pop("creator", None)
                r.pop("prior_mint", None)
                r.pop("tx_signature", None)

        events = _rows_to_numeric_array(
            events_w,
            time_key="event_time_unix",
            value_keys=BROAD_EVENT_NUMERIC,
            transforms=transforms,
        )
        curve = _rows_to_numeric_array(
            curve_w,
            time_key="event_time_unix",
            value_keys=BROAD_CURVE_NUMERIC,
            transforms=transforms,
        )
        creator = _rows_to_numeric_array(
            creator_w,
            time_key="event_time_unix",
            value_keys=BROAD_CREATOR_NUMERIC,
            transforms=transforms,
        )
        return {
            "events": events,
            "events_mask": np.ones((events.shape[0],), dtype=bool),
            "curve": curve,
            "curve_mask": np.ones((curve.shape[0],), dtype=bool),
            "creator": creator,
            "creator_mask": np.ones((creator.shape[0],), dtype=bool),
            "modality_masks": {
                "modality_mask_events": 1.0 if events.shape[0] else 0.0,
                "modality_mask_curve": 1.0 if curve.shape[0] else 0.0,
                "modality_mask_creator": 1.0 if creator.shape[0] else 0.0,
            },
            "window_meta": {"events": emeta, "curve": cmeta, "creator": crmeta},
            "truncation": {
                "events": bool(emeta.get("truncated")),
                "curve": bool(cmeta.get("truncated")),
                "creator": bool(crmeta.get("truncated")),
            },
            "causal_ok": all(
                (m.get("max_event_time") is None or m["max_event_time"] <= T)
                for m in (emeta, cmeta, crmeta)
            ),
        }


class RichSequenceReader:
    def __init__(self, world_root: Path) -> None:
        self.world_root = Path(world_root)
        self.seq_dir = self.world_root / "sequences" / "sequence_dataset_v2"
        self._refs_by_oid: dict[str, dict[str, Any]] | None = None
        self._store_cache: dict[str, list[dict[str, Any]]] = {}

    def _refs(self) -> dict[str, dict[str, Any]]:
        if self._refs_by_oid is None:
            path = self.seq_dir / "observation_sequence_refs_v2.parquet"
            self._refs_by_oid = {
                str(r["observation_id"]): r for r in pq.read_table(path).to_pylist()
            }
        return self._refs_by_oid

    def _store(self, name: str) -> list[dict[str, Any]]:
        if name not in self._store_cache:
            path = self.seq_dir / name
            self._store_cache[name] = (
                pq.read_table(path).to_pylist() if path.exists() else []
            )
        return self._store_cache[name]

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
        mint = str(assignment["mint"])
        ref = self._refs().get(oid)
        empty = {
            "events": np.zeros((0, 1 + len(RICH_EVENT_NUMERIC)), np.float32),
            "events_mask": np.zeros((0,), dtype=bool),
            "curve": np.zeros((0, 5), np.float32),
            "curve_mask": np.zeros((0,), dtype=bool),
            "micro": np.zeros((0, 2), np.float32),
            "micro_mask": np.zeros((0,), dtype=bool),
            "ownership": np.zeros((0, 2), np.float32),
            "ownership_mask": np.zeros((0,), dtype=bool),
            "graph_token": np.zeros((0, 2), np.float32),
            "graph_token_mask": np.zeros((0,), dtype=bool),
            "wallet_prior": np.zeros((0, 2), np.float32),
            "wallet_prior_mask": np.zeros((0,), dtype=bool),
            "modality_masks": {},
            "window_meta": {},
            "truncation": {},
            "causal_ok": True,
        }
        if ref is None:
            empty["modality_masks"] = {
                "modality_mask_events": 0.0,
                "modality_mask_curve": 0.0,
                "modality_mask_micro": 0.0,
                "modality_mask_ownership": 0.0,
                "modality_mask_graph": 0.0,
                "modality_mask_wallet_prior": 0.0,
            }
            return empty

        def _mod(store_file: str, start_k: str, end_k: str) -> list[dict]:
            rows = self._store(store_file)
            lo = int(ref.get(start_k) or 0)
            hi = int(ref.get(end_k) or lo)
            return rows[lo:hi]

        events_raw = _mod(
            "sequence_events_v2.parquet", "events_start_idx", "events_end_idx"
        )
        curve_raw = _mod("sequence_curve_v2.parquet", "curve_start_idx", "curve_end_idx")
        micro_raw = _mod(
            "sequence_microstructure_v2.parquet", "micro_start_idx", "micro_end_idx"
        )
        own_raw = _mod(
            "sequence_ownership_v2.parquet", "ownership_start_idx", "ownership_end_idx"
        )
        graph_raw = _mod(
            "sequence_graph_token_v2.parquet",
            "graph_token_start_idx",
            "graph_token_end_idx",
        )

        events_w, emeta = _window_rows(
            events_raw, time_key="event_time_unix", observation_time_t=T, preset=preset
        )
        for r in events_w:
            r.pop("mint", None)
            r.pop("wallet", None)
            r.pop("tx_signature", None)
        events = _rows_to_numeric_array(
            events_w,
            time_key="event_time_unix",
            value_keys=RICH_EVENT_NUMERIC,
            transforms=transforms,
        )

        def _presence_seq(
            rows: list[dict], *, available: bool
        ) -> tuple[np.ndarray, dict[str, Any]]:
            if not rows or not available:
                return np.zeros((0, 2), np.float32), {
                    "truncated": False,
                    "raw_len": 0,
                    "max_event_time": None,
                }
            w, meta = _window_rows(
                rows, time_key="event_time_unix", observation_time_t=T, preset=preset
            )
            arr = np.zeros((len(w), 2), np.float32)
            if w:
                t0 = float(w[0]["event_time_unix"])
                for i, r in enumerate(w):
                    arr[i, 0] = float(r["event_time_unix"]) - t0
                    arr[i, 1] = 1.0
            return arr, meta

        curve_keys = tuple(
            k
            for k in (
                "price_sol",
                "v_sol",
                "v_tok",
                "v_sol_bonding_curve",
                "curve_pct_depleted",
            )
            if curve_raw and k in curve_raw[0]
        )
        if not curve_keys:
            curve_keys = ("price_sol",)
        curve_w, cmeta = _window_rows(
            curve_raw, time_key="event_time_unix", observation_time_t=T, preset=preset
        )
        for r in curve_w:
            r.pop("mint", None)
            r.pop("wallet", None)
        curve = _rows_to_numeric_array(
            curve_w,
            time_key="event_time_unix",
            value_keys=curve_keys,
            transforms=None,
        )

        micro_arr, micro_meta = _presence_seq(
            micro_raw, available=bool(ref.get("mask_micro"))
        )
        own_arr, own_meta = _presence_seq(
            own_raw, available=bool(ref.get("mask_ownership"))
        )
        graph_arr, graph_meta = _presence_seq(
            graph_raw, available=bool(ref.get("mask_graph_token"))
        )

        # Wallet prior via existing causal graph loader (indexing only; no raw IDs in tensors)
        prior_arr = np.zeros((0, 2), np.float32)
        try:
            nb = load_observation_graph_neighborhood(
                out_dir=self.seq_dir, mint=mint, observation_time_t=T, last_n=256
            )
            prior_refs = nb.get("wallet_prior_refs") or []
            # Encode as count + mean prior length (no wallet ids)
            if prior_refs:
                lengths = [int(p.get("count") or 0) for p in prior_refs]
                prior_arr = np.array(
                    [[float(len(lengths)), float(np.mean(lengths))]], dtype=np.float32
                )
        except Exception:
            prior_refs = []

        modality_masks = {
            "modality_mask_events": 1.0
            if bool(ref.get("mask_events")) and events.shape[0]
            else 0.0,
            "modality_mask_curve": 1.0
            if bool(ref.get("mask_curve")) and curve.shape[0]
            else 0.0,
            "modality_mask_micro": 1.0
            if bool(ref.get("mask_micro")) and micro_arr.shape[0]
            else 0.0,
            "modality_mask_ownership": 1.0
            if bool(ref.get("mask_ownership")) and own_arr.shape[0]
            else 0.0,
            "modality_mask_graph": 1.0
            if bool(ref.get("mask_graph_token")) and graph_arr.shape[0]
            else 0.0,
            "modality_mask_wallet_prior": 1.0 if prior_arr.shape[0] else 0.0,
        }
        return {
            "events": events,
            "events_mask": np.ones((events.shape[0],), dtype=bool),
            "curve": curve,
            "curve_mask": np.ones((curve.shape[0],), dtype=bool),
            "micro": micro_arr,
            "micro_mask": np.ones((micro_arr.shape[0],), dtype=bool),
            "ownership": own_arr,
            "ownership_mask": np.ones((own_arr.shape[0],), dtype=bool),
            "graph_token": graph_arr,
            "graph_token_mask": np.ones((graph_arr.shape[0],), dtype=bool),
            "wallet_prior": prior_arr,
            "wallet_prior_mask": np.ones((prior_arr.shape[0],), dtype=bool),
            "modality_masks": modality_masks,
            "window_meta": {"events": emeta, "curve": cmeta},
            "truncation": {
                "events": bool(emeta.get("truncated")),
                "curve": bool(cmeta.get("truncated")),
                "micro": bool(micro_meta.get("truncated")),
                "ownership": bool(own_meta.get("truncated")),
                "graph": bool(graph_meta.get("truncated")),
            },
            "causal_ok": (emeta.get("max_event_time") is None or emeta["max_event_time"] <= T),
            "structured_state": strip_forbidden(
                {
                    "token_age_seconds": float(assignment.get("token_age_seconds") or 0.0),
                    "ownership_any": 1.0
                    if ref.get("ownership_level_at_t")
                    not in (None, "", "none", "unavailable")
                    else 0.0,
                }
            ),
        }
