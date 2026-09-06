"""Causal rich-modality aggregates from frozen ``sequence_dataset_v2``.

Uses the same PIT rules as ``RichSequenceReader``:
- modality windows from ``observation_sequence_refs_v2`` cut at T
- wallet prior: wallets observable on the focal mint by T, each wallet's
  prior history truncated with ``searchsorted(..., side="right")``

Never reads lifetime ``graph/edges_wallet_token.parquet`` aggregates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trencher.baselines.features import rich_features_from_arrays

SEQ_DIR_REL = Path("sequences") / "sequence_dataset_v2"

MICRO_STORE = "sequence_microstructure_v2.parquet"
OWN_STORE = "sequence_ownership_v2.parquet"
GRAPH_STORE = "sequence_graph_token_v2.parquet"
PRIOR_STORE = "sequence_graph_wallet_prior_v2.parquet"
REFS_STORE = "observation_sequence_refs_v2.parquet"

_WALLET_PRIOR_FANOUT = 64
_WALLET_PRIOR_LAST_N = 256


def _as_unix_seconds(ts: Any) -> float | None:
    """Coerce parquet timestamp / numeric / datetime values to unix seconds."""
    if ts is None:
        return None
    if hasattr(ts, "timestamp"):
        return float(ts.timestamp())
    try:
        return float(ts)
    except (TypeError, ValueError):
        return None


def _as_float_col(table: pa.Table, name: str, n: int) -> np.ndarray:
    if name not in table.column_names:
        return np.full(n, np.nan, dtype=np.float64)
    return np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype=np.float64)


def _as_str_col(table: pa.Table, name: str, n: int) -> list[str]:
    if name not in table.column_names:
        return [""] * n
    return [str(x or "") for x in table.column(name).to_pylist()]


def causal_slice_arrays(
    pack: dict[str, Any] | None, T: float
) -> dict[str, np.ndarray] | None:
    """Truncate a time-sorted pack to events with time <= T."""
    if not pack:
        return None
    times = np.asarray(pack["times"], dtype=np.float64)
    hi = int(np.searchsorted(times, T, side="right"))
    if hi <= 0:
        return None
    out: dict[str, Any] = {}
    for k, v in pack.items():
        if k == "times":
            continue
        if isinstance(v, list):
            out[k] = v[:hi]
        else:
            out[k] = np.asarray(v)[:hi]
    out["times"] = times[:hi]
    return out


class CausalRichSequenceIndex:
    """Lightweight PIT aggregate reader over frozen rich sequence stores."""

    def __init__(self, world_root: Path) -> None:
        self.world_root = Path(world_root)
        self.seq_dir = self.world_root / SEQ_DIR_REL
        self._refs_by_oid: dict[str, dict[str, Any]] | None = None
        self._tables: dict[str, pa.Table | None] = {}
        self._entity: dict[str, dict[str, tuple[int, int]]] = {}

    def _table(self, name: str) -> pa.Table | None:
        if name not in self._tables:
            path = self.seq_dir / name
            self._tables[name] = pq.read_table(path) if path.exists() else None
        return self._tables[name]

    def _entity_index(self, store_file: str) -> dict[str, tuple[int, int]]:
        if store_file not in self._entity:
            ipath = self.seq_dir / store_file.replace(".parquet", "_entity_index.parquet")
            idx: dict[str, tuple[int, int]] = {}
            if ipath.exists():
                for r in pq.read_table(ipath).to_pylist():
                    idx[str(r["entity"])] = (int(r["start_idx"]), int(r["end_idx"]))
            self._entity[store_file] = idx
        return self._entity[store_file]

    def refs(self) -> dict[str, dict[str, Any]]:
        if self._refs_by_oid is None:
            path = self.seq_dir / REFS_STORE
            if not path.exists():
                self._refs_by_oid = {}
            else:
                self._refs_by_oid = {
                    str(r["observation_id"]): r for r in pq.read_table(path).to_pylist()
                }
        return self._refs_by_oid

    def _mod_slice(
        self,
        store: str,
        ref: dict[str, Any],
        start_k: str,
        end_k: str,
        T: float,
    ) -> pa.Table | None:
        table = self._table(store)
        if table is None:
            return None
        lo0 = int(ref.get(start_k) or 0)
        hi0 = int(ref.get(end_k) or lo0)
        if hi0 <= lo0:
            return None
        sl = table.slice(lo0, hi0 - lo0)
        times = _as_float_col(sl, "event_time_unix", sl.num_rows)
        hi = int(np.searchsorted(times, T, side="right"))
        if hi <= 0:
            return None
        return sl.slice(0, hi)

    def _micro_pack(self, sl: pa.Table | None) -> dict[str, np.ndarray] | None:
        if sl is None or sl.num_rows == 0:
            return None
        n = sl.num_rows
        return {
            "times": _as_float_col(sl, "event_time_unix", n),
            "fee_lamports": _as_float_col(sl, "fee_lamports", n),
            "compute_units_consumed": _as_float_col(sl, "compute_units_consumed", n),
            "priority_fee_lamports": _as_float_col(sl, "priority_fee_lamports", n),
            "success": _as_float_col(sl, "success", n),
            "account_key_count": _as_float_col(sl, "account_key_count", n),
        }

    def _ownership_pack(self, sl: pa.Table | None) -> dict[str, np.ndarray] | None:
        """Use only fields present on the frozen ownership sequence (no pre=0 synthesis)."""
        if sl is None or sl.num_rows == 0:
            return None
        n = sl.num_rows
        value = _as_float_col(sl, "value", n)
        pre = _as_float_col(sl, "pre_ui_amount", n)
        post = _as_float_col(sl, "post_ui_amount", n)
        # delta only where both sides are finite — never invent pre=0 from value
        delta = np.full(n, np.nan, dtype=np.float64)
        both = np.isfinite(pre) & np.isfinite(post)
        delta[both] = post[both] - pre[both]
        return {
            "times": _as_float_col(sl, "event_time_unix", n),
            "value": value,
            "pre_ui_amount": pre,
            "post_ui_amount": post,
            "delta_ui": delta,
        }

    def _graph_pack(self, sl: pa.Table | None) -> dict[str, Any] | None:
        if sl is None or sl.num_rows == 0:
            return None
        n = sl.num_rows
        return {
            "times": _as_float_col(sl, "event_time_unix", n),
            "sol_amount": _as_float_col(sl, "sol_amount", n),
            "token_amount": _as_float_col(sl, "token_amount", n),
            "edge_type": _as_str_col(sl, "edge_type", n),
            "wallet": _as_str_col(sl, "wallet", n),
        }

    def _wallet_prior_pack(self, graph: dict[str, Any] | None, T: float) -> dict[str, Any] | None:
        if not graph:
            return None
        wallets = sorted({w for w in graph.get("wallet", []) if w})[:_WALLET_PRIOR_FANOUT]
        if not wallets:
            return None
        prior_table = self._table(PRIOR_STORE)
        prior_idx = self._entity_index(PRIOR_STORE)
        if prior_table is None:
            return None
        times_all: list[float] = []
        sol_all: list[float] = []
        wallet_ids: list[str] = []
        for w in wallets:
            wlo, whi = prior_idx.get(w, (0, 0))
            if whi <= wlo:
                continue
            wsl = prior_table.slice(wlo, whi - wlo)
            wtimes = _as_float_col(wsl, "event_time_unix", wsl.num_rows)
            end = int(np.searchsorted(wtimes, T, side="right")) if wtimes.size else 0
            start = max(0, end - _WALLET_PRIOR_LAST_N)
            if end <= start:
                continue
            chunk = wsl.slice(start, end - start)
            cn = chunk.num_rows
            sols = _as_float_col(chunk, "sol_amount", cn)
            cts = _as_float_col(chunk, "event_time_unix", cn)
            for i in range(cn):
                times_all.append(float(cts[i]))
                sol_all.append(float(sols[i]) if np.isfinite(sols[i]) else 0.0)
                wallet_ids.append(w)
        if not times_all:
            return None
        return {
            "times": np.asarray(times_all, dtype=np.float64),
            "sol_amount": np.asarray(sol_all, dtype=np.float64),
            "wallet": wallet_ids,
        }

    def packs_for_observation(
        self, observation_id: str, observation_time_t: float
    ) -> dict[str, dict[str, Any] | None]:
        T = float(observation_time_t)
        ref = self.refs().get(str(observation_id)) or {}
        micro_sl = (
            self._mod_slice(MICRO_STORE, ref, "micro_start_idx", "micro_end_idx", T)
            if bool(ref.get("mask_micro"))
            else None
        )
        own_sl = (
            self._mod_slice(
                OWN_STORE, ref, "ownership_start_idx", "ownership_end_idx", T
            )
            if bool(ref.get("mask_ownership"))
            else None
        )
        graph_sl = (
            self._mod_slice(
                GRAPH_STORE, ref, "graph_token_start_idx", "graph_token_end_idx", T
            )
            if bool(ref.get("mask_graph_token"))
            else None
        )
        graph = self._graph_pack(graph_sl)
        return {
            "micro": self._micro_pack(micro_sl),
            "ownership": self._ownership_pack(own_sl),
            "graph": graph,
            "wallet_prior": self._wallet_prior_pack(graph, T),
        }

    def features_for_observation(
        self, observation_id: str, observation_time_t: float
    ) -> dict[str, float]:
        packs = self.packs_for_observation(observation_id, observation_time_t)
        feat = rich_features_from_arrays(
            micro=packs["micro"],
            ownership=packs["ownership"],
            graph=packs["graph"],
            wallet_prior=packs["wallet_prior"],
        )
        # Prefer unique wallet count from prior pack when available
        wp = packs["wallet_prior"]
        if wp and wp.get("wallet"):
            feat["wallet_prior_unique_approx"] = float(len(set(wp["wallet"])))
        return feat


# Back-compat aliases — must not load lifetime edges_wallet_token.
def load_wallet_prior_by_mint(world_root: Path) -> dict[str, dict[str, np.ndarray]]:
    """Deprecated: lifetime mint-edge priors are forbidden.

    Returns empty; callers must use ``CausalRichSequenceIndex``.
    """
    del world_root
    return {}


def load_micro_by_mint(world_root: Path) -> dict[str, dict[str, np.ndarray]]:
    del world_root
    return {}


def load_ownership_by_mint(world_root: Path) -> dict[str, dict[str, np.ndarray]]:
    del world_root
    return {}


def load_graph_by_mint(world_root: Path) -> dict[str, dict[str, np.ndarray]]:
    del world_root
    return {}
