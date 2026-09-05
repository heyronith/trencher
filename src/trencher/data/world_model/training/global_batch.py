"""Global PIT-safe context sequences (last-completed-second policy)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from trencher.data.world_model.pit import last_completed_second
from trencher.data.world_model.training.schemas import GLOBAL_BUCKET_CHANNELS
from trencher.data.world_model.training.transforms import apply_numeric_channel


@lru_cache(maxsize=4)
def _load_trade_buckets(path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    table = pq.read_table(path)
    seconds = np.asarray(table.column("second_unix").to_numpy(zero_copy_only=False), dtype=np.int64)
    cols = {}
    for c in ("trades", "buys", "sells", "buy_sol", "sell_sol", "unique_mints", "unique_wallets"):
        if c in table.column_names:
            cols[c] = np.asarray(table.column(c).to_numpy(zero_copy_only=False), dtype=np.float64)
        else:
            cols[c] = np.zeros(len(seconds), dtype=np.float64)
    order = np.argsort(seconds)
    seconds = seconds[order]
    cols = {k: v[order] for k, v in cols.items()}
    return seconds, cols


@lru_cache(maxsize=8)
def _load_timeline(path: str) -> np.ndarray:
    if not Path(path).exists():
        return np.zeros((0,), dtype=np.float64)
    table = pq.read_table(path, columns=["event_time_unix"])
    arr = np.asarray(table.column(0).to_numpy(zero_copy_only=False), dtype=np.float64)
    return np.sort(arr[np.isfinite(arr)])


@lru_cache(maxsize=4)
def _load_daily_network(path: str) -> dict[str, dict[str, float]]:
    if not Path(path).exists():
        return {}
    rows = pq.read_table(path).to_pylist()
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        day = str(r.get("day") or "")
        out[day] = {
            "launches": float(r.get("launches") or 0.0),
            "graduations": float(r.get("graduations") or 0.0),
            "trades": float(r.get("trades") or 0.0),
            "buy_volume_sol": float(r.get("buy_volume_sol") or 0.0),
            "graduation_rate": float(r.get("graduation_rate") or 0.0),
        }
    return out


class GlobalContextReader:
    def __init__(self, world_root: Path, *, last_n_seconds: int = 128) -> None:
        self.world_root = Path(world_root)
        self.last_n = int(last_n_seconds)
        self.trade_path = (
            self.world_root / "pit" / "global" / "trade_second_buckets.parquet"
        )
        self.launches_path = (
            self.world_root / "pit" / "global" / "launches_timeline.parquet"
        )
        self.migrations_path = (
            self.world_root / "pit" / "global" / "migrations_timeline.parquet"
        )
        self.network_path = (
            self.world_root / "global_context" / "daily_pump_network_context.parquet"
        )
        self._by_last_sec: dict[int, dict[str, Any]] = {}

    def load_for_t(
        self,
        observation_time_t: float,
        *,
        transforms: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        T = float(observation_time_t)
        last_sec = last_completed_second(T)
        # Cache by last completed second (transforms rarely touch global buckets).
        hit = self._by_last_sec.get(last_sec)
        if hit is not None:
            return hit
        out = self._load_uncached(T, last_sec, transforms=transforms)
        self._by_last_sec[last_sec] = out
        if len(self._by_last_sec) > 8192:
            self._by_last_sec.pop(next(iter(self._by_last_sec)))
        return out

    def _load_uncached(
        self,
        T: float,
        last_sec: int,
        *,
        transforms: dict[str, Any] | None,
    ) -> dict[str, Any]:
        empty = np.zeros((0, len(GLOBAL_BUCKET_CHANNELS)), dtype=np.float32)
        if not self.trade_path.exists():
            return {
                "global": empty,
                "global_mask": np.zeros((0,), dtype=bool),
                "global_scalars": {},
                "modality_mask_global": 0.0,
                "pit_last_completed_second": last_sec,
                "max_bucket_second": None,
                "causal_ok": True,
            }

        seconds, cols = _load_trade_buckets(str(self.trade_path))
        end = int(np.searchsorted(seconds, last_sec, side="right"))
        start = max(0, end - self.last_n)
        sl = slice(start, end)
        n = end - start
        arr = np.zeros((n, len(GLOBAL_BUCKET_CHANNELS)), dtype=np.float32)
        col_stats = (transforms or {}).get("columns") or {}
        if n:
            sec = seconds[sl].astype(np.float64)
            arr[:, 0] = (sec - sec[0]).astype(np.float32)
            for j, name in enumerate(GLOBAL_BUCKET_CHANNELS[1:], start=1):
                raw = cols[name][sl].astype(np.float64)
                stats = col_stats.get(name)
                if stats is not None:
                    raw = apply_numeric_channel(raw, stats)
                arr[:, j] = raw.astype(np.float32)

        launches = _load_timeline(str(self.launches_path))
        migrations = _load_timeline(str(self.migrations_path))

        def _count(arr_t: np.ndarray, window: float) -> float:
            if arr_t.size == 0:
                return 0.0
            hi = int(np.searchsorted(arr_t, float(last_sec), side="right"))
            lo = int(np.searchsorted(arr_t, float(last_sec) - window, side="right"))
            return float(hi - lo)

        from datetime import UTC, datetime

        day = datetime.fromtimestamp(T, tz=UTC).strftime("%Y-%m-%d")
        net = _load_daily_network(str(self.network_path)).get(day, {})

        scalars = {
            "global_launches_1h": _count(launches, 3600.0),
            "global_launches_24h": _count(launches, 86400.0),
            "global_migrations_1h": _count(migrations, 3600.0),
            "global_migrations_24h": _count(migrations, 86400.0),
            "daily_network_launches": float(net.get("launches") or 0.0),
            "daily_network_graduations": float(net.get("graduations") or 0.0),
            "daily_network_trades": float(net.get("trades") or 0.0),
            "daily_network_buy_volume_sol": float(net.get("buy_volume_sol") or 0.0),
            "daily_network_graduation_rate": float(net.get("graduation_rate") or 0.0),
            "pit_last_completed_second": float(last_sec),
        }
        max_sec = int(seconds[end - 1]) if end > start else None
        return {
            "global": arr,
            "global_mask": np.ones((n,), dtype=bool),
            "global_scalars": scalars,
            "modality_mask_global": 1.0 if n else 0.0,
            "pit_last_completed_second": last_sec,
            "max_bucket_second": max_sec,
            "causal_ok": max_sec is None or max_sec <= last_sec,
        }
