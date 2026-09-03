"""GeckoTerminal acquisition for SOL market and post-grad pools."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from trencher.data.world_model.constants import (
    CORPUS_END,
    CORPUS_START,
    GECKO_BASE_URL,
    GECKO_MIN_INTERVAL_SEC,
    GECKO_NETWORK,
    GECKO_SOL_USDC_POOL,
)
from trencher.data.world_model.io import ensure_dir, file_provenance, read_json, utc_now, write_json

logger = logging.getLogger(__name__)


class GeckoClient:
    def __init__(
        self,
        *,
        base_url: str = GECKO_BASE_URL,
        min_interval: float = GECKO_MIN_INTERVAL_SEC,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.min_interval = min_interval
        self.session = session or requests.Session()
        self.session.headers.setdefault("Accept", "application/json")
        self._last = 0.0
        self.calls = 0

    def get(self, path: str, params: dict[str, Any] | None = None) -> tuple[int, Any]:
        url = f"{self.base_url}{path}"
        for attempt in range(6):
            elapsed = time.monotonic() - self._last
            wait = (
                self.min_interval if attempt == 0 else min(60.0, self.min_interval * (2**attempt))
            )
            if elapsed < wait:
                time.sleep(wait - elapsed)
            resp = self.session.get(url, params=params, timeout=60)
            self._last = time.monotonic()
            self.calls += 1
            try:
                body = resp.json()
            except Exception:  # noqa: BLE001
                body = {"raw": resp.text[:500]}
            if resp.status_code != 429:
                return resp.status_code, body
            logger.warning("GeckoTerminal 429; backing off (attempt %s)", attempt + 1)
        return 429, body


def _ohlcv_to_rows(pool: str, ohlcv: list[Any], timeframe: str) -> list[dict[str, Any]]:
    rows = []
    for item in ohlcv:
        if not item or len(item) < 6:
            continue
        ts, o, h, low, c, vol = item[:6]
        rows.append(
            {
                "pool_address": pool,
                "timeframe": timeframe,
                "ts": int(ts),
                "open": o,
                "high": h,
                "low": low,
                "close": c,
                "volume": vol,
                "quality_flag": "geckoterminal_observed",
            }
        )
    return rows


def fetch_sol_usdc_ohlcv(client: GeckoClient, out_dir: Path) -> dict[str, Any]:
    ensure_dir(out_dir)
    path = f"/networks/{GECKO_NETWORK}/pools/{GECKO_SOL_USDC_POOL}/ohlcv/day"
    code, body = client.get(path, params={"aggregate": 1, "limit": 1000, "currency": "usd"})
    ohlcv = []
    if isinstance(body, dict):
        ohlcv = body.get("data", {}).get("attributes", {}).get("ohlcv_list") or []
    rows = _ohlcv_to_rows(GECKO_SOL_USDC_POOL, ohlcv, "day")
    out = out_dir / "sol_usdc_ohlcv_day.parquet"
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), out)
    else:
        pq.write_table(pa.table({"pool_address": pa.array([], pa.string())}), out)

    start_ts = datetime(
        CORPUS_START.year, CORPUS_START.month, CORPUS_START.day, tzinfo=UTC
    ).timestamp()
    end_ts = datetime(
        CORPUS_END.year, CORPUS_END.month, CORPUS_END.day, tzinfo=UTC
    ).timestamp()
    in_window = [r for r in rows if start_ts <= r["ts"] <= end_ts + 86400]
    return {
        "status_code": code,
        "rows": len(rows),
        "corpus_window_rows": len(in_window),
        "covers_corpus": len(in_window) >= (CORPUS_END - CORPUS_START).days * 0.8,
        "path": file_provenance(out, source="geckoterminal"),
    }


def fetch_pool_ohlcv_batch(
    client: GeckoClient,
    pools: list[str],
    out_dir: Path,
    *,
    checkpoint: Path,
    max_pools: int,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    done: set[str] = set()
    if checkpoint.exists():
        done = set(read_json(checkpoint).get("completed_pools", []))

    targets = [p for p in pools if p not in done][: max(0, max_pools - len(done))]
    all_rows: list[dict[str, Any]] = []
    # Reload existing parquet if present.
    out_path = out_dir / "graduated_pool_ohlcv_day.parquet"
    if out_path.exists():
        existing = pq.read_table(out_path).to_pylist()
        all_rows.extend(existing)

    failures = []
    for pool in targets:
        code, body = client.get(
            f"/networks/{GECKO_NETWORK}/pools/{pool}/ohlcv/day",
            params={"aggregate": 1, "limit": 200, "currency": "usd"},
        )
        if code != 200:
            failures.append({"pool": pool, "status_code": code})
            done.add(pool)  # don't hammer forever
            write_json(checkpoint, {"completed_pools": sorted(done), "updated_at": utc_now()})
            continue
        ohlcv = body.get("data", {}).get("attributes", {}).get("ohlcv_list") or []
        all_rows.extend(_ohlcv_to_rows(pool, ohlcv, "day"))
        done.add(pool)
        write_json(checkpoint, {"completed_pools": sorted(done), "updated_at": utc_now()})

    if all_rows:
        # de-dupe by pool+ts
        uniq = {(r["pool_address"], r["ts"]): r for r in all_rows}
        pq.write_table(pa.Table.from_pylist(list(uniq.values())), out_path)
    else:
        pq.write_table(pa.table({"pool_address": pa.array([], pa.string())}), out_path)

    return {
        "pools_attempted": len(targets),
        "pools_completed_total": len(done),
        "failures": failures[:50],
        "failure_count": len(failures),
        "rows": pq.ParquetFile(out_path).metadata.num_rows,
        "path": file_provenance(out_path, source="geckoterminal"),
        "calls": client.calls,
    }


def run_gecko_acquisition(
    *,
    world_root: Path,
    pools: list[str],
    max_pools: int = 200,
    client: GeckoClient | None = None,
    skip_live: bool = False,
) -> dict[str, Any]:
    out_dir = ensure_dir(world_root / "gecko")
    if skip_live:
        payload = {
            "skipped": True,
            "reason": "skip_live",
            "generated_at": utc_now(),
        }
        write_json(out_dir / "gecko_manifest.json", payload)
        return payload
    client = client or GeckoClient()
    sol = fetch_sol_usdc_ohlcv(client, out_dir)
    pools_result = fetch_pool_ohlcv_batch(
        client,
        pools,
        out_dir,
        checkpoint=out_dir / "pool_ohlcv_checkpoint.json",
        max_pools=max_pools,
    )
    payload = {
        "generated_at": utc_now(),
        "sol_usdc": sol,
        "graduated_pools": pools_result,
        "api_calls": client.calls,
    }
    write_json(out_dir / "gecko_manifest.json", payload)
    return payload
