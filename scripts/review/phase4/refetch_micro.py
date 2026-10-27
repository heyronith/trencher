#!/usr/bin/env python3
# ruff: noqa: E501,E402
"""Leak-free microstructure signature selection + optional RPC refetch.

Selection rule: for every rich train+val observation, take the last K=16
pump trades with event_time strictly before observation T
(``select_causal_trades_before``).

Writes under ``data/external/world_model_clean_v1/microstructure/`` only.
Never touches ``data/external/world_model/`` freeze paths.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from trencher.data.world_model.acquire.local_corpus import (  # noqa: E402
    select_causal_trades_before,
)
from trencher.data.world_model.io import ensure_dir, write_json  # noqa: E402

WORLD = REPO / "data" / "external" / "world_model"
SPLIT = WORLD / "splits" / "split_v1"
CLEAN = REPO / "data" / "external" / "world_model_clean_v1" / "microstructure"
K = 16
Q85 = {"q85_test_accessed": False}


def _rpc_configured() -> tuple[bool, str | None]:
    url = os.environ.get("SOLANA_RPC_URL") or None
    helius = os.environ.get("HELIUS_API_KEY") or None
    if url:
        return True, url
    if helius:
        return True, f"https://mainnet.helius-rpc.com/?api-key={helius}"
    return False, None


def _rpc_host_redacted(url: str) -> str:
    p = urlparse(url)
    host = p.netloc.split("@")[-1]
    # strip query api-key
    return host.split("?")[0]


def _load_rich_obs() -> list[dict[str, Any]]:
    """Rich train+val observation_id, mint, observation_time_t. No TEST."""
    path = SPLIT / "baseline_features_v1" / "rich" / "by_mint_partition"
    rows: list[dict[str, Any]] = []
    for part in sorted(path.glob("*/features.parquet")):
        table = pq.read_table(
            part,
            columns=["observation_id", "mint", "observation_time_t", "base_split"],
        )
        table = table.filter(
            pc.is_in(table["base_split"], value_set=pa.array(["train", "val"]))
        )
        for oid, mint, t, split in zip(
            table["observation_id"].to_pylist(),
            table["mint"].to_pylist(),
            table["observation_time_t"].to_pylist(),
            table["base_split"].to_pylist(),
            strict=True,
        ):
            if split not in ("train", "val"):
                continue
            if t is None or mint is None or oid is None:
                continue
            rows.append(
                {
                    "observation_id": str(oid),
                    "mint": str(mint),
                    "observation_time_t": float(t),
                    "base_split": str(split),
                }
            )
    return rows


def _scan_trades_for_mints(mints: set[str]) -> dict[str, list[dict[str, Any]]]:
    pump = WORLD / "raw" / "pumpfun"
    # Prefer local corpus trades if present; else sequences are not full trade lists.
    trades_root = None
    for cand in (
        pump / "trades",
        WORLD / ".." / "pumpfun" / "trades",
        REPO / "data" / "external" / "pumpfun" / "trades",
    ):
        if Path(cand).exists():
            trades_root = Path(cand)
            break
    if trades_root is None:
        # Fall back: reconstruct from sequence_dataset_v2 micro event times is
        # insufficient for signature refetch. Use broad/sequence refs if pump trades
        # live under a known path from freeze docs.
        alt = REPO / "data" / "external" / "red" / "pumpfun" / "trades"
        if alt.exists():
            trades_root = alt
    if trades_root is None:
        raise FileNotFoundError(
            "pumpfun trades parquet not found locally; cannot enumerate causal sigs"
        )
    import pyarrow.dataset as ds

    dataset = ds.dataset(trades_root, format="parquet")
    by_mint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cols = ["mint", "tx_signature", "event_time", "sol_amount", "is_buy", "curve_pct_depleted"]
    have = set(dataset.schema.names)
    use_cols = [c for c in cols if c in have]
    scanner = dataset.scanner(columns=use_cols, batch_size=256_000)
    for batch in scanner.to_batches():
        data = {c: batch.column(c).to_pylist() for c in use_cols}
        n = len(data["mint"])
        for i in range(n):
            mint = data["mint"][i]
            if mint not in mints:
                continue
            sig = data.get("tx_signature", [None] * n)[i]
            if not sig:
                continue
            by_mint[str(mint)].append(
                {
                    "mint": str(mint),
                    "tx_signature": str(sig),
                    "event_time": data.get("event_time", [None] * n)[i],
                    "sol_amount": data.get("sol_amount", [None] * n)[i],
                    "is_buy": data.get("is_buy", [None] * n)[i],
                    "curve_pct_depleted": data.get("curve_pct_depleted", [None] * n)[i],
                }
            )
    return by_mint


def collect_causal_signatures(*, k: int = K) -> dict[str, Any]:
    obs = _load_rich_obs()
    mints = {r["mint"] for r in obs}
    trades = _scan_trades_for_mints(mints)
    selected_rows: list[dict[str, Any]] = []
    sigs: set[str] = set()
    for row in obs:
        picks = select_causal_trades_before(
            trades.get(row["mint"], []), row["observation_time_t"], k
        )
        for p in picks:
            sig = p["tx_signature"]
            sigs.add(sig)
            selected_rows.append(
                {
                    "observation_id": row["observation_id"],
                    "mint": row["mint"],
                    "base_split": row["base_split"],
                    "observation_time_t": row["observation_time_t"],
                    "tx_signature": sig,
                    "event_time": str(p.get("event_time")),
                }
            )
    return {
        **Q85,
        "k": k,
        "n_observations": len(obs),
        "n_mints": len(mints),
        "n_selected_pairs": len(selected_rows),
        "n_unique_signatures": len(sigs),
        "signatures": sorted(sigs),
        "selected_rows": selected_rows,
    }


def dry_run() -> dict[str, Any]:
    ensure_dir(CLEAN)
    try:
        payload = collect_causal_signatures(k=K)
    except FileNotFoundError as exc:
        # Phase-3 estimated 54144 from rich obs × last-K without full trade scan.
        # Reuse that estimate when pump trades are absent locally.
        # Count unique (mint,T) approx via rich feature cache train+val oids
        rich_root = SPLIT / "baseline_features_v1" / "rich" / "by_mint_partition"
        n_obs = 0
        for part in rich_root.glob("*/features.parquet"):
            t = pq.read_table(part, columns=["base_split"])
            t = t.filter(pc.is_in(t["base_split"], value_set=pa.array(["train", "val"])))
            n_obs += t.num_rows
        # Prior phase-3 dry-run unique sig count
        est_unique = 54144
        out = {
            **Q85,
            "status": "dry_run_estimate",
            "reason": str(exc),
            "k": K,
            "n_observations_train_val": n_obs,
            "n_unique_signatures_estimate": est_unique,
            "selection_rule": "last_K_pump_trades_strictly_before_T",
            "rpc_configured": False,
            "note": (
                "Full trade corpus unavailable locally; signature count estimate "
                "from phase-3 causal dry-run (54144). RPC not configured."
            ),
        }
        write_json(CLEAN / "dry_run_manifest.json", out)
        print(json.dumps({k: v for k, v in out.items() if k != "signatures"}, indent=2))
        return out

    out = {
        **Q85,
        "status": "dry_run",
        "k": K,
        "n_observations": payload["n_observations"],
        "n_mints": payload["n_mints"],
        "n_selected_pairs": payload["n_selected_pairs"],
        "n_unique_signatures": payload["n_unique_signatures"],
        "selection_rule": "last_K_pump_trades_strictly_before_T",
        "rpc_configured": _rpc_configured()[0],
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    write_json(CLEAN / "dry_run_manifest.json", out)
    # Persist signature list for a later fetch
    pq.write_table(
        pa.Table.from_pylist(
            [{"tx_signature": s} for s in payload["signatures"]]
        ),
        CLEAN / "causal_signatures_k16.parquet",
    )
    print(json.dumps(out, indent=2))
    return out


def fetch_if_configured(*, rps: float = 5.0) -> dict[str, Any]:
    ok, url = _rpc_configured()
    if not ok or url is None:
        return dry_run()

    ensure_dir(CLEAN)
    payload = collect_causal_signatures(k=K)
    sigs = payload["signatures"]
    ckpt_path = CLEAN / "fetch_checkpoint.json"
    out_dir = ensure_dir(CLEAN / "txs")
    done: dict[str, Any] = {}
    if ckpt_path.exists():
        done = json.loads(ckpt_path.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    fetched = 0
    t0 = time.time()
    delay = 1.0 / max(rps, 0.1)
    import urllib.request

    for sig in sigs:
        if sig in done:
            continue
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [
                    sig,
                    {"encoding": "json", "maxSupportedTransactionVersion": 0},
                ],
            }
        ).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = json.loads(resp.read().decode())
            (out_dir / f"{sig}.json").write_text(
                json.dumps(raw), encoding="utf-8"
            )
            done[sig] = {"ok": True, "at": datetime.now(tz=UTC).isoformat()}
            fetched += 1
        except Exception as exc:  # noqa: BLE001
            failures.append({"tx_signature": sig, "error": str(exc)})
            done[sig] = {"ok": False, "error": str(exc)}
        write_json(ckpt_path, done)
        time.sleep(delay)

    manifest = {
        **Q85,
        "status": "fetch_complete" if not failures else "fetch_partial",
        "selection_rule": "last_K_pump_trades_strictly_before_T",
        "k": K,
        "rpc_host": _rpc_host_redacted(url),
        "n_unique_signatures": len(sigs),
        "n_fetched_this_run": fetched,
        "n_checkpointed": len(done),
        "n_failures": len(failures),
        "failures_sample": failures[:20],
        "started_at": datetime.fromtimestamp(t0, tz=UTC).isoformat(),
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "elapsed_sec": time.time() - t0,
    }
    write_json(CLEAN / "provenance_manifest.json", manifest)
    return manifest


def main() -> int:
    ok, _ = _rpc_configured()
    if ok:
        print(json.dumps(fetch_if_configured(), indent=2, default=str))
    else:
        dry_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
