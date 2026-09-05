"""Causal rich-modality indexes for RICH-additive baseline features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def load_micro_by_mint(world_root: Path) -> dict[str, dict[str, np.ndarray]]:
    path = world_root / "microstructure" / "rpc_transactions.parquet"
    if not path.exists():
        return {}
    cols = [
        "mint",
        "block_time",
        "fee_lamports",
        "compute_units_consumed",
        "success",
        "account_key_count",
    ]
    schema = set(pq.read_schema(path).names)
    use = [c for c in cols if c in schema]
    if "mint" not in use or "block_time" not in use:
        return {}
    table = pq.read_table(path, columns=use)
    out: dict[str, list[dict[str, Any]]] = {}
    for r in table.to_pylist():
        mint = str(r.get("mint") or "")
        if not mint or r.get("block_time") is None:
            continue
        out.setdefault(mint, []).append(r)
    packed: dict[str, dict[str, np.ndarray]] = {}
    for mint, rows in out.items():
        rows.sort(key=lambda x: float(x["block_time"]))
        packed[mint] = {
            "times": np.array([float(r["block_time"]) for r in rows], dtype=np.float64),
            "fee_lamports": np.array(
                [float(r.get("fee_lamports") or 0.0) for r in rows], dtype=np.float64
            ),
            "compute_units_consumed": np.array(
                [float(r.get("compute_units_consumed") or 0.0) for r in rows],
                dtype=np.float64,
            ),
            "success": np.array(
                [1.0 if r.get("success") else 0.0 for r in rows], dtype=np.float64
            ),
            "account_key_count": np.array(
                [float(r.get("account_key_count") or 0.0) for r in rows], dtype=np.float64
            ),
            "priority_fee_lamports": np.zeros(len(rows), dtype=np.float64),
        }
    return packed


def load_ownership_by_mint(world_root: Path) -> dict[str, dict[str, np.ndarray]]:
    path = world_root / "bigquery" / "token_transfers_rich.parquet"
    if not path.exists():
        return {}
    schema = set(pq.read_schema(path).names)
    # block_timestamp may be timestamp type
    use = [c for c in ("mint", "block_timestamp", "value") if c in schema]
    if "mint" not in use or "block_timestamp" not in use:
        return {}
    table = pq.read_table(path, columns=use)
    buckets: dict[str, list[tuple[float, float]]] = {}
    for r in table.to_pylist():
        mint = str(r.get("mint") or "")
        ts = r.get("block_timestamp")
        if not mint or ts is None:
            continue
        t = float(ts.timestamp()) if hasattr(ts, "timestamp") else float(ts)
        val = float(r.get("value") or 0.0)
        buckets.setdefault(mint, []).append((t, val))
    packed: dict[str, dict[str, np.ndarray]] = {}
    for mint, rows in buckets.items():
        rows.sort(key=lambda x: x[0])
        times = np.array([r[0] for r in rows], dtype=np.float64)
        val = np.array([r[1] for r in rows], dtype=np.float64)
        packed[mint] = {
            "times": times,
            "value": val,
            "pre_ui_amount": np.zeros_like(val),
            "post_ui_amount": val,
        }
    return packed


def load_graph_by_mint(world_root: Path) -> dict[str, dict[str, np.ndarray]]:
    path = world_root / "graph" / "edges_cobuy_rich.parquet"
    if not path.exists():
        return {}
    schema = set(pq.read_schema(path).names)
    if "mint" not in schema:
        return {}
    cols = [c for c in ("mint", "block_time", "sol_amount", "token_amount", "edge_type") if c in schema]
    if "block_time" not in cols:
        return {}
    table = pq.read_table(path, columns=cols)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in table.to_pylist():
        mint = str(r.get("mint") or "")
        if not mint or r.get("block_time") is None:
            continue
        buckets.setdefault(mint, []).append(r)
    packed: dict[str, dict[str, Any]] = {}
    for mint, rows in buckets.items():
        rows.sort(key=lambda x: float(x["block_time"]))
        packed[mint] = {
            "times": np.array([float(r["block_time"]) for r in rows], dtype=np.float64),
            "sol_amount": np.array(
                [float(r.get("sol_amount") or 0.0) for r in rows], dtype=np.float64
            ),
            "token_amount": np.array(
                [float(r.get("token_amount") or 0.0) for r in rows], dtype=np.float64
            ),
            "edge_type": [str(r.get("edge_type") or "") for r in rows],
        }
    return packed


def load_wallet_prior_by_mint(world_root: Path) -> dict[str, dict[str, np.ndarray]]:
    """Causal wallet-prior history from frozen ``edges_wallet_token`` (mint=dst).

    Uses ``first_t`` as the causal timestamp (wallet's first observed interaction
    with the mint). Aggregates buy/sell SOL as prior activity mass.
    """
    path = world_root / "graph" / "edges_wallet_token.parquet"
    if not path.exists():
        return {}
    schema = set(pq.read_schema(path).names)
    # dst is mint in this store
    mint_col = "dst" if "dst" in schema else ("mint" if "mint" in schema else None)
    t_col = "first_t" if "first_t" in schema else ("last_t" if "last_t" in schema else None)
    if mint_col is None or t_col is None:
        return {}
    cols = [c for c in (mint_col, t_col, "buy_sol", "sell_sol", "buys", "sells", "src") if c in schema]
    table = pq.read_table(path, columns=cols)
    buckets: dict[str, list[tuple[float, float]]] = {}
    for r in table.to_pylist():
        mint = str(r.get(mint_col) or "")
        ts = r.get(t_col)
        if not mint or ts is None:
            continue
        t = float(ts)
        sol = float(r.get("buy_sol") or 0.0) + float(r.get("sell_sol") or 0.0)
        buckets.setdefault(mint, []).append((t, sol))
    packed: dict[str, dict[str, np.ndarray]] = {}
    for mint, rows in buckets.items():
        rows.sort(key=lambda x: x[0])
        packed[mint] = {
            "times": np.array([r[0] for r in rows], dtype=np.float64),
            "sol_amount": np.array([r[1] for r in rows], dtype=np.float64),
        }
    return packed


def causal_slice_arrays(
    pack: dict[str, Any] | None, T: float
) -> dict[str, np.ndarray] | None:
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
    return out
