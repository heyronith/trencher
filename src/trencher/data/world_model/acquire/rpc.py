"""Solana RPC acquisition for transaction microstructure enrichment."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from trencher.data.world_model.constants import DEFAULT_RPC_URL, RPC_MIN_INTERVAL_SEC
from trencher.data.world_model.io import ensure_dir, file_provenance, read_json, utc_now, write_json

logger = logging.getLogger(__name__)


class SolanaRpc:
    def __init__(
        self,
        url: str = DEFAULT_RPC_URL,
        *,
        min_interval: float = RPC_MIN_INTERVAL_SEC,
        session: requests.Session | None = None,
    ) -> None:
        self.url = url
        self.min_interval = min_interval
        self.session = session or requests.Session()
        self._last = 0.0
        self.calls = 0

    def call(self, method: str, params: list[Any]) -> Any:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        last_error: Exception | None = None
        for attempt in range(8):
            elapsed = time.monotonic() - self._last
            wait = (
                self.min_interval
                if attempt == 0
                else min(30.0, self.min_interval * (2**attempt))
            )
            if elapsed < wait:
                time.sleep(wait - elapsed)
            try:
                resp = self.session.post(self.url, json=payload, timeout=60)
                self._last = time.monotonic()
                self.calls += 1
                if resp.status_code == 429:
                    logger.warning("RPC 429; backing off (attempt %s)", attempt + 1)
                    continue
                resp.raise_for_status()
                body = resp.json()
                if body.get("error"):
                    return {"_rpc_error": body["error"]}
                return body.get("result")
            except requests.HTTPError as exc:
                last_error = exc
                if exc.response is not None and exc.response.status_code == 429:
                    continue
                raise
        if last_error:
            raise last_error
        return {"_rpc_error": {"message": "rate_limited"}}


def _parse_tx(signature: str, mint: str, result: Any) -> dict[str, Any]:
    if result is None:
        return {
            "tx_signature": signature,
            "mint": mint,
            "found": False,
            "quality_flag": "rpc_null_result",
        }
    if isinstance(result, dict) and result.get("_rpc_error"):
        return {
            "tx_signature": signature,
            "mint": mint,
            "found": False,
            "quality_flag": "rpc_error",
            "error": str(result["_rpc_error"]),
        }
    meta = result.get("meta") or {}
    tx = result.get("transaction") or {}
    message = tx.get("message") or {}
    account_keys = message.get("accountKeys") or []
    # versioned tx may encode keys differently
    if account_keys and isinstance(account_keys[0], dict):
        keys = [a.get("pubkey") for a in account_keys]
    else:
        keys = list(account_keys)
    pre = meta.get("preBalances") or []
    post = meta.get("postBalances") or []
    balance_changes = []
    for i, key in enumerate(keys):
        if i >= len(pre) or i >= len(post):
            break
        delta = post[i] - pre[i]
        if delta:
            balance_changes.append({"account": key, "delta_lamports": delta})

    # Priority fee / tip heuristic: fee beyond base signature fee when available.
    fee = meta.get("fee")
    cu = meta.get("computeUnitsConsumed")
    return {
        "tx_signature": signature,
        "mint": mint,
        "found": True,
        "slot": result.get("slot"),
        "block_time": result.get("blockTime"),
        "success": meta.get("err") is None,
        "err": str(meta.get("err")) if meta.get("err") is not None else None,
        "fee_lamports": fee,
        "compute_units_consumed": cu,
        "account_key_count": len(keys),
        "account_keys": keys[:64],
        "balance_change_count": len(balance_changes),
        "balance_changes": balance_changes[:64],
        "log_message_count": len(meta.get("logMessages") or []),
        "inner_instruction_groups": len(meta.get("innerInstructions") or []),
        "quality_flag": "rpc_getTransaction_observed",
    }


def extract_funding_edges(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive SOL funding-like edges from non-zero balance deltas in enriched txs."""
    edges = []
    for row in rows:
        if not row.get("found"):
            continue
        changes = row.get("balance_changes") or []
        # Pair largest debit with largest credit as a weak funding edge proxy.
        debits = sorted(
            [c for c in changes if (c.get("delta_lamports") or 0) < 0],
            key=lambda c: c["delta_lamports"],
        )
        credits = sorted(
            [c for c in changes if (c.get("delta_lamports") or 0) > 0],
            key=lambda c: c["delta_lamports"],
            reverse=True,
        )
        if not debits or not credits:
            continue
        src = debits[0]["account"]
        dst = credits[0]["account"]
        if src == dst:
            continue
        edges.append(
            {
                "edge_type": "sol_balance_delta_proxy",
                "src": src,
                "dst": dst,
                "tx_signature": row["tx_signature"],
                "lamports": credits[0]["delta_lamports"],
                "slot": row.get("slot"),
                "quality_flag": "derived_from_rpc_balance_changes_not_parsed_system_transfer",
            }
        )
    return edges


def run_rpc_acquisition(
    *,
    world_root: Path,
    max_signatures: int = 5_000,
    rpc: SolanaRpc | None = None,
    skip_live: bool = False,
) -> dict[str, Any]:
    out_dir = ensure_dir(world_root / "microstructure")
    targets_path = out_dir / "signature_targets.parquet"
    checkpoint = out_dir / "rpc_checkpoint.json"
    out_path = out_dir / "rpc_transactions.parquet"
    funding_path = world_root / "graph" / "edges_funding_rpc.parquet"
    ensure_dir(funding_path.parent)

    if skip_live:
        payload = {"skipped": True, "reason": "skip_live", "generated_at": utc_now()}
        write_json(out_dir / "rpc_manifest.json", payload)
        return payload

    if not targets_path.exists():
        payload = {"error": "missing signature_targets.parquet", "generated_at": utc_now()}
        write_json(out_dir / "rpc_manifest.json", payload)
        return payload

    targets = pq.read_table(targets_path, columns=["mint", "tx_signature"]).to_pylist()
    done: set[str] = set()
    if checkpoint.exists():
        done = set(read_json(checkpoint).get("completed_signatures", []))

    existing_rows: list[dict[str, Any]] = []
    if out_path.exists():
        existing_rows = pq.read_table(out_path).to_pylist()
        done.update(r["tx_signature"] for r in existing_rows if r.get("tx_signature"))

    rpc = rpc or SolanaRpc()
    new_rows: list[dict[str, Any]] = []
    matched = sum(1 for r in existing_rows if r.get("found"))
    attempted = 0
    interrupted = None
    try:
        for row in targets:
            sig = row.get("tx_signature")
            if not sig or sig in done:
                continue
            if len(done) >= max_signatures:
                break
            result = rpc.call(
                "getTransaction",
                [sig, {"encoding": "json", "maxSupportedTransactionVersion": 0}],
            )
            parsed = _parse_tx(sig, row.get("mint") or "", result)
            new_rows.append(parsed)
            done.add(sig)
            attempted += 1
            if parsed.get("found"):
                matched += 1
            if attempted % 25 == 0:
                write_json(
                    checkpoint,
                    {
                        "completed_signatures": sorted(done)[-max_signatures:],
                        "updated_at": utc_now(),
                        "attempted_session": attempted,
                    },
                )
    except Exception as exc:  # noqa: BLE001
        interrupted = str(exc)
        logger.exception("RPC acquisition interrupted; persisting partial results")

    all_rows = existing_rows + new_rows
    # de-dupe by signature keeping last
    by_sig = {r["tx_signature"]: r for r in all_rows if r.get("tx_signature")}
    final_rows = list(by_sig.values())
    if final_rows:
        # Nested columns: store balance_changes as JSON string for parquet simplicity
        flat = []
        for r in final_rows:
            item = dict(r)
            item["account_keys"] = item.get("account_keys") or []
            item["balance_changes_json"] = __import__("json").dumps(
                item.pop("balance_changes", [])
            )
            flat.append(item)
        pq.write_table(pa.Table.from_pylist(flat), out_path)
    else:
        pq.write_table(pa.table({"tx_signature": pa.array([], pa.string())}), out_path)

    funding = extract_funding_edges(final_rows)
    if funding:
        pq.write_table(pa.Table.from_pylist(funding), funding_path)
    else:
        pq.write_table(pa.table({"edge_type": pa.array([], pa.string())}), funding_path)

    found = sum(1 for r in final_rows if r.get("found"))
    payload = {
        "generated_at": utc_now(),
        "attempted_session": attempted,
        "total_enriched": len(final_rows),
        "found": found,
        "signature_match_rate": (found / len(final_rows)) if final_rows else None,
        "rpc_calls": rpc.calls,
        "funding_edges": len(funding),
        "path": file_provenance(out_path, source="solana_rpc.getTransaction"),
        "funding_path": file_provenance(funding_path, source="solana_rpc.balance_changes"),
        "max_signatures": max_signatures,
        "interrupted": interrupted,
        "quality_flag": "partial" if interrupted else "ok",
    }
    write_json(checkpoint, {"completed_signatures": sorted(done), "updated_at": utc_now()})
    write_json(out_dir / "rpc_manifest.json", payload)
    return payload
