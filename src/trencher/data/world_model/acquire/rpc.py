"""Solana RPC acquisition for transaction microstructure enrichment."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from trencher.data.world_model.constants import (
    DEFAULT_RPC_URL,
    HELIUS_RPC_TEMPLATE,
    RPC_BATCH_SIZE,
    RPC_MIN_INTERVAL_SEC,
)
from trencher.data.world_model.io import ensure_dir, file_provenance, utc_now, write_json

logger = logging.getLogger(__name__)


def resolve_rpc_url() -> str:
    """Optional Helius URL/key; otherwise public Solana RPC. No paid Helius APIs."""
    url = os.environ.get("SOLANA_RPC_URL") or os.environ.get("HELIUS_RPC_URL")
    if url:
        return url
    key = os.environ.get("HELIUS_API_KEY")
    if key:
        return HELIUS_RPC_TEMPLATE.format(key=key)
    return DEFAULT_RPC_URL


class SolanaRpc:
    def __init__(
        self,
        url: str | None = None,
        *,
        min_interval: float = RPC_MIN_INTERVAL_SEC,
        session: requests.Session | None = None,
        provider: str | None = None,
    ) -> None:
        self.url = url or resolve_rpc_url()
        self.min_interval = min_interval
        self.session = session or requests.Session()
        self._last = 0.0
        self.calls = 0
        if provider:
            self.provider = provider
        elif "helius" in self.url:
            self.provider = "helius"
        elif "mainnet-beta.solana.com" in self.url:
            self.provider = "solana_rpc"
            if min_interval == RPC_MIN_INTERVAL_SEC:
                self.min_interval = max(min_interval, 0.85)
        else:
            self.provider = "alt_rpc"
            if min_interval == RPC_MIN_INTERVAL_SEC:
                self.min_interval = 0.12

    def _post(self, payload: Any) -> Any:
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
                return resp.json()
            except requests.HTTPError as exc:
                last_error = exc
                if exc.response is not None and exc.response.status_code == 429:
                    continue
                raise
        if last_error:
            raise last_error
        return {"error": {"message": "rate_limited"}}

    def call(self, method: str, params: list[Any]) -> Any:
        body = self._post({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        if isinstance(body, dict) and body.get("error"):
            return {"_rpc_error": body["error"]}
        return body.get("result") if isinstance(body, dict) else None

    def call_batch(self, calls: list[tuple[str, list[Any]]]) -> list[Any]:
        if not calls:
            return []
        payload = [
            {"jsonrpc": "2.0", "id": i, "method": method, "params": params}
            for i, (method, params) in enumerate(calls)
        ]
        body = self._post(payload)
        if isinstance(body, dict) and body.get("error"):
            return [{"_rpc_error": body["error"]} for _ in calls]
        if not isinstance(body, list):
            return [{"_rpc_error": {"message": "unexpected_batch_response"}} for _ in calls]
        by_id = {item.get("id"): item for item in body if isinstance(item, dict)}
        out = []
        for i in range(len(calls)):
            item = by_id.get(i, {})
            if item.get("error"):
                out.append({"_rpc_error": item["error"]})
            else:
                out.append(item.get("result"))
        return out


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

    fee = meta.get("fee")
    cu = meta.get("computeUnitsConsumed")
    pre_tok = meta.get("preTokenBalances") or []
    post_tok = meta.get("postTokenBalances") or []
    token_balance_changes = _token_balance_deltas(pre_tok, post_tok, mint)
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
        "token_balance_change_count": len(token_balance_changes),
        "token_balance_changes": token_balance_changes[:64],
        "log_message_count": len(meta.get("logMessages") or []),
        "inner_instruction_groups": len(meta.get("innerInstructions") or []),
        "quality_flag": "rpc_getTransaction_observed",
    }


def _ui_amount(entry: dict[str, Any]) -> float | None:
    ui = (entry.get("uiTokenAmount") or {})
    if ui.get("uiAmount") is not None:
        return float(ui["uiAmount"])
    amount = ui.get("amount")
    decimals = ui.get("decimals")
    if amount is None or decimals is None:
        return None
    try:
        return float(amount) / (10 ** int(decimals))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _token_balance_deltas(
    pre_tok: list[dict[str, Any]],
    post_tok: list[dict[str, Any]],
    mint: str,
) -> list[dict[str, Any]]:
    """Genuine SPL token account balance deltas from runtime meta."""
    pre_map = {}
    for e in pre_tok:
        if not isinstance(e, dict):
            continue
        if mint and e.get("mint") not in {None, mint}:
            # keep all mints but prefer matching later
            pass
        key = (e.get("mint"), e.get("owner"), e.get("accountIndex"))
        pre_map[key] = e
    post_map = {}
    for e in post_tok:
        if not isinstance(e, dict):
            continue
        key = (e.get("mint"), e.get("owner"), e.get("accountIndex"))
        post_map[key] = e
    keys = set(pre_map) | set(post_map)
    out = []
    for key in keys:
        m, owner, _idx = key
        if mint and m != mint:
            continue
        pre_a = _ui_amount(pre_map[key]) if key in pre_map else 0.0
        post_a = _ui_amount(post_map[key]) if key in post_map else 0.0
        if pre_a is None or post_a is None:
            continue
        delta = post_a - pre_a
        if delta == 0:
            continue
        out.append(
            {
                "mint": m,
                "owner": owner,
                "pre_ui_amount": pre_a,
                "post_ui_amount": post_a,
                "delta_ui_amount": delta,
            }
        )
    return out


def extract_token_ownership_edges(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wallet ownership deltas from pre/post token balances (genuine ledger meta)."""
    edges = []
    for row in rows:
        if not row.get("found"):
            continue
        changes = row.get("token_balance_changes") or []
        if isinstance(changes, str):
            try:
                changes = json.loads(changes)
            except json.JSONDecodeError:
                changes = []
        for c in changes:
            owner = c.get("owner")
            if not owner:
                continue
            edges.append(
                {
                    "edge_type": "rpc_token_balance_delta",
                    "src": owner,
                    "dst": row.get("mint") or c.get("mint"),
                    "mint": row.get("mint") or c.get("mint"),
                    "owner": owner,
                    "delta_ui_amount": c.get("delta_ui_amount"),
                    "pre_ui_amount": c.get("pre_ui_amount"),
                    "post_ui_amount": c.get("post_ui_amount"),
                    "tx_signature": row.get("tx_signature"),
                    "slot": row.get("slot"),
                    "block_time": row.get("block_time"),
                    "source": "solana_rpc.getTransaction.meta.pre_post_token_balances",
                    "provenance": "runtime_token_balance_meta",
                    "quality_flag": "genuine_spl_token_balance_delta_observed",
                }
            )
    return edges


def extract_funding_edges(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """SOL balance-delta proxy. Not genuine System Program funding."""
    edges = []
    for row in rows:
        if not row.get("found"):
            continue
        changes = row.get("balance_changes") or []
        if isinstance(changes, str):
            try:
                changes = json.loads(changes)
            except json.JSONDecodeError:
                changes = []
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


def _flush_rpc_rows(out_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        pq.write_table(pa.table({"tx_signature": pa.array([], pa.string())}), out_path)
        return
    flat = []
    for r in rows:
        item = dict(r)
        item["account_keys"] = item.get("account_keys") or []
        if "balance_changes" in item:
            changes = item.pop("balance_changes")
            item["balance_changes_json"] = (
                changes if isinstance(changes, str) else json.dumps(changes or [])
            )
        elif "balance_changes_json" not in item:
            item["balance_changes_json"] = "[]"
        if "token_balance_changes" in item:
            tchanges = item.pop("token_balance_changes")
            item["token_balance_changes_json"] = (
                tchanges if isinstance(tchanges, str) else json.dumps(tchanges or [])
            )
        elif "token_balance_changes_json" not in item:
            item["token_balance_changes_json"] = "[]"
        flat.append(item)
    pq.write_table(pa.Table.from_pylist(flat), out_path)


def run_rpc_acquisition(
    *,
    world_root: Path,
    max_signatures: int = 5_000,
    rpc: SolanaRpc | None = None,
    skip_live: bool = False,
    batch_size: int = RPC_BATCH_SIZE,
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
    existing_rows: list[dict[str, Any]] = []
    if out_path.exists():
        existing_rows = pq.read_table(out_path).to_pylist()
    done: set[str] = {
        r["tx_signature"] for r in existing_rows if r.get("tx_signature")
    }

    rpc = rpc or SolanaRpc()
    if rpc.provider == "solana_rpc":
        batch_size = 1
    pending = [
        row
        for row in targets
        if row.get("tx_signature") and row["tx_signature"] not in done
    ]
    pending = pending[: max(0, max_signatures - len(done))]
    new_rows: list[dict[str, Any]] = []
    matched = sum(1 for r in existing_rows if r.get("found"))
    attempted = 0
    interrupted = None
    tx_opts = {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}
    try:
        for i in range(0, len(pending), batch_size):
            chunk = pending[i : i + batch_size]
            if batch_size == 1:
                results = [
                    rpc.call("getTransaction", [row["tx_signature"], tx_opts])
                    for row in chunk
                ]
            else:
                results = rpc.call_batch(
                    [
                        ("getTransaction", [row["tx_signature"], tx_opts])
                        for row in chunk
                    ]
                )
                if len(results) != len(chunk):
                    results = [
                        rpc.call("getTransaction", [row["tx_signature"], tx_opts])
                        for row in chunk
                    ]
            for row, result in zip(chunk, results, strict=True):
                sig = row["tx_signature"]
                parsed = _parse_tx(sig, row.get("mint") or "", result)
                new_rows.append(parsed)
                done.add(sig)
                attempted += 1
                if parsed.get("found"):
                    matched += 1
            if attempted and attempted % 200 < batch_size:
                write_json(
                    checkpoint,
                    {
                        "n_done": len(done),
                        "updated_at": utc_now(),
                        "attempted_session": attempted,
                    },
                )
                _flush_rpc_rows(out_path, existing_rows + new_rows)
    except Exception as exc:  # noqa: BLE001
        interrupted = str(exc)
        logger.exception("RPC acquisition interrupted; persisting partial results")

    all_rows = existing_rows + new_rows
    by_sig = {r["tx_signature"]: r for r in all_rows if r.get("tx_signature")}
    final_rows = list(by_sig.values())
    _flush_rpc_rows(out_path, final_rows)

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
        "rpc_provider": rpc.provider,
        "rpc_url_host": rpc.url.split("?")[0],
        "funding_edges": len(funding),
        "path": file_provenance(out_path, source="solana_rpc.getTransaction"),
        "funding_path": file_provenance(
            funding_path, source="solana_rpc.balance_changes_proxy_not_native_funding"
        ),
        "max_signatures": max_signatures,
        "interrupted": interrupted,
        "quality_flag": "partial" if interrupted else "ok",
    }
    write_json(checkpoint, {"n_done": len(done), "updated_at": utc_now()})
    write_json(out_dir / "rpc_manifest.json", payload)
    return payload


def run_rpc_ownership_acquisition(
    *,
    world_root: Path,
    need_mints: set[str],
    rpc: SolanaRpc | None = None,
    skip_live: bool = False,
    per_mint_signatures: int = 2,
) -> dict[str, Any]:
    """Re-fetch trade txs for need-own mints; parse genuine token balance deltas."""
    out_dir = ensure_dir(world_root / "microstructure")
    graph_dir = ensure_dir(world_root / "graph")
    own_path = graph_dir / "edges_ownership_rpc_token_balances.parquet"
    ck = out_dir / "rpc_ownership_checkpoint.json"
    manifest = out_dir / "rpc_ownership_manifest.json"
    if skip_live:
        payload = {"skipped": True, "reason": "skip_live", "generated_at": utc_now()}
        write_json(manifest, payload)
        return payload

    targets_path = out_dir / "signature_targets.parquet"
    if not targets_path.exists():
        payload = {"error": "missing signature_targets", "generated_at": utc_now()}
        write_json(manifest, payload)
        return payload

    targets = pq.read_table(
        targets_path, columns=["mint", "tx_signature", "event_time"]
    ).to_pylist()
    by_mint: dict[str, list[dict[str, Any]]] = {}
    for row in targets:
        mint = row.get("mint")
        if mint not in need_mints or not row.get("tx_signature"):
            continue
        by_mint.setdefault(mint, []).append(row)
    pending: list[dict[str, Any]] = []
    for mint in sorted(by_mint):
        rows = sorted(
            by_mint[mint],
            key=lambda r: (str(r.get("event_time") or ""), r.get("tx_signature") or ""),
        )
        pending.extend(rows[:per_mint_signatures])

    done: set[str] = set()
    existing: list[dict[str, Any]] = []
    if own_path.exists() and pq.ParquetFile(own_path).metadata.num_rows:
        existing = pq.read_table(own_path).to_pylist()
        done = {e["tx_signature"] for e in existing if e.get("tx_signature")}
    if ck.exists():
        from trencher.data.world_model.io import read_json

        done |= set(read_json(ck).get("completed_signatures", []))

    rpc = rpc or SolanaRpc()
    batch_size = 1 if rpc.provider == "solana_rpc" else RPC_BATCH_SIZE
    tx_opts = {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}
    new_edges: list[dict[str, Any]] = []
    mints_unlocked: set[str] = {
        e["mint"] for e in existing if e.get("mint")
    }
    attempted = 0
    interrupted = None
    pending = [p for p in pending if p["tx_signature"] not in done]
    try:
        for i in range(0, len(pending), batch_size):
            chunk = pending[i : i + batch_size]
            if batch_size == 1:
                results = [
                    rpc.call("getTransaction", [row["tx_signature"], tx_opts])
                    for row in chunk
                ]
            else:
                results = rpc.call_batch(
                    [("getTransaction", [row["tx_signature"], tx_opts]) for row in chunk]
                )
            for row, result in zip(chunk, results, strict=True):
                parsed = _parse_tx(row["tx_signature"], row.get("mint") or "", result)
                done.add(row["tx_signature"])
                attempted += 1
                edges = extract_token_ownership_edges([parsed])
                if edges:
                    new_edges.extend(edges)
                    if row.get("mint"):
                        mints_unlocked.add(row["mint"])
            if attempted % 100 < batch_size:
                write_json(
                    ck,
                    {
                        "completed_signatures": sorted(done)[-50_000:],
                        "mints_unlocked": len(mints_unlocked),
                        "updated_at": utc_now(),
                        "attempted_session": attempted,
                    },
                )
                merged = existing + new_edges
                if merged:
                    pq.write_table(pa.Table.from_pylist(merged), own_path)
    except Exception as exc:  # noqa: BLE001
        interrupted = str(exc)
        logger.exception("RPC ownership acquisition interrupted")

    merged = existing + new_edges
    # de-dupe
    by_key = {}
    for e in merged:
        key = (
            e.get("tx_signature"),
            e.get("owner"),
            e.get("mint"),
            e.get("delta_ui_amount"),
        )
        by_key[key] = e
    final = list(by_key.values())
    if final:
        pq.write_table(pa.Table.from_pylist(final), own_path)
    else:
        pq.write_table(pa.table({"edge_type": pa.array([], pa.string())}), own_path)

    payload = {
        "generated_at": utc_now(),
        "need_mints": len(need_mints),
        "signatures_targeted": len(pending) + len(done),
        "attempted_session": attempted,
        "ownership_edges": len(final),
        "mints_with_rpc_ownership": len({e.get("mint") for e in final if e.get("mint")}),
        "mints_unlocked_session": len(mints_unlocked),
        "interrupted": interrupted,
        "path": file_provenance(
            own_path, source="solana_rpc.getTransaction.token_balances"
        ),
        "quality_flag": "partial" if interrupted else "genuine_rpc_token_balances",
        "notes": [
            "Edges come from meta.preTokenBalances/postTokenBalances on real txs.",
            "Not inferred from Pump trade event amounts.",
        ],
    }
    write_json(ck, {"completed_signatures": sorted(done), "updated_at": utc_now()})
    write_json(manifest, payload)
    return payload
