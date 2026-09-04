"""True native-SOL funding from System Program transfers (pre-T only)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.constants import SYSTEM_PROGRAM_ID
from trencher.data.world_model.io import ensure_dir, file_provenance, utc_now, write_json

logger = logging.getLogger(__name__)


def observation_unix(ts: Any) -> float | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return ts.timestamp()
    if isinstance(ts, (int, float)):
        return float(ts)
    text = str(ts)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()
    except ValueError:
        return None


def parse_system_sol_transfers(
    signature: str,
    result: Any,
    *,
    max_block_time: float | None,
) -> list[dict[str, Any]]:
    """Parse System Program transfer instructions. Never infer from SPL."""
    if not isinstance(result, dict) or result.get("_rpc_error") or not result:
        return []
    block_time = result.get("blockTime")
    if block_time is None:
        return []
    if max_block_time is not None and float(block_time) >= float(max_block_time):
        return []
    slot = result.get("slot")
    edges: list[dict[str, Any]] = []
    tx = result.get("transaction") or {}
    message = tx.get("message") or {}
    instructions = list(message.get("instructions") or [])
    meta = result.get("meta") or {}
    for group in meta.get("innerInstructions") or []:
        instructions.extend(group.get("instructions") or [])
    for instr in instructions:
        if not isinstance(instr, dict):
            continue
        program_id = instr.get("programId") or instr.get("programIdIndex")
        program = instr.get("program")
        parsed = instr.get("parsed")
        is_system = program == "system" or program_id == SYSTEM_PROGRAM_ID
        if not is_system or not isinstance(parsed, dict):
            continue
        if parsed.get("type") not in {"transfer", "transferChecked"}:
            continue
        info = parsed.get("info") or {}
        src = info.get("source")
        dst = info.get("destination")
        lamports = info.get("lamports")
        if not src or not dst or src == dst or lamports is None:
            continue
        edges.append(
            {
                "edge_type": "system_sol_transfer",
                "src": src,
                "dst": dst,
                "lamports": int(lamports),
                "tx_signature": signature,
                "slot": slot,
                "block_time": block_time,
                "source": "solana_rpc.getTransaction.jsonParsed",
                "provenance": "system_program_transfer_pre_T",
                "quality_flag": "native_sol_system_transfer_observed",
            }
        )
    return edges


def run_native_sol_funding_acquisition(
    *,
    world_root: Path,
    rpc: Any,
    skip_live: bool = False,
    max_wallets: int = 1_200,
    signatures_per_wallet: int = 12,
    txs_per_wallet: int = 4,
) -> dict[str, Any]:
    """Creators + earliest buyers: signatures strictly before observation T."""
    out_dir = ensure_dir(world_root / "graph")
    path = out_dir / "edges_funding_native_sol.parquet"
    manifest_path = world_root / "microstructure" / "native_sol_funding_manifest.json"
    checkpoint = world_root / "microstructure" / "native_sol_funding_checkpoint.json"
    if skip_live:
        payload = {"skipped": True, "reason": "skip_live", "generated_at": utc_now()}
        write_json(manifest_path, payload)
        return payload

    subset_path = world_root / "universe" / "rich_subset_mints.parquet"
    if not subset_path.exists():
        payload = {"error": "missing rich subset", "generated_at": utc_now()}
        write_json(manifest_path, payload)
        return payload

    subset = pq.read_table(subset_path).to_pylist()
    targets_path = world_root / "microstructure" / "signature_targets.parquet"
    early_by_mint: dict[str, list[str]] = {}
    if targets_path.exists():
        tgt = pq.read_table(
            targets_path, columns=["mint", "user_wallet", "event_time"]
        ).to_pylist()
        for row in tgt:
            mint = row.get("mint")
            wallet = row.get("user_wallet")
            if mint and wallet:
                early_by_mint.setdefault(mint, [])
                if wallet not in early_by_mint[mint]:
                    early_by_mint[mint].append(wallet)

    wallets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in subset:
        t_unix = observation_unix(row.get("observation_time_t") or row.get("detected_at"))
        if t_unix is None:
            continue
        mint = row.get("mint")
        creator = row.get("creator")
        if creator and creator not in seen:
            seen.add(creator)
            wallets.append({"wallet": creator, "mint": mint, "role": "creator", "t": t_unix})
        for w in (early_by_mint.get(mint) or [])[:2]:
            if w not in seen:
                seen.add(w)
                wallets.append(
                    {"wallet": w, "mint": mint, "role": "early_participant", "t": t_unix}
                )
        if len(wallets) >= max_wallets:
            break

    done_wallets: set[str] = set()
    existing: list[dict[str, Any]] = []
    if path.exists() and pq.ParquetFile(path).metadata.num_rows:
        existing = pq.read_table(path).to_pylist()
    if checkpoint.exists():
        from trencher.data.world_model.io import read_json

        done_wallets = set(read_json(checkpoint).get("completed_wallets", []))

    new_edges: list[dict[str, Any]] = []
    attempted = 0
    wallets_with_history = 0
    interrupted = None
    try:
        for item in wallets:
            wallet = item["wallet"]
            if wallet in done_wallets:
                continue
            t_unix = item["t"]
            sig_result = rpc.call(
                "getSignaturesForAddress",
                [wallet, {"limit": signatures_per_wallet}],
            )
            attempted += 1
            if not isinstance(sig_result, list):
                done_wallets.add(wallet)
                continue
            pre_t = [
                s
                for s in sig_result
                if isinstance(s, dict)
                and s.get("signature")
                and s.get("blockTime") is not None
                and float(s["blockTime"]) < t_unix
            ]
            if pre_t:
                wallets_with_history += 1
            for rec in pre_t[:txs_per_wallet]:
                sig = rec["signature"]
                tx = rpc.call(
                    "getTransaction",
                    [
                        sig,
                        {
                            "encoding": "jsonParsed",
                            "maxSupportedTransactionVersion": 0,
                        },
                    ],
                )
                attempted += 1
                new_edges.extend(
                    {
                        **edge,
                        "mint": item.get("mint"),
                        "wallet_role": item.get("role"),
                    }
                    for edge in parse_system_sol_transfers(
                        sig, tx, max_block_time=t_unix
                    )
                )
            done_wallets.add(wallet)
            if len(done_wallets) % 20 == 0:
                write_json(
                    checkpoint,
                    {
                        "completed_wallets": sorted(done_wallets),
                        "updated_at": utc_now(),
                    },
                )
    except Exception as exc:  # noqa: BLE001
        interrupted = str(exc)
        logger.exception("Native SOL funding interrupted")

    by_key = {}
    for edge in existing + new_edges:
        key = (edge.get("tx_signature"), edge.get("src"), edge.get("dst"), edge.get("lamports"))
        by_key[key] = edge
    final = list(by_key.values())
    if final:
        pq.write_table(pa.Table.from_pylist(final), path)
    else:
        pq.write_table(pa.table({"edge_type": pa.array([], pa.string())}), path)

    # Shared native funders
    from collections import defaultdict

    dests: dict[str, set[str]] = defaultdict(set)
    for e in final:
        dests[e["src"]].add(e["dst"])
    shared = []
    for funder, ds in dests.items():
        if len(ds) < 2:
            continue
        ordered = sorted(ds)[:40]
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                shared.append(
                    {
                        "edge_type": "shared_native_funder",
                        "funder": funder,
                        "dst_a": a,
                        "dst_b": b,
                        "quality_flag": "derived_from_system_sol_transfers",
                    }
                )
    shared_path = out_dir / "edges_shared_native_funder.parquet"
    if shared:
        pq.write_table(pa.Table.from_pylist(shared), shared_path)
    else:
        pq.write_table(pa.table({"edge_type": pa.array([], pa.string())}), shared_path)

    payload = {
        "generated_at": utc_now(),
        "wallets_targeted": len(wallets),
        "wallets_completed": len(done_wallets),
        "wallets_with_pre_t_history": wallets_with_history,
        "native_sol_edges": len(final),
        "shared_native_funder_edges": len(shared),
        "rpc_calls_session": attempted,
        "interrupted": interrupted,
        "path": file_provenance(path, source="solana_rpc.system_program_transfer"),
        "quality_flag": "partial" if interrupted else "native_sol_pre_T",
        "notes": [
            "Edges are System Program SOL transfers only.",
            "SPL token transfers are never treated as native SOL funding.",
            "Signatures and transfers with blockTime >= observation T are dropped.",
        ],
    }
    write_json(checkpoint, {"completed_wallets": sorted(done_wallets), "updated_at": utc_now()})
    write_json(manifest_path, payload)
    return payload
