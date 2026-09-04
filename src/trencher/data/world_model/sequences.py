"""Causal sequence stores + observation references for the world-model input layer.

Builds normalized, timestamp-sorted modality stores once, then attaches compact
per-observation cutoffs so every referenced row satisfies event_time <= T.
Does not copy full histories into observation rows. No labels / no training.
"""

from __future__ import annotations

import json
import logging
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from trencher.data.feasibility.concepts import SYSTEM_PROGRAM_WALLET
from trencher.data.world_model.freeze import FREEZE_ID
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.pit import (
    PIT_ENGINE_VERSION,
    _to_unix,
    ensure_atomic_trades_index,
    ensure_global_context_index,
    last_completed_second,
)

logger = logging.getLogger(__name__)

SEQUENCE_DATASET_VERSION = "sequence_dataset_v1"
SEQUENCE_STORE_VERSION = "sequence_stores_v1"
MODEL_INPUT_CONTRACT_VERSION = "model_input_contract_v1"

# Cap prior wallet↔token corpus history per wallet (feasibility; not outcome-based).
WALLET_PRIOR_EVENT_CAP = 2048

WINDOW_PRESETS = (
    {"name": "last_n_64", "kind": "last_n", "n": 64},
    {"name": "last_n_256", "kind": "last_n", "n": 256},
    {"name": "last_10s", "kind": "last_seconds", "seconds": 10},
    {"name": "last_30s", "kind": "last_seconds", "seconds": 30},
    {"name": "last_2m", "kind": "last_seconds", "seconds": 120},
    {"name": "full_capped_4096", "kind": "full_capped", "cap": 4096},
)

MODALITIES = ("events", "curve", "micro", "ownership", "graph_token")


def _order_key(
    *,
    event_time_unix: float,
    tx_signature: str | None,
    wallet: str | None,
    side: str | None,
    extra: str = "",
) -> str:
    return (
        f"{event_time_unix:.6f}|{tx_signature or ''}|{wallet or ''}|"
        f"{side or ''}|{extra}"
    )


def _dedupe_key_trade(
    *,
    mint: str,
    tx_signature: str | None,
    event_time_unix: float,
    wallet: str | None,
    is_buy: bool | None,
    sol_amount: float | None,
    token_amount: float | None,
) -> str:
    if tx_signature:
        return f"sig:{mint}:{tx_signature}"
    sol = None if sol_amount is None else round(float(sol_amount), 9)
    tok = None if token_amount is None else round(float(token_amount), 6)
    return f"fallback:{mint}:{event_time_unix:.6f}:{wallet}:{is_buy}:{sol}:{tok}"


def _pct(values: list[int], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return float(s[idx])


def _length_stats(counts: list[int]) -> dict[str, Any]:
    return {
        "min": min(counts) if counts else 0,
        "median": _pct(counts, 0.5),
        "p90": _pct(counts, 0.9),
        "p99": _pct(counts, 0.99),
        "max": max(counts) if counts else 0,
        "mean": (sum(counts) / len(counts)) if counts else 0.0,
        "zero_count": sum(1 for c in counts if c == 0),
        "n": len(counts),
    }


def _write_sorted_store(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    entity_key: str,
) -> dict[str, Any]:
    """Sort by (entity, time, order_key), assign seq_idx, write store + entity index."""
    idx_path = path.with_name(path.stem + "_entity_index.parquet")
    if not rows:
        pq.write_table(pa.table({"seq_idx": pa.array([], type=pa.int64())}), path)
        pq.write_table(
            pa.table(
                {
                    "entity": pa.array([], type=pa.string()),
                    "start_idx": pa.array([], type=pa.int64()),
                    "end_idx": pa.array([], type=pa.int64()),
                }
            ),
            idx_path,
        )
        return {
            "rows": 0,
            "entities": 0,
            "path": str(path),
            "entity_index": str(idx_path),
        }

    rows.sort(
        key=lambda r: (
            r[entity_key] or "",
            float(r["event_time_unix"]),
            r.get("order_key") or "",
        )
    )
    for i, row in enumerate(rows):
        row["seq_idx"] = i

    pq.write_table(pa.Table.from_pylist(rows), path)

    entity_index: list[dict[str, Any]] = []
    i = 0
    n = len(rows)
    while i < n:
        ent = rows[i][entity_key]
        j = i + 1
        while j < n and rows[j][entity_key] == ent:
            j += 1
        entity_index.append({"entity": ent, "start_idx": i, "end_idx": j})
        i = j
    pq.write_table(pa.Table.from_pylist(entity_index), idx_path)
    return {
        "rows": n,
        "entities": len(entity_index),
        "path": str(path),
        "entity_index": str(idx_path),
    }


def _load_entity_index(path: Path) -> dict[str, tuple[int, int]]:
    if not path.exists() or pq.ParquetFile(path).metadata.num_rows == 0:
        return {}
    table = pq.read_table(path)
    out: dict[str, tuple[int, int]] = {}
    for ent, lo, hi in zip(
        table.column("entity").to_pylist(),
        table.column("start_idx").to_pylist(),
        table.column("end_idx").to_pylist(),
        strict=True,
    ):
        if ent is not None:
            out[str(ent)] = (int(lo), int(hi))
    return out


def _end_idx_for_t(
    times: list[float], start: int, end: int, observation_time_t: float
) -> int:
    """Exclusive end index within [start, end) for times <= T."""
    if start >= end:
        return start
    lo, hi = start, end
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] <= observation_time_t:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _load_times_array(store_path: Path) -> list[float]:
    if not store_path.exists() or pq.ParquetFile(store_path).metadata.num_rows == 0:
        return []
    return [
        float(x)
        for x in pq.read_table(store_path, columns=["event_time_unix"])
        .column(0)
        .to_pylist()
    ]


def build_model_input_contract_v1() -> dict[str, Any]:
    """Explicit separation of identifiers vs model-consumable values."""
    return {
        "contract_version": MODEL_INPUT_CONTRACT_VERSION,
        "generated_at": utc_now(),
        "pit_engine_version": PIT_ENGINE_VERSION,
        "sequence_dataset_version": SEQUENCE_DATASET_VERSION,
        "freeze_id": FREEZE_ID,
        "identifiers_not_structured_features": [
            "mint",
            "observation_id",
            "source_freeze_version",
            "pit_engine_version",
            "observation_dataset_version",
            "sequence_store_version",
            "artifact_id",
            "tx_signature",
            "model_state_digest",
        ],
        "structured_encoder_forbidden": [
            "mint_as_categorical_embedding",
            "base58_address_as_tabular_string_feature",
            "red_cohort_full_history_membership_at_earlier_T",
            "lifetime_edges_wallet_token_aggregates",
            "post_T_sequence_rows",
            "graduation_or_outcome_labels",
        ],
        "graph_node_identities_allowed": [
            "wallet_id (cross-token history is an intended TGN signal)",
            "token_mint as graph node id (not tabular categorical feature)",
            "creator_wallet as graph node id",
        ],
        "modality_stores": {
            "token_events": "sequence_events_v1",
            "microstructure": "sequence_microstructure_v1",
            "ownership": "sequence_ownership_v1",
            "graph_token": "sequence_graph_token_v1",
            "graph_wallet_prior": "sequence_graph_wallet_prior_v1",
            "curve_execution": "sequence_curve_v1",
            "global": "pit/global + blocks_daily (referenced, last_completed_second)",
        },
        "observation_payload_composition": [
            "observation metadata (identifiers)",
            "structured PIT model_state summary (tabular/masks)",
            "causal token event sequence via refs/cutoffs",
            "causal graph history via token + wallet-prior refs",
            "causal ownership / microstructure / curve / global modalities",
        ],
        "availability_quality_masks": [
            "modality_mask_*",
            "ownership_level_at_t",
            "source_quality_flag",
            "evidence_level",
        ],
        "provenance_debug_metadata": [
            "store paths",
            "construction stats",
            "excluded_system_program_counts",
            "duplicate_raw_event_rate",
        ],
        "training_window_presets": list(WINDOW_PRESETS),
        "causal_invariant": (
            "event_time_unix <= observation_time_t for every referenced row"
        ),
        "labels": "not_generated",
        "training": "not_started",
    }


def apply_training_window(
    *,
    times: list[float],
    start_idx: int,
    end_idx: int,
    observation_time_t: float,
    preset: dict[str, Any],
) -> tuple[int, int]:
    """Return [lo, hi) within [start_idx, end_idx) for a window preset."""
    lo, hi = start_idx, end_idx
    if hi <= lo:
        return lo, hi
    kind = preset.get("kind")
    if kind == "last_n":
        n = int(preset["n"])
        lo = max(lo, hi - n)
    elif kind == "last_seconds":
        seconds = float(preset["seconds"])
        cutoff = observation_time_t - seconds
        left, right = lo, hi
        while left < right:
            mid = (left + right) // 2
            if times[mid] < cutoff:
                left = mid + 1
            else:
                right = mid
        lo = left
    elif kind == "full_capped":
        cap = int(preset["cap"])
        lo = max(lo, hi - cap)
    return lo, hi



def _build_token_event_store(
    *,
    world_root: Path,
    out_dir: Path,
    subset: set[str],
) -> dict[str, Any]:
    atomic_path = world_root / "pit" / "atomic_trades_subset.parquet"
    curve_path = world_root / "execution" / "execution_curve_states.parquet"

    curve_by_key: dict[str, dict[str, Any]] = {}
    if curve_path.exists() and pq.ParquetFile(curve_path).metadata.num_rows:
        for row in pq.read_table(curve_path).to_pylist():
            mint = row.get("mint")
            if mint not in subset:
                continue
            et = _to_unix(row.get("event_time"))
            if et is None:
                continue
            key = (
                f"{mint}|{et:.6f}|{row.get('is_buy')}|{row.get('token_amount')}|"
                f"{row.get('sol_amount')}"
            )
            curve_by_key[key] = row

    best: dict[str, dict[str, Any]] = {}
    raw_n = 0
    excluded_system = 0
    if atomic_path.exists() and pq.ParquetFile(atomic_path).metadata.num_rows:
        for row in pq.read_table(atomic_path).to_pylist():
            mint = row.get("mint")
            if mint not in subset:
                continue
            wallet = row.get("user_wallet")
            if wallet == SYSTEM_PROGRAM_WALLET:
                excluded_system += 1
                continue
            et = _to_unix(row.get("event_time"))
            if et is None:
                continue
            raw_n += 1
            dk = _dedupe_key_trade(
                mint=mint,
                tx_signature=row.get("tx_signature"),
                event_time_unix=et,
                wallet=wallet,
                is_buy=row.get("is_buy"),
                sol_amount=row.get("sol_amount"),
                token_amount=row.get("token_amount"),
            )
            side = "buy" if row.get("is_buy") else "sell"
            ck = (
                f"{mint}|{et:.6f}|{row.get('is_buy')}|{row.get('token_amount')}|"
                f"{row.get('sol_amount')}"
            )
            curve = curve_by_key.get(ck) or {}
            cand = {
                "mint": mint,
                "event_time_unix": et,
                "slot": None,
                "tx_signature": row.get("tx_signature"),
                "order_key": _order_key(
                    event_time_unix=et,
                    tx_signature=row.get("tx_signature"),
                    wallet=wallet,
                    side=side,
                ),
                "wallet": wallet,
                "side": side,
                "is_buy": (
                    bool(row.get("is_buy")) if row.get("is_buy") is not None else None
                ),
                "token_amount": row.get("token_amount"),
                "sol_amount": row.get("sol_amount"),
                "price_sol": row.get("price_sol")
                if row.get("price_sol") is not None
                else curve.get("price_sol"),
                "v_sol_bonding_curve": row.get("v_sol_bonding_curve")
                if row.get("v_sol_bonding_curve") is not None
                else curve.get("v_sol_bonding_curve"),
                "v_tokens_bonding_curve": row.get("v_tokens_bonding_curve")
                if row.get("v_tokens_bonding_curve") is not None
                else curve.get("v_tokens_bonding_curve"),
                "real_sol_reserves": None,
                "real_token_reserves": None,
                "curve_pct_depleted": row.get("curve_pct_depleted")
                if row.get("curve_pct_depleted") is not None
                else curve.get("curve_pct_depleted"),
                "executable_price_sol": curve.get("executable_price_sol"),
                "source": "pumpfun.trades.atomic_subset",
                "source_quality_flag": curve.get("quality_flag") or "observed",
                "has_reserves": bool(
                    (
                        row.get("v_sol_bonding_curve")
                        if row.get("v_sol_bonding_curve") is not None
                        else curve.get("v_sol_bonding_curve")
                    )
                    is not None
                    or (
                        row.get("v_tokens_bonding_curve")
                        if row.get("v_tokens_bonding_curve") is not None
                        else curve.get("v_tokens_bonding_curve")
                    )
                    is not None
                ),
            }
            prev = best.get(dk)
            if prev is None:
                best[dk] = cand
            else:
                fields = (
                    "price_sol",
                    "v_sol_bonding_curve",
                    "v_tokens_bonding_curve",
                    "tx_signature",
                )
                score = sum(1 for k in fields if cand.get(k) is not None)
                prev_score = sum(1 for k in fields if prev.get(k) is not None)
                if score >= prev_score:
                    best[dk] = cand

    rows = list(best.values())
    stats = _write_sorted_store(
        out_dir / "sequence_events_v1.parquet", rows, entity_key="mint"
    )
    stats.update(
        {
            "raw_input_rows": raw_n,
            "deduped_rows": len(rows),
            "duplicate_raw_event_rate": (
                (raw_n - len(rows)) / raw_n if raw_n else 0.0
            ),
            "excluded_system_program_rows": excluded_system,
        }
    )
    return stats


def _build_curve_store(
    *, world_root: Path, out_dir: Path, subset: set[str]
) -> dict[str, Any]:
    path = world_root / "execution" / "execution_curve_states.parquet"
    rows: list[dict[str, Any]] = []
    if path.exists() and pq.ParquetFile(path).metadata.num_rows:
        for row in pq.read_table(path).to_pylist():
            mint = row.get("mint")
            if mint not in subset:
                continue
            et = _to_unix(row.get("event_time"))
            if et is None:
                continue
            side = None
            if row.get("is_buy") is not None:
                side = "buy" if row.get("is_buy") else "sell"
            rows.append(
                {
                    "mint": mint,
                    "event_time_unix": et,
                    "order_key": _order_key(
                        event_time_unix=et,
                        tx_signature=None,
                        wallet=None,
                        side=side,
                        extra=str(row.get("token_amount") or ""),
                    ),
                    "side": side,
                    "is_buy": row.get("is_buy"),
                    "sol_amount": row.get("sol_amount"),
                    "token_amount": row.get("token_amount"),
                    "price_sol": row.get("price_sol"),
                    "executable_price_sol": row.get("executable_price_sol"),
                    "v_sol_bonding_curve": row.get("v_sol_bonding_curve"),
                    "v_tokens_bonding_curve": row.get("v_tokens_bonding_curve"),
                    "curve_pct_depleted": row.get("curve_pct_depleted"),
                    "has_reserves": row.get("has_reserves"),
                    "source_quality_flag": row.get("quality_flag"),
                    "source": "execution_curve_states",
                }
            )
    return _write_sorted_store(
        out_dir / "sequence_curve_v1.parquet", rows, entity_key="mint"
    )


def _build_microstructure_store(
    *, world_root: Path, out_dir: Path, subset: set[str]
) -> dict[str, Any]:
    path = world_root / "microstructure" / "rpc_transactions.parquet"
    rows: list[dict[str, Any]] = []
    if path.exists() and pq.ParquetFile(path).metadata.num_rows:
        for row in pq.read_table(path).to_pylist():
            mint = row.get("mint")
            if mint not in subset or not row.get("found"):
                continue
            bt = row.get("block_time")
            et = float(bt) if bt is not None else None
            if et is None:
                continue
            rows.append(
                {
                    "mint": mint,
                    "event_time_unix": et,
                    "order_key": _order_key(
                        event_time_unix=et,
                        tx_signature=row.get("tx_signature"),
                        wallet=None,
                        side=None,
                        extra=str(row.get("slot") or ""),
                    ),
                    "tx_signature": row.get("tx_signature"),
                    "slot": row.get("slot"),
                    "block_time": et,
                    "success": row.get("success"),
                    "err": row.get("err"),
                    "fee_lamports": row.get("fee_lamports"),
                    "compute_units_consumed": row.get("compute_units_consumed"),
                    "priority_fee_lamports": None,
                    "account_key_count": row.get("account_key_count"),
                    "account_keys": row.get("account_keys"),
                    "balance_change_count": row.get("balance_change_count"),
                    "log_message_count": row.get("log_message_count"),
                    "inner_instruction_groups": row.get("inner_instruction_groups"),
                    "balance_changes_json": row.get("balance_changes_json"),
                    "source_quality_flag": row.get("quality_flag"),
                    "source": "solana_rpc.getTransaction",
                    "availability_priority_fee": False,
                }
            )
    return _write_sorted_store(
        out_dir / "sequence_microstructure_v1.parquet", rows, entity_key="mint"
    )



def _build_ownership_store(
    *, world_root: Path, out_dir: Path, subset: set[str]
) -> dict[str, Any]:
    own_level: dict[str, str] = {}
    own_path = world_root / "universe" / "ownership_evidence.parquet"
    if own_path.exists() and pq.ParquetFile(own_path).metadata.num_rows:
        ot = pq.read_table(own_path, columns=["mint", "ownership_evidence_level"])
        for m, level in zip(
            ot.column("mint").to_pylist(),
            ot.column("ownership_evidence_level").to_pylist(),
            strict=True,
        ):
            if m in subset:
                own_level[m] = level

    rows: list[dict[str, Any]] = []
    for name in (
        "token_transfers_unbiased_days.parquet",
        "token_transfers_rich.parquet",
    ):
        path = world_root / "bigquery" / name
        if not path.exists() or pq.ParquetFile(path).metadata.num_rows == 0:
            continue
        for row in pq.read_table(path).to_pylist():
            mint = row.get("mint")
            if mint not in subset:
                continue
            et = _to_unix(row.get("block_timestamp"))
            if et is None:
                continue
            level = own_level.get(mint, "transfer_ledger_observed_partial")
            if level not in {
                "holder_state_reconstructable",
                "transfer_ledger_observed_partial",
            }:
                level = "transfer_ledger_observed_partial"
            rows.append(
                {
                    "mint": mint,
                    "event_time_unix": et,
                    "order_key": _order_key(
                        event_time_unix=et,
                        tx_signature=row.get("tx_signature"),
                        wallet=row.get("authority") or row.get("source"),
                        side=row.get("transfer_type"),
                    ),
                    "event_kind": "bq_spl_transfer",
                    "evidence_level": level,
                    "transfer_type": row.get("transfer_type"),
                    "source_account": row.get("source"),
                    "destination_account": row.get("destination"),
                    "authority": row.get("authority"),
                    "value": (
                        float(row["value"]) if row.get("value") is not None else None
                    ),
                    "decimals": row.get("decimals"),
                    "tx_signature": row.get("tx_signature"),
                    "slot": row.get("block_slot"),
                    "source": f"bigquery.{name}",
                    "source_quality_flag": "ledger_observed",
                    "is_complete_holder_ledger": level
                    == "holder_state_reconstructable",
                }
            )

    rpc_path = world_root / "graph" / "edges_ownership_rpc_token_balances.parquet"
    if rpc_path.exists() and pq.ParquetFile(rpc_path).metadata.num_rows:
        for row in pq.read_table(rpc_path).to_pylist():
            mint = row.get("mint")
            if mint not in subset:
                continue
            bt = row.get("block_time")
            et = float(bt) if bt is not None else None
            if et is None:
                continue
            rows.append(
                {
                    "mint": mint,
                    "event_time_unix": et,
                    "order_key": _order_key(
                        event_time_unix=et,
                        tx_signature=row.get("tx_signature"),
                        wallet=row.get("owner"),
                        side="rpc_balance",
                    ),
                    "event_kind": "rpc_pre_post_token_balance",
                    "evidence_level": "rpc_balance_observed",
                    "transfer_type": None,
                    "source_account": None,
                    "destination_account": None,
                    "authority": row.get("owner"),
                    "value": row.get("delta_ui_amount"),
                    "decimals": None,
                    "tx_signature": row.get("tx_signature"),
                    "slot": row.get("slot"),
                    "pre_ui_amount": row.get("pre_ui_amount"),
                    "post_ui_amount": row.get("post_ui_amount"),
                    "source": "solana_rpc.pre_post_token_balances",
                    "source_quality_flag": row.get("quality_flag") or "sampled_tx",
                    "is_complete_holder_ledger": False,
                }
            )

    stats = _write_sorted_store(
        out_dir / "sequence_ownership_v1.parquet", rows, entity_key="mint"
    )
    stats["evidence_level_counts"] = dict(Counter(r["evidence_level"] for r in rows))
    stats["event_kind_counts"] = dict(Counter(r["event_kind"] for r in rows))
    return stats


def _build_graph_token_store(
    *,
    world_root: Path,
    out_dir: Path,
    subset: set[str],
    event_rows_path: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    if event_rows_path.exists() and pq.ParquetFile(event_rows_path).metadata.num_rows:
        for row in pq.read_table(event_rows_path).to_pylist():
            wallet = row.get("wallet")
            if not wallet:
                continue
            side = row.get("side") or ("buy" if row.get("is_buy") else "sell")
            rows.append(
                {
                    "mint": row["mint"],
                    "event_time_unix": row["event_time_unix"],
                    "order_key": row["order_key"],
                    "edge_type": f"wallet_token_{side}",
                    "src": wallet,
                    "dst": row["mint"],
                    "wallet": wallet,
                    "token_mint": row["mint"],
                    "sol_amount": row.get("sol_amount"),
                    "token_amount": row.get("token_amount"),
                    "tx_signature": row.get("tx_signature"),
                    "slot": row.get("slot"),
                    "source": "sequence_events_v1",
                    "source_quality_flag": row.get("source_quality_flag"),
                    "is_cohort_derived": False,
                }
            )

    ct = world_root / "graph" / "edges_creator_token.parquet"
    if ct.exists() and pq.ParquetFile(ct).metadata.num_rows:
        for row in pq.read_table(ct).to_pylist():
            mint = row.get("dst")
            if mint not in subset:
                continue
            et = _to_unix(row.get("t"))
            if et is None or not row.get("src"):
                continue
            rows.append(
                {
                    "mint": mint,
                    "event_time_unix": et,
                    "order_key": _order_key(
                        event_time_unix=et,
                        tx_signature=None,
                        wallet=row.get("src"),
                        side="create",
                    ),
                    "edge_type": "creator_token",
                    "src": row["src"],
                    "dst": mint,
                    "wallet": row["src"],
                    "token_mint": mint,
                    "sol_amount": None,
                    "token_amount": None,
                    "tx_signature": None,
                    "slot": None,
                    "source": "edges_creator_token",
                    "source_quality_flag": row.get("quality_flag"),
                    "is_cohort_derived": False,
                }
            )

    fn = world_root / "graph" / "edges_funding_native_sol.parquet"
    if fn.exists() and pq.ParquetFile(fn).metadata.num_rows:
        for row in pq.read_table(fn).to_pylist():
            mint = row.get("mint")
            if mint not in subset:
                continue
            bt = row.get("block_time")
            et = float(bt) if bt is not None else None
            if et is None:
                continue
            rows.append(
                {
                    "mint": mint,
                    "event_time_unix": et,
                    "order_key": _order_key(
                        event_time_unix=et,
                        tx_signature=row.get("tx_signature"),
                        wallet=row.get("src"),
                        side="native_sol",
                    ),
                    "edge_type": "native_sol_funding",
                    "src": row.get("src"),
                    "dst": row.get("dst"),
                    "wallet": row.get("dst"),
                    "token_mint": mint,
                    "sol_amount": (
                        float(row["lamports"]) / 1e9
                        if row.get("lamports") is not None
                        else None
                    ),
                    "token_amount": None,
                    "tx_signature": row.get("tx_signature"),
                    "slot": row.get("slot"),
                    "source": "edges_funding_native_sol",
                    "source_quality_flag": row.get("quality_flag"),
                    "is_cohort_derived": False,
                }
            )

    own_path = out_dir / "sequence_ownership_v1.parquet"
    if own_path.exists() and pq.ParquetFile(own_path).metadata.num_rows:
        for row in pq.read_table(own_path).to_pylist():
            if row.get("event_kind") != "bq_spl_transfer":
                continue
            src = row.get("source_account")
            dst = row.get("destination_account")
            if not src and not dst:
                continue
            rows.append(
                {
                    "mint": row["mint"],
                    "event_time_unix": row["event_time_unix"],
                    "order_key": row["order_key"],
                    "edge_type": "spl_transfer_ownership",
                    "src": src,
                    "dst": dst,
                    "wallet": dst or src,
                    "token_mint": row["mint"],
                    "sol_amount": None,
                    "token_amount": row.get("value"),
                    "tx_signature": row.get("tx_signature"),
                    "slot": row.get("slot"),
                    "source": "sequence_ownership_v1",
                    "source_quality_flag": row.get("source_quality_flag"),
                    "evidence_level": row.get("evidence_level"),
                    "is_cohort_derived": False,
                }
            )

    stats = _write_sorted_store(
        out_dir / "sequence_graph_token_v1.parquet", rows, entity_key="mint"
    )
    stats["edge_type_counts"] = dict(Counter(r["edge_type"] for r in rows))
    stats["excludes_red_cohort_membership"] = True
    stats["excludes_lifetime_wallet_token_aggregates"] = True
    return stats


def _build_graph_wallet_prior_store(
    *,
    world_root: Path,
    out_dir: Path,
    pump_root: Path | None,
    focal_wallets: set[str],
) -> dict[str, Any]:
    """Corpus wallet↔token priors for focal wallets (Arrow-filtered, capped)."""
    pump = pump_root or Path("data/external/pumpfun_memecoin_corpus")
    trades_dir = pump / "trades"
    scanned = 0
    excluded_system = 0
    parts: list[pa.Table] = []
    counts: dict[str, int] = defaultdict(int)
    wallets_full: set[str] = set()
    kept = 0
    # Feasible local budget: prefer breadth across wallets over unbounded depth.
    per_wallet_cap = min(256, WALLET_PRIOR_EVENT_CAP)
    global_budget = max(750_000, len(focal_wallets) * per_wallet_cap)

    if trades_dir.exists() and focal_wallets:
        dataset = ds.dataset(trades_dir, format="parquet")
        use_cols = [
            c
            for c in [
                "mint",
                "user_wallet",
                "event_time",
                "is_buy",
                "sol_amount",
                "token_amount",
                "tx_signature",
            ]
            if c in set(dataset.schema.names)
        ]
        wallet_set = pa.array(list(focal_wallets), type=pa.string())
        for batch in dataset.scanner(columns=use_cols, batch_size=256_000).to_batches():
            scanned += batch.num_rows
            wallets_col = batch.column("user_wallet")
            excluded_system += int(
                pc.sum(
                    pc.cast(pc.equal(wallets_col, SYSTEM_PROGRAM_WALLET), pa.int64())
                ).as_py()
                or 0
            )
            keep = pc.and_(
                pc.is_in(wallets_col, value_set=wallet_set),
                pc.not_equal(wallets_col, SYSTEM_PROGRAM_WALLET),
            )
            if wallets_full:
                keep = pc.and_(
                    keep,
                    pc.invert(
                        pc.is_in(
                            wallets_col,
                            value_set=pa.array(list(wallets_full), type=pa.string()),
                        )
                    ),
                )
            filtered = batch.filter(keep)
            if filtered.num_rows == 0:
                continue

            et = filtered.column("event_time")
            if pa.types.is_timestamp(et.type):
                et_unix = pc.cast(
                    pc.divide(pc.cast(et, pa.int64()), 1_000_000), pa.float64()
                )
            else:
                et_unix = pa.array(
                    [_to_unix(x) for x in et.to_pylist()], type=pa.float64()
                )

            is_buy_list = filtered.column("is_buy").to_pylist()
            edge_types = [
                "wallet_token_buy" if b else "wallet_token_sell" for b in is_buy_list
            ]
            wallets = filtered.column("user_wallet").to_pylist()
            take_idx: list[int] = []
            for i, w in enumerate(wallets):
                if w in wallets_full:
                    continue
                if counts[w] >= per_wallet_cap:
                    wallets_full.add(w)
                    continue
                take_idx.append(i)
                counts[w] += 1
                kept += 1
                if counts[w] >= per_wallet_cap:
                    wallets_full.add(w)
                if kept >= global_budget:
                    break
            if not take_idx:
                if kept >= global_budget or len(wallets_full) >= len(focal_wallets):
                    break
                continue

            idx = pa.array(take_idx, type=pa.int64())
            part = pa.table(
                {
                    "wallet": pc.take(filtered.column("user_wallet"), idx),
                    "event_time_unix": pc.take(et_unix, idx),
                    "token_mint": pc.take(filtered.column("mint"), idx),
                    "dst": pc.take(filtered.column("mint"), idx),
                    "src": pc.take(filtered.column("user_wallet"), idx),
                    "sol_amount": pc.take(filtered.column("sol_amount"), idx),
                    "token_amount": pc.take(filtered.column("token_amount"), idx),
                    "tx_signature": pc.take(filtered.column("tx_signature"), idx),
                    "edge_type": pa.array(
                        [edge_types[i] for i in take_idx], type=pa.string()
                    ),
                }
            )
            parts.append(part)
            if kept >= global_budget or len(wallets_full) >= len(focal_wallets):
                break

    rows: list[dict[str, Any]] = []
    if parts:
        table = pa.concat_tables(parts)
        df = table.to_pandas()
        df = df.sort_values(
            ["wallet", "event_time_unix", "tx_signature"], kind="mergesort"
        )
        for rec in df.to_dict(orient="records"):
            et = float(rec["event_time_unix"])
            side = "buy" if str(rec["edge_type"]).endswith("buy") else "sell"
            rows.append(
                {
                    "wallet": rec["wallet"],
                    "event_time_unix": et,
                    "order_key": _order_key(
                        event_time_unix=et,
                        tx_signature=rec.get("tx_signature"),
                        wallet=rec["wallet"],
                        side=side,
                        extra=str(rec.get("token_mint") or ""),
                    ),
                    "edge_type": rec["edge_type"],
                    "src": rec["src"],
                    "dst": rec["dst"],
                    "token_mint": rec["token_mint"],
                    "sol_amount": rec.get("sol_amount"),
                    "token_amount": rec.get("token_amount"),
                    "tx_signature": rec.get("tx_signature"),
                    "source": "pumpfun_memecoin_corpus.trades",
                    "source_quality_flag": "corpus_prior_observed",
                    "is_cohort_derived": False,
                    "capped": bool(counts.get(rec["wallet"], 0) >= per_wallet_cap),
                }
            )

    stats = _write_sorted_store(
        out_dir / "sequence_graph_wallet_prior_v1.parquet",
        rows,
        entity_key="wallet",
    )
    stats.update(
        {
            "corpus_rows_scanned": scanned,
            "rows_kept": kept,
            "focal_wallets": len(focal_wallets),
            "wallets_with_prior": len({r["wallet"] for r in rows}),
            "excluded_system_program_rows": excluded_system,
            "per_wallet_cap": per_wallet_cap,
            "global_row_budget": global_budget,
            "wallets_hit_cap": sum(1 for _w, n in counts.items() if n >= per_wallet_cap),
            "excludes_red_cohort_membership": True,
        }
    )
    return stats


def _build_global_reference_manifest(
    *, world_root: Path, out_dir: Path
) -> dict[str, Any]:
    global_paths = {
        "launches": world_root / "pit" / "global" / "launches_timeline.parquet",
        "trade_buckets": world_root / "pit" / "global" / "trade_second_buckets.parquet",
        "migrations": world_root / "pit" / "global" / "migrations_timeline.parquet",
        "blocks_daily": world_root / "bigquery" / "blocks_daily.parquet",
        "daily_pump_network": world_root
        / "global_context"
        / "daily_pump_network_context.parquet",
    }
    manifest = {
        "store_id": "sequence_global_v1",
        "version": SEQUENCE_STORE_VERSION,
        "policy": "last_completed_second_only_for_trade_buckets",
        "paths": {k: str(v) for k, v in global_paths.items()},
        "row_counts": {
            k: (pq.ParquetFile(v).metadata.num_rows if v.exists() else 0)
            for k, v in global_paths.items()
        },
        "note": (
            "References PIT v4 global indexes; consumers must apply "
            "last_completed_second(T) for trade flow."
        ),
    }
    write_json(out_dir / "sequence_global_v1_manifest.json", manifest)
    return manifest



def build_observation_sequence_refs(
    *,
    world_root: Path,
    out_dir: Path,
    store_stats: dict[str, Any],
) -> dict[str, Any]:
    """Compact causal cutoffs for every observation_index_v1 row."""
    index_path = (
        world_root
        / "observations"
        / "observation_dataset_v1"
        / "observation_index_v1.parquet"
    )
    obs = pq.read_table(index_path).to_pylist()

    stores = {
        "events": (
            out_dir / "sequence_events_v1.parquet",
            out_dir / "sequence_events_v1_entity_index.parquet",
        ),
        "curve": (
            out_dir / "sequence_curve_v1.parquet",
            out_dir / "sequence_curve_v1_entity_index.parquet",
        ),
        "micro": (
            out_dir / "sequence_microstructure_v1.parquet",
            out_dir / "sequence_microstructure_v1_entity_index.parquet",
        ),
        "ownership": (
            out_dir / "sequence_ownership_v1.parquet",
            out_dir / "sequence_ownership_v1_entity_index.parquet",
        ),
        "graph_token": (
            out_dir / "sequence_graph_token_v1.parquet",
            out_dir / "sequence_graph_token_v1_entity_index.parquet",
        ),
    }

    times: dict[str, list[float]] = {}
    indexes: dict[str, dict[str, tuple[int, int]]] = {}
    for name, (spath, ipath) in stores.items():
        times[name] = _load_times_array(spath)
        indexes[name] = _load_entity_index(ipath)

    launches_path = world_root / "pit" / "global" / "launches_timeline.parquet"
    mig_path = world_root / "pit" / "global" / "migrations_timeline.parquet"
    launch_times = (
        sorted(
            float(x)
            for x in pq.read_table(launches_path, columns=["event_time_unix"])
            .column(0)
            .to_pylist()
            if x is not None
        )
        if launches_path.exists()
        else []
    )
    mig_times = (
        sorted(
            float(x)
            for x in pq.read_table(mig_path, columns=["event_time_unix"])
            .column(0)
            .to_pylist()
            if x is not None
        )
        if mig_path.exists()
        else []
    )

    ref_rows: list[dict[str, Any]] = []
    for row in obs:
        mint = row["mint"]
        t = float(row["observation_time_t"])
        ref: dict[str, Any] = {
            "observation_id": row["observation_id"],
            "mint": mint,
            "observation_time_t": t,
            "token_age_seconds": row.get("token_age_seconds"),
            "observation_trigger": row.get("observation_trigger"),
            "trigger_value": row.get("trigger_value"),
            "launch_day": row.get("launch_day"),
            "launch_regime": row.get("launch_regime"),
            "ownership_level_at_t": row.get("ownership_level_at_t"),
            "sequence_store_version": SEQUENCE_STORE_VERSION,
            "sequence_dataset_version": SEQUENCE_DATASET_VERSION,
            "window_presets_available": True,
        }
        for name in stores:
            lo, hi = indexes[name].get(mint, (0, 0))
            end = _end_idx_for_t(times[name], lo, hi, t) if hi > lo else lo
            count = max(0, end - lo)
            ref[f"{name}_store_id"] = f"sequence_{name}_v1"
            ref[f"{name}_start_idx"] = lo
            ref[f"{name}_end_idx"] = end
            ref[f"{name}_count"] = count
            ref[f"mask_{name}"] = count > 0

        end_sec = last_completed_second(t)
        ref["global_store_id"] = "sequence_global_v1"
        ref["global_last_completed_second"] = end_sec
        ref["global_launch_end_idx"] = bisect_right(launch_times, t)
        ref["global_launch_count"] = ref["global_launch_end_idx"]
        ref["global_migration_end_idx"] = bisect_right(mig_times, t)
        ref["global_migration_count"] = ref["global_migration_end_idx"]
        ref["mask_global"] = True
        ref["graph_wallet_prior_strategy"] = (
            "wallets_from_graph_token_events_leq_T_then_wallet_store_cutoff"
        )
        ref["graph_wallet_prior_store_id"] = "sequence_graph_wallet_prior_v1"
        ref_rows.append(ref)

    refs_path = out_dir / "observation_sequence_refs_v1.parquet"
    pq.write_table(pa.Table.from_pylist(ref_rows), refs_path)
    return {
        "path": str(refs_path),
        "rows": len(ref_rows),
        "store_stats": store_stats,
    }


def _stratified_sample_refs(
    refs: list[dict[str, Any]], *, n: int = 48
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in refs:
        activity = (
            "high"
            if int(r.get("events_count") or 0) >= 32
            else ("low" if int(r.get("events_count") or 0) <= 2 else "mid")
        )
        key = "|".join(
            [
                str(r.get("launch_day") or "na"),
                str(r.get("launch_regime") or "na"),
                str(r.get("observation_trigger") or "na"),
                str(r.get("trigger_value") or "na"),
                str(r.get("ownership_level_at_t") or "na"),
                activity,
            ]
        )
        buckets[key].append(r)

    chosen: list[dict[str, Any]] = []
    for key in sorted(buckets):
        rows = sorted(buckets[key], key=lambda r: r["observation_id"])
        chosen.append(rows[0])
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        for key in sorted(buckets):
            for row in sorted(buckets[key], key=lambda r: r["observation_id"])[1:]:
                chosen.append(row)
                if len(chosen) >= n:
                    break
            if len(chosen) >= n:
                break
    return chosen[:n]


def validate_observation_causality(
    *,
    world_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Exhaustive cheap boundary checks for all observation refs."""
    refs = pq.read_table(out_dir / "observation_sequence_refs_v1.parquet").to_pylist()
    store_paths = {
        "events": out_dir / "sequence_events_v1.parquet",
        "curve": out_dir / "sequence_curve_v1.parquet",
        "micro": out_dir / "sequence_microstructure_v1.parquet",
        "ownership": out_dir / "sequence_ownership_v1.parquet",
        "graph_token": out_dir / "sequence_graph_token_v1.parquet",
    }
    times = {m: _load_times_array(store_paths[m]) for m in MODALITIES}

    counts_by_mod: dict[str, list[int]] = {m: [] for m in MODALITIES}
    post_t_violations = 0
    ordering_violations = 0
    coverage_by_age: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    trigger_cov: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for m in MODALITIES:
        idx = _load_entity_index(
            store_paths[m].with_name(store_paths[m].stem + "_entity_index.parquet")
        )
        for _ent, (lo, hi) in idx.items():
            for i in range(lo + 1, hi):
                if times[m][i] < times[m][i - 1]:
                    ordering_violations += 1

    for ref in refs:
        t = float(ref["observation_time_t"])
        age = str(ref.get("trigger_value") or ref.get("observation_trigger"))
        trig = str(ref.get("observation_trigger"))
        for m in MODALITIES:
            lo = int(ref[f"{m}_start_idx"])
            end = int(ref[f"{m}_end_idx"])
            count = int(ref[f"{m}_count"])
            counts_by_mod[m].append(count)
            if count:
                coverage_by_age[age][m] += 1
                trigger_cov[trig][m] += 1
            if end < lo:
                post_t_violations += 1
                continue
            for i in range(lo, end):
                if times[m][i] > t:
                    post_t_violations += 1
                    break

    sample = _stratified_sample_refs(refs, n=48)
    sample_results = []
    graph_prior_path = out_dir / "sequence_graph_wallet_prior_v1.parquet"
    graph_prior_times = _load_times_array(graph_prior_path)
    graph_prior_idx = _load_entity_index(
        out_dir / "sequence_graph_wallet_prior_v1_entity_index.parquet"
    )
    graph_token_path = out_dir / "sequence_graph_token_v1.parquet"
    graph_token_wallets: list[tuple[Any, Any]] = []
    if graph_token_path.exists() and pq.ParquetFile(graph_token_path).metadata.num_rows:
        gt = pq.read_table(graph_token_path, columns=["wallet", "event_time_unix"])
        graph_token_wallets = list(
            zip(
                gt.column("wallet").to_pylist(),
                gt.column("event_time_unix").to_pylist(),
                strict=True,
            )
        )

    for ref in sample:
        t = float(ref["observation_time_t"])
        mint = ref["mint"]
        ok = True
        details: dict[str, Any] = {}
        for m in MODALITIES:
            lo = int(ref[f"{m}_start_idx"])
            end = int(ref[f"{m}_end_idx"])
            bad = sum(1 for i in range(lo, end) if times[m][i] > t)
            details[m] = {"count": end - lo, "post_t": bad}
            if bad:
                ok = False
        lo = int(ref["graph_token_start_idx"])
        end = int(ref["graph_token_end_idx"])
        wallets = {
            graph_token_wallets[i][0]
            for i in range(lo, end)
            if i < len(graph_token_wallets) and graph_token_wallets[i][0]
        }
        prior_post = 0
        prior_n = 0
        for w in wallets:
            wlo, whi = graph_prior_idx.get(w, (0, 0))
            wend = _end_idx_for_t(graph_prior_times, wlo, whi, t)
            prior_n += max(0, wend - wlo)
            for i in range(wlo, wend):
                if graph_prior_times[i] > t:
                    prior_post += 1
        details["graph_wallet_prior"] = {"count": prior_n, "post_t": prior_post}
        if prior_post:
            ok = False
        end_sec = int(ref["global_last_completed_second"])
        details["global_last_completed_second"] = end_sec
        details["global_completed_second_ok"] = end_sec == last_completed_second(t)
        if not details["global_completed_second_ok"]:
            ok = False
        sample_results.append(
            {
                "observation_id": ref["observation_id"],
                "mint": mint,
                "observation_time_t": t,
                "trigger": ref.get("observation_trigger"),
                "trigger_value": ref.get("trigger_value"),
                "launch_day": ref.get("launch_day"),
                "launch_regime": ref.get("launch_regime"),
                "ownership_level_at_t": ref.get("ownership_level_at_t"),
                "passed": ok,
                "details": details,
            }
        )

    events_dup_rate = 0.0
    man = out_dir / "sequence_dataset_manifest.json"
    if man.exists():
        payload = json.loads(man.read_text(encoding="utf-8"))
        events_dup_rate = ((payload.get("stores") or {}).get("events") or {}).get(
            "duplicate_raw_event_rate", 0.0
        )

    return {
        "observations_validated": len(refs),
        "post_t_violations": post_t_violations,
        "ordering_violations": ordering_violations,
        "duplicate_raw_event_rate": events_dup_rate,
        "sequence_length_stats": {
            m: _length_stats(counts_by_mod[m]) for m in MODALITIES
        },
        "observations_with_no_sequence": {
            m: sum(1 for c in counts_by_mod[m] if c == 0) for m in MODALITIES
        },
        "token_event_coverage_by_age": {
            age: dict(v) for age, v in coverage_by_age.items()
        },
        "coverage_by_trigger": {k: dict(v) for k, v in trigger_cov.items()},
        "temporal_graph_coverage": {
            "obs_with_graph_token": sum(
                1 for c in counts_by_mod["graph_token"] if c > 0
            ),
            "obs_total": len(refs),
        },
        "microstructure_coverage": {
            "obs_with_micro": sum(1 for c in counts_by_mod["micro"] if c > 0),
            "obs_total": len(refs),
        },
        "ownership_coverage": {
            "obs_with_ownership": sum(1 for c in counts_by_mod["ownership"] if c > 0),
            "obs_total": len(refs),
        },
        "stratified_sample": {
            "n": len(sample_results),
            "all_passed": all(r["passed"] for r in sample_results),
            "results": sample_results,
        },
        "causal_boundary_ok": post_t_violations == 0 and ordering_violations == 0,
    }



def build_sequence_dataset_v1(
    *,
    world_root: Path,
    manifest_dir: Path,
    pump_root: Path | None = None,
) -> dict[str, Any]:
    """Materialize sequence stores + observation refs + contract + audit."""
    ensure_atomic_trades_index(world_root=world_root, pump_root=pump_root, force=False)
    ensure_global_context_index(world_root=world_root, pump_root=pump_root, force=False)

    out_dir = ensure_dir(world_root / "sequences" / SEQUENCE_DATASET_VERSION)
    subset = set(
        pq.read_table(
            world_root / "universe" / "rich_subset_mints.parquet", columns=["mint"]
        )
        .column(0)
        .to_pylist()
    ) - {None}

    logger.info("Building token event sequence store")
    events_stats = _build_token_event_store(
        world_root=world_root, out_dir=out_dir, subset=subset
    )
    logger.info("Building curve/execution sequence store")
    curve_stats = _build_curve_store(
        world_root=world_root, out_dir=out_dir, subset=subset
    )
    logger.info("Building microstructure sequence store")
    micro_stats = _build_microstructure_store(
        world_root=world_root, out_dir=out_dir, subset=subset
    )
    logger.info("Building ownership sequence store")
    own_stats = _build_ownership_store(
        world_root=world_root, out_dir=out_dir, subset=subset
    )
    logger.info("Building token temporal graph store")
    graph_token_stats = _build_graph_token_store(
        world_root=world_root,
        out_dir=out_dir,
        subset=subset,
        event_rows_path=out_dir / "sequence_events_v1.parquet",
    )

    focal_wallets: set[str] = set()
    ev_path = out_dir / "sequence_events_v1.parquet"
    if ev_path.exists() and pq.ParquetFile(ev_path).metadata.num_rows:
        for w in pq.read_table(ev_path, columns=["wallet"]).column(0).to_pylist():
            if w and w != SYSTEM_PROGRAM_WALLET:
                focal_wallets.add(w)

    logger.info(
        "Building wallet-prior graph store (%s focal wallets)", len(focal_wallets)
    )
    graph_prior_stats = _build_graph_wallet_prior_store(
        world_root=world_root,
        out_dir=out_dir,
        pump_root=pump_root,
        focal_wallets=focal_wallets,
    )
    global_manifest = _build_global_reference_manifest(
        world_root=world_root, out_dir=out_dir
    )

    store_stats = {
        "events": events_stats,
        "curve": curve_stats,
        "microstructure": micro_stats,
        "ownership": own_stats,
        "graph_token": graph_token_stats,
        "graph_wallet_prior": graph_prior_stats,
        "global": global_manifest,
    }
    write_json(
        out_dir / "sequence_dataset_manifest.json",
        {
            "generated_at": utc_now(),
            "version": SEQUENCE_DATASET_VERSION,
            "store_version": SEQUENCE_STORE_VERSION,
            "freeze_id": FREEZE_ID,
            "pit_engine_version": PIT_ENGINE_VERSION,
            "subset_mints": len(subset),
            "stores": store_stats,
        },
    )

    logger.info("Building observation sequence references")
    refs_stats = build_observation_sequence_refs(
        world_root=world_root, out_dir=out_dir, store_stats=store_stats
    )
    logger.info("Validating observation causality")
    validation = validate_observation_causality(
        world_root=world_root, out_dir=out_dir
    )

    contract = build_model_input_contract_v1()
    write_json(out_dir / "MODEL_INPUT_CONTRACT_V1.json", contract)
    ensure_dir(manifest_dir)
    write_json(manifest_dir / "MODEL_INPUT_CONTRACT_V1.json", contract)

    disk_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    for p in (
        world_root / "pit" / "global" / "trade_second_buckets.parquet",
        world_root / "pit" / "global" / "launches_timeline.parquet",
        world_root / "pit" / "global" / "migrations_timeline.parquet",
    ):
        if p.exists():
            disk_bytes += p.stat().st_size

    audit = {
        "audit_id": SEQUENCE_DATASET_VERSION,
        "generated_at": utc_now(),
        "acquisition_freeze_id": FREEZE_ID,
        "pit_engine_version": PIT_ENGINE_VERSION,
        "sequence_store_version": SEQUENCE_STORE_VERSION,
        "subset_mints": len(subset),
        "sequence_row_counts_by_modality": {
            "events": events_stats.get("rows"),
            "curve": curve_stats.get("rows"),
            "microstructure": micro_stats.get("rows"),
            "ownership": own_stats.get("rows"),
            "graph_token": graph_token_stats.get("rows"),
            "graph_wallet_prior": graph_prior_stats.get("rows"),
            "global_trade_second_buckets": (global_manifest.get("row_counts") or {}).get(
                "trade_buckets"
            ),
            "global_launches": (global_manifest.get("row_counts") or {}).get("launches"),
            "global_migrations": (global_manifest.get("row_counts") or {}).get(
                "migrations"
            ),
        },
        "construction_stats": store_stats,
        "observation_refs": {
            "rows": refs_stats.get("rows"),
            "path": refs_stats.get("path"),
        },
        "causality_validation": validation,
        "estimated_disk_bytes": disk_bytes,
        "estimated_disk_mb": round(disk_bytes / (1024 * 1024), 3),
        "artifacts": {
            "sequence_events_v1": str(out_dir / "sequence_events_v1.parquet"),
            "sequence_microstructure_v1": str(
                out_dir / "sequence_microstructure_v1.parquet"
            ),
            "sequence_ownership_v1": str(out_dir / "sequence_ownership_v1.parquet"),
            "sequence_graph_token_v1": str(out_dir / "sequence_graph_token_v1.parquet"),
            "sequence_graph_wallet_prior_v1": str(
                out_dir / "sequence_graph_wallet_prior_v1.parquet"
            ),
            "sequence_curve_v1": str(out_dir / "sequence_curve_v1.parquet"),
            "observation_sequence_refs_v1": str(
                out_dir / "observation_sequence_refs_v1.parquet"
            ),
            "model_input_contract_v1": str(out_dir / "MODEL_INPUT_CONTRACT_V1.json"),
        },
        "labels": "not_generated",
        "targets": "not_generated",
        "training": "not_started",
        "known_limitations": [
            "Wallet-prior corpus history uses Arrow-filtered full-corpus scan with "
            "per-wallet and global row budgets for local feasibility "
            f"(cap min(256,{WALLET_PRIOR_EVENT_CAP})).",
            "Priority-fee fields were not recovered in the freeze; left null.",
            "Real bonding-curve reserves are not in local trade artifacts; left null.",
            "Trade slots are unavailable on Pump trade parquet; left null.",
            "RED-COHORT membership is intentionally excluded from temporal graph inputs.",
            "Lifetime edges_wallet_token aggregates are not used as temporal inputs.",
            "Global trade flow uses completed-second buckets only (<1s latency).",
            "Ownership RPC rows remain sampled-tx evidence, not a full holder ledger.",
        ],
    }
    write_json(out_dir / "SEQUENCE_DATASET_V1_AUDIT.json", audit)
    write_json(manifest_dir / "SEQUENCE_DATASET_V1_AUDIT.json", audit)
    (manifest_dir / "SEQUENCE_DATASET_V1_AUDIT.md").write_text(
        render_sequence_audit_md(audit), encoding="utf-8"
    )
    return audit


def render_sequence_audit_md(audit: dict[str, Any]) -> str:
    counts = audit.get("sequence_row_counts_by_modality") or {}
    val = audit.get("causality_validation") or {}
    lines = [
        f"# Sequence Dataset v1 (`{audit['audit_id']}`)",
        "",
        f"Generated at: `{audit['generated_at']}`",
        f"Freeze: `{audit['acquisition_freeze_id']}`",
        f"PIT engine: `{audit['pit_engine_version']}`",
        "",
        "## Sequence row counts",
        "",
    ]
    for k, v in counts.items():
        lines.append(f"- `{k}`: **{v}**")
    lines.extend(
        [
            "",
            "## Observation refs / causality",
            "",
            f"- observation refs: "
            f"**{(audit.get('observation_refs') or {}).get('rows')}**",
            f"- post_t_violations: **{val.get('post_t_violations')}**",
            f"- ordering_violations: **{val.get('ordering_violations')}**",
            f"- causal_boundary_ok: **{val.get('causal_boundary_ok')}**",
            f"- stratified_sample_all_passed: **"
            f"{(val.get('stratified_sample') or {}).get('all_passed')}**",
            f"- disk MB: {audit.get('estimated_disk_mb')}",
            "",
            "## Known limitations",
            "",
        ]
    )
    for item in audit.get("known_limitations") or []:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
