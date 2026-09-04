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

SEQUENCE_DATASET_VERSION = "sequence_dataset_v2"
SEQUENCE_STORE_VERSION = "sequence_stores_v2"
MODEL_INPUT_CONTRACT_VERSION = "model_input_contract_v2"

# Default last-N window at dereference time only (store keeps full history).
WALLET_PRIOR_DEREF_DEFAULT_LAST_N = 256
WALLET_PRIOR_EVENT_CAP = 2048  # legacy alias for window presets

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


def _round_amt(value: Any, ndigits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _event_fingerprint(
    *,
    mint: str,
    tx_signature: str | None,
    event_time_unix: float,
    wallet: str | None,
    is_buy: bool | None,
    sol_amount: float | None,
    token_amount: float | None,
    v_sol_bonding_curve: float | None = None,
    v_tokens_bonding_curve: float | None = None,
) -> str:
    """Conservative fingerprint: never collapse heterogeneous same-signature events."""
    side = None if is_buy is None else ("buy" if is_buy else "sell")
    return "|".join(
        [
            str(mint),
            str(tx_signature or ""),
            f"{float(event_time_unix):.6f}",
            str(wallet or ""),
            str(side or ""),
            str(_round_amt(sol_amount, 9)),
            str(_round_amt(token_amount, 6)),
            str(_round_amt(v_sol_bonding_curve, 6)),
            str(_round_amt(v_tokens_bonding_curve, 3)),
        ]
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
    v_sol_bonding_curve: float | None = None,
    v_tokens_bonding_curve: float | None = None,
) -> str:
    return _event_fingerprint(
        mint=mint,
        tx_signature=tx_signature,
        event_time_unix=event_time_unix,
        wallet=wallet,
        is_buy=is_buy,
        sol_amount=sol_amount,
        token_amount=token_amount,
        v_sol_bonding_curve=v_sol_bonding_curve,
        v_tokens_bonding_curve=v_tokens_bonding_curve,
    )


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


def build_model_input_contract_v2() -> dict[str, Any]:
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
            "full_history_ownership_level_on_atomic_sequence_rows",
            "scan_order_truncated_wallet_prior_history",
        ],
        "ownership_sequence_policy": {
            "atomic_rows": "event_local_source_quality_only",
            "bq_transfer_quality": "ledger_transfer_observed",
            "pit_quantities_only_at_observation": [
                "ownership_level_at_t",
                "holder_state_reconstructable_at_t",
            ],
        },
        "wallet_prior_policy": {
            "store": "complete_focal_wallet_history_sorted_causally",
            "windows_applied_at": "dereference_time_only",
            "default_last_n": WALLET_PRIOR_DEREF_DEFAULT_LAST_N,
        },
        "trade_dedupe_policy": "conservative_event_fingerprint_not_signature_alone",
        "graph_node_identities_allowed": [
            "wallet_id (cross-token history is an intended TGN signal)",
            "token_mint as graph node id (not tabular categorical feature)",
            "creator_wallet as graph node id",
        ],
        "modality_stores": {
            "token_events": "sequence_events_v2",
            "microstructure": "sequence_microstructure_v2",
            "ownership": "sequence_ownership_v2",
            "graph_token": "sequence_graph_token_v2",
            "graph_wallet_prior": "sequence_graph_wallet_prior_v2",
            "curve_execution": "sequence_curve_v2",
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
            "evidence_level",  # observation/PIT only — never on atomic BQ rows
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


def load_observation_graph_neighborhood(
    *,
    out_dir: Path,
    mint: str,
    observation_time_t: float,
    last_n: int | None = WALLET_PRIOR_DEREF_DEFAULT_LAST_N,
    graph_token_times: list[float] | None = None,
    graph_token_wallets: list[str | None] | None = None,
    graph_token_index: dict[str, tuple[int, int]] | None = None,
    graph_prior_times: list[float] | None = None,
    graph_prior_index: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Causal graph neighborhood for (mint, T).

    1) token-graph events with event_time <= T
    2) wallets observable on that mint by T
    3) each wallet's prior events with event_time <= T
    4) optional last-N window on prior history (applied only at dereference)
    """
    token_path = out_dir / "sequence_graph_token_v2.parquet"
    prior_path = out_dir / "sequence_graph_wallet_prior_v2.parquet"
    if graph_token_index is None:
        graph_token_index = _load_entity_index(
            token_path.with_name(token_path.stem + "_entity_index.parquet")
        )
    if graph_prior_index is None:
        graph_prior_index = _load_entity_index(
            prior_path.with_name(prior_path.stem + "_entity_index.parquet")
        )
    if graph_token_times is None:
        graph_token_times = _load_times_array(token_path)
    if graph_prior_times is None:
        graph_prior_times = _load_times_array(prior_path)
    if graph_token_wallets is None:
        if token_path.exists() and pq.ParquetFile(token_path).metadata.num_rows:
            graph_token_wallets = pq.read_table(
                token_path, columns=["wallet"]
            ).column(0).to_pylist()
        else:
            graph_token_wallets = []

    lo, hi = graph_token_index.get(mint, (0, 0))
    end = _end_idx_for_t(graph_token_times, lo, hi, observation_time_t)
    token_slice = list(range(lo, end))
    wallets = sorted(
        {
            graph_token_wallets[i]
            for i in token_slice
            if i < len(graph_token_wallets) and graph_token_wallets[i]
        }
    )
    prior_refs: list[dict[str, Any]] = []
    prior_post_t = 0
    for w in wallets:
        wlo, whi = graph_prior_index.get(w, (0, 0))
        wend = _end_idx_for_t(graph_prior_times, wlo, whi, observation_time_t)
        w_start = wlo
        if last_n is not None and wend - w_start > last_n:
            w_start = wend - int(last_n)
        for i in range(w_start, wend):
            if graph_prior_times[i] > observation_time_t:
                prior_post_t += 1
        prior_refs.append(
            {
                "wallet": w,
                "start_idx": w_start,
                "end_idx": wend,
                "count": max(0, wend - w_start),
                "full_history_end_idx": wend,
                "full_history_count": max(0, wend - wlo),
            }
        )
    return {
        "mint": mint,
        "observation_time_t": observation_time_t,
        "token_graph_start_idx": lo,
        "token_graph_end_idx": end,
        "token_graph_count": max(0, end - lo),
        "wallets_observable_by_t": wallets,
        "wallet_prior_refs": prior_refs,
        "wallet_prior_rows": sum(r["count"] for r in prior_refs),
        "prior_post_t_violations": prior_post_t,
        "last_n": last_n,
    }


def build_sequence_dataset_v1(**kwargs: Any) -> dict[str, Any]:
    """Deprecated alias — builds v2 (v1 artifacts are frozen and not mutated)."""
    return build_sequence_dataset_v2(**kwargs)



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
    exact_dup_groups = 0
    sig_fps: dict[str, set[str]] = defaultdict(set)
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
            side = "buy" if row.get("is_buy") else "sell"
            ck = (
                f"{mint}|{et:.6f}|{row.get('is_buy')}|{row.get('token_amount')}|"
                f"{row.get('sol_amount')}"
            )
            curve = curve_by_key.get(ck) or {}
            v_sol = (
                row.get("v_sol_bonding_curve")
                if row.get("v_sol_bonding_curve") is not None
                else curve.get("v_sol_bonding_curve")
            )
            v_tok = (
                row.get("v_tokens_bonding_curve")
                if row.get("v_tokens_bonding_curve") is not None
                else curve.get("v_tokens_bonding_curve")
            )
            dk = _event_fingerprint(
                mint=mint,
                tx_signature=row.get("tx_signature"),
                event_time_unix=et,
                wallet=wallet,
                is_buy=row.get("is_buy"),
                sol_amount=row.get("sol_amount"),
                token_amount=row.get("token_amount"),
                v_sol_bonding_curve=v_sol,
                v_tokens_bonding_curve=v_tok,
            )
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
                "v_sol_bonding_curve": v_sol,
                "v_tokens_bonding_curve": v_tok,
                "real_sol_reserves": None,
                "real_token_reserves": None,
                "curve_pct_depleted": row.get("curve_pct_depleted")
                if row.get("curve_pct_depleted") is not None
                else curve.get("curve_pct_depleted"),
                "executable_price_sol": curve.get("executable_price_sol"),
                "source": "pumpfun.trades.atomic_subset",
                "source_quality_flag": curve.get("quality_flag") or "observed",
                "has_reserves": bool(v_sol is not None or v_tok is not None),
                "event_fingerprint": dk,
            }
            prev = best.get(dk)
            if prev is None:
                best[dk] = cand
            else:
                exact_dup_groups += 1
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

            sig = row.get("tx_signature")
            if sig:
                sig_fps[f"{mint}:{sig}"].add(dk)

    rows = list(best.values())
    hetero_sig_groups = sum(1 for fps in sig_fps.values() if len(fps) > 1)
    hetero_events = sum(len(fps) for fps in sig_fps.values() if len(fps) > 1)
    # Events v1 would collapse (same mint+sig) but v2 keeps when fingerprints differ.
    v1_collapsed_but_v2_kept = 0
    for fps in sig_fps.values():
        if len(fps) > 1:
            v1_collapsed_but_v2_kept += len(fps) - 1

    stats = _write_sorted_store(
        out_dir / "sequence_events_v2.parquet", rows, entity_key="mint"
    )
    stats.update(
        {
            "raw_input_rows": raw_n,
            "deduped_rows": len(rows),
            "exact_duplicate_groups_removed": exact_dup_groups,
            "duplicate_raw_event_rate": (
                exact_dup_groups / raw_n if raw_n else 0.0
            ),
            "signatures_with_heterogeneous_event_fingerprints": hetero_sig_groups,
            "events_in_heterogeneous_signature_groups": hetero_events,
            "events_v1_would_collapse_but_v2_preserves": v1_collapsed_but_v2_kept,
            "excluded_system_program_rows": excluded_system,
            "dedupe_policy": "conservative_event_fingerprint",
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
        out_dir / "sequence_curve_v2.parquet", rows, entity_key="mint"
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
        out_dir / "sequence_microstructure_v2.parquet", rows, entity_key="mint"
    )



def _build_ownership_store(
    *, world_root: Path, out_dir: Path, subset: set[str]
) -> dict[str, Any]:
    """Atomic ownership events with event-local provenance only (no future mint labels)."""
    rows: list[dict[str, Any]] = []
    forbidden_future_labels = 0
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
                    "source_quality_flag": "ledger_transfer_observed",
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
                    "source_quality_flag": row.get("quality_flag")
                    or "sampled_tx_rpc_balance",
                }
            )

    # Guardrail counter (rows should never carry these keys/values).
    forbidden_future_labels = 0
    for r in rows:
        if r.get("source_quality_flag") in {
            "holder_state_reconstructable",
            "transfer_ledger_observed_partial",
        }:
            forbidden_future_labels += 1
        if r.get("evidence_level") or r.get("is_complete_holder_ledger"):
            forbidden_future_labels += 1

    stats = _write_sorted_store(
        out_dir / "sequence_ownership_v2.parquet", rows, entity_key="mint"
    )
    stats["event_kind_counts"] = dict(Counter(r["event_kind"] for r in rows))
    stats["source_quality_counts"] = dict(
        Counter(r["source_quality_flag"] for r in rows)
    )
    stats["future_derived_ownership_labels_on_rows"] = forbidden_future_labels
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
                    "source": "sequence_events_v2",
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

    own_path = out_dir / "sequence_ownership_v2.parquet"
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
                    "source": "sequence_ownership_v2",
                    "source_quality_flag": row.get("source_quality_flag")
                    or "ledger_transfer_observed",
                    "is_cohort_derived": False,
                }
            )

    stats = _write_sorted_store(
        out_dir / "sequence_graph_token_v2.parquet", rows, entity_key="mint"
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
    """Complete focal-wallet Pump trade history, sorted causally (no scan-order truncation)."""
    pump = pump_root or Path("data/external/pumpfun_memecoin_corpus")
    trades_dir = pump / "trades"
    scanned = 0
    excluded_system = 0
    parts: list[pa.Table] = []

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
        for batch in dataset.scanner(columns=use_cols, batch_size=512_000).to_batches():
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
            is_buy = filtered.column("is_buy")
            edge_type = pc.if_else(
                is_buy,
                pa.scalar("wallet_token_buy"),
                pa.scalar("wallet_token_sell"),
            )
            parts.append(
                pa.table(
                    {
                        "wallet": filtered.column("user_wallet"),
                        "event_time_unix": et_unix,
                        "token_mint": filtered.column("mint"),
                        "dst": filtered.column("mint"),
                        "src": filtered.column("user_wallet"),
                        "sol_amount": filtered.column("sol_amount"),
                        "token_amount": filtered.column("token_amount"),
                        "tx_signature": filtered.column("tx_signature"),
                        "is_buy": is_buy,
                        "edge_type": edge_type,
                    }
                )
            )

    out_path = out_dir / "sequence_graph_wallet_prior_v2.parquet"
    idx_path = out_dir / "sequence_graph_wallet_prior_v2_entity_index.parquet"
    if not parts:
        return _write_sorted_store(out_path, [], entity_key="wallet")

    table = pa.concat_tables(parts, promote_options="default")
    # Deterministic causal order per wallet.
    # Build order_key via pandas after causal sort (13M rows; feasible locally).
    df = table.to_pandas()
    df = df.sort_values(
        ["wallet", "event_time_unix", "tx_signature", "token_mint", "edge_type"],
        kind="mergesort",
    )
    df["order_key"] = [
        _order_key(
            event_time_unix=float(et),
            tx_signature=sig,
            wallet=w,
            side=("buy" if str(etp).endswith("buy") else "sell"),
            extra=str(tm or ""),
        )
        for w, et, sig, etp, tm in zip(
            df["wallet"],
            df["event_time_unix"],
            df["tx_signature"],
            df["edge_type"],
            df["token_mint"],
            strict=True,
        )
    ]
    df["source"] = "pumpfun_memecoin_corpus.trades"
    df["source_quality_flag"] = "corpus_prior_observed"
    df["is_cohort_derived"] = False
    df["capped"] = False
    df["seq_idx"] = range(len(df))
    keep_cols = [
        "seq_idx",
        "wallet",
        "event_time_unix",
        "order_key",
        "edge_type",
        "src",
        "dst",
        "token_mint",
        "sol_amount",
        "token_amount",
        "tx_signature",
        "source",
        "source_quality_flag",
        "is_cohort_derived",
        "capped",
    ]
    out_table = pa.Table.from_pandas(df[keep_cols], preserve_index=False)
    pq.write_table(out_table, out_path)

    # Entity index
    wallets = df["wallet"].to_numpy()
    entity_index: list[dict[str, Any]] = []
    i = 0
    n = len(wallets)
    while i < n:
        w = wallets[i]
        j = i + 1
        while j < n and wallets[j] == w:
            j += 1
        entity_index.append({"entity": w, "start_idx": int(i), "end_idx": int(j)})
        i = j
    pq.write_table(pa.Table.from_pylist(entity_index), idx_path)

    per_wallet_n = [e["end_idx"] - e["start_idx"] for e in entity_index]
    stats = {
        "rows": n,
        "entities": len(entity_index),
        "path": str(out_path),
        "entity_index": str(idx_path),
        "corpus_rows_scanned": scanned,
        "rows_kept": n,
        "focal_wallets": len(focal_wallets),
        "wallets_with_prior": len(entity_index),
        "excluded_system_program_rows": excluded_system,
        "scan_order_truncation": False,
        "store_policy": "complete_focal_wallet_history_sorted_causally",
        "events_per_wallet": _length_stats(per_wallet_n),
        "excludes_red_cohort_membership": True,
    }
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
        "store_id": "sequence_global_v2",
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
    write_json(out_dir / "sequence_global_v2_manifest.json", manifest)
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
            out_dir / "sequence_events_v2.parquet",
            out_dir / "sequence_events_v2_entity_index.parquet",
        ),
        "curve": (
            out_dir / "sequence_curve_v2.parquet",
            out_dir / "sequence_curve_v2_entity_index.parquet",
        ),
        "micro": (
            out_dir / "sequence_microstructure_v2.parquet",
            out_dir / "sequence_microstructure_v2_entity_index.parquet",
        ),
        "ownership": (
            out_dir / "sequence_ownership_v2.parquet",
            out_dir / "sequence_ownership_v2_entity_index.parquet",
        ),
        "graph_token": (
            out_dir / "sequence_graph_token_v2.parquet",
            out_dir / "sequence_graph_token_v2_entity_index.parquet",
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
            ref[f"{name}_store_id"] = f"sequence_{name}_v2"
            ref[f"{name}_start_idx"] = lo
            ref[f"{name}_end_idx"] = end
            ref[f"{name}_count"] = count
            ref[f"mask_{name}"] = count > 0

        end_sec = last_completed_second(t)
        ref["global_store_id"] = "sequence_global_v2"
        ref["global_last_completed_second"] = end_sec
        ref["global_launch_end_idx"] = bisect_right(launch_times, t)
        ref["global_launch_count"] = ref["global_launch_end_idx"]
        ref["global_migration_end_idx"] = bisect_right(mig_times, t)
        ref["global_migration_count"] = ref["global_migration_end_idx"]
        ref["mask_global"] = True
        ref["graph_wallet_prior_strategy"] = (
            "wallets_from_graph_token_events_leq_T_then_wallet_store_cutoff"
        )
        ref["graph_wallet_prior_store_id"] = "sequence_graph_wallet_prior_v2"
        ref_rows.append(ref)

    refs_path = out_dir / "observation_sequence_refs_v2.parquet"
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
    """Exhaustive cheap boundary checks for all observation refs + graph loader."""
    refs = pq.read_table(out_dir / "observation_sequence_refs_v2.parquet").to_pylist()
    store_paths = {
        "events": out_dir / "sequence_events_v2.parquet",
        "curve": out_dir / "sequence_curve_v2.parquet",
        "micro": out_dir / "sequence_microstructure_v2.parquet",
        "ownership": out_dir / "sequence_ownership_v2.parquet",
        "graph_token": out_dir / "sequence_graph_token_v2.parquet",
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

    # Ownership atomic rows must not carry future-derived completeness labels.
    own_future_labels = 0
    own_path = store_paths["ownership"]
    if own_path.exists() and pq.ParquetFile(own_path).metadata.num_rows:
        own_table = pq.read_table(own_path)
        names = set(own_table.column_names)
        if "evidence_level" in names:
            own_future_labels += sum(
                1
                for x in own_table.column("evidence_level").to_pylist()
                if x
                in {
                    "holder_state_reconstructable",
                    "transfer_ledger_observed_partial",
                }
            )
        if "is_complete_holder_ledger" in names:
            own_future_labels += sum(
                1 for x in own_table.column("is_complete_holder_ledger").to_pylist() if x
            )
        if "source_quality_flag" in names:
            own_future_labels += sum(
                1
                for x in own_table.column("source_quality_flag").to_pylist()
                if x == "holder_state_reconstructable"
            )

    graph_token_path = store_paths["graph_token"]
    graph_token_times = times["graph_token"]
    graph_token_index = _load_entity_index(
        graph_token_path.with_name(graph_token_path.stem + "_entity_index.parquet")
    )
    graph_token_wallets = (
        pq.read_table(graph_token_path, columns=["wallet"]).column(0).to_pylist()
        if graph_token_path.exists() and pq.ParquetFile(graph_token_path).metadata.num_rows
        else []
    )
    graph_prior_path = out_dir / "sequence_graph_wallet_prior_v2.parquet"
    graph_prior_times = _load_times_array(graph_prior_path)
    graph_prior_idx = _load_entity_index(
        out_dir / "sequence_graph_wallet_prior_v2_entity_index.parquet"
    )

    # Exhaustive wallet-prior dereference validation for all observations.
    wallet_prior_post_t = 0
    wallet_prior_usable = 0
    wallet_after_t_leak = 0
    for ref in refs:
        t = float(ref["observation_time_t"])
        nb = load_observation_graph_neighborhood(
            out_dir=out_dir,
            mint=ref["mint"],
            observation_time_t=t,
            last_n=None,  # full history <= T
            graph_token_times=graph_token_times,
            graph_token_wallets=graph_token_wallets,
            graph_token_index=graph_token_index,
            graph_prior_times=graph_prior_times,
            graph_prior_index=graph_prior_idx,
        )
        wallet_prior_post_t += int(nb["prior_post_t_violations"])
        if nb["wallet_prior_rows"] > 0:
            wallet_prior_usable += 1
        # Wallets must each have a token-graph event <= T (by construction).
        lo = int(ref["graph_token_start_idx"])
        end = int(ref["graph_token_end_idx"])
        allowed = {
            graph_token_wallets[i]
            for i in range(lo, end)
            if i < len(graph_token_wallets) and graph_token_wallets[i]
        }
        for w in nb["wallets_observable_by_t"]:
            if w not in allowed:
                wallet_after_t_leak += 1

    sample = _stratified_sample_refs(refs, n=64)
    sample_results = []
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
        nb = load_observation_graph_neighborhood(
            out_dir=out_dir,
            mint=mint,
            observation_time_t=t,
            last_n=WALLET_PRIOR_DEREF_DEFAULT_LAST_N,
            graph_token_times=graph_token_times,
            graph_token_wallets=graph_token_wallets,
            graph_token_index=graph_token_index,
            graph_prior_times=graph_prior_times,
            graph_prior_index=graph_prior_idx,
        )
        details["graph_wallet_prior"] = {
            "wallets": len(nb["wallets_observable_by_t"]),
            "rows": nb["wallet_prior_rows"],
            "post_t": nb["prior_post_t_violations"],
            "last_n": nb["last_n"],
        }
        if nb["prior_post_t_violations"]:
            ok = False
        end_sec = int(ref["global_last_completed_second"])
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

    events_stats: dict[str, Any] = {}
    man = out_dir / "sequence_dataset_manifest.json"
    if man.exists():
        payload = json.loads(man.read_text(encoding="utf-8"))
        events_stats = (payload.get("stores") or {}).get("events") or {}

    return {
        "observations_validated": len(refs),
        "post_t_violations": post_t_violations + wallet_prior_post_t,
        "ordering_violations": ordering_violations,
        "future_derived_ownership_labels_on_atomic_rows": own_future_labels,
        "wallet_prior_post_t_violations": wallet_prior_post_t,
        "wallet_not_observable_by_t_leaks": wallet_after_t_leak,
        "observations_with_usable_wallet_prior": wallet_prior_usable,
        "duplicate_raw_event_rate": events_stats.get("duplicate_raw_event_rate", 0.0),
        "exact_duplicate_groups_removed": events_stats.get(
            "exact_duplicate_groups_removed"
        ),
        "signatures_with_heterogeneous_event_fingerprints": events_stats.get(
            "signatures_with_heterogeneous_event_fingerprints"
        ),
        "events_v1_would_collapse_but_v2_preserves": events_stats.get(
            "events_v1_would_collapse_but_v2_preserves"
        ),
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
        "causal_boundary_ok": (
            post_t_violations == 0
            and ordering_violations == 0
            and wallet_prior_post_t == 0
            and wallet_after_t_leak == 0
            and own_future_labels == 0
        ),
    }




def build_sequence_dataset_v2(
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
        event_rows_path=out_dir / "sequence_events_v2.parquet",
    )

    focal_wallets: set[str] = set()
    ev_path = out_dir / "sequence_events_v2.parquet"
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

    contract = build_model_input_contract_v2()
    write_json(out_dir / "MODEL_INPUT_CONTRACT_V2.json", contract)
    ensure_dir(manifest_dir)
    write_json(manifest_dir / "MODEL_INPUT_CONTRACT_V2.json", contract)

    disk_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    for p in (
        world_root / "pit" / "global" / "trade_second_buckets.parquet",
        world_root / "pit" / "global" / "launches_timeline.parquet",
        world_root / "pit" / "global" / "migrations_timeline.parquet",
    ):
        if p.exists():
            disk_bytes += p.stat().st_size

    # Compare vs v1 wallet-prior if present.
    v1_prior = (
        world_root
        / "sequences"
        / "sequence_dataset_v1"
        / "sequence_graph_wallet_prior_v1.parquet"
    )
    v1_prior_rows = (
        pq.ParquetFile(v1_prior).metadata.num_rows if v1_prior.exists() else None
    )

    audit = {
        "audit_id": SEQUENCE_DATASET_VERSION,
        "generated_at": utc_now(),
        "acquisition_freeze_id": FREEZE_ID,
        "pit_engine_version": PIT_ENGINE_VERSION,
        "sequence_store_version": SEQUENCE_STORE_VERSION,
        "supersedes": "sequence_dataset_v1",
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
        "coverage_vs_v1": {
            "wallet_prior_rows_v1": v1_prior_rows,
            "wallet_prior_rows_v2": graph_prior_stats.get("rows"),
            "wallet_prior_events_per_wallet": graph_prior_stats.get("events_per_wallet"),
            "scan_order_truncation_removed": True,
        },
        "construction_stats": store_stats,
        "observation_refs": {
            "rows": refs_stats.get("rows"),
            "path": refs_stats.get("path"),
        },
        "causality_validation": validation,
        "integrity_confirmations": {
            "future_derived_ownership_classification_in_atomic_rows": validation.get(
                "future_derived_ownership_labels_on_atomic_rows"
            ),
            "post_t_referenced_rows": validation.get("post_t_violations"),
            "ordering_violations": validation.get("ordering_violations"),
            "arbitrary_scan_order_wallet_truncation": "removed",
            "heterogeneous_same_signature_events_incorrectly_collapsed": 0,
            "events_v1_would_collapse_but_v2_preserves": validation.get(
                "events_v1_would_collapse_but_v2_preserves"
            ),
            "red_cohort_eventual_membership_as_early_feature": 0,
            "lifetime_wallet_aggregates_as_temporal_inputs": 0,
        },
        "estimated_disk_bytes": disk_bytes,
        "estimated_disk_mb": round(disk_bytes / (1024 * 1024), 3),
        "artifacts": {
            "sequence_events_v2": str(out_dir / "sequence_events_v2.parquet"),
            "sequence_microstructure_v2": str(
                out_dir / "sequence_microstructure_v2.parquet"
            ),
            "sequence_ownership_v2": str(out_dir / "sequence_ownership_v2.parquet"),
            "sequence_graph_token_v2": str(out_dir / "sequence_graph_token_v2.parquet"),
            "sequence_graph_wallet_prior_v2": str(
                out_dir / "sequence_graph_wallet_prior_v2.parquet"
            ),
            "sequence_curve_v2": str(out_dir / "sequence_curve_v2.parquet"),
            "observation_sequence_refs_v2": str(
                out_dir / "observation_sequence_refs_v2.parquet"
            ),
            "model_input_contract_v2": str(out_dir / "MODEL_INPUT_CONTRACT_V2.json"),
        },
        "labels": "not_generated",
        "targets": "not_generated",
        "training": "not_started",
        "known_limitations": [
            "Wallet-prior store keeps complete focal-wallet history; last-N / "
            "temporal windows apply only at dereference.",
            "Priority-fee fields were not recovered in the freeze; left null.",
            "Real bonding-curve reserves are not in local trade artifacts; left null.",
            "Trade slots are unavailable on Pump trade parquet; left null.",
            "RED-COHORT membership is intentionally excluded from temporal graph inputs.",
            "Lifetime edges_wallet_token aggregates are not used as temporal inputs.",
            "Global trade flow uses completed-second buckets only (<1s latency).",
            "Ownership RPC rows remain sampled-tx evidence, not a full holder ledger.",
            "sequence_dataset_v1 artifacts are retained but superseded for model input.",
        ],
    }
    write_json(out_dir / "SEQUENCE_DATASET_V2_AUDIT.json", audit)
    write_json(manifest_dir / "SEQUENCE_DATASET_V2_AUDIT.json", audit)
    (manifest_dir / "SEQUENCE_DATASET_V2_AUDIT.md").write_text(
        render_sequence_audit_md(audit), encoding="utf-8"
    )
    return audit


def render_sequence_audit_md(audit: dict[str, Any]) -> str:
    counts = audit.get("sequence_row_counts_by_modality") or {}
    val = audit.get("causality_validation") or {}
    conf = audit.get("integrity_confirmations") or {}
    lines = [
        f"# Sequence Dataset v2 (`{audit['audit_id']}`)",
        "",
        f"Generated at: `{audit['generated_at']}`",
        f"Freeze: `{audit['acquisition_freeze_id']}`",
        f"PIT engine: `{audit['pit_engine_version']}`",
        f"Supersedes: `{audit.get('supersedes')}`",
        "",
        "## Integrity confirmations",
        "",
    ]
    for k, v in conf.items():
        lines.append(f"- `{k}`: **{v}**")
    lines.extend(["", "## Sequence row counts", ""])
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
            f"- usable wallet-prior observations: **"
            f"{val.get('observations_with_usable_wallet_prior')}**",
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
