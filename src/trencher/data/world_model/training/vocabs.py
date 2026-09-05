"""Deterministic category ID vocabs for predictive tensors.

UNK=0 for all unknown values. Never use Python ``hash()`` (PYTHONHASHSEED-dependent).
"""

from __future__ import annotations

from typing import Any

UNK = 0

SIDE_TO_ID: dict[str, int] = {
    "": UNK,
    "unknown": UNK,
    "buy": 1,
    "sell": 2,
}

# Exact frozen graph edge types observed in sequence_graph_*_v2
EDGE_TYPE_VOCAB: dict[str, int] = {
    "": UNK,
    "unknown": UNK,
    "creator_token": 1,
    "wallet_token_buy": 2,
    "wallet_token_sell": 3,
    "spl_transfer_ownership": 4,
    "native_sol_funding": 5,
}

# Frozen ownership event kinds
EVENT_KIND_VOCAB: dict[str, int] = {
    "": UNK,
    "unknown": UNK,
    "rpc_pre_post_token_balance": 1,
    "bq_spl_transfer": 2,
}

TRANSFER_TYPE_VOCAB: dict[str, int] = {
    "": UNK,
    "unknown": UNK,
    "spl-transfer": 1,
    "burn": 2,
    "mintTo": 3,
}

SOURCE_QUALITY_VOCAB: dict[str, int] = {
    "": UNK,
    "unknown": UNK,
    "observed": 1,
    "ok": 1,
    "high": 1,
    "good": 1,
    "derived": 2,
    "low": 3,
    "bad": 3,
    "missing": UNK,
}


def _norm_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # numeric quality flags: map coarsely
        if value == 0:
            return "0"
        return "1"
    return str(value).strip()


def lookup_id(vocab: dict[str, int], value: Any) -> float:
    key = _norm_key(value)
    if key in vocab:
        return float(vocab[key])
    # case-insensitive fallback for known keys
    low = key.lower()
    for k, v in vocab.items():
        if k.lower() == low:
            return float(v)
    return float(UNK)


def side_id(side: Any, is_buy: Any = None) -> float:
    if side is None or side == "":
        if is_buy is True:
            return float(SIDE_TO_ID["buy"])
        if is_buy is False:
            return float(SIDE_TO_ID["sell"])
        return float(UNK)
    return lookup_id(SIDE_TO_ID, side)


def edge_type_id(edge_type: Any) -> float:
    return lookup_id(EDGE_TYPE_VOCAB, edge_type)


def event_kind_id(kind: Any) -> float:
    return lookup_id(EVENT_KIND_VOCAB, kind)


def transfer_type_id(t: Any) -> float:
    return lookup_id(TRANSFER_TYPE_VOCAB, t)


def source_quality_id(q: Any) -> float:
    return lookup_id(SOURCE_QUALITY_VOCAB, q)


def vocab_manifest() -> dict[str, Any]:
    return {
        "unk": UNK,
        "hash_policy": "never_python_hash",
        "unknown_policy": "UNK=0",
        "side": SIDE_TO_ID,
        "edge_type": EDGE_TYPE_VOCAB,
        "event_kind": EVENT_KIND_VOCAB,
        "transfer_type": TRANSFER_TYPE_VOCAB,
        "source_quality": SOURCE_QUALITY_VOCAB,
    }
