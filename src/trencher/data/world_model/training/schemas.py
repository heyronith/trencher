"""Modality channel schemas for production training tensors.

Continuous channels may receive train-only clip→zscore.
Binary / categorical / graph-index channels must NOT be z-scored as continuous.
"""

from __future__ import annotations

from typing import Any

from trencher.data.world_model.training.vocabs import (
    vocab_manifest,
)

# --- Broad / rich event sequences ---
# Layout: [rel_dt, interarrival, is_buy, side_id, sol, token, price, v_sol, v_tok,
#          curve_pct, has_reserves, has_finite_price, source_quality]
EVENT_CONTINUOUS = (
    "rel_dt",
    "interarrival",
    "sol_amount",
    "token_amount",
    "price_sol",
    "v_sol",
    "v_tok",
    "curve_pct_depleted",
)
EVENT_BINARY = ("is_buy", "has_reserves", "has_finite_price")
EVENT_CATEGORICAL = ("side_id", "source_quality")
EVENT_CHANNELS: tuple[str, ...] = (
    "rel_dt",
    "interarrival",
    "is_buy",
    "side_id",
    "sol_amount",
    "token_amount",
    "price_sol",
    "v_sol",
    "v_tok",
    "curve_pct_depleted",
    "has_reserves",
    "has_finite_price",
    "source_quality",
)

CURVE_CHANNELS: tuple[str, ...] = (
    "rel_dt",
    "price_sol",
    "v_sol",
    "v_tok",
    "curve_pct_depleted",
    "has_reserves",
    "source_quality",
)
CURVE_CONTINUOUS = ("rel_dt", "price_sol", "v_sol", "v_tok", "curve_pct_depleted")
CURVE_BINARY = ("has_reserves",)
CURVE_CATEGORICAL = ("source_quality",)

CREATOR_CHANNELS: tuple[str, ...] = (
    "rel_dt",
    "prior_rank",
    "prior_is_mayhem_mode",
    "source_quality",
)
CREATOR_CONTINUOUS = ("rel_dt", "prior_rank")
CREATOR_BINARY = ("prior_is_mayhem_mode",)

# Microstructure
MICRO_CHANNELS: tuple[str, ...] = (
    "rel_dt",
    "success",
    "fee_lamports",
    "compute_units_consumed",
    "priority_fee_lamports",
    "account_key_count",
    "balance_change_count",
    "log_message_count",
    "inner_instruction_groups",
    "availability_priority_fee",
    "source_quality",
)
MICRO_CONTINUOUS = (
    "rel_dt",
    "fee_lamports",
    "compute_units_consumed",
    "priority_fee_lamports",
    "account_key_count",
    "balance_change_count",
    "log_message_count",
    "inner_instruction_groups",
)
MICRO_BINARY = ("success", "availability_priority_fee")

# Ownership (event-local; no invented holder state)
OWN_CHANNELS: tuple[str, ...] = (
    "rel_dt",
    "event_kind_id",
    "transfer_type_id",
    "value",
    "decimals",
    "pre_ui_amount",
    "post_ui_amount",
    "delta_ui",
    "source_quality",
)
OWN_CONTINUOUS = (
    "rel_dt",
    "value",
    "decimals",
    "pre_ui_amount",
    "post_ui_amount",
    "delta_ui",
)
OWN_CATEGORICAL = ("event_kind_id", "transfer_type_id", "source_quality")

# Token graph edges (IDs live in separate index tensors)
GRAPH_CHANNELS: tuple[str, ...] = (
    "rel_dt",
    "edge_type_id",
    "direction",  # +1 src→dst focal-out, -1 inbound, 0 unknown
    "sol_amount",
    "token_amount",
    "is_cohort_derived",
    "source_quality",
)
GRAPH_CONTINUOUS = ("rel_dt", "sol_amount", "token_amount")
GRAPH_BINARY = ("is_cohort_derived",)
GRAPH_CATEGORICAL = ("edge_type_id", "direction", "source_quality")

# Wallet prior (same edge attrs; node indices separate)
WALLET_PRIOR_CHANNELS: tuple[str, ...] = GRAPH_CHANNELS

# Global last-N second buckets + daily context scalars elsewhere
GLOBAL_BUCKET_CHANNELS: tuple[str, ...] = (
    "rel_dt",
    "trades",
    "buys",
    "sells",
    "buy_sol",
    "sell_sol",
    "unique_mints",
    "unique_wallets",
)
GLOBAL_CONTINUOUS = tuple(c for c in GLOBAL_BUCKET_CHANNELS if c != "rel_dt")
GLOBAL_SCALAR_CONTINUOUS = (
    "global_launches_1h",
    "global_launches_24h",
    "global_migrations_1h",
    "global_migrations_24h",
    "daily_network_launches",
    "daily_network_graduations",
    "daily_network_trades",
    "daily_network_buy_volume_sol",
    "daily_network_graduation_rate",
)

SIDE_TO_ID = {"buy": 1, "sell": 2, "unknown": 0, "": 0}


def continuous_channel_names(*, include_relative_time: bool = False) -> list[str]:
    """Unbounded continuous channels that receive clip→zscore by default."""
    names: list[str] = []
    for group in (
        EVENT_CONTINUOUS,
        CURVE_CONTINUOUS,
        CREATOR_CONTINUOUS,
        MICRO_CONTINUOUS,
        OWN_CONTINUOUS,
        GRAPH_CONTINUOUS,
        GLOBAL_CONTINUOUS,
        GLOBAL_SCALAR_CONTINUOUS,
    ):
        for n in group:
            if n in ("rel_dt", "interarrival") and not include_relative_time:
                continue
            if n == "decimals":
                continue  # bounded ordinal — explicit exemption
            if n not in names:
                names.append(n)
    return names


# Relative time: documented exemption (remain raw relative / optional log1p upstream)
RELATIVE_TIME_EXEMPT = ("rel_dt", "interarrival")
BOUNDED_ORDINAL_EXEMPT = ("decimals",)
NEVER_NORMALIZE = (
    "masks",
    "binary_flags",
    "categorical_ids",
    "graph_node_ids",
    *RELATIVE_TIME_EXEMPT,
    *BOUNDED_ORDINAL_EXEMPT,
)


def modality_schema_manifest() -> dict[str, Any]:
    return {
        "events": {
            "channels": list(EVENT_CHANNELS),
            "continuous": list(EVENT_CONTINUOUS),
            "binary": list(EVENT_BINARY),
            "categorical": list(EVENT_CATEGORICAL),
        },
        "curve": {
            "channels": list(CURVE_CHANNELS),
            "continuous": list(CURVE_CONTINUOUS),
            "binary": list(CURVE_BINARY),
            "categorical": list(CURVE_CATEGORICAL),
        },
        "creator": {
            "channels": list(CREATOR_CHANNELS),
            "continuous": list(CREATOR_CONTINUOUS),
            "binary": list(CREATOR_BINARY),
        },
        "micro": {
            "channels": list(MICRO_CHANNELS),
            "continuous": list(MICRO_CONTINUOUS),
            "binary": list(MICRO_BINARY),
        },
        "ownership": {
            "channels": list(OWN_CHANNELS),
            "continuous": list(OWN_CONTINUOUS),
            "categorical": list(OWN_CATEGORICAL),
        },
        "graph_token": {
            "channels": list(GRAPH_CHANNELS),
            "continuous": list(GRAPH_CONTINUOUS),
            "binary": list(GRAPH_BINARY),
            "categorical": list(GRAPH_CATEGORICAL),
            "node_ids": "graph_src_idx/graph_dst_idx indexing only",
        },
        "wallet_prior": {
            "channels": list(WALLET_PRIOR_CHANNELS),
            "node_ids": "wallet_node_idx indexing only",
        },
        "global": {
            "channels": list(GLOBAL_BUCKET_CHANNELS),
            "policy": "last_completed_second_only",
            "daily_policy": "last_fully_completed_utc_day_before_T",
        },
        "vocabs": vocab_manifest(),
        "transform_exemptions": {
            "relative_time_raw": list(RELATIVE_TIME_EXEMPT),
            "bounded_ordinals": list(BOUNDED_ORDINAL_EXEMPT),
            "never_normalize": list(NEVER_NORMALIZE),
        },
    }
