"""Trades and protocol events: bonding-curve, PumpSwap, and Solana DEX."""

from __future__ import annotations

import pyarrow as pa

from trencher.schemas._common import ORDERING_FIELDS, SCHEMA_VERSION, with_provenance

SCHEMA_VERSION = SCHEMA_VERSION

TRADES_EVENTS_SCHEMA = with_provenance(
    ORDERING_FIELDS
    + [
        ("venue", pa.string()),
        ("event_type", pa.string()),
        ("instruction_name", pa.string()),
        ("instruction_discriminator", pa.string()),
        ("side", pa.string()),
        ("mint", pa.string()),
        ("base_mint", pa.string()),
        ("quote_mint", pa.string()),
        ("pool", pa.string()),
        ("trader_wallet", pa.string()),
        ("creator_wallet", pa.string()),
        ("token_program", pa.string()),
        ("is_token_2022", pa.bool_()),
        ("base_amount", pa.decimal128(38, 0)),
        ("quote_amount", pa.decimal128(38, 0)),
        ("sol_amount", pa.decimal128(38, 0)),
        ("token_amount", pa.decimal128(38, 0)),
        ("is_buy", pa.bool_()),
        ("virtual_base_reserves", pa.decimal128(38, 0)),
        ("virtual_quote_reserves", pa.decimal128(38, 0)),
        ("real_base_reserves", pa.decimal128(38, 0)),
        ("real_quote_reserves", pa.decimal128(38, 0)),
        ("virtual_sol_reserves", pa.decimal128(38, 0)),
        ("virtual_token_reserves", pa.decimal128(38, 0)),
        ("real_sol_reserves", pa.decimal128(38, 0)),
        ("real_token_reserves", pa.decimal128(38, 0)),
        ("lp_fee", pa.decimal128(38, 0)),
        ("protocol_fee", pa.decimal128(38, 0)),
        ("coin_creator_fee", pa.decimal128(38, 0)),
        ("fee", pa.decimal128(38, 0)),
        ("fee_basis_points", pa.int64()),
        ("event_timestamp", pa.int64()),
        ("project", pa.string()),
        ("project_version", pa.string()),
        ("project_version_name", pa.string()),
        ("trade_source", pa.string()),
        ("project_program_id", pa.string()),
        ("project_main_id", pa.string()),
        ("token_bought_mint", pa.string()),
        ("token_sold_mint", pa.string()),
        ("token_bought_amount_raw", pa.decimal128(38, 0)),
        ("token_sold_amount_raw", pa.decimal128(38, 0)),
        ("token_bought_vault", pa.string()),
        ("token_sold_vault", pa.string()),
        ("token_pair", pa.string()),
        ("amount_usd", pa.float64()),
        ("fee_usd", pa.float64()),
        ("fee_tier", pa.float64()),
    ]
)
