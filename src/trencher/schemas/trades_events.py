"""Trades and protocol events: bonding-curve and PumpSwap."""

from __future__ import annotations

import pyarrow as pa

from trencher.schemas._common import ORDERING_FIELDS, SCHEMA_VERSION, with_provenance

SCHEMA_VERSION = SCHEMA_VERSION

TRADES_EVENTS_SCHEMA = with_provenance(
    ORDERING_FIELDS
    + [
        ("venue", pa.string()),
        ("event_type", pa.string()),
        ("side", pa.string()),
        ("mint", pa.string()),
        ("pool", pa.string()),
        ("trader_wallet", pa.string()),
        ("creator_wallet", pa.string()),
        ("quote_mint", pa.string()),
        ("base_amount", pa.decimal128(38, 0)),
        ("quote_amount", pa.decimal128(38, 0)),
        ("sol_amount", pa.decimal128(38, 0)),
        ("token_amount", pa.decimal128(38, 0)),
        ("is_buy", pa.bool_()),
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
    ]
)
