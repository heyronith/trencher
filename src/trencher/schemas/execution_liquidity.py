"""Execution and liquidity state (PumpSwap pools, reserves)."""

from __future__ import annotations

import pyarrow as pa

from trencher.schemas._common import ORDERING_FIELDS, SCHEMA_VERSION, with_provenance

SCHEMA_VERSION = SCHEMA_VERSION

EXECUTION_LIQUIDITY_SCHEMA = with_provenance(
    ORDERING_FIELDS
    + [
        ("event_type", pa.string()),
        ("pool", pa.string()),
        ("mint", pa.string()),
        ("quote_mint", pa.string()),
        ("creator_wallet", pa.string()),
        ("base_mint", pa.string()),
        ("lp_mint", pa.string()),
        ("base_amount", pa.decimal128(38, 0)),
        ("quote_amount", pa.decimal128(38, 0)),
        ("pool_base_token_reserves", pa.decimal128(38, 0)),
        ("pool_quote_token_reserves", pa.decimal128(38, 0)),
        ("initial_virtual_base_reserves", pa.decimal128(38, 0)),
        ("initial_virtual_quote_reserves", pa.decimal128(38, 0)),
    ]
)
