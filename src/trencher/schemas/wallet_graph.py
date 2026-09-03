"""Wallet graph edges derived from on-chain transfers."""

from __future__ import annotations

import pyarrow as pa

from trencher.schemas._common import ORDERING_FIELDS, SCHEMA_VERSION, with_provenance

SCHEMA_VERSION = SCHEMA_VERSION

WALLET_GRAPH_SCHEMA = with_provenance(
    ORDERING_FIELDS
    + [
        ("edge_type", pa.string()),
        ("from_wallet", pa.string()),
        ("to_wallet", pa.string()),
        ("mint", pa.string()),
        ("amount_raw", pa.decimal128(38, 0)),
        ("amount_sol", pa.float64()),
        ("from_is_participant", pa.bool_()),
        ("to_is_participant", pa.bool_()),
        ("tx_signer", pa.string()),
        ("outer_executing_account", pa.string()),
    ]
)
