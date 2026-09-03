"""Wallet SOL funding and native transfer history."""

from __future__ import annotations

import pyarrow as pa

from trencher.schemas._common import ORDERING_FIELDS, SCHEMA_VERSION, with_provenance

SCHEMA_VERSION = SCHEMA_VERSION

WALLET_FUNDING_SCHEMA = with_provenance(
    ORDERING_FIELDS
    + [
        ("action", pa.string()),
        ("from_owner", pa.string()),
        ("to_owner", pa.string()),
        ("from_token_account", pa.string()),
        ("to_token_account", pa.string()),
        ("amount_lamports", pa.decimal128(38, 0)),
        ("amount_sol", pa.float64()),
        ("from_is_participant", pa.bool_()),
        ("to_is_participant", pa.bool_()),
        ("tx_signer", pa.string()),
        ("outer_executing_account", pa.string()),
    ]
)
