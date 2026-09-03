"""Token transfers and ownership-changing SPL events."""

from __future__ import annotations

import pyarrow as pa

from trencher.schemas._common import ORDERING_FIELDS, SCHEMA_VERSION, with_provenance

SCHEMA_VERSION = SCHEMA_VERSION

TOKEN_TRANSFERS_SCHEMA = with_provenance(
    ORDERING_FIELDS
    + [
        ("action", pa.string()),
        ("mint", pa.string()),
        ("symbol", pa.string()),
        ("token_version", pa.string()),
        ("is_token_2022", pa.bool_()),
        ("from_token_account", pa.string()),
        ("to_token_account", pa.string()),
        ("from_owner", pa.string()),
        ("to_owner", pa.string()),
        ("amount_raw", pa.decimal128(38, 0)),
        ("amount_display", pa.float64()),
        ("amount_usd", pa.float64()),
        ("tx_signer", pa.string()),
        ("outer_executing_account", pa.string()),
        ("unique_instruction_key", pa.string()),
    ]
)
