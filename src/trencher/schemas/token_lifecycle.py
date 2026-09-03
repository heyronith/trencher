"""Token lifecycle: Pump creation, migration, graduation events."""

from __future__ import annotations

import pyarrow as pa

from trencher.schemas._common import ORDERING_FIELDS, SCHEMA_VERSION, with_provenance

SCHEMA_VERSION = SCHEMA_VERSION

TOKEN_LIFECYCLE_SCHEMA = with_provenance(
    ORDERING_FIELDS
    + [
        ("event_type", pa.string()),
        ("mint", pa.string()),
        ("creator_wallet", pa.string()),
        ("name", pa.string()),
        ("symbol", pa.string()),
        ("uri", pa.string()),
        ("bonding_curve", pa.string()),
        ("associated_bonding_curve", pa.string()),
        ("migrated", pa.bool_()),
        ("migration_tx_id", pa.string()),
        ("call_tx_signer", pa.string()),
        ("call_tx_success", pa.bool_()),
        ("call_outer_executing_account", pa.string()),
        ("call_inner_executing_account", pa.string()),
    ]
)
