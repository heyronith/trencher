"""Raw transaction / instruction microstructure."""

from __future__ import annotations

import pyarrow as pa

from trencher.schemas._common import ORDERING_FIELDS, SCHEMA_VERSION, with_provenance

SCHEMA_VERSION = SCHEMA_VERSION

TRANSACTION_MICROSTRUCTURE_SCHEMA = with_provenance(
    ORDERING_FIELDS
    + [
        ("tx_signer", pa.string()),
        ("tx_success", pa.bool_()),
        ("executing_account", pa.string()),
        ("outer_executing_account", pa.string()),
        ("inner_executing_account", pa.string()),
        ("is_inner", pa.bool_()),
        ("stack_height", pa.int64()),
        ("instruction_data_hex", pa.string()),
        ("account_arguments_json", pa.string()),
        ("fee_lamports", pa.int64()),
        ("compute_units_consumed", pa.int64()),
        ("cost_units", pa.int64()),
        ("log_messages_json", pa.string()),
        ("pre_balances_json", pa.string()),
        ("post_balances_json", pa.string()),
        ("pre_token_balances_json", pa.string()),
        ("post_token_balances_json", pa.string()),
        ("error_message", pa.string()),
    ]
)
