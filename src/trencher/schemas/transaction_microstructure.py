"""Raw transaction / instruction microstructure."""

from __future__ import annotations

import pyarrow as pa

from trencher.schemas._common import ORDERING_FIELDS, SCHEMA_VERSION, with_provenance

SCHEMA_VERSION = SCHEMA_VERSION

TRANSACTION_MICROSTRUCTURE_SCHEMA = with_provenance(
    ORDERING_FIELDS
    + [
        ("record_type", pa.string()),
        ("instruction_name", pa.string()),
        ("instruction_discriminator", pa.string()),
        ("tx_signer", pa.string()),
        ("tx_success", pa.bool_()),
        ("tx_version", pa.string()),
        ("block_hash", pa.string()),
        ("recent_block_hash", pa.string()),
        ("executing_account", pa.string()),
        ("outer_executing_account", pa.string()),
        ("inner_executing_account", pa.string()),
        ("is_inner", pa.bool_()),
        ("stack_height", pa.int64()),
        ("instruction_data_hex", pa.string()),
        ("account_arguments_json", pa.string()),
        ("inner_instructions_json", pa.string()),
        ("instructions_json", pa.string()),
        ("account_keys_json", pa.string()),
        ("signatures_json", pa.string()),
        ("signers_json", pa.string()),
        ("address_table_lookups_json", pa.string()),
        ("loaded_addresses_json", pa.string()),
        ("return_data_program_id", pa.string()),
        ("return_data_json", pa.string()),
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
