"""Common schema field helpers."""

from __future__ import annotations

import pyarrow as pa

SCHEMA_VERSION = "1.0.0"

# Shared provenance / ordering columns appended to every raw dataset.
PROVENANCE_FIELDS = [
    ("source_table", pa.string()),
    ("source_partition_col", pa.string()),
    ("protocol", pa.string()),
    ("protocol_version", pa.string()),
    ("schema_version", pa.string()),
    ("raw_payload_json", pa.string()),
]

ORDERING_FIELDS = [
    ("block_time", pa.timestamp("us", tz="UTC")),
    ("block_slot", pa.int64()),
    ("block_date", pa.date32()),
    ("tx_index", pa.int32()),
    ("tx_id", pa.string()),
    ("outer_instruction_index", pa.int32()),
    ("inner_instruction_index", pa.int32()),
    ("event_index", pa.int32()),
]


def with_provenance(fields: list[tuple[str, pa.DataType]]) -> pa.Schema:
    names = {name for name, _ in fields}
    extra = [f for f in PROVENANCE_FIELDS if f[0] not in names]
    return pa.schema(list(fields) + extra)
