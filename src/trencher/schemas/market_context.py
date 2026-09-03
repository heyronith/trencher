"""Block and network context (fees, throughput)."""

from __future__ import annotations

import pyarrow as pa

from trencher.schemas._common import SCHEMA_VERSION, with_provenance

SCHEMA_VERSION = SCHEMA_VERSION

MARKET_CONTEXT_SCHEMA = with_provenance(
    [
        ("block_time", pa.timestamp("us", tz="UTC")),
        ("block_slot", pa.int64()),
        ("block_date", pa.date32()),
        ("block_height", pa.int64()),
        ("block_hash", pa.string()),
        ("total_transactions", pa.int32()),
        ("successful_transactions", pa.int32()),
        ("failed_transactions", pa.int32()),
        ("total_non_vote_transactions", pa.int32()),
        ("successful_non_vote_transactions", pa.int32()),
        ("failed_non_vote_transactions", pa.int32()),
        ("parent_slot", pa.int64()),
        ("previous_block_hash", pa.string()),
    ]
)
