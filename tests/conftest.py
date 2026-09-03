"""Shared test fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from trencher.schemas.registry import validate_dataframe


@pytest.fixture
def sample_token_lifecycle_df() -> pd.DataFrame:
    row = {
        "block_time": datetime(2025, 1, 1, 12, 0, tzinfo=UTC),
        "block_slot": 100,
        "block_date": datetime(2025, 1, 1).date(),
        "tx_index": 1,
        "tx_id": "sig123",
        "outer_instruction_index": 0,
        "inner_instruction_index": None,
        "event_index": None,
        "event_type": "create_call",
        "instruction_name": "create",
        "instruction_discriminator": "181ec828051c0777",
        "mint": "mint1",
        "creator_wallet": "creator1",
        "user_wallet": "user1",
        "name": "Test",
        "symbol": "TST",
        "uri": "https://example.com",
        "bonding_curve": "bc1",
        "associated_bonding_curve": "abc1",
        "quote_mint": "So11111111111111111111111111111111111111112",
        "token_program": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
        "is_token_2022": False,
        "migrated": False,
        "migration_tx_id": None,
        "pool": None,
        "call_tx_signer": "feePayer1",
        "call_tx_success": True,
        "call_outer_executing_account": "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",
        "call_inner_executing_account": None,
        "source_table": "pumpdotfun_solana.pump_call_create",
        "source_partition_col": "call_block_time",
        "protocol": "pump",
        "protocol_version": "1",
        "schema_version": "1.1.0",
        "raw_payload_json": '{"instruction":"create","mint":"mint1"}',
    }
    df = pd.DataFrame([row])
    validate_dataframe(df, "token_lifecycle")
    return df


@pytest.fixture
def tmp_data_root(tmp_path: Path) -> Path:
    return tmp_path / "data" / "raw" / "dune"
