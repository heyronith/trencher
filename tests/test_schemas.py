"""Schema validation tests."""

from __future__ import annotations

import pyarrow as pa
import pytest

from trencher.schemas.registry import DATASET_SCHEMAS, get_schema, validate_dataframe


@pytest.mark.parametrize("dataset", sorted(DATASET_SCHEMAS))
def test_schema_registry_has_version_fields(dataset: str) -> None:
    schema = get_schema(dataset)
    names = schema.names
    assert "schema_version" in names
    assert "source_table" in names
    assert "block_time" in names or dataset == "market_context"


def test_unknown_dataset_raises() -> None:
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_schema("not_a_dataset")


def test_validate_dataframe_round_trip(sample_token_lifecycle_df) -> None:
    validate_dataframe(sample_token_lifecycle_df, "token_lifecycle")


def test_protocol_safe_fields_present() -> None:
    lifecycle = get_schema("token_lifecycle")
    assert "quote_mint" in lifecycle.names
    assert "token_program" in lifecycle.names
    assert "instruction_name" in lifecycle.names
    assert "user_wallet" in lifecycle.names
    trades = get_schema("trades_events")
    assert "base_mint" in trades.names
    assert "real_base_reserves" in trades.names
    assert "real_quote_reserves" in trades.names
    assert "trade_source" in trades.names
    micro = get_schema("transaction_microstructure")
    assert "instructions_json" in micro.names
    assert "address_table_lookups_json" in micro.names
    graph = get_schema("wallet_graph")
    assert "from_is_participant" in graph.names
    assert get_schema("dex_trades") is trades


def test_all_schemas_are_pyarrow(dataset=None) -> None:
    for name, schema in DATASET_SCHEMAS.items():
        assert isinstance(schema, pa.Schema), name
