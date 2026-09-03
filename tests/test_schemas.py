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


def test_all_schemas_are_pyarrow(dataset=None) -> None:
    for name, schema in DATASET_SCHEMAS.items():
        assert isinstance(schema, pa.Schema), name
