"""Schema registry mapping dataset names to canonical PyArrow schemas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from trencher.schemas.execution_liquidity import EXECUTION_LIQUIDITY_SCHEMA
from trencher.schemas.market_context import MARKET_CONTEXT_SCHEMA
from trencher.schemas.token_lifecycle import TOKEN_LIFECYCLE_SCHEMA
from trencher.schemas.token_transfers import TOKEN_TRANSFERS_SCHEMA
from trencher.schemas.trades_events import TRADES_EVENTS_SCHEMA
from trencher.schemas.transaction_microstructure import TRANSACTION_MICROSTRUCTURE_SCHEMA
from trencher.schemas.wallet_funding import WALLET_FUNDING_SCHEMA
from trencher.schemas.wallet_graph import WALLET_GRAPH_SCHEMA

if TYPE_CHECKING:
    import pandas as pd

DATASET_SCHEMAS: dict[str, pa.Schema] = {
    "token_lifecycle": TOKEN_LIFECYCLE_SCHEMA,
    "trades_events": TRADES_EVENTS_SCHEMA,
    "dex_trades": TRADES_EVENTS_SCHEMA,
    "transaction_microstructure": TRANSACTION_MICROSTRUCTURE_SCHEMA,
    "token_transfers": TOKEN_TRANSFERS_SCHEMA,
    "wallet_funding": WALLET_FUNDING_SCHEMA,
    "wallet_graph": WALLET_GRAPH_SCHEMA,
    "market_context": MARKET_CONTEXT_SCHEMA,
    "execution_liquidity": EXECUTION_LIQUIDITY_SCHEMA,
}


def get_schema(dataset: str) -> pa.Schema:
    try:
        return DATASET_SCHEMAS[dataset]
    except KeyError as exc:
        known = ", ".join(sorted(DATASET_SCHEMAS))
        raise KeyError(f"Unknown dataset {dataset!r}. Known datasets: {known}") from exc


def validate_dataframe(df: pd.DataFrame, dataset: str) -> None:
    """Validate a DataFrame against the canonical schema for a dataset."""
    schema = get_schema(dataset)
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    if table.num_rows != len(df):
        raise ValueError(f"Schema coercion changed row count for dataset {dataset!r}")
