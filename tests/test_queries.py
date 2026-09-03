"""Query template correctness tests."""

from __future__ import annotations

from trencher.dune.config import load_datasets_config, load_sql_template, render_sql


def test_dex_trades_dataset_registered() -> None:
    datasets = load_datasets_config()
    assert "dex_trades" in datasets
    assert "dex_solana.trades" in datasets["dex_trades"].source_tables
    assert datasets["dex_trades"].sql_path.exists()


def test_wallet_graph_keeps_external_counterparties() -> None:
    sql = load_sql_template(load_datasets_config()["wallet_graph"])
    assert "INNER JOIN participants pf" not in sql
    assert "INNER JOIN participants pt" not in sql
    assert "LEFT JOIN participants pf" in sql
    assert "LEFT JOIN participants pt" in sql
    assert "pf.wallet IS NOT NULL OR pt.wallet IS NOT NULL" in sql
    assert "from_is_participant" in sql
    assert "to_is_participant" in sql


def test_trades_reserve_semantics_are_quote_generic() -> None:
    sql = load_sql_template(load_datasets_config()["trades_events"])
    assert "pool_base_token_reserves AS real_base_reserves" in sql
    assert "pool_quote_token_reserves AS real_quote_reserves" in sql
    assert "pool_base_token_reserves AS real_sol_reserves" not in sql
    assert "THEN b.pool_quote_token_reserves" in sql
    assert "THEN s.pool_quote_token_reserves" in sql


def test_lifecycle_creator_is_not_fee_payer() -> None:
    sql = load_sql_template(load_datasets_config()["token_lifecycle"])
    assert "COALESCE(creator, account_user) AS creator_wallet" in sql
    assert "call_tx_signer AS creator_wallet" not in sql
    assert "create_v2" in sql
    assert "account_tokenProgram" in sql or "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb" in sql


def test_microstructure_preserves_raw_tx_fields() -> None:
    sql = load_sql_template(load_datasets_config()["transaction_microstructure"])
    for field in (
        "instructions_json",
        "account_keys_json",
        "signatures_json",
        "signers_json",
        "address_table_lookups_json",
        "loaded_addresses_json",
        "return_data_json",
        "tx_version",
        "recent_block_hash",
        "instruction_data_hex",
        "raw_payload_json",
    ):
        assert field in sql
    assert "CAST(NULL AS varchar) AS raw_payload_json" not in sql


def test_optional_v2_tables_can_be_stripped() -> None:
    template = load_sql_template(load_datasets_config()["token_lifecycle"])
    included = render_sql(template, start_ts="a", end_ts="b", enabled_tables=None)
    assert "pump_call_create_v2" in included
    stripped = render_sql(template, start_ts="a", end_ts="b", enabled_tables=set())
    assert "pump_call_create_v2" not in stripped
    assert "pump_call_create" in stripped


def test_all_sql_templates_parameterized() -> None:
    for cfg in load_datasets_config().values():
        sql = load_sql_template(cfg)
        assert "{{start_ts}}" in sql
        assert "{{end_ts}}" in sql
        rendered = render_sql(sql, start_ts="2025-01-01 00:00:00", end_ts="2025-01-02 00:00:00")
        assert "{{start_ts}}" not in rendered
