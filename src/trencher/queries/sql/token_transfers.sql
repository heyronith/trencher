-- SPL token transfers for Pump-origin mints (creator activity / ownership changes).
-- Verified: tokens_solana.transfers (docs.dune.com/data-catalog/curated/token-transfers/solana).

WITH pump_mints AS (
    SELECT DISTINCT account_mint AS mint
    FROM pumpdotfun_solana.pump_call_create
    WHERE call_block_time < TIMESTAMP '{{end_ts}}'
)
SELECT
    t.block_time,
    t.block_slot,
    t.block_date,
    t.tx_index,
    t.tx_id,
    t.outer_instruction_index,
    t.inner_instruction_index,
    CAST(NULL AS integer) AS event_index,
    t.action,
    t.token_mint_address AS mint,
    t.symbol,
    t.token_version,
    t.from_token_account,
    t.to_token_account,
    t.from_owner,
    t.to_owner,
    CAST(t.amount AS uint256) AS amount_raw,
    t.amount_display,
    t.amount_usd,
    t.tx_signer,
    t.outer_executing_account,
    CAST(t.unique_instruction_key AS varchar) AS unique_instruction_key,
    'tokens_solana.transfers' AS source_table,
    'block_time' AS source_partition_col,
    'spl_token' AS protocol,
    '1' AS protocol_version,
    '1.0.0' AS schema_version,
    CAST(NULL AS varchar) AS raw_payload_json
FROM tokens_solana.transfers t
INNER JOIN pump_mints pm ON t.token_mint_address = pm.mint
WHERE t.block_time >= TIMESTAMP '{{start_ts}}'
  AND t.block_time < TIMESTAMP '{{end_ts}}'
