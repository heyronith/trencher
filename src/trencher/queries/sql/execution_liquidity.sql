-- PumpSwap pool creation events for Pump-origin tokens.
-- Verified: pump_amm_evt_createpoolevent (DefiLlama dimension-adapters pump-swap).

WITH pump_mints AS (
    SELECT DISTINCT account_mint AS mint, call_tx_signer AS creator_wallet
    FROM pumpdotfun_solana.pump_call_create
    WHERE call_block_time < TIMESTAMP '{{end_ts}}'
)
SELECT
    e.evt_block_time AS block_time,
    e.evt_block_slot AS block_slot,
    CAST(e.evt_block_time AS date) AS block_date,
    e.evt_tx_index AS tx_index,
    e.evt_tx_id AS tx_id,
    e.evt_outer_instruction_index AS outer_instruction_index,
    e.evt_inner_instruction_index AS inner_instruction_index,
    e.evt_index AS event_index,
    'create_pool_event' AS event_type,
    e.pool,
    pm.mint,
    e.quote_mint,
    pm.creator_wallet,
    e.base_mint,
    e.lp_mint,
    e.base_amount,
    e.quote_amount,
    e.pool_base_token_reserves,
    e.pool_quote_token_reserves,
    e.initial_virtual_base_reserves,
    e.initial_virtual_quote_reserves,
    'pumpdotfun_solana.pump_amm_evt_createpoolevent' AS source_table,
    'evt_block_time' AS source_partition_col,
    'pump_amm' AS protocol,
    '1' AS protocol_version,
    '1.0.0' AS schema_version,
    CAST(NULL AS varchar) AS raw_payload_json
FROM pumpdotfun_solana.pump_amm_evt_createpoolevent e
INNER JOIN pump_mints pm ON e.base_mint = pm.mint
WHERE e.evt_block_time >= TIMESTAMP '{{start_ts}}'
  AND e.evt_block_time < TIMESTAMP '{{end_ts}}'
