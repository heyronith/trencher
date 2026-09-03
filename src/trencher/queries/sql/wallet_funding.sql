-- Native SOL transfers involving Pump creators and traders.
-- Uses wrapped/native SOL mint So11111111111111111111111111111111111111112 in tokens_solana.transfers.

WITH participants AS (
    SELECT DISTINCT call_tx_signer AS wallet
    FROM pumpdotfun_solana.pump_call_create
    WHERE call_block_time < TIMESTAMP '{{end_ts}}'
    UNION
    SELECT DISTINCT "user" AS wallet
    FROM pumpdotfun_solana.pump_evt_tradeevent
    WHERE evt_block_time < TIMESTAMP '{{end_ts}}'
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
    t.from_owner,
    t.to_owner,
    t.from_token_account,
    t.to_token_account,
    CAST(t.amount AS uint256) AS amount_lamports,
    t.amount_display AS amount_sol,
    t.tx_signer,
    t.outer_executing_account,
    true AS is_pump_participant,
    'tokens_solana.transfers' AS source_table,
    'block_time' AS source_partition_col,
    'solana' AS protocol,
    '1' AS protocol_version,
    '1.0.0' AS schema_version,
    CAST(NULL AS varchar) AS raw_payload_json
FROM tokens_solana.transfers t
INNER JOIN participants p
    ON t.from_owner = p.wallet OR t.to_owner = p.wallet
WHERE t.block_time >= TIMESTAMP '{{start_ts}}'
  AND t.block_time < TIMESTAMP '{{end_ts}}'
  AND t.token_mint_address = 'So11111111111111111111111111111111111111112'
