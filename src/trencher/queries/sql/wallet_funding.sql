-- Native SOL transfers involving Pump creators/traders, including external funders.
-- Point-in-time participant flags: true only if the wallet had created/traded at or before this event.

WITH participants AS (
    SELECT wallet, MIN(first_seen) AS first_seen
    FROM (
        SELECT COALESCE(creator, account_user) AS wallet, call_block_time AS first_seen
        FROM pumpdotfun_solana.pump_call_create
        WHERE call_block_time < TIMESTAMP '{{end_ts}}'
        UNION ALL
        SELECT account_user AS wallet, call_block_time AS first_seen
        FROM pumpdotfun_solana.pump_call_create
        WHERE call_block_time < TIMESTAMP '{{end_ts}}'
        /* optional:pumpdotfun_solana.pump_call_create_v2 */
        UNION ALL
        SELECT COALESCE(creator, account_user) AS wallet, call_block_time AS first_seen
        FROM pumpdotfun_solana.pump_call_create_v2
        WHERE call_block_time < TIMESTAMP '{{end_ts}}'
        UNION ALL
        SELECT account_user AS wallet, call_block_time AS first_seen
        FROM pumpdotfun_solana.pump_call_create_v2
        WHERE call_block_time < TIMESTAMP '{{end_ts}}'
        /* /optional:pumpdotfun_solana.pump_call_create_v2 */
        UNION ALL
        SELECT "user" AS wallet, evt_block_time AS first_seen
        FROM pumpdotfun_solana.pump_evt_tradeevent
        WHERE evt_block_time < TIMESTAMP '{{end_ts}}'
        UNION ALL
        SELECT creator AS wallet, evt_block_time AS first_seen
        FROM pumpdotfun_solana.pump_evt_tradeevent
        WHERE evt_block_time < TIMESTAMP '{{end_ts}}'
          AND creator IS NOT NULL
    ) w
    WHERE wallet IS NOT NULL
    GROUP BY 1
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
    pf.wallet IS NOT NULL AND t.block_time >= pf.first_seen AS from_is_participant,
    pt.wallet IS NOT NULL AND t.block_time >= pt.first_seen AS to_is_participant,
    t.tx_signer,
    t.outer_executing_account,
    'tokens_solana.transfers' AS source_table,
    'block_time' AS source_partition_col,
    'solana' AS protocol,
    '1' AS protocol_version,
    '1.1.0' AS schema_version,
    CONCAT(
        '{"from":"', CAST(t.from_owner AS varchar),
        '","to":"', CAST(t.to_owner AS varchar),
        '","amount":"', CAST(t.amount AS varchar), '"}'
    ) AS raw_payload_json
FROM tokens_solana.transfers t
LEFT JOIN participants pf ON t.from_owner = pf.wallet
LEFT JOIN participants pt ON t.to_owner = pt.wallet
WHERE t.block_time >= TIMESTAMP '{{start_ts}}'
  AND t.block_time < TIMESTAMP '{{end_ts}}'
  AND t.token_mint_address = 'So11111111111111111111111111111111111111112'
  AND (pf.wallet IS NOT NULL OR pt.wallet IS NOT NULL)
