-- Wallet graph edges: transfers where at least one endpoint is a Pump participant.
-- External funders/counterparties are retained. Participant flags are point-in-time.

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
        SELECT COALESCE(/* optional-col:pumpdotfun_solana.pump_call_create_v2.creator:varchar */ creator /* /optional-col:pumpdotfun_solana.pump_call_create_v2.creator */, account_user) AS wallet, call_block_time AS first_seen
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
        SELECT /* optional-col:pumpdotfun_solana.pump_evt_tradeevent.creator:varchar */ creator /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.creator */ AS wallet, evt_block_time AS first_seen
        FROM pumpdotfun_solana.pump_evt_tradeevent
        WHERE evt_block_time < TIMESTAMP '{{end_ts}}'
          AND /* optional-col:pumpdotfun_solana.pump_evt_tradeevent.creator:varchar */ creator /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.creator */ IS NOT NULL
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
    CASE
        WHEN t.token_mint_address = 'So11111111111111111111111111111111111111112' THEN 'sol_transfer'
        ELSE 'spl_transfer'
    END AS edge_type,
    t.from_owner AS from_wallet,
    t.to_owner AS to_wallet,
    t.token_mint_address AS mint,
    CAST(t.amount AS uint256) AS amount_raw,
    CASE
        WHEN t.token_mint_address = 'So11111111111111111111111111111111111111112'
            THEN t.amount_display
        ELSE CAST(NULL AS double)
    END AS amount_sol,
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
        '","mint":"', CAST(t.token_mint_address AS varchar), '"}'
    ) AS raw_payload_json
FROM tokens_solana.transfers t
LEFT JOIN participants pf ON t.from_owner = pf.wallet
LEFT JOIN participants pt ON t.to_owner = pt.wallet
WHERE t.block_time >= TIMESTAMP '{{start_ts}}'
  AND t.block_time < TIMESTAMP '{{end_ts}}'
  AND t.from_owner IS NOT NULL
  AND t.to_owner IS NOT NULL
  AND t.from_owner <> t.to_owner
  AND (pf.wallet IS NOT NULL OR pt.wallet IS NOT NULL)
