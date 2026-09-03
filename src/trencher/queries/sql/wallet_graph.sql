-- Wallet graph edges from SPL/SOL transfers among Pump ecosystem wallets.

WITH participants AS (
    SELECT DISTINCT call_tx_signer AS wallet
    FROM pumpdotfun_solana.pump_call_create
    WHERE call_block_time < TIMESTAMP '{{end_ts}}'
    UNION
    SELECT DISTINCT "user" AS wallet
    FROM pumpdotfun_solana.pump_evt_tradeevent
    WHERE evt_block_time < TIMESTAMP '{{end_ts}}'
),
edges AS (
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
        t.tx_signer,
        t.outer_executing_account,
        'tokens_solana.transfers' AS source_table,
        'block_time' AS source_partition_col,
        'solana' AS protocol,
        '1' AS protocol_version,
        '1.0.0' AS schema_version,
        CAST(NULL AS varchar) AS raw_payload_json
    FROM tokens_solana.transfers t
    INNER JOIN participants pf ON t.from_owner = pf.wallet
    INNER JOIN participants pt ON t.to_owner = pt.wallet
    WHERE t.block_time >= TIMESTAMP '{{start_ts}}'
      AND t.block_time < TIMESTAMP '{{end_ts}}'
      AND t.from_owner IS NOT NULL
      AND t.to_owner IS NOT NULL
      AND t.from_owner <> t.to_owner
)
SELECT * FROM edges
