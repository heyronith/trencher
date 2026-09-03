-- Post-graduation Solana DEX trades for Pump-origin mints.
-- Verified: dex_solana.trades (docs.dune.com/data-catalog/curated/dex-trades/solana).
-- Preserves project/version/trade_source/vaults/program IDs for venue and route identity.

WITH pump_mints AS (
    SELECT DISTINCT account_mint AS mint
    FROM pumpdotfun_solana.pump_call_create
    WHERE call_block_time < TIMESTAMP '{{end_ts}}'
    /* optional:pumpdotfun_solana.pump_call_create_v2 */
    UNION
    SELECT DISTINCT account_mint AS mint
    FROM pumpdotfun_solana.pump_call_create_v2
    WHERE call_block_time < TIMESTAMP '{{end_ts}}'
    /* /optional:pumpdotfun_solana.pump_call_create_v2 */
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
    COALESCE(t.project, 'dex') AS venue,
    'dex_trade' AS event_type,
    CAST(NULL AS varchar) AS instruction_name,
    CAST(NULL AS varchar) AS instruction_discriminator,
    CASE
        WHEN pm_bought.mint IS NOT NULL THEN 'buy'
        WHEN pm_sold.mint IS NOT NULL THEN 'sell'
    END AS side,
    COALESCE(pm_bought.mint, pm_sold.mint) AS mint,
    COALESCE(pm_bought.mint, pm_sold.mint) AS base_mint,
    CASE
        WHEN pm_bought.mint IS NOT NULL THEN t.token_sold_mint_address
        ELSE t.token_bought_mint_address
    END AS quote_mint,
    COALESCE(t.token_bought_vault, t.token_sold_vault) AS pool,
    t.trader_id AS trader_wallet,
    CAST(NULL AS varchar) AS creator_wallet,
    CAST(NULL AS varchar) AS token_program,
    CAST(NULL AS boolean) AS is_token_2022,
    CASE
        WHEN pm_bought.mint IS NOT NULL THEN t.token_bought_amount_raw
        ELSE t.token_sold_amount_raw
    END AS base_amount,
    CASE
        WHEN pm_bought.mint IS NOT NULL THEN t.token_sold_amount_raw
        ELSE t.token_bought_amount_raw
    END AS quote_amount,
    CASE
        WHEN (
            CASE
                WHEN pm_bought.mint IS NOT NULL THEN t.token_sold_mint_address
                ELSE t.token_bought_mint_address
            END
        ) IN (
            'So11111111111111111111111111111111111111112',
            '11111111111111111111111111111111'
        )
        THEN CASE
            WHEN pm_bought.mint IS NOT NULL THEN t.token_sold_amount_raw
            ELSE t.token_bought_amount_raw
        END
    END AS sol_amount,
    CASE
        WHEN pm_bought.mint IS NOT NULL THEN t.token_bought_amount_raw
        ELSE t.token_sold_amount_raw
    END AS token_amount,
    pm_bought.mint IS NOT NULL AS is_buy,
    CAST(NULL AS uint256) AS virtual_base_reserves,
    CAST(NULL AS uint256) AS virtual_quote_reserves,
    CAST(NULL AS uint256) AS real_base_reserves,
    CAST(NULL AS uint256) AS real_quote_reserves,
    CAST(NULL AS uint256) AS virtual_sol_reserves,
    CAST(NULL AS uint256) AS virtual_token_reserves,
    CAST(NULL AS uint256) AS real_sol_reserves,
    CAST(NULL AS uint256) AS real_token_reserves,
    CAST(NULL AS uint256) AS lp_fee,
    CAST(NULL AS uint256) AS protocol_fee,
    CAST(NULL AS uint256) AS coin_creator_fee,
    CAST(NULL AS uint256) AS fee,
    CAST(NULL AS bigint) AS fee_basis_points,
    CAST(NULL AS bigint) AS event_timestamp,
    t.project,
    CAST(t.version AS varchar) AS project_version,
    t.version_name AS project_version_name,
    t.trade_source,
    t.project_program_id,
    t.project_main_id,
    t.token_bought_mint_address AS token_bought_mint,
    t.token_sold_mint_address AS token_sold_mint,
    t.token_bought_amount_raw,
    t.token_sold_amount_raw,
    t.token_bought_vault,
    t.token_sold_vault,
    t.token_pair,
    t.amount_usd,
    t.fee_usd,
    t.fee_tier,
    'dex_solana.trades' AS source_table,
    'block_time' AS source_partition_col,
    COALESCE(t.project, 'dex') AS protocol,
    CAST(t.version AS varchar) AS protocol_version,
    '1.1.0' AS schema_version,
    CONCAT(
        '{"project":"', CAST(t.project AS varchar),
        '","trade_source":"', CAST(t.trade_source AS varchar),
        '","project_program_id":"', CAST(t.project_program_id AS varchar),
        '","token_bought":"', CAST(t.token_bought_mint_address AS varchar),
        '","token_sold":"', CAST(t.token_sold_mint_address AS varchar),
        '","bought_vault":"', CAST(t.token_bought_vault AS varchar),
        '","sold_vault":"', CAST(t.token_sold_vault AS varchar), '"}'
    ) AS raw_payload_json
FROM dex_solana.trades t
LEFT JOIN pump_mints pm_bought ON t.token_bought_mint_address = pm_bought.mint
LEFT JOIN pump_mints pm_sold ON t.token_sold_mint_address = pm_sold.mint
WHERE t.block_time >= TIMESTAMP '{{start_ts}}'
  AND t.block_time < TIMESTAMP '{{end_ts}}'
  AND (pm_bought.mint IS NOT NULL OR pm_sold.mint IS NOT NULL)
