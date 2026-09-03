-- Post-graduation Solana DEX trades for Pump-origin mints.
-- Verified: dex_solana.trades (docs.dune.com/data-catalog/curated/dex-trades/solana).
-- Pool identity: project_main_id (Dune pool/market id). Token vaults are preserved
-- separately and are never used as pool IDs. block_month is the documented partition key.

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
    /* optional-col:dex_solana.trades.project_main_id:varchar */ t.project_main_id /* /optional-col:dex_solana.trades.project_main_id */ AS pool,
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
    /* optional-col:dex_solana.trades.version:varchar */ CAST(t.version AS varchar) /* /optional-col:dex_solana.trades.version */ AS project_version,
    /* optional-col:dex_solana.trades.version_name:varchar */ t.version_name /* /optional-col:dex_solana.trades.version_name */ AS project_version_name,
    t.trade_source,
    t.project_program_id,
    /* optional-col:dex_solana.trades.project_main_id:varchar */ t.project_main_id /* /optional-col:dex_solana.trades.project_main_id */ AS project_main_id,
    t.token_bought_mint_address AS token_bought_mint,
    t.token_sold_mint_address AS token_sold_mint,
    t.token_bought_amount_raw,
    t.token_sold_amount_raw,
    /* optional-col:dex_solana.trades.token_bought_vault:varchar */ t.token_bought_vault /* /optional-col:dex_solana.trades.token_bought_vault */ AS token_bought_vault,
    /* optional-col:dex_solana.trades.token_sold_vault:varchar */ t.token_sold_vault /* /optional-col:dex_solana.trades.token_sold_vault */ AS token_sold_vault,
    /* optional-col:dex_solana.trades.token_pair:varchar */ t.token_pair /* /optional-col:dex_solana.trades.token_pair */ AS token_pair,
    /* optional-col:dex_solana.trades.amount_usd:double */ t.amount_usd /* /optional-col:dex_solana.trades.amount_usd */ AS amount_usd,
    /* optional-col:dex_solana.trades.fee_usd:double */ t.fee_usd /* /optional-col:dex_solana.trades.fee_usd */ AS fee_usd,
    /* optional-col:dex_solana.trades.fee_tier:double */ t.fee_tier /* /optional-col:dex_solana.trades.fee_tier */ AS fee_tier,
    'dex_solana.trades' AS source_table,
    'block_time' AS source_partition_col,
    COALESCE(t.project, 'dex') AS protocol,
    /* optional-col:dex_solana.trades.version:varchar */ CAST(t.version AS varchar) /* /optional-col:dex_solana.trades.version */ AS protocol_version,
    '1.1.0' AS schema_version,
    CONCAT(
        '{"project":"', CAST(t.project AS varchar),
        '","trade_source":"', CAST(t.trade_source AS varchar),
        '","project_program_id":"', CAST(t.project_program_id AS varchar),
        '","token_bought":"', CAST(t.token_bought_mint_address AS varchar),
        '","token_sold":"', CAST(t.token_sold_mint_address AS varchar),
        '","project_main_id":"', CAST(/* optional-col:dex_solana.trades.project_main_id:varchar */ t.project_main_id /* /optional-col:dex_solana.trades.project_main_id */ AS varchar),
        '","bought_vault":"', CAST(/* optional-col:dex_solana.trades.token_bought_vault:varchar */ t.token_bought_vault /* /optional-col:dex_solana.trades.token_bought_vault */ AS varchar),
        '","sold_vault":"', CAST(/* optional-col:dex_solana.trades.token_sold_vault:varchar */ t.token_sold_vault /* /optional-col:dex_solana.trades.token_sold_vault */ AS varchar), '"}'
    ) AS raw_payload_json
FROM dex_solana.trades t
LEFT JOIN pump_mints pm_bought ON t.token_bought_mint_address = pm_bought.mint
LEFT JOIN pump_mints pm_sold ON t.token_sold_mint_address = pm_sold.mint
WHERE t.block_time >= TIMESTAMP '{{start_ts}}'
  AND t.block_time < TIMESTAMP '{{end_ts}}'
  AND t.block_month >= CAST(date_trunc('month', TIMESTAMP '{{start_ts}}') AS date)
  AND t.block_month <= CAST(date_trunc('month', TIMESTAMP '{{end_ts}}' - INTERVAL '1' SECOND) AS date)
  AND (pm_bought.mint IS NOT NULL OR pm_sold.mint IS NOT NULL)
