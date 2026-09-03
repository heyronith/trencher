-- Bonding-curve trades and PumpSwap AMM buy/sell events.
-- Quote-generic base/quote amounts and reserves. SOL aliases only when quote is wSOL.
-- Verified: pump_evt_tradeevent, pump_amm_evt_buyevent, pump_amm_evt_sellevent.

WITH curve_trades AS (
    SELECT
        evt_block_time AS block_time,
        evt_block_slot AS block_slot,
        CAST(evt_block_time AS date) AS block_date,
        evt_tx_index AS tx_index,
        evt_tx_id AS tx_id,
        evt_outer_instruction_index AS outer_instruction_index,
        evt_inner_instruction_index AS inner_instruction_index,
        evt_index AS event_index,
        'pump_bonding_curve' AS venue,
        'trade_event' AS event_type,
        COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_tradeevent.ix_name:varchar */ ix_name /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.ix_name */, CASE WHEN is_buy THEN 'buy' ELSE 'sell' END) AS instruction_name,
        CAST(NULL AS varchar) AS instruction_discriminator,
        CASE WHEN is_buy THEN 'buy' ELSE 'sell' END AS side,
        mint,
        mint AS base_mint,
        COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_tradeevent.quote_mint:varchar */ quote_mint /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.quote_mint */, 'So11111111111111111111111111111111111111112') AS quote_mint,
        CAST(NULL AS varchar) AS pool,
        "user" AS trader_wallet,
        /* optional-col:pumpdotfun_solana.pump_evt_tradeevent.creator:varchar */ creator /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.creator */ AS creator_wallet,
        CAST(NULL AS varchar) AS token_program,
        CAST(NULL AS boolean) AS is_token_2022,
        token_amount AS base_amount,
        COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_tradeevent.quote_amount:uint256 */ quote_amount /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.quote_amount */, sol_amount) AS quote_amount,
        CASE
            WHEN COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_tradeevent.quote_mint:varchar */ quote_mint /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.quote_mint */, 'So11111111111111111111111111111111111111112')
                IN ('So11111111111111111111111111111111111111112', '11111111111111111111111111111111')
            THEN COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_tradeevent.quote_amount:uint256 */ quote_amount /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.quote_amount */, sol_amount)
        END AS sol_amount,
        token_amount,
        is_buy,
        virtual_token_reserves AS virtual_base_reserves,
        COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_tradeevent.virtual_quote_reserves:uint256 */ virtual_quote_reserves /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.virtual_quote_reserves */, virtual_sol_reserves) AS virtual_quote_reserves,
        real_token_reserves AS real_base_reserves,
        COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_tradeevent.real_quote_reserves:uint256 */ real_quote_reserves /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.real_quote_reserves */, real_sol_reserves) AS real_quote_reserves,
        CASE
            WHEN COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_tradeevent.quote_mint:varchar */ quote_mint /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.quote_mint */, 'So11111111111111111111111111111111111111112')
                IN ('So11111111111111111111111111111111111111112', '11111111111111111111111111111111')
            THEN COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_tradeevent.virtual_quote_reserves:uint256 */ virtual_quote_reserves /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.virtual_quote_reserves */, virtual_sol_reserves)
        END AS virtual_sol_reserves,
        CASE
            WHEN COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_tradeevent.quote_mint:varchar */ quote_mint /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.quote_mint */, 'So11111111111111111111111111111111111111112')
                IN ('So11111111111111111111111111111111111111112', '11111111111111111111111111111111')
            THEN virtual_token_reserves
        END AS virtual_token_reserves,
        CASE
            WHEN COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_tradeevent.quote_mint:varchar */ quote_mint /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.quote_mint */, 'So11111111111111111111111111111111111111112')
                IN ('So11111111111111111111111111111111111111112', '11111111111111111111111111111111')
            THEN COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_tradeevent.real_quote_reserves:uint256 */ real_quote_reserves /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.real_quote_reserves */, real_sol_reserves)
        END AS real_sol_reserves,
        CASE
            WHEN COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_tradeevent.quote_mint:varchar */ quote_mint /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.quote_mint */, 'So11111111111111111111111111111111111111112')
                IN ('So11111111111111111111111111111111111111112', '11111111111111111111111111111111')
            THEN real_token_reserves
        END AS real_token_reserves,
        CAST(NULL AS uint256) AS lp_fee,
        fee AS protocol_fee,
        /* optional-col:pumpdotfun_solana.pump_evt_tradeevent.creator_fee:uint256 */ creator_fee /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.creator_fee */ AS coin_creator_fee,
        fee,
        fee_basis_points,
        timestamp AS event_timestamp,
        CAST(NULL AS varchar) AS project,
        CAST(NULL AS varchar) AS project_version,
        CAST(NULL AS varchar) AS project_version_name,
        CAST(NULL AS varchar) AS trade_source,
        CAST(NULL AS varchar) AS project_program_id,
        CAST(NULL AS varchar) AS project_main_id,
        CAST(NULL AS varchar) AS token_bought_mint,
        CAST(NULL AS varchar) AS token_sold_mint,
        CAST(NULL AS uint256) AS token_bought_amount_raw,
        CAST(NULL AS uint256) AS token_sold_amount_raw,
        CAST(NULL AS varchar) AS token_bought_vault,
        CAST(NULL AS varchar) AS token_sold_vault,
        CAST(NULL AS varchar) AS token_pair,
        CAST(NULL AS double) AS amount_usd,
        CAST(NULL AS double) AS fee_usd,
        CAST(NULL AS double) AS fee_tier,
        'pumpdotfun_solana.pump_evt_tradeevent' AS source_table,
        'evt_block_time' AS source_partition_col,
        'pump' AS protocol,
        CASE
            WHEN COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_tradeevent.ix_name:varchar */ ix_name /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.ix_name */, '') LIKE '%v2%' THEN '2'
            WHEN COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_tradeevent.ix_name:varchar */ ix_name /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.ix_name */, '') = 'buy_exact_sol_in' THEN '2'
            ELSE '1'
        END AS protocol_version,
        '1.1.0' AS schema_version,
        CONCAT(
            '{"event":"TradeEvent","mint":"', CAST(mint AS varchar),
            '","ix_name":"', CAST(COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_tradeevent.ix_name:varchar */ ix_name /* /optional-col:pumpdotfun_solana.pump_evt_tradeevent.ix_name */, '') AS varchar),
            '","is_buy":', CASE WHEN is_buy THEN 'true' ELSE 'false' END, '}'
        ) AS raw_payload_json
    FROM pumpdotfun_solana.pump_evt_tradeevent
    WHERE evt_block_time >= TIMESTAMP '{{start_ts}}'
      AND evt_block_time < TIMESTAMP '{{end_ts}}'
),
amm_buys AS (
    SELECT
        b.evt_block_time AS block_time,
        b.evt_block_slot AS block_slot,
        CAST(b.evt_block_time AS date) AS block_date,
        b.evt_tx_index AS tx_index,
        b.evt_tx_id AS tx_id,
        b.evt_outer_instruction_index AS outer_instruction_index,
        b.evt_inner_instruction_index AS inner_instruction_index,
        b.evt_index AS event_index,
        'pumpswap' AS venue,
        'buy_event' AS event_type,
        COALESCE(/* optional-col:pumpdotfun_solana.pump_amm_evt_buyevent.ix_name:varchar */ b.ix_name /* /optional-col:pumpdotfun_solana.pump_amm_evt_buyevent.ix_name */, 'buy') AS instruction_name,
        CAST(NULL AS varchar) AS instruction_discriminator,
        'buy' AS side,
        p.base_mint AS mint,
        p.base_mint,
        p.quote_mint,
        b.pool,
        b."user" AS trader_wallet,
        COALESCE(/* optional-col:pumpdotfun_solana.pump_amm_evt_buyevent.coin_creator:varchar */ b.coin_creator /* /optional-col:pumpdotfun_solana.pump_amm_evt_buyevent.coin_creator */, p.creator) AS creator_wallet,
        CAST(NULL AS varchar) AS token_program,
        CAST(NULL AS boolean) AS is_token_2022,
        b.base_amount_out AS base_amount,
        b.quote_amount_in AS quote_amount,
        CASE
            WHEN p.quote_mint IN (
                'So11111111111111111111111111111111111111112',
                '11111111111111111111111111111111'
            ) THEN b.quote_amount_in
        END AS sol_amount,
        b.base_amount_out AS token_amount,
        true AS is_buy,
        CAST(NULL AS uint256) AS virtual_base_reserves,
        CAST(NULL AS uint256) AS virtual_quote_reserves,
        b.pool_base_token_reserves AS real_base_reserves,
        b.pool_quote_token_reserves AS real_quote_reserves,
        CAST(NULL AS uint256) AS virtual_sol_reserves,
        CAST(NULL AS uint256) AS virtual_token_reserves,
        CASE
            WHEN p.quote_mint IN (
                'So11111111111111111111111111111111111111112',
                '11111111111111111111111111111111'
            ) THEN b.pool_quote_token_reserves
        END AS real_sol_reserves,
        CASE
            WHEN p.quote_mint IN (
                'So11111111111111111111111111111111111111112',
                '11111111111111111111111111111111'
            ) THEN b.pool_base_token_reserves
        END AS real_token_reserves,
        b.lp_fee,
        b.protocol_fee,
        /* optional-col:pumpdotfun_solana.pump_amm_evt_buyevent.coin_creator_fee:uint256 */ b.coin_creator_fee /* /optional-col:pumpdotfun_solana.pump_amm_evt_buyevent.coin_creator_fee */ AS coin_creator_fee,
        b.protocol_fee AS fee,
        CAST(NULL AS bigint) AS fee_basis_points,
        b.timestamp AS event_timestamp,
        'pumpswap' AS project,
        CAST(NULL AS varchar) AS project_version,
        CAST(NULL AS varchar) AS project_version_name,
        CAST(NULL AS varchar) AS trade_source,
        'pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA' AS project_program_id,
        CAST(NULL AS varchar) AS project_main_id,
        CAST(NULL AS varchar) AS token_bought_mint,
        CAST(NULL AS varchar) AS token_sold_mint,
        CAST(NULL AS uint256) AS token_bought_amount_raw,
        CAST(NULL AS uint256) AS token_sold_amount_raw,
        CAST(NULL AS varchar) AS token_bought_vault,
        CAST(NULL AS varchar) AS token_sold_vault,
        CAST(NULL AS varchar) AS token_pair,
        CAST(NULL AS double) AS amount_usd,
        CAST(NULL AS double) AS fee_usd,
        CAST(NULL AS double) AS fee_tier,
        'pumpdotfun_solana.pump_amm_evt_buyevent' AS source_table,
        'evt_block_time' AS source_partition_col,
        'pump_amm' AS protocol,
        '1' AS protocol_version,
        '1.1.0' AS schema_version,
        CONCAT(
            '{"event":"BuyEvent","pool":"', CAST(b.pool AS varchar),
            '","base_mint":"', CAST(p.base_mint AS varchar),
            '","quote_mint":"', CAST(p.quote_mint AS varchar), '"}'
        ) AS raw_payload_json
    FROM pumpdotfun_solana.pump_amm_evt_buyevent b
    LEFT JOIN pumpdotfun_solana.pump_amm_evt_createpoolevent p
        ON b.pool = p.pool
    WHERE b.evt_block_time >= TIMESTAMP '{{start_ts}}'
      AND b.evt_block_time < TIMESTAMP '{{end_ts}}'
),
amm_sells AS (
    SELECT
        s.evt_block_time AS block_time,
        s.evt_block_slot AS block_slot,
        CAST(s.evt_block_time AS date) AS block_date,
        s.evt_tx_index AS tx_index,
        s.evt_tx_id AS tx_id,
        s.evt_outer_instruction_index AS outer_instruction_index,
        s.evt_inner_instruction_index AS inner_instruction_index,
        s.evt_index AS event_index,
        'pumpswap' AS venue,
        'sell_event' AS event_type,
        COALESCE(/* optional-col:pumpdotfun_solana.pump_amm_evt_sellevent.ix_name:varchar */ s.ix_name /* /optional-col:pumpdotfun_solana.pump_amm_evt_sellevent.ix_name */, 'sell') AS instruction_name,
        CAST(NULL AS varchar) AS instruction_discriminator,
        'sell' AS side,
        p.base_mint AS mint,
        p.base_mint,
        p.quote_mint,
        s.pool,
        s."user" AS trader_wallet,
        COALESCE(/* optional-col:pumpdotfun_solana.pump_amm_evt_sellevent.coin_creator:varchar */ s.coin_creator /* /optional-col:pumpdotfun_solana.pump_amm_evt_sellevent.coin_creator */, p.creator) AS creator_wallet,
        CAST(NULL AS varchar) AS token_program,
        CAST(NULL AS boolean) AS is_token_2022,
        s.base_amount_in AS base_amount,
        s.quote_amount_out AS quote_amount,
        CASE
            WHEN p.quote_mint IN (
                'So11111111111111111111111111111111111111112',
                '11111111111111111111111111111111'
            ) THEN s.quote_amount_out
        END AS sol_amount,
        s.base_amount_in AS token_amount,
        false AS is_buy,
        CAST(NULL AS uint256) AS virtual_base_reserves,
        CAST(NULL AS uint256) AS virtual_quote_reserves,
        s.pool_base_token_reserves AS real_base_reserves,
        s.pool_quote_token_reserves AS real_quote_reserves,
        CAST(NULL AS uint256) AS virtual_sol_reserves,
        CAST(NULL AS uint256) AS virtual_token_reserves,
        CASE
            WHEN p.quote_mint IN (
                'So11111111111111111111111111111111111111112',
                '11111111111111111111111111111111'
            ) THEN s.pool_quote_token_reserves
        END AS real_sol_reserves,
        CASE
            WHEN p.quote_mint IN (
                'So11111111111111111111111111111111111111112',
                '11111111111111111111111111111111'
            ) THEN s.pool_base_token_reserves
        END AS real_token_reserves,
        s.lp_fee,
        s.protocol_fee,
        /* optional-col:pumpdotfun_solana.pump_amm_evt_sellevent.coin_creator_fee:uint256 */ s.coin_creator_fee /* /optional-col:pumpdotfun_solana.pump_amm_evt_sellevent.coin_creator_fee */ AS coin_creator_fee,
        s.protocol_fee AS fee,
        CAST(NULL AS bigint) AS fee_basis_points,
        s.timestamp AS event_timestamp,
        'pumpswap' AS project,
        CAST(NULL AS varchar) AS project_version,
        CAST(NULL AS varchar) AS project_version_name,
        CAST(NULL AS varchar) AS trade_source,
        'pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA' AS project_program_id,
        CAST(NULL AS varchar) AS project_main_id,
        CAST(NULL AS varchar) AS token_bought_mint,
        CAST(NULL AS varchar) AS token_sold_mint,
        CAST(NULL AS uint256) AS token_bought_amount_raw,
        CAST(NULL AS uint256) AS token_sold_amount_raw,
        CAST(NULL AS varchar) AS token_bought_vault,
        CAST(NULL AS varchar) AS token_sold_vault,
        CAST(NULL AS varchar) AS token_pair,
        CAST(NULL AS double) AS amount_usd,
        CAST(NULL AS double) AS fee_usd,
        CAST(NULL AS double) AS fee_tier,
        'pumpdotfun_solana.pump_amm_evt_sellevent' AS source_table,
        'evt_block_time' AS source_partition_col,
        'pump_amm' AS protocol,
        '1' AS protocol_version,
        '1.1.0' AS schema_version,
        CONCAT(
            '{"event":"SellEvent","pool":"', CAST(s.pool AS varchar),
            '","base_mint":"', CAST(p.base_mint AS varchar),
            '","quote_mint":"', CAST(p.quote_mint AS varchar), '"}'
        ) AS raw_payload_json
    FROM pumpdotfun_solana.pump_amm_evt_sellevent s
    LEFT JOIN pumpdotfun_solana.pump_amm_evt_createpoolevent p
        ON s.pool = p.pool
    WHERE s.evt_block_time >= TIMESTAMP '{{start_ts}}'
      AND s.evt_block_time < TIMESTAMP '{{end_ts}}'
)
SELECT * FROM curve_trades
UNION ALL
SELECT * FROM amm_buys
UNION ALL
SELECT * FROM amm_sells
