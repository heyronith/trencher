-- Bonding-curve trades and PumpSwap AMM buy/sell events.
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
        CASE WHEN is_buy THEN 'buy' ELSE 'sell' END AS side,
        mint,
        CAST(NULL AS varchar) AS pool,
        "user" AS trader_wallet,
        creator AS creator_wallet,
        CAST(NULL AS varchar) AS quote_mint,
        token_amount AS base_amount,
        sol_amount AS quote_amount,
        sol_amount,
        token_amount,
        is_buy,
        virtual_sol_reserves,
        virtual_token_reserves,
        real_sol_reserves,
        real_token_reserves,
        CAST(NULL AS uint256) AS lp_fee,
        fee AS protocol_fee,
        CAST(NULL AS uint256) AS coin_creator_fee,
        fee,
        fee_basis_points,
        timestamp AS event_timestamp,
        'pumpdotfun_solana.pump_evt_tradeevent' AS source_table,
        'evt_block_time' AS source_partition_col,
        'pump' AS protocol,
        '1' AS protocol_version,
        '1.0.0' AS schema_version,
        CAST(NULL AS varchar) AS raw_payload_json
    FROM pumpdotfun_solana.pump_evt_tradeevent
    WHERE evt_block_time >= TIMESTAMP '{{start_ts}}'
      AND evt_block_time < TIMESTAMP '{{end_ts}}'
),
amm_buys AS (
    SELECT
        evt_block_time AS block_time,
        evt_block_slot AS block_slot,
        CAST(evt_block_time AS date) AS block_date,
        evt_tx_index AS tx_index,
        evt_tx_id AS tx_id,
        evt_outer_instruction_index AS outer_instruction_index,
        evt_inner_instruction_index AS inner_instruction_index,
        evt_index AS event_index,
        'pumpswap' AS venue,
        'buy_event' AS event_type,
        'buy' AS side,
        base_mint AS mint,
        pool,
        "user" AS trader_wallet,
        coin_creator AS creator_wallet,
        quote_mint,
        base_amount_out AS base_amount,
        quote_amount_in AS quote_amount,
        quote_amount_in AS sol_amount,
        base_amount_out AS token_amount,
        true AS is_buy,
        CAST(NULL AS uint256) AS virtual_sol_reserves,
        CAST(NULL AS uint256) AS virtual_token_reserves,
        pool_base_token_reserves AS real_sol_reserves,
        pool_quote_token_reserves AS real_token_reserves,
        lp_fee,
        protocol_fee,
        coin_creator_fee,
        protocol_fee AS fee,
        CAST(NULL AS bigint) AS fee_basis_points,
        timestamp AS event_timestamp,
        'pumpdotfun_solana.pump_amm_evt_buyevent' AS source_table,
        'evt_block_time' AS source_partition_col,
        'pump_amm' AS protocol,
        '1' AS protocol_version,
        '1.0.0' AS schema_version,
        CAST(NULL AS varchar) AS raw_payload_json
    FROM pumpdotfun_solana.pump_amm_evt_buyevent
    WHERE evt_block_time >= TIMESTAMP '{{start_ts}}'
      AND evt_block_time < TIMESTAMP '{{end_ts}}'
),
amm_sells AS (
    SELECT
        evt_block_time AS block_time,
        evt_block_slot AS block_slot,
        CAST(evt_block_time AS date) AS block_date,
        evt_tx_index AS tx_index,
        evt_tx_id AS tx_id,
        evt_outer_instruction_index AS outer_instruction_index,
        evt_inner_instruction_index AS inner_instruction_index,
        evt_index AS event_index,
        'pumpswap' AS venue,
        'sell_event' AS event_type,
        'sell' AS side,
        base_mint AS mint,
        pool,
        "user" AS trader_wallet,
        coin_creator AS creator_wallet,
        quote_mint,
        base_amount_in AS base_amount,
        quote_amount_out AS quote_amount,
        quote_amount_out AS sol_amount,
        base_amount_in AS token_amount,
        false AS is_buy,
        CAST(NULL AS uint256) AS virtual_sol_reserves,
        CAST(NULL AS uint256) AS virtual_token_reserves,
        pool_base_token_reserves AS real_sol_reserves,
        pool_quote_token_reserves AS real_token_reserves,
        lp_fee,
        protocol_fee,
        coin_creator_fee,
        protocol_fee AS fee,
        CAST(NULL AS bigint) AS fee_basis_points,
        timestamp AS event_timestamp,
        'pumpdotfun_solana.pump_amm_evt_sellevent' AS source_table,
        'evt_block_time' AS source_partition_col,
        'pump_amm' AS protocol,
        '1' AS protocol_version,
        '1.0.0' AS schema_version,
        CAST(NULL AS varchar) AS raw_payload_json
    FROM pumpdotfun_solana.pump_amm_evt_sellevent
    WHERE evt_block_time >= TIMESTAMP '{{start_ts}}'
      AND evt_block_time < TIMESTAMP '{{end_ts}}'
)
SELECT * FROM curve_trades
UNION ALL
SELECT * FROM amm_buys
UNION ALL
SELECT * FROM amm_sells
