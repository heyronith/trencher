-- Raw Pump / PumpSwap instructions joined to parent transactions.
-- Verified: solana.instruction_calls, solana.transactions (docs.dune.com/data-catalog/solana).
-- Program IDs: Pump 6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P
--              PumpAMM pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA

WITH pump_instructions AS (
    SELECT
        ic.block_time,
        ic.block_slot,
        ic.block_date,
        ic.tx_index,
        ic.tx_id,
        ic.outer_instruction_index,
        ic.inner_instruction_index,
        CAST(NULL AS integer) AS event_index,
        ic.tx_signer,
        ic.tx_success,
        ic.executing_account,
        ic.outer_executing_account,
        ic.inner_executing_account,
        ic.is_inner,
        ic.stack_height,
        to_hex(ic.data) AS instruction_data_hex,
        CAST(json_format(CAST(ic.account_arguments AS json)) AS varchar) AS account_arguments_json,
        CAST(NULL AS bigint) AS fee_lamports,
        CAST(NULL AS bigint) AS compute_units_consumed,
        CAST(NULL AS bigint) AS cost_units,
        CAST(json_format(CAST(ic.log_messages AS json)) AS varchar) AS log_messages_json,
        CAST(NULL AS varchar) AS pre_balances_json,
        CAST(NULL AS varchar) AS post_balances_json,
        CAST(NULL AS varchar) AS pre_token_balances_json,
        CAST(NULL AS varchar) AS post_token_balances_json,
        CAST(NULL AS varchar) AS error_message,
        'solana.instruction_calls' AS source_table,
        'block_time' AS source_partition_col,
        CASE
            WHEN ic.executing_account = '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P' THEN 'pump'
            ELSE 'pump_amm'
        END AS protocol,
        '1' AS protocol_version,
        '1.0.0' AS schema_version,
        CAST(NULL AS varchar) AS raw_payload_json
    FROM solana.instruction_calls ic
    WHERE ic.block_time >= TIMESTAMP '{{start_ts}}'
      AND ic.block_time < TIMESTAMP '{{end_ts}}'
      AND ic.executing_account IN (
          '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P',
          'pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA'
      )
),
pump_transactions AS (
    SELECT
        t.block_time,
        t.block_slot,
        t.block_date,
        t.index AS tx_index,
        t.id AS tx_id,
        CAST(NULL AS integer) AS outer_instruction_index,
        CAST(NULL AS integer) AS inner_instruction_index,
        CAST(NULL AS integer) AS event_index,
        t.signer AS tx_signer,
        t.success AS tx_success,
        CAST(NULL AS varchar) AS executing_account,
        CAST(NULL AS varchar) AS outer_executing_account,
        CAST(NULL AS varchar) AS inner_executing_account,
        CAST(NULL AS boolean) AS is_inner,
        CAST(NULL AS bigint) AS stack_height,
        CAST(NULL AS varchar) AS instruction_data_hex,
        CAST(NULL AS varchar) AS account_arguments_json,
        t.fee AS fee_lamports,
        t.compute_units_consumed,
        t.cost_units,
        CAST(json_format(CAST(t.log_messages AS json)) AS varchar) AS log_messages_json,
        CAST(json_format(CAST(t.pre_balances AS json)) AS varchar) AS pre_balances_json,
        CAST(json_format(CAST(t.post_balances AS json)) AS varchar) AS post_balances_json,
        CAST(json_format(CAST(t.pre_token_balances AS json)) AS varchar) AS pre_token_balances_json,
        CAST(json_format(CAST(t.post_token_balances AS json)) AS varchar) AS post_token_balances_json,
        t.error AS error_message,
        'solana.transactions' AS source_table,
        'block_time' AS source_partition_col,
        'solana' AS protocol,
        '1' AS protocol_version,
        '1.0.0' AS schema_version,
        CAST(NULL AS varchar) AS raw_payload_json
    FROM solana.transactions t
    WHERE t.block_time >= TIMESTAMP '{{start_ts}}'
      AND t.block_time < TIMESTAMP '{{end_ts}}'
      AND EXISTS (
          SELECT 1
          FROM solana.instruction_calls ic
          WHERE ic.tx_id = t.id
            AND ic.block_time >= TIMESTAMP '{{start_ts}}'
            AND ic.block_time < TIMESTAMP '{{end_ts}}'
            AND ic.executing_account IN (
                '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P',
                'pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA'
            )
      )
)
SELECT * FROM pump_instructions
UNION ALL
SELECT * FROM pump_transactions
