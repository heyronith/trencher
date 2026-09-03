-- Raw Pump / PumpSwap instructions and parent transactions.
-- Preserves instruction bytes, accounts, full tx instructions, ALTs, signers,
-- hashes, return data, and pre/post balances for future re-decoding.
-- Verified: solana.instruction_calls, solana.transactions (docs.dune.com/data-catalog/solana).

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
        'instruction' AS record_type,
        CASE substr(to_hex(ic.data), 1, 16)
            WHEN '181ec828051c0777' THEN 'create'
            WHEN 'd6904cec5f8b31b4' THEN 'create_v2'
            WHEN '66063d1201daebea' THEN 'buy'
            WHEN 'b817ee6167c5d33d' THEN 'buy_v2'
            WHEN '38fc74089edfcd5f' THEN 'buy_exact_sol_in'
            WHEN '33e685a4017f83ad' THEN 'sell'
            WHEN '5df6823ce7e940b2' THEN 'sell_v2'
            WHEN '9beae792ec9ea21e' THEN 'migrate'
        END AS instruction_name,
        substr(to_hex(ic.data), 1, 16) AS instruction_discriminator,
        ic.tx_signer,
        ic.tx_success,
        CAST(NULL AS varchar) AS tx_version,
        ic.block_hash,
        CAST(NULL AS varchar) AS recent_block_hash,
        ic.executing_account,
        ic.outer_executing_account,
        ic.inner_executing_account,
        ic.is_inner,
        ic.stack_height,
        to_hex(ic.data) AS instruction_data_hex,
        CAST(json_format(CAST(ic.account_arguments AS json)) AS varchar) AS account_arguments_json,
        CAST(json_format(CAST(ic.inner_instructions AS json)) AS varchar) AS inner_instructions_json,
        CAST(NULL AS varchar) AS instructions_json,
        CAST(NULL AS varchar) AS account_keys_json,
        CAST(NULL AS varchar) AS signatures_json,
        CAST(NULL AS varchar) AS signers_json,
        CAST(NULL AS varchar) AS address_table_lookups_json,
        CAST(NULL AS varchar) AS loaded_addresses_json,
        CAST(NULL AS varchar) AS return_data_program_id,
        CAST(NULL AS varchar) AS return_data_json,
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
        CASE substr(to_hex(ic.data), 1, 16)
            WHEN 'd6904cec5f8b31b4' THEN '2'
            WHEN 'b817ee6167c5d33d' THEN '2'
            WHEN '38fc74089edfcd5f' THEN '2'
            WHEN '5df6823ce7e940b2' THEN '2'
            ELSE '1'
        END AS protocol_version,
        '1.1.0' AS schema_version,
        CONCAT(
            '{"instruction_data_hex":"', COALESCE(to_hex(ic.data), ''),
            '","account_arguments":', COALESCE(json_format(CAST(ic.account_arguments AS json)), 'null'),
            ',"inner_instructions":', COALESCE(json_format(CAST(ic.inner_instructions AS json)), 'null'),
            '}'
        ) AS raw_payload_json
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
        'transaction' AS record_type,
        CAST(NULL AS varchar) AS instruction_name,
        CAST(NULL AS varchar) AS instruction_discriminator,
        t.signer AS tx_signer,
        t.success AS tx_success,
        t.version AS tx_version,
        t.block_hash,
        t.recent_block_hash,
        CAST(NULL AS varchar) AS executing_account,
        CAST(NULL AS varchar) AS outer_executing_account,
        CAST(NULL AS varchar) AS inner_executing_account,
        CAST(NULL AS boolean) AS is_inner,
        CAST(NULL AS bigint) AS stack_height,
        CAST(NULL AS varchar) AS instruction_data_hex,
        CAST(NULL AS varchar) AS account_arguments_json,
        CAST(NULL AS varchar) AS inner_instructions_json,
        CAST(json_format(CAST(t.instructions AS json)) AS varchar) AS instructions_json,
        CAST(json_format(CAST(t.account_keys AS json)) AS varchar) AS account_keys_json,
        CAST(json_format(CAST(t.signatures AS json)) AS varchar) AS signatures_json,
        CAST(json_format(CAST(t.signers AS json)) AS varchar) AS signers_json,
        CAST(json_format(CAST(t.address_table_lookups AS json)) AS varchar) AS address_table_lookups_json,
        CAST(json_format(CAST(t.loaded_addresses AS json)) AS varchar) AS loaded_addresses_json,
        /* optional-col:solana.transactions.return_data_program_id:varchar */ t.return_data_program_id /* /optional-col:solana.transactions.return_data_program_id */ AS return_data_program_id,
        /* optional-col:solana.transactions.return_data:varchar */ CAST(json_format(CAST(t.return_data AS json)) AS varchar) /* /optional-col:solana.transactions.return_data */ AS return_data_json,
        t.fee AS fee_lamports,
        t.compute_units_consumed,
        /* optional-col:solana.transactions.cost_units:bigint */ t.cost_units /* /optional-col:solana.transactions.cost_units */ AS cost_units,
        CAST(json_format(CAST(t.log_messages AS json)) AS varchar) AS log_messages_json,
        CAST(json_format(CAST(t.pre_balances AS json)) AS varchar) AS pre_balances_json,
        CAST(json_format(CAST(t.post_balances AS json)) AS varchar) AS post_balances_json,
        CAST(json_format(CAST(t.pre_token_balances AS json)) AS varchar) AS pre_token_balances_json,
        CAST(json_format(CAST(t.post_token_balances AS json)) AS varchar) AS post_token_balances_json,
        t.error AS error_message,
        'solana.transactions' AS source_table,
        'block_time' AS source_partition_col,
        'solana' AS protocol,
        COALESCE(t.version, 'legacy') AS protocol_version,
        '1.1.0' AS schema_version,
        CONCAT(
            '{"instructions":', COALESCE(json_format(CAST(t.instructions AS json)), 'null'),
            ',"account_keys":', COALESCE(json_format(CAST(t.account_keys AS json)), 'null'),
            ',"signatures":', COALESCE(json_format(CAST(t.signatures AS json)), 'null'),
            ',"signers":', COALESCE(json_format(CAST(t.signers AS json)), 'null'),
            ',"address_table_lookups":', COALESCE(json_format(CAST(t.address_table_lookups AS json)), 'null'),
            ',"loaded_addresses":', COALESCE(json_format(CAST(t.loaded_addresses AS json)), 'null'),
            ',"return_data_program_id":"', COALESCE(/* optional-col:solana.transactions.return_data_program_id:varchar */ t.return_data_program_id /* /optional-col:solana.transactions.return_data_program_id */, ''),
            '","return_data":', COALESCE(/* optional-col:solana.transactions.return_data:varchar */ json_format(CAST(t.return_data AS json)) /* /optional-col:solana.transactions.return_data */, 'null'),
            '}'
        ) AS raw_payload_json
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
