-- Pump token lifecycle: creation + migration + anchor events.
-- Verified source tables: pumpdotfun_solana.pump_call_create, pump_call_migrate,
-- pump_evt_createevent, pump_evt_completeevent (Dune IDL decoded, call_/evt_ prefixes).

WITH creates AS (
    SELECT
        call_block_time AS block_time,
        call_block_slot AS block_slot,
        CAST(call_block_time AS date) AS block_date,
        call_tx_index AS tx_index,
        call_tx_id AS tx_id,
        call_outer_instruction_index AS outer_instruction_index,
        call_inner_instruction_index AS inner_instruction_index,
        CAST(NULL AS integer) AS event_index,
        'create_call' AS event_type,
        account_mint AS mint,
        call_tx_signer AS creator_wallet,
        name,
        symbol,
        uri,
        account_bondingCurve AS bonding_curve,
        account_associatedBondingCurve AS associated_bonding_curve,
        false AS migrated,
        CAST(NULL AS varchar) AS migration_tx_id,
        call_tx_signer,
        call_tx_success AS call_tx_success,
        call_outer_executing_account,
        call_inner_executing_account,
        'pumpdotfun_solana.pump_call_create' AS source_table,
        'call_block_time' AS source_partition_col,
        'pump' AS protocol,
        '1' AS protocol_version,
        '1.0.0' AS schema_version,
        CAST(NULL AS varchar) AS raw_payload_json
    FROM pumpdotfun_solana.pump_call_create
    WHERE call_block_time >= TIMESTAMP '{{start_ts}}'
      AND call_block_time < TIMESTAMP '{{end_ts}}'
),
migrations AS (
    SELECT
        call_block_time AS block_time,
        call_block_slot AS block_slot,
        CAST(call_block_time AS date) AS block_date,
        call_tx_index AS tx_index,
        call_tx_id AS tx_id,
        call_outer_instruction_index AS outer_instruction_index,
        call_inner_instruction_index AS inner_instruction_index,
        CAST(NULL AS integer) AS event_index,
        'migrate_call' AS event_type,
        account_mint AS mint,
        call_tx_signer AS creator_wallet,
        CAST(NULL AS varchar) AS name,
        CAST(NULL AS varchar) AS symbol,
        CAST(NULL AS varchar) AS uri,
        account_bondingCurve AS bonding_curve,
        account_associatedBondingCurve AS associated_bonding_curve,
        true AS migrated,
        call_tx_id AS migration_tx_id,
        call_tx_signer,
        call_tx_success AS call_tx_success,
        call_outer_executing_account,
        call_inner_executing_account,
        'pumpdotfun_solana.pump_call_migrate' AS source_table,
        'call_block_time' AS source_partition_col,
        'pump' AS protocol,
        '1' AS protocol_version,
        '1.0.0' AS schema_version,
        CAST(NULL AS varchar) AS raw_payload_json
    FROM pumpdotfun_solana.pump_call_migrate
    WHERE call_block_time >= TIMESTAMP '{{start_ts}}'
      AND call_block_time < TIMESTAMP '{{end_ts}}'
),
create_events AS (
    SELECT
        evt_block_time AS block_time,
        evt_block_slot AS block_slot,
        CAST(evt_block_time AS date) AS block_date,
        evt_tx_index AS tx_index,
        evt_tx_id AS tx_id,
        evt_outer_instruction_index AS outer_instruction_index,
        evt_inner_instruction_index AS inner_instruction_index,
        evt_index AS event_index,
        'create_event' AS event_type,
        mint,
        user AS creator_wallet,
        name,
        symbol,
        uri,
        bonding_curve,
        CAST(NULL AS varchar) AS associated_bonding_curve,
        false AS migrated,
        CAST(NULL AS varchar) AS migration_tx_id,
        evt_tx_signer AS call_tx_signer,
        evt_tx_success AS call_tx_success,
        evt_outer_executing_account AS call_outer_executing_account,
        evt_inner_executing_account AS call_inner_executing_account,
        'pumpdotfun_solana.pump_evt_createevent' AS source_table,
        'evt_block_time' AS source_partition_col,
        'pump' AS protocol,
        '1' AS protocol_version,
        '1.0.0' AS schema_version,
        CAST(NULL AS varchar) AS raw_payload_json
    FROM pumpdotfun_solana.pump_evt_createevent
    WHERE evt_block_time >= TIMESTAMP '{{start_ts}}'
      AND evt_block_time < TIMESTAMP '{{end_ts}}'
),
complete_events AS (
    SELECT
        evt_block_time AS block_time,
        evt_block_slot AS block_slot,
        CAST(evt_block_time AS date) AS block_date,
        evt_tx_index AS tx_index,
        evt_tx_id AS tx_id,
        evt_outer_instruction_index AS outer_instruction_index,
        evt_inner_instruction_index AS inner_instruction_index,
        evt_index AS event_index,
        'complete_event' AS event_type,
        mint,
        user AS creator_wallet,
        CAST(NULL AS varchar) AS name,
        CAST(NULL AS varchar) AS symbol,
        CAST(NULL AS varchar) AS uri,
        bonding_curve,
        CAST(NULL AS varchar) AS associated_bonding_curve,
        true AS migrated,
        evt_tx_id AS migration_tx_id,
        evt_tx_signer AS call_tx_signer,
        evt_tx_success AS call_tx_success,
        evt_outer_executing_account AS call_outer_executing_account,
        evt_inner_executing_account AS call_inner_executing_account,
        'pumpdotfun_solana.pump_evt_completeevent' AS source_table,
        'evt_block_time' AS source_partition_col,
        'pump' AS protocol,
        '1' AS protocol_version,
        '1.0.0' AS schema_version,
        CAST(NULL AS varchar) AS raw_payload_json
    FROM pumpdotfun_solana.pump_evt_completeevent
    WHERE evt_block_time >= TIMESTAMP '{{start_ts}}'
      AND evt_block_time < TIMESTAMP '{{end_ts}}'
)
SELECT * FROM creates
UNION ALL
SELECT * FROM migrations
UNION ALL
SELECT * FROM create_events
UNION ALL
SELECT * FROM complete_events
