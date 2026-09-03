-- Pump token lifecycle: creation + migration + anchor events (legacy + v2).
-- Creator identity: decoded `creator` arg / CreateEvent.creator / account_user.
-- Fee payer is preserved separately as call_tx_signer.

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
    'create' AS instruction_name,
    '181ec828051c0777' AS instruction_discriminator,
    account_mint AS mint,
    COALESCE(creator, account_user) AS creator_wallet,
    account_user AS user_wallet,
    name,
    symbol,
    uri,
    account_bondingCurve AS bonding_curve,
    account_associatedBondingCurve AS associated_bonding_curve,
    CAST(NULL AS varchar) AS quote_mint,
    COALESCE(/* optional-col:pumpdotfun_solana.pump_call_create.account_tokenProgram:varchar */ account_tokenProgram /* /optional-col:pumpdotfun_solana.pump_call_create.account_tokenProgram */, 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA') AS token_program,
    COALESCE(/* optional-col:pumpdotfun_solana.pump_call_create.account_tokenProgram:varchar */ account_tokenProgram /* /optional-col:pumpdotfun_solana.pump_call_create.account_tokenProgram */, 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA')
        = 'TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb' AS is_token_2022,
    false AS migrated,
    CAST(NULL AS varchar) AS migration_tx_id,
    CAST(NULL AS varchar) AS pool,
    call_tx_signer,
    call_tx_success,
    call_outer_executing_account,
    call_inner_executing_account,
    'pumpdotfun_solana.pump_call_create' AS source_table,
    'call_block_time' AS source_partition_col,
    'pump' AS protocol,
    '1' AS protocol_version,
    '1.1.0' AS schema_version,
    CONCAT(
        '{"instruction":"create","mint":"', CAST(account_mint AS varchar),
        '","creator":"', CAST(COALESCE(creator, account_user) AS varchar),
        '","user":"', CAST(account_user AS varchar), '"}'
    ) AS raw_payload_json
FROM pumpdotfun_solana.pump_call_create
WHERE call_block_time >= TIMESTAMP '{{start_ts}}'
  AND call_block_time < TIMESTAMP '{{end_ts}}'

UNION ALL

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
    'migrate' AS instruction_name,
    '9beae792ec9ea21e' AS instruction_discriminator,
    account_mint AS mint,
    COALESCE(account_user, call_tx_signer) AS creator_wallet,
    account_user AS user_wallet,
    CAST(NULL AS varchar) AS name,
    CAST(NULL AS varchar) AS symbol,
    CAST(NULL AS varchar) AS uri,
    account_bondingCurve AS bonding_curve,
    account_associatedBondingCurve AS associated_bonding_curve,
    CAST(NULL AS varchar) AS quote_mint,
    CAST(NULL AS varchar) AS token_program,
    CAST(NULL AS boolean) AS is_token_2022,
    true AS migrated,
    call_tx_id AS migration_tx_id,
    CAST(NULL AS varchar) AS pool,
    call_tx_signer,
    call_tx_success,
    call_outer_executing_account,
    call_inner_executing_account,
    'pumpdotfun_solana.pump_call_migrate' AS source_table,
    'call_block_time' AS source_partition_col,
    'pump' AS protocol,
    '1' AS protocol_version,
    '1.1.0' AS schema_version,
    CONCAT('{"instruction":"migrate","mint":"', CAST(account_mint AS varchar), '"}') AS raw_payload_json
FROM pumpdotfun_solana.pump_call_migrate
WHERE call_block_time >= TIMESTAMP '{{start_ts}}'
  AND call_block_time < TIMESTAMP '{{end_ts}}'

UNION ALL

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
    'CreateEvent' AS instruction_name,
    CAST(NULL AS varchar) AS instruction_discriminator,
    mint,
    COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_createevent.creator:varchar */ creator /* /optional-col:pumpdotfun_solana.pump_evt_createevent.creator */, "user") AS creator_wallet,
    "user" AS user_wallet,
    name,
    symbol,
    uri,
    bonding_curve,
    CAST(NULL AS varchar) AS associated_bonding_curve,
    COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_createevent.quote_mint:varchar */ quote_mint /* /optional-col:pumpdotfun_solana.pump_evt_createevent.quote_mint */, 'So11111111111111111111111111111111111111112') AS quote_mint,
    /* optional-col:pumpdotfun_solana.pump_evt_createevent.token_program:varchar */ token_program /* /optional-col:pumpdotfun_solana.pump_evt_createevent.token_program */ AS token_program,
    /* optional-col:pumpdotfun_solana.pump_evt_createevent.token_program:varchar */ token_program /* /optional-col:pumpdotfun_solana.pump_evt_createevent.token_program */ = 'TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb' AS is_token_2022,
    false AS migrated,
    CAST(NULL AS varchar) AS migration_tx_id,
    CAST(NULL AS varchar) AS pool,
    evt_tx_signer AS call_tx_signer,
    evt_tx_success AS call_tx_success,
    evt_outer_executing_account AS call_outer_executing_account,
    evt_inner_executing_account AS call_inner_executing_account,
    'pumpdotfun_solana.pump_evt_createevent' AS source_table,
    'evt_block_time' AS source_partition_col,
    'pump' AS protocol,
    CASE WHEN /* optional-col:pumpdotfun_solana.pump_evt_createevent.token_program:varchar */ token_program /* /optional-col:pumpdotfun_solana.pump_evt_createevent.token_program */ = 'TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb' THEN '2' ELSE '1' END
        AS protocol_version,
    '1.1.0' AS schema_version,
    CONCAT(
        '{"event":"CreateEvent","mint":"', CAST(mint AS varchar),
        '","creator":"', CAST(COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_createevent.creator:varchar */ creator /* /optional-col:pumpdotfun_solana.pump_evt_createevent.creator */, "user") AS varchar), '"}'
    ) AS raw_payload_json
FROM pumpdotfun_solana.pump_evt_createevent
WHERE evt_block_time >= TIMESTAMP '{{start_ts}}'
  AND evt_block_time < TIMESTAMP '{{end_ts}}'

UNION ALL

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
    'CompleteEvent' AS instruction_name,
    CAST(NULL AS varchar) AS instruction_discriminator,
    mint,
    "user" AS creator_wallet,
    "user" AS user_wallet,
    CAST(NULL AS varchar) AS name,
    CAST(NULL AS varchar) AS symbol,
    CAST(NULL AS varchar) AS uri,
    bonding_curve,
    CAST(NULL AS varchar) AS associated_bonding_curve,
    COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_completeevent.quote_mint:varchar */ quote_mint /* /optional-col:pumpdotfun_solana.pump_evt_completeevent.quote_mint */, 'So11111111111111111111111111111111111111112') AS quote_mint,
    CAST(NULL AS varchar) AS token_program,
    CAST(NULL AS boolean) AS is_token_2022,
    true AS migrated,
    evt_tx_id AS migration_tx_id,
    CAST(NULL AS varchar) AS pool,
    evt_tx_signer AS call_tx_signer,
    evt_tx_success AS call_tx_success,
    evt_outer_executing_account AS call_outer_executing_account,
    evt_inner_executing_account AS call_inner_executing_account,
    'pumpdotfun_solana.pump_evt_completeevent' AS source_table,
    'evt_block_time' AS source_partition_col,
    'pump' AS protocol,
    '1' AS protocol_version,
    '1.1.0' AS schema_version,
    CONCAT('{"event":"CompleteEvent","mint":"', CAST(mint AS varchar), '"}') AS raw_payload_json
FROM pumpdotfun_solana.pump_evt_completeevent
WHERE evt_block_time >= TIMESTAMP '{{start_ts}}'
  AND evt_block_time < TIMESTAMP '{{end_ts}}'

/* optional:pumpdotfun_solana.pump_call_create_v2 */
UNION ALL

SELECT
    call_block_time AS block_time,
    call_block_slot AS block_slot,
    CAST(call_block_time AS date) AS block_date,
    call_tx_index AS tx_index,
    call_tx_id AS tx_id,
    call_outer_instruction_index AS outer_instruction_index,
    call_inner_instruction_index AS inner_instruction_index,
    CAST(NULL AS integer) AS event_index,
    'create_v2_call' AS event_type,
    'create_v2' AS instruction_name,
    'd6904cec5f8b31b4' AS instruction_discriminator,
    account_mint AS mint,
    COALESCE(/* optional-col:pumpdotfun_solana.pump_call_create_v2.creator:varchar */ creator /* /optional-col:pumpdotfun_solana.pump_call_create_v2.creator */, account_user) AS creator_wallet,
    account_user AS user_wallet,
    name,
    symbol,
    uri,
    account_bondingCurve AS bonding_curve,
    account_associatedBondingCurve AS associated_bonding_curve,
    CAST(NULL AS varchar) AS quote_mint,
    COALESCE(/* optional-col:pumpdotfun_solana.pump_call_create_v2.account_tokenProgram:varchar */ account_tokenProgram /* /optional-col:pumpdotfun_solana.pump_call_create_v2.account_tokenProgram */, 'TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb') AS token_program,
    true AS is_token_2022,
    false AS migrated,
    CAST(NULL AS varchar) AS migration_tx_id,
    CAST(NULL AS varchar) AS pool,
    call_tx_signer,
    call_tx_success,
    call_outer_executing_account,
    call_inner_executing_account,
    'pumpdotfun_solana.pump_call_create_v2' AS source_table,
    'call_block_time' AS source_partition_col,
    'pump' AS protocol,
    '2' AS protocol_version,
    '1.1.0' AS schema_version,
    CONCAT(
        '{"instruction":"create_v2","mint":"', CAST(account_mint AS varchar),
        '","creator":"', CAST(COALESCE(/* optional-col:pumpdotfun_solana.pump_call_create_v2.creator:varchar */ creator /* /optional-col:pumpdotfun_solana.pump_call_create_v2.creator */, account_user) AS varchar), '"}'
    ) AS raw_payload_json
FROM pumpdotfun_solana.pump_call_create_v2
WHERE call_block_time >= TIMESTAMP '{{start_ts}}'
  AND call_block_time < TIMESTAMP '{{end_ts}}'
/* /optional:pumpdotfun_solana.pump_call_create_v2 */

/* optional:pumpdotfun_solana.pump_evt_completepumpammmigrationevent */
UNION ALL

SELECT
    evt_block_time AS block_time,
    evt_block_slot AS block_slot,
    CAST(evt_block_time AS date) AS block_date,
    evt_tx_index AS tx_index,
    evt_tx_id AS tx_id,
    evt_outer_instruction_index AS outer_instruction_index,
    evt_inner_instruction_index AS inner_instruction_index,
    evt_index AS event_index,
    'complete_pump_amm_migration_event' AS event_type,
    'CompletePumpAmmMigrationEvent' AS instruction_name,
    CAST(NULL AS varchar) AS instruction_discriminator,
    mint,
    "user" AS creator_wallet,
    "user" AS user_wallet,
    CAST(NULL AS varchar) AS name,
    CAST(NULL AS varchar) AS symbol,
    CAST(NULL AS varchar) AS uri,
    bonding_curve,
    CAST(NULL AS varchar) AS associated_bonding_curve,
    COALESCE(/* optional-col:pumpdotfun_solana.pump_evt_completepumpammmigrationevent.quote_mint:varchar */ quote_mint /* /optional-col:pumpdotfun_solana.pump_evt_completepumpammmigrationevent.quote_mint */, 'So11111111111111111111111111111111111111112') AS quote_mint,
    CAST(NULL AS varchar) AS token_program,
    CAST(NULL AS boolean) AS is_token_2022,
    true AS migrated,
    evt_tx_id AS migration_tx_id,
    pool,
    evt_tx_signer AS call_tx_signer,
    evt_tx_success AS call_tx_success,
    evt_outer_executing_account AS call_outer_executing_account,
    evt_inner_executing_account AS call_inner_executing_account,
    'pumpdotfun_solana.pump_evt_completepumpammmigrationevent' AS source_table,
    'evt_block_time' AS source_partition_col,
    'pump' AS protocol,
    '1' AS protocol_version,
    '1.1.0' AS schema_version,
    CONCAT(
        '{"event":"CompletePumpAmmMigrationEvent","mint":"', CAST(mint AS varchar),
        '","pool":"', CAST(pool AS varchar), '"}'
    ) AS raw_payload_json
FROM pumpdotfun_solana.pump_evt_completepumpammmigrationevent
WHERE evt_block_time >= TIMESTAMP '{{start_ts}}'
  AND evt_block_time < TIMESTAMP '{{end_ts}}'
/* /optional:pumpdotfun_solana.pump_evt_completepumpammmigrationevent */
