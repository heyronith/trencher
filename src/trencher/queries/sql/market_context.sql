-- Solana block-level network context.
-- Verified: solana.blocks (docs.dune.com/data-catalog/solana/blocks).

SELECT
    b.time AS block_time,
    b.slot AS block_slot,
    b.date AS block_date,
    b.height AS block_height,
    b.hash AS block_hash,
    b.total_transactions,
    b.successful_transactions,
    b.failed_transactions,
    b.total_non_vote_transactions,
    b.successful_non_vote_transactions,
    b.failed_non_vote_transactions,
    b.parent_slot,
    b.previous_block_hash,
    'solana.blocks' AS source_table,
    'time' AS source_partition_col,
    'solana' AS protocol,
    '1' AS protocol_version,
    '1.0.0' AS schema_version,
    CAST(NULL AS varchar) AS raw_payload_json
FROM solana.blocks b
WHERE b.time >= TIMESTAMP '{{start_ts}}'
  AND b.time < TIMESTAMP '{{end_ts}}'
