# Sequence Dataset v2 (`sequence_dataset_v2`)

Generated at: `2026-09-04T22:26:04.640002+00:00`
Freeze: `world_model_freeze_v1`
PIT engine: `pit_v4`
Supersedes: `sequence_dataset_v1`

## Integrity confirmations

- `future_derived_ownership_classification_in_atomic_rows`: **0**
- `post_t_referenced_rows`: **0**
- `ordering_violations`: **0**
- `arbitrary_scan_order_wallet_truncation`: **removed**
- `heterogeneous_same_signature_events_incorrectly_collapsed`: **0**
- `events_v1_would_collapse_but_v2_preserves`: **1332**
- `red_cohort_eventual_membership_as_early_feature`: **0**
- `lifetime_wallet_aggregates_as_temporal_inputs`: **0**

## Sequence row counts

- `events`: **136196**
- `curve`: **102164**
- `microstructure`: **15241**
- `ownership`: **10797**
- `graph_token`: **144953**
- `graph_wallet_prior`: **13444425**
- `global_trade_second_buckets`: **2425001**
- `global_launches`: **798430**
- `global_migrations`: **5701**

## Observation refs / causality

- observation refs: **51353**
- post_t_violations: **0**
- ordering_violations: **0**
- causal_boundary_ok: **True**
- usable wallet-prior observations: **42539**
- stratified_sample_all_passed: **True**
- disk MB: 3639.413

## Known limitations

- Wallet-prior store keeps complete focal-wallet history; last-N / temporal windows apply only at dereference.
- Priority-fee fields were not recovered in the freeze; left null.
- Real bonding-curve reserves are not in local trade artifacts; left null.
- Trade slots are unavailable on Pump trade parquet; left null.
- RED-COHORT membership is intentionally excluded from temporal graph inputs.
- Lifetime edges_wallet_token aggregates are not used as temporal inputs.
- Global trade flow uses completed-second buckets only (<1s latency).
- Ownership RPC rows remain sampled-tx evidence, not a full holder ledger.
- sequence_dataset_v1 artifacts are retained but superseded for model input.
