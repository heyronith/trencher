# Sequence Dataset v1 (`sequence_dataset_v1`)

Generated at: `2026-09-04T19:44:01.287626+00:00`
Freeze: `world_model_freeze_v1`
PIT engine: `pit_v4`

## Sequence row counts

- `events`: **134864**
- `curve`: **102164**
- `microstructure`: **15241**
- `ownership`: **10797**
- `graph_token`: **143621**
- `graph_wallet_prior`: **3210201**
- `global_trade_second_buckets`: **2425001**
- `global_launches`: **798430**
- `global_migrations`: **5701**

## Observation refs / causality

- observation refs: **51353**
- post_t_violations: **0**
- ordering_violations: **0**
- causal_boundary_ok: **True**
- stratified_sample_all_passed: **True**
- disk MB: 1035.025

## Known limitations

- Wallet-prior corpus history uses Arrow-filtered full-corpus scan with per-wallet and global row budgets for local feasibility (cap min(256,2048)).
- Priority-fee fields were not recovered in the freeze; left null.
- Real bonding-curve reserves are not in local trade artifacts; left null.
- Trade slots are unavailable on Pump trade parquet; left null.
- RED-COHORT membership is intentionally excluded from temporal graph inputs.
- Lifetime edges_wallet_token aggregates are not used as temporal inputs.
- Global trade flow uses completed-second buckets only (<1s latency).
- Ownership RPC rows remain sampled-tx evidence, not a full holder ledger.
