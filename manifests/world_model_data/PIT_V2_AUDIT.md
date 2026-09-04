# PIT v2 Audit (`pit_v2`)

Generated at: `2026-09-04T11:27:36.517953+00:00`
Acquisition freeze: `world_model_freeze_v1` (unchanged)

## Ownership evidence counts

- `holder_state_reconstructable`: **14**
- `transfer_ledger_observed_partial`: **303**
- `rpc_balance_observed`: **2845**
- `trade_derived_proxy`: **1**
- `missing`: **837**

- Truly reconstructable holder-state: **14**
- Core + any genuine ownership: **3132**
- Core + reconstructable holder-state: **13**

## Arbitrary-T reconstruction coverage

- Mints materialized: 50
- Rows materialized: 400
- Age `launch`: 50 mints with any modality
- Age `10s`: 50 mints with any modality
- Age `30s`: 50 mints with any modality
- Age `1m`: 50 mints with any modality
- Age `2m`: 50 mints with any modality
- Age `5m`: 50 mints with any modality
- Age `10m`: 50 mints with any modality
- Age `30m`: 50 mints with any modality

## Leakage tests

- All passed: **True**
- Mints tested: 5

## Known remaining limitations

- holder_state_reconstructable is rare under documented completeness; most BQ ledgers remain transfer_ledger_observed_partial.
- RPC token-balance ownership is sampled-tx evidence only.
- Lifetime SOL funding covers a wallet subset only.
- Global same-day context uses rolling subset launches + mint trades; full-universe same-day aggregates are not used.
- Frozen lifetime edges_wallet_token.parquet is lineage/QA only — never a PIT model input.
- No supervised labels, training splits, or models in this audit.
