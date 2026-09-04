# PIT v4 Audit (`pit_v4`)

Generated at: `2026-09-04T17:20:38.257720+00:00`
Acquisition freeze: `world_model_freeze_v1` (unchanged)
PIT engine: `pit_v4`
Fusion status: `MASKED`

## Same-second global trade policy

- Rule: last completed second only (`{'last_completed_second_rule': True, 'example_T': 1000.2, 'last_completed_second_at_example': 999, 'intentional_latency': '<1s', 'does_not_round_T_upward': True}`)

## Global trade stream cleaning

- `{'includes_user_wallet': True, 'excludes_system_program_wallet': True, 'excluded_system_program_wallet_rows': 6690901, 'skipped_missing_event_time_rows': 0, 'raw_local_artifacts_unchanged': True}`

## As-of-T ownership by age

- `launch`: {'transfer_ledger_observed_partial': 2, 'missing': 37, 'rpc_balance_observed': 11}
- `10s`: {'holder_state_reconstructable': 2, 'trade_derived_proxy': 7, 'transfer_ledger_observed_partial': 1, 'rpc_balance_observed': 34, 'missing': 6}
- `30s`: {'holder_state_reconstructable': 2, 'trade_derived_proxy': 5, 'transfer_ledger_observed_partial': 2, 'rpc_balance_observed': 36, 'missing': 5}
- `1m`: {'holder_state_reconstructable': 2, 'trade_derived_proxy': 4, 'transfer_ledger_observed_partial': 3, 'rpc_balance_observed': 36, 'missing': 5}
- `2m`: {'holder_state_reconstructable': 2, 'trade_derived_proxy': 4, 'transfer_ledger_observed_partial': 3, 'rpc_balance_observed': 36, 'missing': 5}
- `5m`: {'holder_state_reconstructable': 2, 'transfer_ledger_observed_partial': 4, 'rpc_balance_observed': 37, 'missing': 4, 'trade_derived_proxy': 3}
- `10m`: {'holder_state_reconstructable': 3, 'transfer_ledger_observed_partial': 3, 'rpc_balance_observed': 38, 'missing': 3, 'trade_derived_proxy': 3}
- `30m`: {'holder_state_reconstructable': 3, 'transfer_ledger_observed_partial': 3, 'rpc_balance_observed': 38, 'missing': 3, 'trade_derived_proxy': 3}
- `1h`: {'holder_state_reconstructable': 3, 'transfer_ledger_observed_partial': 3, 'rpc_balance_observed': 38, 'missing': 3, 'trade_derived_proxy': 3}
- `6h`: {'holder_state_reconstructable': 6, 'rpc_balance_observed': 38, 'missing': 3, 'trade_derived_proxy': 2, 'transfer_ledger_observed_partial': 1}

## Holder-state reconstructable at T

- `launch`: **0**
- `10s`: **2**
- `30s`: **2**
- `1m`: **2**
- `2m`: **2**
- `5m`: **2**
- `10m`: **3**
- `30m`: **3**
- `1h`: **3**
- `6h`: **6**

## Global context coverage

- `launch`: 50
- `10s`: 50
- `30s`: 50
- `1m`: 50
- `2m`: 50
- `5m`: 50
- `10m`: 50
- `30m`: 50
- `1h`: 50
- `6h`: 50

## Leakage / structural checks

- cases: 15
- all_passed: **True**
- same_second_bucket_case: **True**
- future fields absent: **True**

## Known limitations

- holder_state_reconstructable requires continuous queried day partitions launch→T; sparse day extracts usually force partial.
- Global Pump trade context intentionally lags by <1 completed second.
- Window unique mint/wallet counts are not materialized in second buckets (volume/flow only).
- RPC ownership remains sampled-tx evidence.
- Native SOL funding covers a wallet subset only.
- Fusion remains MASKED; do not claim full ownership coverage.
- No supervised labels, splits, or models in this audit.
