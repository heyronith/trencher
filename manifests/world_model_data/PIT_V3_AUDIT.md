# PIT v3 Audit (`pit_v3`)

Generated at: `2026-09-04T11:58:13.972507+00:00`
Acquisition freeze: `world_model_freeze_v1` (unchanged)
Fusion status: `MASKED`

## As-of-T ownership by age

- `launch`: {'missing': 37, 'transfer_ledger_observed_partial': 4, 'rpc_balance_observed': 9}
- `10s`: {'trade_derived_proxy': 12, 'holder_state_reconstructable': 2, 'transfer_ledger_observed_partial': 4, 'rpc_balance_observed': 26, 'missing': 6}
- `30s`: {'trade_derived_proxy': 8, 'transfer_ledger_observed_partial': 7, 'holder_state_reconstructable': 2, 'rpc_balance_observed': 28, 'missing': 5}
- `1m`: {'trade_derived_proxy': 7, 'transfer_ledger_observed_partial': 8, 'holder_state_reconstructable': 2, 'rpc_balance_observed': 28, 'missing': 5}
- `2m`: {'trade_derived_proxy': 6, 'transfer_ledger_observed_partial': 9, 'holder_state_reconstructable': 2, 'rpc_balance_observed': 28, 'missing': 5}
- `5m`: {'transfer_ledger_observed_partial': 12, 'holder_state_reconstructable': 2, 'rpc_balance_observed': 29, 'missing': 4, 'trade_derived_proxy': 3}
- `10m`: {'transfer_ledger_observed_partial': 11, 'holder_state_reconstructable': 3, 'rpc_balance_observed': 30, 'missing': 3, 'trade_derived_proxy': 3}
- `30m`: {'holder_state_reconstructable': 5, 'transfer_ledger_observed_partial': 9, 'rpc_balance_observed': 30, 'missing': 3, 'trade_derived_proxy': 3}

## Holder-state reconstructable at T

- `launch`: **0**
- `10s`: **2**
- `30s`: **2**
- `1m`: **2**
- `2m`: **2**
- `5m`: **2**
- `10m`: **3**
- `30m`: **5**

## Global context coverage

- `launch`: 50
- `10s`: 50
- `30s`: 50
- `1m`: 50
- `2m`: 50
- `5m`: 50
- `10m`: 50
- `30m`: 50

## Leakage / structural checks

- Leakage all passed: **True**
- Future labels absent from model_state: **True**
- generated_at only in provenance: **True**

## Known remaining limitations

- Full-history holder_state_reconstructable remains rare; as-of-T is usually lower.
- Window unique mint/wallet counts are not materialized in v3 second buckets (volume/flow only).
- RPC ownership remains sampled-tx evidence.
- Native SOL funding covers a wallet subset only.
- Fusion remains MASKED; do not claim full ownership coverage.
- No supervised labels, splits, or models in this audit.
