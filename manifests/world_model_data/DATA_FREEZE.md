# Data Freeze `world_model_freeze_v1`

Generated at: `2026-09-04T11:07:26.344115+00:00`

## Policy

- Historical raw/canonical acquisition artifacts are **append-only**.
- No further external bulk acquisition unless explicitly requested.

## Fusion coverage (authoritative)

- `core_event_graph_micro_exec_market`: **3133**
- `core_plus_any_genuine_ownership`: **3132**
- `core_plus_reconstructable_ownership`: **287**
- `with_true_native_sol_history`: **84**
- `core_plus_native_sol`: **46**

## Ownership evidence levels

- `ledger_full_or_reconstructable`: **317**
- `rpc_balance_observed`: **2845**
- `trade_derived_proxy`: **1**
- `missing`: **837**

## BigQuery lifetime billed: **901.051 GiB**

## Artifacts

| artifact | rows | mints | quality | ownership level |
| -------- | ---: | ----: | ------- | --------------- |
| universe_tokens | 798430 | 798430 | authoritative_local |  |
| rich_subset_mints | 4000 | 4000 | outcome_independent |  |
| ownership_evidence | 4000 | 4000 | evidence_leveled | mixed |
| fusion_coverage_truth | 4000 | 4000 | source_of_truth |  |
| token_transfers_bq | 562175 | 2936 | ledger | ledger_full_or_reconstructable |
| token_transfers_unbiased_days | 2472 | 314 | ledger_day_extracts | ledger_full_or_reconstructable |
| rpc_token_balance_ownership | 5691 | 2845 | sampled_tx_observations | rpc_balance_observed |
| native_sol_funding | 716 | 84 | native_sol_pre_T |  |
| rpc_transactions | 17741 | 3215 | archival_rpc_subset |  |
| signature_targets | 15241 | 3133 | stratified_per_token |  |
| execution_curve_states | 102164 | 3133 | curve_observed_subset |  |
| edges_wallet_token | 9644468 | 617150 | trade_participation | trade_derived_proxy |
| daily_pump_network_context | 38 |  | derived_daily |  |
| blocks_daily | 61 |  | network_context |  |

## Known limitations

- RPC ownership observations are sampled-tx facts, not full ledgers.
- Reconstructable ownership multimodal overlap is much smaller than any-genuine-ownership overlap.
- Solarchive has no June–July 2026 coverage.
- Native SOL funding covers a wallet subset only.
