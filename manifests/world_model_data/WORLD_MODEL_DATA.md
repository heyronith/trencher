# World Model Historical Data

Generated at: `2026-09-03T18:44:31.138241+00:00`

## Coverage matrix

| Layer | tokens | wallets | rows/edges | date coverage | source | quality |
| ----- | -----: | ------: | ---------: | ------------- | ------ | ------- |
| transaction_microstructure | 82 |  | 2500 | 2026-06-05..2026-07-14 (enriched subset) | pumpfun.tx_signature + solana_rpc.getTransaction | high_subset |
| token_ownership_transfers | 2627 | 332489 | 559703 | 2026-06-05..2026-07-14 | bigquery.Token Transfers (rich mints) | ledger |
| wallet_history_graph | 2721 | 1006640 | 11550229 | corpus window | pumpfun + red_cohort + bq SPL transfers + rpc proxies | strong_with_spl_funding |
| global_market_network_context | 798430 |  | 140 | 2026-06-05..2026-07-14 | pumpfun daily + geckoterminal SOL/USDC + bigquery.Blocks | high |
| post_graduation | 5669 |  | 1392133 | graduated subset | pumpfun.postgard_* (+ gecko pool OHLCV) | high_local_postgard |
| execution_state | 1173 |  | 182297 | rich subset pre-grad | pumpfun.trades curve reserves | curve_observed_subset |
| account_ownership_snapshots | 100 | 4 | 222 | 2026-06..2026-07 | bigquery.Accounts | ledger_snapshots |

## Key metrics

- True SPL transfer count: 559703
- Mints with genuine ownership histories: 2627
- Wallets with prior history: 1006640
- Genuine wallet funding edges: 523086
- Wallets with genuine funding edges: 332442
- Shared-funder graph edges: 335850
- Creator ownership transfer rows: 107896
- Transaction microstructure: 82 tokens / 2500 txs (match rate=1.0)
- Execution-state coverage: 1173 mints / 182297 rows
- Account snapshots: 100 mints / 222 rows
- BigQuery bytes processed / billed: 656323392556 / 656323641344
- Solarchive bytes downloaded: 0

## Encoder readiness

- **Event Encoder**: `READY`
- **Wallet/Graph Encoder**: `READY`
- **Token/Ownership/Curve Encoder**: `READY`
- **Global Market Encoder**: `READY`
- **Fusion/world-state model**: `PARTIAL`
- **prediction heads**: `READY`

**World-model dataset ready (strict core encoders):** True

## Remaining missing / irrecoverable under current constraints

- Native SOL system-program transfer ledger (Instructions SELECT ~7.7 TiB for corpus window; cost-aborted)
- Full Transactions table scans for microstructure (~370+ GiB/day for signature column alone; use RPC subset instead)
- Solarchive txs/accounts for June–July 2026 (index ends 2025-12-05)
- Real pool addresses for synthetic/null migrations without discovery
- Row-level Pump fee bps history (protocol metadata only)
- Transaction microstructure still covers only 82 tokens / 2500 enriched txs — not the full 798k universe

## Source decisions

- BigQuery: `usable_with_partition_filters`
- Solarchive: `skip_no_corpus_coverage`
- GeckoTerminal: `usable`
