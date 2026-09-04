# World Model Historical Data

Generated at: `2026-09-04T03:28:30.696355+00:00`

## Coverage matrix

| Layer | tokens | wallets | rows/edges | date coverage | source | quality |
| ----- | -----: | ------: | ---------: | ------------- | ------ | ------- |
| transaction_microstructure | 3215 |  | 17741 | 2026-06-05..2026-07-14 (enriched subset) | pumpfun.tx_signature + solana_rpc.getTransaction | high_subset |
| token_ownership_transfers | 2936 | 334012 | 562175 | representative corpus days (partition-pruned) | bigquery.Token Transfers (unbiased rich mints) | ledger |
| wallet_history_graph | 4000 | 1006640 | 11481707 | corpus window | pumpfun + red_cohort + bq SPL + native SOL system transfers | strong_with_spl_and_native_sol |
| global_market_network_context | 798430 |  | 140 | 2026-06-05..2026-07-14 | pumpfun daily + geckoterminal SOL/USDC + bigquery.Blocks | high |
| post_graduation | 5669 |  | 1392133 | graduated subset (label, not fusion inclusion) | pumpfun.postgard_* (+ gecko pool OHLCV) | high_local_postgard |
| execution_state | 3133 |  | 102164 | unbiased rich subset | pumpfun.trades curve reserves | curve_observed_subset |
| account_ownership_snapshots | 100 | 4 | 222 | 2026-06..2026-07 | bigquery.Accounts | ledger_snapshots |
| native_sol_funding | 84 | 517 | 716 | pre-T wallet history | solana_rpc.getSignaturesForAddress + parsed System Program transfer | native_sol |

## Key metrics

- True SPL transfer count: 562175
- Mints with genuine ownership histories: 2936
- Unbiased rich subset: 4000 (graduated=20, non-graduated=3980)
- Fusion multimodal overlap: 284 (graduated=8, non-graduated=276)
- Selection is not a graduation proxy: True
- Wallets with prior history: 1006640
- SPL transfer edges (not native SOL funding): 525053
- True native-SOL funding edges: 716
- Wallets with native-SOL funding: 517
- Shared-funder graph edges (SPL): 337457
- Shared native funder edges: 2115
- Creator ownership transfer rows: 108199
- Transaction microstructure: 3215 tokens / 17741 txs (match rate=1.0)
- Execution-state coverage: 3133 mints / 102164 rows
- Account snapshots: 100 mints / 222 rows
- BigQuery bytes processed / billed: 176934370471 / 176940908544
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

- Full-universe native SOL Instructions table (SELECT ~7.7 TiB; recovered only via RPC System Program parses on subset)
- Full Transactions table scans for microstructure (~370+ GiB/day for signature column alone; use RPC subset instead)
- Solarchive txs/accounts for June–July 2026 (index ends 2025-12-05)
- Real pool addresses for synthetic/null migrations without discovery
- Row-level Pump fee bps history (protocol metadata only)
- Complete fusion population is the multimodal overlap (284 tokens), not the full 798k universe

## Source decisions

- BigQuery: `usable_with_partition_filters`
- Solarchive: `skip_no_corpus_coverage`
- GeckoTerminal: `usable`
