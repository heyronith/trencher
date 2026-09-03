# World Model Historical Data

Generated at: `2026-09-03T05:51:25.074936+00:00`

## Coverage matrix

| World-model layer | tokens covered | events/rows | date coverage | quality | source |
| ----------------- | -------------: | ----------: | ------------- | ------- | ------ |
| transaction_microstructure | 82 | 2500 | 2026-06-05..2026-07-14 (subset) | high | pumpfun.trades.tx_signature + solana_rpc.getTransaction |
| token_ownership_transfers | 798430 | 9644468 | corpus window via trades | trade_derived_only | pumpfun.trades (+ BigQuery Token Transfers if auth) |
| wallet_history_graph | 2721 | 10691293 | corpus window | strong_participation_partial_funding | pumpfun + red_cohort + rpc balance-delta proxies |
| global_market_network_context | 798430 | 79 | 2026-06-05..2026-07-14 | high_pump_aggregates + gecko_sol_usdc | pumpfun daily aggregates + geckoterminal SOL/USDC |
| post_graduation | 5669 | 1392133 | graduated subset | high_local_postgard | pumpfun.postgard_* (+ gecko pool OHLCV supplement) |
| execution_state | 1173 | 182297 | rich subset pre-grad | high_curve_reserves | pumpfun.trades/snapshots curve fields |

## Key metrics

- Transaction-signature match rate: 1.0
- Wallet–mint ownership/flow edges: 9644468
- Tokens with trade-derived ownership history: 798430
- Wallets with usable prior history: 1006640
- Wallets with funding-proxy edges: 1494
- Funding proxy edges: 2169
- Graph edge counts: `{'wallet_token': 9644468, 'creator_token': 798430, 'cobuy': 243261, 'cohort_member': 2965, 'sol_balance_delta_proxy': 2169}`
- Tokens with post-grad history: 5669
- Network-context corpus days: 38
- Execution-state coverage: 1173 mints / 182297 rows
- BigQuery bytes processed: 0
- Solarchive bytes downloaded: 0
- Gecko API calls / pools completed: 1 / 150
- RPC transactions found: 2500

## Encoder readiness

- **Event Encoder**: `READY`
- **Wallet/Graph Encoder**: `READY`
- **Token/Ownership/Curve Encoder**: `READY`
- **Global Market Encoder**: `READY`
- **Fusion/world-state model**: `READY`
- **prediction heads**: `READY`

**World-model dataset ready (core encoders):** True

## Remaining irrecoverable / blocked without new sources

- Full SPL/Token-2022 transfer ledger without BigQuery auth or Solarchive 2026
- Parsed System-program funding graph for all wallets (only RPC balance-delta proxies on subset)
- Exact historical priority-fee/Jito tip fields when RPC meta lacks them
- Solarchive txs/accounts for June–July 2026 (index ends 2025-12-05)
- Real pool addresses for synthetic_graduation_queue / null migrations without discovery
- Row-level Pump fee bps history (protocol metadata only)

## Source decisions

- BigQuery: `auth_missing`
- Solarchive: `skip_no_corpus_coverage`
- GeckoTerminal: `usable`
