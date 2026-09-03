# Trencher

On-chain raw data extraction for the meme-coin world model. Pulls verified Pump / PumpSwap / Solana data from Dune into immutable, partitioned Parquet with full provenance. No social data.

## Setup

```bash
uv sync --all-extras
cp .env.example .env   # add DUNE_API_KEY for backfill / live preflight
```

## Validate (no Dune credits)

```bash
./scripts/ci.sh
# or manually:
uv run ruff check .
uv run pytest
uv run trencher dune doctor
```

## Live Dune preflight (low-cost, required before historical backfill)

```bash
export DUNE_API_KEY=...
uv run trencher dune validate-queries
```

This checks current Dune tables/columns via `information_schema`, persists the
enabled-table cache under the data root, and semantically compiles every rendered
dataset query with `EXPLAIN (TYPE VALIDATE)` (falling back to a zero-row compile
if Dune rejects EXPLAIN). It does not run the historical backfill.

## Backfill one day

```bash
export DUNE_API_KEY=...
uv run trencher dune backfill \
  --dataset token_lifecycle \
  --start 2025-01-01 \
  --end 2025-01-01 \
  --chunk-days 1
```

`--chunk-days` is the Dune query window. Results are still written as daily `dt=YYYY-MM-DD` partitions. Re-running a completed partition is skipped unless `--force` is passed.

Dry-run (no API calls):

```bash
uv run trencher dune backfill --dataset trades_events --start 2025-01-01 --end 2025-01-01 --dry-run
```

## Architecture

| Layer | Role |
|-------|------|
| `src/trencher/schemas/` | Versioned PyArrow canonical schemas (v1.1) |
| `src/trencher/queries/` | Parameterized Dune SQL + `datasets.yaml` registry |
| `src/trencher/dune/` | Client, backfill engine, preflight, manifests, checkpoints |
| `data/raw/dune/` | Immutable Parquet output (gitignored) |

**Datasets:** `token_lifecycle`, `trades_events`, `dex_trades`, `transaction_microstructure`, `token_transfers`, `wallet_funding`, `wallet_graph`, `market_context`, `execution_liquidity`

**Dune sources:** `pumpdotfun_solana.*` (including v2 where exposed), `dex_solana.trades`, `solana.instruction_calls`, `solana.transactions`, `solana.blocks`, `tokens_solana.transfers`.

Each partition manifest records SQL hash, parameters (including the query window), Dune execution ID, row counts, source time bounds, and file checksums.
