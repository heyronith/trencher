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

## Historical public datasets

```bash
uv run trencher data download-historical
```

Downloads finalized public corpora into `data/external/` (gitignored), verifies
publisher checksums when available, and writes provenance/inventory/registry
metadata under `manifests/external/` (committed). Sources:

- Hugging Face `Slinky21/Pumpfun_Memecoin_Corpus`
- Zenodo RED-PUMP-2026 (`21383616`)
- Zenodo RED-COHORT-2026 (`21765387`) archive + extracted tree
- Hugging Face `solarchive/solarchive` README/schemas/indexes only (no full chain)

## Feasibility audit

```bash
uv run trencher data audit-feasibility -v
```

Writes machine-readable reports and `FEASIBILITY.md` under `manifests/feasibility/`.
CI uses fixture-sized samples only; the full local corpus is not required.

## World-model historical enrichment

```bash
uv run trencher data probe-world-model-sources -v
uv run trencher data recover-world-model-data -v
```

Recovers microstructure, ownership/graph, global context, post-grad, and execution
layers into `data/external/world_model/` with manifests under
`manifests/world_model_data/`. BigQuery requires ADC/`GOOGLE_CLOUD_PROJECT`.
CI must pass `--skip-live` (no paid/external bulk queries).

## Architecture

| Layer | Role |
|-------|------|
| `src/trencher/schemas/` | Versioned PyArrow canonical schemas (v1.1) |
| `src/trencher/queries/` | Parameterized Dune SQL + `datasets.yaml` registry |
| `src/trencher/dune/` | Client, backfill engine, preflight, manifests, checkpoints |
| `src/trencher/data/` | Public historical dataset download, verify, inventory, registry |
| `data/raw/dune/` | Immutable Dune Parquet output (gitignored) |
| `data/external/` | Downloaded public corpora (gitignored) |
| `manifests/external/` | Provenance + inventory metadata (committed) |

**Datasets:** `token_lifecycle`, `trades_events`, `dex_trades`, `transaction_microstructure`, `token_transfers`, `wallet_funding`, `wallet_graph`, `market_context`, `execution_liquidity`

**Dune sources:** `pumpdotfun_solana.*` (including v2 where exposed), `dex_solana.trades`, `solana.instruction_calls`, `solana.transactions`, `solana.blocks`, `tokens_solana.transfers`.

Each partition manifest records SQL hash, parameters (including the query window), Dune execution ID, row counts, source time bounds, and file checksums.
