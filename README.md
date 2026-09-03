# Trencher

On-chain raw data extraction for the meme-coin world model. Pulls verified Pump / PumpSwap / Solana data from Dune into immutable, partitioned Parquet with full provenance. No social data.

## Setup

```bash
uv sync --all-extras
cp .env.example .env   # add DUNE_API_KEY for backfill
```

## Validate (no Dune credits)

```bash
./scripts/ci.sh
# or manually:
uv run ruff check .
uv run pytest
uv run trencher dune doctor
```

Enable GitHub Actions CI: copy `ci/github-actions-ci.yml.example` to `.github/workflows/ci.yml`.

## Backfill one day

```bash
export DUNE_API_KEY=...
uv run trencher dune backfill \
  --dataset token_lifecycle \
  --start 2025-01-01 \
  --end 2025-01-01
```

Partitions land at `data/raw/dune/<dataset>/dt=YYYY-MM-DD/part-00000.parquet` with a sidecar `manifest.json`. Re-running a completed partition is skipped unless `--force` is passed.

Dry-run (no API calls):

```bash
uv run trencher dune backfill --dataset trades_events --start 2025-01-01 --end 2025-01-01 --dry-run
```

## Architecture

| Layer | Role |
|-------|------|
| `src/trencher/schemas/` | Versioned PyArrow canonical schemas (8 raw dataset types) |
| `src/trencher/queries/` | Parameterized Dune SQL + `datasets.yaml` registry |
| `src/trencher/dune/` | Client, backfill engine, manifests, checkpoints |
| `data/raw/dune/` | Immutable Parquet output (gitignored) |

**Datasets:** `token_lifecycle`, `trades_events`, `transaction_microstructure`, `token_transfers`, `wallet_funding`, `wallet_graph`, `market_context`, `execution_liquidity`

**Dune sources (verified):** `pumpdotfun_solana.*` decoded call/evt tables, `solana.instruction_calls`, `solana.transactions`, `solana.blocks`, `tokens_solana.transfers`.

Each partition manifest records SQL hash, parameters, Dune execution ID, row counts, source time bounds, and file checksums for auditability.
