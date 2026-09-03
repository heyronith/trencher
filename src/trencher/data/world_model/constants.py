"""Constants for world-model historical data recovery."""

from __future__ import annotations

from datetime import date

CORPUS_START = date(2026, 6, 5)
CORPUS_END = date(2026, 7, 14)

BQ_PROJECT_ENV = "GOOGLE_CLOUD_PROJECT"
BQ_DATASET = "bigquery-public-data.crypto_solana_mainnet_us"
BQ_TABLES = (
    "Transactions",
    "Token Transfers",
    "Blocks",
    "Accounts",
    "Rewards",
)
# Hard safety caps — stay well under free 1 TiB/month.
BQ_MAX_BYTES_BILLED_DEFAULT = 25 * 1024**3  # 25 GiB per query
BQ_SESSION_BUDGET_BYTES = 200 * 1024**3  # 200 GiB soft session budget
BQ_DRY_RUN_ABORT_BYTES = 50 * 1024**3  # abort if dry-run exceeds 50 GiB

SOLARCHIVE_TXS_REL = "txs"
SOLARCHIVE_ACCOUNTS_REL = "accounts"

# High-liquidity SOL/USDC reference pool (Raydium CLMM) on GeckoTerminal.
GECKO_SOL_USDC_POOL = "Czfq3xZZDmsdGdUyrNLtRhGc47cXcZtLG4crryfu44zE"
GECKO_NETWORK = "solana"
GECKO_BASE_URL = "https://api.geckoterminal.com/api/v2"
GECKO_MIN_INTERVAL_SEC = 7.5  # stay under ~8 calls/minute to reduce 429s

DEFAULT_RPC_URL = "https://api.mainnet-beta.solana.com"
RPC_MIN_INTERVAL_SEC = 0.75

SYNTHETIC_POOLS = frozenset(
    {
        "synthetic_graduation_queue",
        "backfilled_from_pumpswap_trade",
    }
)

WORLD_MODEL_DATA_DIR = "world_model"
MANIFEST_DIR = "world_model_data"
