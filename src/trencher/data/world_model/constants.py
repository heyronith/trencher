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
BQ_MAX_BYTES_BILLED_DEFAULT = 25 * 1024**3  # 25 GiB per small query
BQ_SESSION_BUDGET_BYTES = 800 * 1024**3  # 800 GiB soft session budget
BQ_DRY_RUN_ABORT_BYTES = 50 * 1024**3  # abort small queries if dry-run exceeds 50 GiB
BQ_TT_ABORT_BYTES = 650 * 1024**3  # allow one planned Token Transfers extract

SOLARCHIVE_TXS_REL = "txs"
SOLARCHIVE_ACCOUNTS_REL = "accounts"

# High-liquidity SOL/USDC reference pool (Raydium CLMM) on GeckoTerminal.
GECKO_SOL_USDC_POOL = "Czfq3xZZDmsdGdUyrNLtRhGc47cXcZtLG4crryfu44zE"
GECKO_NETWORK = "solana"
GECKO_BASE_URL = "https://api.geckoterminal.com/api/v2"
GECKO_MIN_INTERVAL_SEC = 7.5  # stay under ~8 calls/minute to reduce 429s

DEFAULT_RPC_URL = "https://api.mainnet-beta.solana.com"
HELIUS_RPC_TEMPLATE = "https://mainnet.helius-rpc.com/?api-key={key}"
RPC_MIN_INTERVAL_SEC = 0.75
RPC_BATCH_SIZE = 8
SYSTEM_PROGRAM_ID = "11111111111111111111111111111111"
BQ_FREE_TIER_BYTES = 1024 * 1024**3  # 1 TiB / month query processing
BQ_HEADROOM_RESERVE_BYTES = 80 * 1024**3  # keep 80 GiB unused
REPRESENTATIVE_BQ_DAYS = (
    "2026-06-05",
    "2026-06-06",
    "2026-06-07",
    "2026-06-08",
    "2026-06-09",
    "2026-06-10",
    "2026-06-12",
    "2026-06-15",
    "2026-06-18",
    "2026-06-22",
    "2026-06-26",
    "2026-06-29",
    "2026-07-01",
    "2026-07-02",
    "2026-07-04",
    "2026-07-07",
    "2026-07-10",
    "2026-07-12",
    "2026-07-14",
)

SYNTHETIC_POOLS = frozenset(
    {
        "synthetic_graduation_queue",
        "backfilled_from_pumpswap_trade",
    }
)

WORLD_MODEL_DATA_DIR = "world_model"
MANIFEST_DIR = "world_model_data"
