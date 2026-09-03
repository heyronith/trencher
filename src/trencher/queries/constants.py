"""Shared program IDs, mints, and instruction discriminators."""

from __future__ import annotations

SCHEMA_VERSION = "1.1.0"

WSOL_MINT = "So11111111111111111111111111111111111111112"
NATIVE_SOL_MINT = "11111111111111111111111111111111"
PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMP_AMM_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"

# Anchor discriminators as lowercase hex (8 bytes), matching Dune to_hex(varbinary).
DISCRIMINATORS: dict[str, str] = {
    "create": "181ec828051c0777",
    "create_v2": "d6904cec5f8b31b4",
    "buy": "66063d1201daebea",
    "buy_v2": "b817ee6167c5d33d",
    "buy_exact_sol_in": "38fc74089edfcd5f",
    "sell": "33e685a4017f83ad",
    "sell_v2": "5df6823ce7e940b2",
    "migrate": "9beae792ec9ea21e",
}

SOL_QUOTE_MINTS = (WSOL_MINT, NATIVE_SOL_MINT)

OPTIONAL_DUNE_TABLES: frozenset[str] = frozenset(
    {
        "pumpdotfun_solana.pump_call_create_v2",
        "pumpdotfun_solana.pump_call_buy_v2",
        "pumpdotfun_solana.pump_call_sell_v2",
        "pumpdotfun_solana.pump_call_buy_exact_sol_in",
        "pumpdotfun_solana.pump_evt_completepumpammmigrationevent",
    }
)
