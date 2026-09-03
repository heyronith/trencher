"""Reconstruction and label-feasibility assessment."""

from __future__ import annotations

from typing import Any


def reconstruction_assessment() -> dict[str, Any]:
    """Can PumpFun corpus reconstruct state/history at arbitrary observation time T?"""
    return {
        "ordered_trade_event_history": {
            "supported": "full",
            "evidence": [
                "trades.*.event_time + seconds_since_launch + mint + tx_signature",
                "sortable per-mint trade streams",
            ],
            "caveats": [
                "exclude System Program wallet rows for participant history",
                "exclude sol_amount-inconsistent and null-price rows for SOL P&L",
                "Jul 3 2026 collection outage (zero trades)",
            ],
        },
        "price_reserve_trajectory": {
            "supported": "full",
            "evidence": [
                "trades.price_sol, market_cap_sol, v_sol/v_tokens bonding curve",
                "snapshots bucket OHLCV + curve_pct_depleted_eob",
            ],
            "caveats": [
                "~7% null price_sol rows (mostly whole-token early failures)",
                "curve_pct_depleted can exceed 100 (cap 110) especially post Jul 4",
                "raw reserve units need documented scaling relationship",
            ],
        },
        "wallet_participation_history": {
            "supported": "full",
            "evidence": [
                "trades.user_wallet with event_time per mint",
                "can recompute first/last/activity from trades",
            ],
            "caveats": [
                "do not use wallet_stats activity/volume aggregates (stale)",
                "exclude System Program address",
            ],
        },
        "bonding_curve_state": {
            "supported": "full",
            "evidence": [
                "per-trade and per-bucket reserve/price/curve depletion fields",
            ],
            "caveats": [
                "concentration fields need corrected columns + suspect exclusions",
            ],
        },
        "graduation": {
            "supported": "full",
            "evidence": [
                "tokens.graduated_at / seconds_to_graduation",
                "migrations.migrated_at",
                "red_pump outcomes GRADUATED vs TIMEOUT",
            ],
            "caveats": [
                "migrations.pool_address partially synthetic",
                "source populations differ slightly across tables",
            ],
        },
        "post_graduation_trajectory": {
            "supported": "partial",
            "evidence": [
                "postgard_snapshots price/liquidity/volume after graduation",
                "postgard_outcomes peak/horizon fields for graduated subset",
            ],
            "caveats": [
                "only ~5.7k graduated tokens, not full 798k launch universe",
                "source outcome_label is source-specific; regenerate labels ourselves",
            ],
        },
    }


def label_feasibility() -> dict[str, Any]:
    """Which desired labels can we generate ourselves from these datasets?"""
    return {
        "survival_horizons": {
            "supported": "full",
            "method": (
                "From observation time T, use future trades/snapshots/postgard "
                "or red_pump TIMEOUT censoring at 24h for survival analysis."
            ),
            "sources": ["pumpfun.trades", "pumpfun.snapshots", "red_pump.outcomes"],
        },
        "plus_x_before_minus_y_barriers": {
            "supported": "full",
            "method": (
                "Walk forward price_sol / postgrad price_usd path after T; "
                "detect first barrier hits."
            ),
            "sources": ["pumpfun.trades", "pumpfun.snapshots", "pumpfun.postgard_snapshots"],
            "caveats": ["requires clean price path; drop null/inconsistent trade prices"],
        },
        "future_return_distributions": {
            "supported": "full",
            "method": "Compute returns at fixed horizons from T using price path.",
            "sources": ["pumpfun.trades", "pumpfun.snapshots", "pumpfun.postgard_snapshots"],
        },
        "mfe_mae": {
            "supported": "full",
            "method": (
                "Max favorable/adverse excursion from price path after T within horizon."
            ),
            "sources": ["pumpfun.trades", "pumpfun.snapshots", "pumpfun.postgard_snapshots"],
        },
        "time_to_graduation_crash_inactivity": {
            "supported": "full",
            "method": (
                "graduation via graduated_at/migrations; crash via drawdown barriers; "
                "inactivity via time since last trade/unique wallet activity."
            ),
            "sources": [
                "pumpfun.tokens",
                "pumpfun.migrations",
                "pumpfun.trades",
                "red_pump.outcomes",
            ],
        },
        "buyer_activity_persistence": {
            "supported": "full",
            "method": (
                "Recompute per-wallet trade counts/volumes over windows from trades; "
                "optionally join RED-COHORT membership as coordination feature."
            ),
            "sources": ["pumpfun.trades", "red_cohort.sniper_cohorts"],
            "caveats": ["ignore wallet_stats aggregates; exclude system program"],
        },
        "source_specific_labels_to_ignore_as_canonical": [
            "postgard_outcomes.outcome_label (dead/pump_dump/etc are source taxonomy)",
            "postgard_outcomes.success_label / target_label_legacy",
            "quickstart rug filter on outcome_label='rug' (does not exist)",
        ],
    }
