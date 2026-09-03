"""Canonical concept mapping and constants for feasibility audit."""

from __future__ import annotations

SYSTEM_PROGRAM_WALLET = "BwWK17cbHxwWBKZkUYvzxLcNQ1YVyaFezduWbtm2de6s"

# Map useful source columns -> canonical research concepts (documentation only).
COLUMN_CONCEPT_MAP: dict[str, list[dict[str, str]]] = {
    "mint": [
        {"source": "pumpfun.tokens.mint", "role": "primary_key"},
        {"source": "pumpfun.trades.mint", "role": "foreign_key"},
        {"source": "pumpfun.snapshots.mint", "role": "foreign_key"},
        {"source": "pumpfun.migrations.mint", "role": "foreign_key"},
        {"source": "pumpfun.postgard_*.mint", "role": "foreign_key"},
        {"source": "red_pump.launches.mint", "role": "primary_key"},
        {"source": "red_pump.outcomes.mint", "role": "foreign_key"},
        {"source": "red_cohort.sniper_cohorts.mints_hit[].mint", "role": "set_membership"},
    ],
    "wallet": [
        {"source": "pumpfun.trades.user_wallet", "role": "trader_wallet"},
        {"source": "pumpfun.wallet_stats.wallet", "role": "wallet_id_unreliable_aggs"},
        {"source": "red_cohort.sniper_cohorts.wallets", "role": "cohort_member"},
    ],
    "creator": [
        {"source": "pumpfun.tokens.creator", "role": "token_creator"},
    ],
    "event_time": [
        {"source": "pumpfun.trades.event_time", "role": "trade_timestamp"},
        {"source": "pumpfun.snapshots.bucket_start", "role": "snapshot_timestamp"},
        {"source": "pumpfun.tokens.detected_at", "role": "launch_observation_time"},
        {"source": "red_pump.launches.created_timestamp", "role": "chain_create_time_ms"},
        {"source": "red_pump.launches.t", "role": "observer_seen_time_ms"},
    ],
    "price": [
        {"source": "pumpfun.trades.price_sol", "role": "bonding_curve_price_sol"},
        {"source": "pumpfun.snapshots.price_sol_eob", "role": "bucket_end_price_sol"},
        {"source": "pumpfun.postgard_snapshots.price_usd", "role": "post_grad_dex_price_usd"},
    ],
    "reserves": [
        {"source": "pumpfun.trades.v_sol_bonding_curve", "role": "raw_curve_sol_reserves"},
        {"source": "pumpfun.trades.v_tokens_bonding_curve", "role": "raw_curve_token_reserves"},
        {"source": "pumpfun.snapshots.v_sol_eob", "role": "bucket_end_sol_reserves"},
        {"source": "pumpfun.snapshots.v_tokens_eob", "role": "bucket_end_token_reserves"},
    ],
    "graduation": [
        {"source": "pumpfun.tokens.graduated_at", "role": "graduation_time"},
        {"source": "pumpfun.migrations.migrated_at", "role": "migration_time"},
        {"source": "pumpfun.migrations.pool_address", "role": "pool_id_sometimes_synthetic"},
        {"source": "red_pump.outcomes.outcome", "role": "graduated_or_timeout_label"},
    ],
    "outcome_labels_source": [
        {
            "source": "pumpfun.postgard_outcomes.outcome_label",
            "role": "source_specific_postgrad_label_ignore_for_canonical",
        },
        {
            "source": "pumpfun.postgard_outcomes.rug_detected",
            "role": "source_rug_flag_optional",
        },
        {
            "source": "red_pump.outcomes.outcome",
            "role": "survival_censoring_label_grad_vs_timeout",
        },
    ],
}

LEAKAGE_RISK_FIELDS = [
    "pumpfun.tokens.entry_price_20s_usd",
    "pumpfun.tokens.entry_price_30s_usd",
    "pumpfun.tokens.entry_price_1m_usd",
    "pumpfun.tokens.graduated_at",  # label for graduation prediction
    "pumpfun.tokens.seconds_to_graduation",
    "pumpfun.tokens.peak_market_cap_sol",
    "pumpfun.tokens.peak_market_cap_at",
    "pumpfun.postgard_outcomes.*",
]

UNRELIABLE_FIELDS = [
    "pumpfun.tokens.initial_top1_pct",
    "pumpfun.tokens.initial_top5_pct",
    "pumpfun.tokens.initial_top10_pct",
    "pumpfun.tokens.dev_buy_pct",
    "pumpfun.wallet_stats.tokens_traded",
    "pumpfun.wallet_stats.total_trades",
    "pumpfun.wallet_stats.total_buy_volume_sol",
    "pumpfun.wallet_stats.total_sell_volume_sol",
    "pumpfun.migrations.pool_address (synthetic sentinels)",
]

EXCLUDE_WALLETS = [SYSTEM_PROGRAM_WALLET]
