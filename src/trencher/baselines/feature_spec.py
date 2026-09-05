"""Baseline feature contract: BROAD core + RICH-additive families."""

from __future__ import annotations

from typing import Any

FEATURE_CACHE_VERSION = "baseline_features_v1"
WINDOWS_SEC: tuple[int, ...] = (10, 30, 120)

# Explicit missingness masks — never semantic zero for absent modalities
MASK_PREFIX = "mask_"


def _w(name: str, window: int | str) -> str:
    return f"{name}_w{window}"


def broad_feature_names() -> list[str]:
    names: list[str] = [
        "token_age_seconds",
        "token_age_log1p",
        "mask_events",
        "mask_curve",
        "mask_creator",
        "mask_global",
    ]
    for w in (*WINDOWS_SEC, "full"):
        names.extend(
            [
                _w("event_count", w),
                _w("buy_count", w),
                _w("sell_count", w),
                _w("buy_fraction", w),
                _w("buy_sol", w),
                _w("sell_sol", w),
                _w("net_sol", w),
                _w("flow_imbalance", w),
                _w("sol_mean", w),
                _w("sol_std", w),
                _w("sol_max", w),
                _w("price_return", w),
                _w("price_slope", w),
                _w("realized_vol", w),
                _w("drawdown", w),
                _w("price_accel", w),
                _w("interarrival_mean", w),
                _w("interarrival_std", w),
                _w("burst_intensity", w),
                _w("intensity_accel", w),
                _w("reserve_delta_sol", w),
                _w("curve_delta", w),
                _w("curve_velocity", w),
                _w("curve_accel", w),
            ]
        )
    names.extend(
        [
            "latest_price",
            "latest_v_sol",
            "latest_v_tok",
            "latest_curve_pct",
            "latest_is_buy",
            "creator_prior_rank",
            "creator_prior_is_mayhem",
            "creator_history_n",
            "global_launches_1h",
            "global_launches_24h",
            "global_migrations_1h",
            "global_migrations_24h",
            "daily_network_launches",
            "daily_network_graduations",
            "daily_network_trades",
            "daily_network_buy_volume_sol",
            "daily_network_graduation_rate",
            "global_bucket_trades_sum",
            "global_bucket_buy_sol_sum",
            "global_bucket_sell_sol_sum",
            "global_bucket_imbalance",
            "global_bucket_unique_mints_mean",
        ]
    )
    return names


def rich_additive_feature_names() -> list[str]:
    return [
        "mask_micro",
        "mask_ownership",
        "mask_graph",
        "mask_wallet_prior",
        "micro_n",
        "micro_fee_mean",
        "micro_cu_mean",
        "micro_priority_fee_mean",
        "micro_success_rate",
        "micro_account_key_mean",
        "own_n",
        "own_value_sum",
        "own_delta_sum",
        "own_delta_abs_mean",
        "graph_n",
        "graph_buy_edges",
        "graph_sell_edges",
        "graph_sol_sum",
        "graph_token_sum",
        "wallet_prior_n",
        "wallet_prior_sol_sum",
        "wallet_prior_unique_approx",
    ]


def all_feature_names(*, include_rich: bool = True) -> list[str]:
    names = broad_feature_names()
    if include_rich:
        names = names + rich_additive_feature_names()
    return names


def feature_contract_manifest() -> dict[str, Any]:
    return {
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "windows_sec": list(WINDOWS_SEC),
        "full_window": "causal_sequence_or_last_n",
        "broad_features": broad_feature_names(),
        "rich_additive_features": rich_additive_feature_names(),
        "missing_modality_policy": "explicit_mask_never_semantic_zero",
        "forbidden_predictive": [
            "mint",
            "wallet",
            "creator",
            "observation_id",
            "base_split",
            "source_identity",
        ],
        "pit_global": "TRAINING_LOADER_CONTRACT_V4 last_completed_second + completed UTC day",
        "no_post_T": True,
    }
