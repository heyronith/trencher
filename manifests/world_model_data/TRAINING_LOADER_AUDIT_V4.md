# TRAINING_LOADER_AUDIT_V4

Generated: `2026-09-05T19:28:33.693076+00:00`

## Transform coverage

```json
{
  "transform_order": "clip_raw_p01_p99_then_zscore",
  "fit_scope": "train_obs_unique_rows_pre_q70",
  "max_event_time": 1782869641.017306,
  "transformed_continuous_channels": [
    "account_key_count",
    "balance_change_count",
    "buy_sol",
    "buys",
    "compute_units_consumed",
    "curve_pct_depleted",
    "daily_network_buy_volume_sol",
    "daily_network_graduation_rate",
    "daily_network_graduations",
    "daily_network_launches",
    "daily_network_trades",
    "delta_ui",
    "fee_lamports",
    "global_launches_1h",
    "global_launches_24h",
    "global_migrations_1h",
    "global_migrations_24h",
    "inner_instruction_groups",
    "log_message_count",
    "post_ui_amount",
    "pre_ui_amount",
    "price_sol",
    "prior_rank",
    "sell_sol",
    "sells",
    "sol_amount",
    "token_amount",
    "trades",
    "unique_mints",
    "unique_wallets",
    "v_sol",
    "v_tok",
    "value"
  ],
  "empty_train_fit_identity_passthrough": [
    "priority_fee_lamports"
  ],
  "missing_or_empty_fit": [
    "priority_fee_lamports"
  ],
  "intentionally_untransformed": {
    "relative_time_raw": [
      "rel_dt",
      "interarrival"
    ],
    "bounded_ordinals": [
      "decimals"
    ],
    "never_normalize": [
      "masks",
      "binary_flags",
      "categorical_ids",
      "graph_node_ids"
    ],
    "empty_train_channels_note": "Channels with n=0 in train fit pass through unscaled (e.g. priority_fee_lamports often unavailable)."
  },
  "rows_seen": 718471,
  "column_stats_n": {
    "sol_amount": 626991.0,
    "token_amount": 718471.0,
    "price_sol": 626991.0,
    "v_sol": 515580.0,
    "v_tok": 590942.0,
    "curve_pct_depleted": 515580.0,
    "trades": 1666258.0,
    "buys": 1666258.0,
    "sells": 1666258.0,
    "buy_sol": 1666258.0,
    "sell_sol": 1666258.0,
    "unique_mints": 1666258.0,
    "unique_wallets": 1666258.0,
    "daily_network_launches": 26.0,
    "daily_network_graduations": 26.0,
    "daily_network_trades": 26.0,
    "daily_network_buy_volume_sol": 26.0,
    "daily_network_graduation_rate": 26.0,
    "fee_lamports": 10831.0,
    "compute_units_consumed": 10831.0,
    "priority_fee_lamports": 0.0,
    "account_key_count": 10831.0,
    "balance_change_count": 10831.0,
    "log_message_count": 10831.0,
    "inner_instruction_groups": 10831.0,
    "value": 6641.0,
    "pre_ui_amount": 4045.0,
    "post_ui_amount": 4045.0,
    "delta_ui": 4045.0,
    "prior_rank": 2298423.0,
    "global_launches_1h": 5000.0,
    "global_launches_24h": 5000.0,
    "global_migrations_1h": 1177.0,
    "global_migrations_24h": 1177.0
  }
}
```

## TTE counts (sample part)

```json
{
  "graduation_tte": {
    "at_risk": 2935,
    "event": 2905,
    "censor": 30,
    "unavailable": 495999,
    "n": 500000
  },
  "explosion_tte": {
    "at_risk": 437471,
    "event": 9667,
    "censor": 427804,
    "unavailable": 21057,
    "n": 500000
  },
  "sample_part": "part-00000.parquet"
}
```

## Benchmark (light)

```json
{
  "batch_size": 64,
  "batches": 20,
  "examples": 1280,
  "shuffle": true,
  "used_training_cache": true,
  "cold_start_examples_per_sec": 188.09394766437654,
  "steady_state_examples_per_sec": 1175.8989608083018,
  "examples_per_sec": 770.9984777376309,
  "batches_per_sec": 12.046851214650482,
  "elapsed_sec": 1.6601848602294922,
  "sequence_shapes": {
    "creator": [
      64,
      256,
      4
    ],
    "curve": [
      64,
      130,
      7
    ],
    "events": [
      64,
      130,
      13
    ],
    "global": [
      64,
      128,
      8
    ]
  },
  "channel_schemas": {
    "events": [
      "rel_dt",
      "interarrival",
      "is_buy",
      "side_id",
      "sol_amount",
      "token_amount",
      "price_sol",
      "v_sol",
      "v_tok",
      "curve_pct_depleted",
      "has_reserves",
      "has_finite_price",
      "source_quality"
    ],
    "curve": [
      "rel_dt",
      "price_sol",
      "v_sol",
      "v_tok",
      "curve_pct_depleted",
      "has_reserves",
      "source_quality"
    ],
    "creator": [
      "rel_dt",
      "prior_rank",
      "prior_is_mayhem_mode",
      "source_quality"
    ]
  },
  "length_means": {
    "creator": 80.201562
```
