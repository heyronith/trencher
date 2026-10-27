# PHASE5_REPORT

## Summary
1. **Alert feed (XGB champion):** top-10/hour mint-deduped mean explode@300s = **13.1%** vs base **2.1%** (~6.2×). Pump recall on 50 primary val events: top-1% catches **6%** (lead p50 ≈ 15 s); top-2% catches **24%** (lead p50 ≈ 3 s) with ~2078 false alerts per caught pump.
2. **Tuning helps lightly for LGBM, not XGB:** best WF pick is V2-like params **without** `scale_pos_weight`. LGBM@60s permanent-val ROC **0.8735** (ΔROC vs frozen phase4 **+0.0044**, CI excludes 0); ΔPR tiny/CI includes 0. XGB tuned configs slightly **below** phase4 frozen preds on both heads.
3. **More history would help:** learning curves on explosion_60s still rising at 100% train (LGBM PR 0.077→0.090; XGB 0.080→0.091). Public corpora already cover ~Jun 5–Jul 14 2026; extending to 12 months ≈ 5.5 TiB / ~$33 BQ-scale + ~4 wall-days @8 CPU (order-of-magnitude).
4. **Clean micro:** skipped (no `HELIUS_API_KEY` / `SOLANA_RPC_URL`).
5. **Sanity:** `n_trades_before_T` was missing from `feature_spec` (not constant)—now injected (125 feats); A+n_trades ROC@60 **0.8389** vs A **0.8346**. Deterministic LGBM: two runs identical. Phase4 graph/wallet 0.7962 vs phase3 0.8173 explained by non-deterministic bagging; deterministic re-run → **0.8233** (+0.026 vs A, CI excludes 0).
6. `q85_test_accessed: false`. Permanent val scored only once per locked WF config (logged in JSON).

## 1. Sanity checks

**n_trades root cause:** n_trades_before_T is not part of all_feature_names()/feature_spec; Phase-4 looked it up in name_to_i and got [], so A_plus_n_trades == A (identical n_features and predictions). Not constant/NaN in data — never injected.

### n_trades_before_T distribution (rich)

```json
{
  "source": "sequence_events_v2 via load_event_times_by_mint + n_before (<=T)",
  "train": {
    "is_constant": false,
    "max": 3891.0,
    "mean": 16.759246991277465,
    "min": 0.0,
    "n": 36228,
    "n_nan": 0,
    "n_unique": 382,
    "n_zero": 9786,
    "quantiles": {
      "p0": 0.0,
      "p1": 0.0,
      "p100": 3891.0,
      "p25": 0.0,
      "p5": 0.0,
      "p50": 2.0,
      "p75": 8.0,
      "p95": 60.0,
      "p99": 248.0
    },
    "std": 83.12746473581696
  },
  "val": {
    "is_constant": false,
    "max": 1454.0,
    "mean": 16.64327197558815,
    "min": 0.0,
    "n": 10159,
    "n_nan": 0,
    "n_unique": 212,
    "n_zero": 3008,
    "quantiles": {
      "p0": 0.0,
      "p1": 0.0,
      "p100": 1454.0,
      "p25": 0.0,
      "p5": 0.0,
      "p50": 2.0,
      "p75": 8.0,
      "p95": 72.0,
      "p99": 228.0
    },
    "std": 66.28657297397024
  }
}
```

### Re-run rich arms (deterministic)

| head | arm | n_features | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|
| explosion_300s | A | 124 | 0.7973 | 0.0567 |
| explosion_300s | A_plus_graph_wallet | 134 | 0.8233 | 0.0666 |
| explosion_300s | A_plus_graph_wallet_n_trades | 135 | 0.8258 | 0.0716 |
| explosion_300s | A_plus_n_trades | 125 | 0.8028 | 0.0719 |
| explosion_60s | A | 124 | 0.8346 | 0.0604 |
| explosion_60s | A_plus_graph_wallet | 134 | 0.8375 | 0.0575 |
| explosion_60s | A_plus_graph_wallet_n_trades | 135 | 0.8393 | 0.0583 |
| explosion_60s | A_plus_n_trades | 125 | 0.8389 | 0.0577 |

### Determinism

```json
{
  "explosion_300s": {
    "arm": "A_plus_graph_wallet",
    "identical_predictions": true,
    "max_abs_diff": 0.0,
    "n_compared": 10159,
    "params": {
      "bagging_seed": 0,
      "deterministic": true,
      "feature_fraction_seed": 0,
      "force_row_wise": true,
      "num_threads": 1,
      "seed": 0
    }
  },
  "explosion_60s": {
    "arm": "A_plus_graph_wallet",
    "identical_predictions": true,
    "max_abs_diff": 0.0,
    "n_compared": 10159,
    "params": {
      "bagging_seed": 0,
      "deterministic": true,
      "feature_fraction_seed": 0,
      "force_row_wise": true,
      "num_threads": 1,
      "seed": 0
    }
  }
}
```

### Phase3 vs Phase4 graph/wallet 300s

```json
{
  "investigation": {
    "columns": "Same CLEAN set: mask_graph + graph_* + mask_wallet_prior + wallet_prior_* (134 features). Phase-4 did not accidentally include micro/ownership.",
    "fixed_iterations": {
      "explosion_300s": {
        "phase3_clean_used_B_fixed_iters": 50,
        "phase4_used_A_fixed_iters_for_all_arms": 50,
        "same": true
      },
      "explosion_60s": {
        "phase3_clean_used_B_fixed_iters": 57,
        "phase4_used_A_fixed_iters_for_all_arms": 50,
        "same": false
      }
    },
    "protocol_fix": "Phase-5 re-runs with deterministic LightGBM and Phase-3 B fixed_iters for graph/wallet arms.",
    "rows": "Same rich train/val via load_joined_matrices on freeze split_v1.",
    "seed": "Both used seed=0; Phase-4 lacked deterministic/force_row_wise/bagging_seed/feature_fraction_seed, so bagging+feature subsample was not bit-stable."
  },
  "phase3_lgbm_clean_only_explosion_300s_roc": 0.8173310701014166,
  "phase4_A_plus_graph_wallet_explosion_300s_roc": 0.7962
}
```

## 2. Pump-level evaluation (broad permanent val, frozen V2 trees)

TTE overlay join: `{"columns_read": ["observation_id", "mint", "observation_time_t", "base_split", "explosion_tte_effective_tte_sec", "explosion_tte_effective_event_observed", "explosion_tte_raw_tte_sec", "explosion_tte_raw_event_observed"], "elapsed_sec": 0.5204591751098633, "n_joined": 1275120, "n_overlay_val_rows_scanned": 1275120, "n_pred_oids": 1275120, "predicate": "base_split == 'val' then observation_id in pred set", "q85_test_accessed": false, "rss_delta_mb": 122.59375, "rss_mb_after": 1105.9375, "rss_mb_before": 983.34375}`

already_started join: `{"elapsed_sec": 3.385512113571167, "n_matched": 1275120, "n_parts": 256, "q85_test_accessed": false, "rss_mb_after": 1353.65625, "rss_mb_before": 1105.9375}`

### Metrics by variant

| head | model | variant | ROC | PR | P@0.1% | P@0.5% | P@1% | P@2% |
|---|---|---|---:|---:|---:|---:|---:|---:|
| explosion_300s | lgb | a_exclude_already_started | 0.8550 | 0.1095 | 0.2052 | 0.1795 | 0.1654 | 0.1571 |
| explosion_300s | lgb | all | 0.8536 | 0.0951 | 0.1396 | 0.1372 | 0.1284 | 0.1347 |
| explosion_300s | lgb | c_onset_ge_5s | 0.8360 | 0.0611 | 0.1084 | 0.0999 | 0.0934 | 0.0861 |
| explosion_300s | xgb | a_exclude_already_started | 0.8583 | 0.1121 | 0.2276 | 0.1849 | 0.1725 | 0.1615 |
| explosion_300s | xgb | all | 0.8577 | 0.0983 | 0.1709 | 0.1454 | 0.1423 | 0.1414 |
| explosion_300s | xgb | c_onset_ge_5s | 0.8387 | 0.0621 | 0.1274 | 0.1064 | 0.0985 | 0.0878 |
| explosion_60s | lgb | a_exclude_already_started | 0.8686 | 0.1013 | 0.1683 | 0.1548 | 0.1510 | 0.1386 |
| explosion_60s | lgb | all | 0.8690 | 0.0892 | 0.1299 | 0.1206 | 0.1278 | 0.1205 |
| explosion_60s | lgb | c_onset_ge_5s | 0.8516 | 0.0513 | 0.0785 | 0.0753 | 0.0743 | 0.0683 |
| explosion_60s | xgb | a_exclude_already_started | 0.8729 | 0.1042 | 0.1576 | 0.1638 | 0.1541 | 0.1535 |
| explosion_60s | xgb | all | 0.8734 | 0.0919 | 0.1235 | 0.1230 | 0.1260 | 0.1345 |
| explosion_60s | xgb | c_onset_ge_5s | 0.8555 | 0.0526 | 0.0894 | 0.0846 | 0.0773 | 0.0740 |

### Pump-level recall

**lgb_explosion_300s** (n_events=50)

| top | fraction_caught | lead_p50 | FA/caught |
|---|---:|---:|---:|
| top_0.01 | 0.06 | 14.884522914886475 | 4153.666666666667 |
| top_0.02 | 0.08 | 8.062109470367432 | 6229.5 |

**lgb_explosion_60s** (n_events=50)

| top | fraction_caught | lead_p50 | FA/caught |
|---|---:|---:|---:|
| top_0.01 | 0.06 | 14.884522914886475 | 4155.666666666667 |
| top_0.02 | 0.1 | 14.884522914886475 | 4987.6 |

**xgb_explosion_300s** (n_events=50)

| top | fraction_caught | lead_p50 | FA/caught |
|---|---:|---:|---:|
| top_0.01 | 0.06 | 14.884522914886475 | 4151.666666666667 |
| top_0.02 | 0.16 | 3.115750551223755 | 3113.75 |

**xgb_explosion_60s** (n_events=50)

| top | fraction_caught | lead_p50 | FA/caught |
|---|---:|---:|---:|
| top_0.01 | 0.06 | 14.884522914886475 | 4167.333333333333 |
| top_0.02 | 0.24 | 2.985063076019287 | 2077.5833333333335 |

### Alert feed (top-10/hour, mint-deduped)

| model | mean_frac@300s | base_rate | lift | n_hours |
|---|---:|---:|---:|---:|
| lgb | 0.12612612612612611 | 0.02108409085438747 | 5.982051917589799 | 111 |
| xgb | 0.1306306306306306 | 0.02108409085438747 | 6.195696628932291 | 111 |

## 3. WF tuning

### lightgbm/explosion_300s

- Best WF mean PR-AUC: **0.07681790545542805**
- Config: `{"bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "learning_rate": 0.05, "min_data_in_leaf": 100, "num_leaves": 63, "use_scale_pos_weight": false}`
- Fixed iters (median best): **50**
- Permanent val ROC/PR: **0.8560427464629519** / **0.09513737493211791**
- ΔPR vs frozen V2 (tuned−frozen): 7.036795599975332e-05 CI [-0.0011997313935116255, 0.0012931563328661862]
- ΔROC vs frozen V2: 0.002395419865095727 CI [0.001781804153879779, 0.0030809296297117593]

### lightgbm/explosion_60s

- Best WF mean PR-AUC: **0.07239928918241079**
- Config: `{"bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "learning_rate": 0.05, "min_data_in_leaf": 100, "num_leaves": 63, "use_scale_pos_weight": false}`
- Fixed iters (median best): **50**
- Permanent val ROC/PR: **0.8734519873901027** / **0.08981779108446239**
- ΔPR vs frozen V2 (tuned−frozen): 0.0006257691091995504 CI [-0.0006005395212504674, 0.0018962469704660327]
- ΔROC vs frozen V2: 0.004427299169641574 CI [0.0036573154740123263, 0.005161746696066987]

### xgboost/explosion_300s

- Best WF mean PR-AUC: **0.07649621930071021**
- Config: `{"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 1, "subsample": 0.8, "use_scale_pos_weight": false}`
- Fixed iters (median best): **71**
- Permanent val ROC/PR: **0.8565570161906079** / **0.09778063457531723**
- ΔPR vs frozen V2 (tuned−frozen): -0.0005039064767345947 CI [-0.0008983019749672118, -0.00011762075489051546]
- ΔROC vs frozen V2: -0.0011015027566922964 CI [-0.0014681339160946161, -0.0007564872711644555]

### xgboost/explosion_60s

- Best WF mean PR-AUC: **0.0724615873529**
- Config: `{"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 1, "subsample": 0.8, "use_scale_pos_weight": false}`
- Fixed iters (median best): **64**
- Permanent val ROC/PR: **0.8714137501263695** / **0.09107407426970178**
- ΔPR vs frozen V2 (tuned−frozen): -0.0008293427819379973 CI [-0.0010131919318084846, -0.0006454820469550872]
- ΔROC vs frozen V2: -0.001988712918292612 CI [-0.002231170649957265, -0.0017451860507039535]

Permanent-val configs scored log: `[
  {
    "config": {
      "bagging_fraction": 0.8,
      "bagging_freq": 1,
      "feature_fraction": 0.8,
      "learning_rate": 0.05,
      "min_data_in_leaf": 100,
      "num_leaves": 63,
      "use_scale_pos_weight": false
    },
    "config_id": 1,
    "family": "lightgbm",
    "fixed_iterations": 50,
    "head_id": "explosion_60s"
  },
  {
    "config": {
      "colsample_bytree": 0.8,
      "learning_rate": 0.05,
      "max_depth": 8,
      "min_child_weight": 1,
      "subsample": 0.8,
      "use_scale_pos_weight": false
    },
    "config_id": 1,
    "family": "xgboost",
    "fixed_iterations": 64,
    "head_id": "explosion_60s"
  },
  {
    "config": {
      "bagging_fraction": 0.8,
      "bagging_freq": 1,
      "feature_fraction": 0.8,
      "learning_rate": 0.05,
      "min_data_in_leaf": 100,
      "num_leaves": 63,
      "use_scale_pos_weight": false
    },
    "config_id": 1,
    "family": "lightgbm",
    "fixed_iterations": 50,
    "head_id": "explosion_300s"
  },
  {
    "config": {
      "colsample_bytree": 0.8,
      "learning_rate": 0.05,
      "max_depth": 8,
      "min_child_weight": 1,
      "subsample": 0.8,
      "use_scale_pos_weight": false
    },
    "config_id": 1,
    "family": "xgboost",
    "fixed_iterations": 71,
    "head_id": "explosion_300s"
  }
]`

## 4. History lever

### Coverage
- Freeze: `{"launch_max_iso": "2026-07-14T15:01:41.919140+00:00", "launch_min_iso": "2026-06-05T09:12:26.343743+00:00", "note": "Freeze covers ~5 weeks: train from 2026-06-05, val [2026-07-01, 2026-07-07).", "train": {"end_unix_exclusive": 1782869641.017306, "rule": "[launch_min, q70)", "start_unix": 1780650746.343743}, "val": {"end_unix_exclusive": 1783459140.3081825, "rule": "[q70, q85)", "start_unix": 1782869641.017306}}`
- Pumpfun trades: `{"covers": "Pump trades (bonding-curve); overlaps freeze Jun\u2013Jul 2026", "kind": "huggingface Slinky21/Pumpfun_Memecoin_Corpus", "local_trades_event_time_max": "2026-07-14 16:02:54.115887+01:00", "local_trades_event_time_min": "2026-06-05 10:12:28.604950+01:00", "n_trade_rows_local": 33581765}`
- RED-PUMP: `{'covers': 'creator/launch cohort metadata, not full trade tape', 'kind': 'zenodo 21383616', 'title_coverage': 'May–June 2026 pump.fun launches (survival outcomes)'}`
- RED-COHORT: `{'covers': 'global/wallet cohort context', 'kind': 'zenodo 21765387', 'title_coverage': 'June 11–25, 2026 wallet cohorts'}`
- Solarchive: `{'covers': 'index/metadata from ~2020-10; freeze note: no June–July 2026 trade coverage', 'kind': 'huggingface solarchive/solarchive'}`
- Dune: `{"example_backfill_start_documented": "2025-01-01", "note": "Dune SQL is parameterized by date; historical coverage is limited by Pump.fun program history on Solana (~2024-01 onward for pump.fun) and credit budget. Repo docs do not store a live Dune credit balance.", "queries": ["trades_events", "dex_trades", "token_lifecycle", "token_transfers", "transaction_microstructure", "execution_liquidity", "wallet_funding", "wallet_graph", "market_context"], "sql_dir": "src/trencher/queries/sql/"}`

### Cost to extend
```json
{
  "assumptions": {
    "caveat": "Repo has BQ partition GiB estimates, not live Dune credit quotes. Dune credit cost for equivalent Pump trades is separate and vendor-priced.",
    "freeze_days": 39.242541381909895,
    "gib_per_day": 15.0,
    "sources": [
      "manifests/world_model_data/FUSION_EXPANSION_COST_PLAN.md",
      "manifests/world_model_data/DATA_FREEZE.md",
      "cutoffs.json quantile token count / launch span"
    ],
    "tokens_per_day": 15857.866949638243,
    "usd_per_tib": 6.25
  },
  "fusion_plan_snapshot": {
    "bytes_consumed_month_gib": 901.051,
    "free_tier_remaining_gib": 122.949,
    "paid_usd_if_executed": 1.09,
    "unqueried_gib": 301.185,
    "unqueried_partitions_days": 20
  },
  "horizons": {
    "12_months": {
      "days": 365,
      "est_bq_gib": 5475.0,
      "est_bq_usd_on_demand": 33.42,
      "est_feature_build_cpu_hours": 730.0,
      "est_tokens": 5788121,
      "est_wall_clock_days_8cpu": 3.8,
      "months": 12
    },
    "3_months": {
      "days": 91,
      "est_bq_gib": 1365.0,
      "est_bq_usd_on_demand": 8.33,
      "est_feature_build_cpu_hours": 182.0,
      "est_tokens": 1443065,
      "est_wall_clock_days_8cpu": 0.95,
      "months": 3
    },
    "6_months": {
      "days": 182,
      "est_bq_gib": 2730.0,
      "est_bq_usd_on_demand": 16.66,
      "est_feature_build_cpu_hours": 364.0,
      "est_tokens": 2886131,
      "est_wall_clock_days_8cpu": 1.9,
      "months": 6
    }
  },
  "q85_test_accessed": false
}
```

### Learning curve

**lightgbm/explosion_60s** — PR rising@100%=True

| fraction | n_train | ROC | PR |
|---:|---:|---:|---:|
| 0.25 | 1391360 | 0.8603 | 0.0773 |
| 0.5 | 2782720 | 0.8640 | 0.0789 |
| 0.75 | 4174080 | 0.8672 | 0.0869 |
| 1.0 | 5565440 | 0.8735 | 0.0898 |

**xgboost/explosion_60s** — PR rising@100%=True

| fraction | n_train | ROC | PR |
|---:|---:|---:|---:|
| 0.25 | 1391360 | 0.8600 | 0.0803 |
| 0.5 | 2782720 | 0.8689 | 0.0894 |
| 0.75 | 4174080 | 0.8700 | 0.0881 |
| 1.0 | 5565440 | 0.8714 | 0.0911 |

## 5. Causal micro refetch

Status: **skipped** — Neither HELIUS_API_KEY nor SOLANA_RPC_URL is set

---
q85_test_accessed: false

