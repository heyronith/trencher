# BASELINE_AUDIT_V1

Generated: `2026-09-05T20:19:10.504576+00:00`

- Feature count: **124**
- Families: `['events', 'curve', 'creator', 'global', 'rich_additive']`

## Results summary

```json
{
  "lightgbm": {
    "heads": [
      "explosion_60s",
      "explosion_300s",
      "barrier_u25_l15_5m",
      "barrier_u100_l30_1h",
      "return_5m",
      "return_1h",
      "survival_30m",
      "survival_24h"
    ],
    "train_n": 47443,
    "val_n": 9595
  },
  "xgboost": {
    "heads": [
      "explosion_60s",
      "explosion_300s",
      "barrier_u25_l15_5m",
      "barrier_u100_l30_1h",
      "return_5m",
      "return_1h",
      "survival_30m",
      "survival_24h"
    ],
    "train_n": 47443,
    "val_n": 9595
  },
  "catboost": {
    "heads": [
      "explosion_60s",
      "explosion_300s",
      "barrier_u25_l15_5m",
      "barrier_u100_l30_1h",
      "return_5m",
      "return_1h",
      "survival_30m",
      "survival_24h"
    ],
    "train_n": 47443,
    "val_n": 9595
  },
  "survival": {
    "heads": [
      "explosion_tte",
      "graduation_tte"
    ],
    "train_n": 47443,
    "val_n": 9595
  }
}
```

## Modal usage

```json
{
  "app": "trencher-baselines-v1",
  "volume": "trencher-world-model-v1",
  "artifacts_volume": "trencher-baseline-artifacts-v1",
  "status": "syncing_splits_and_selective_broad_32_partitions",
  "local_validation": "2_partition_all_core_heads_no_test_access",
  "note": "Full population GPU fits launch on Modal after volume sync; local numbers are pipeline validation only"
}
```

## Core metrics (local 2-partition validation; Modal full pending)

```json
{
  "lightgbm": {
    "explosion_60s": {
      "pr_auc": 0.08596751073716741,
      "roc_auc": 0.8617303782386798,
      "brier": 0.021618270053716047,
      "ece": 0.022421972335773244,
      "cal_brier": 0.015196694903961448,
      "cal_ece": 0.008220586710833805,
      "topk_1pct_prec": 0.09574468085106383,
      "runtime_sec": 1.7922098636627197,
      "train_eligible": 45101,
      "val_eligible": 9390,
      "val_pos": 146
    }
  },
  "xgboost": {
    "explosion_60s": {
      "pr_auc": 0.06737092277284512,
      "roc_auc": 0.8355845776305104,
      "brier": 0.027153374736409077,
      "ece": 0.0397829429383109,
      "cal_brier": 0.015197827564541124,
      "cal_ece": 0.008313007300809699,
      "topk_1pct_prec": 0.11702127659574468,
      "runtime_sec": 0.6380903720855713,
      "train_eligible": 45101,
      "val_eligible": 9390,
      "val_pos": 146
    }
  },
  "catboost": {
    "explosion_60s": {
      "pr_auc": 0.0845799376732814,
      "roc_auc": 0.8773902953711552,
      "brier": 0.08865794181219618,
      "ece": 0.15388241589630533,
      "cal_brier": 0.014879686691444554,
      "cal_ece": 0.007490172158490915,
      "topk_1pct_prec": 0.10638297872340426,
      "runtime_sec": 1.208409309387207,
      "train_eligible": 45101,
      "val_eligible": 9390,
      "val_pos": 146
    }
  }
}
```

### Survival
```json
{
  "explosion_tte": {
    "c_index": 0.8012261260517535,
    "at_risk_val": 8443,
    "events_val": 205,
    "note": "Explosion TTE uses Pump continuous-coverage censoring."
  },
  "graduation_tte": {
    "c_index": 0.6410632447296059,
    "at_risk_val": 112,
    "events_val": 97,
    "note": "Graduation TTE is sparse/event-only under V4 source-aware censoring; report at-risk counts honestly."
  }
}
```
