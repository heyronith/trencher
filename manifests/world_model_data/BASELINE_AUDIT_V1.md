> **Label:** `provisional_32_partition_development_run` (preserved; see BASELINE_AUDIT_V2 for final freeze).

# BASELINE_AUDIT_V1

Generated: `2026-09-05T22:05:47.390805+00:00`

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
    "train_n": 731899,
    "val_n": 157188
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
    "train_n": 731899,
    "val_n": 157188
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
    "train_n": 731899,
    "val_n": 157188
  },
  "survival": {
    "heads": [
      "explosion_tte",
      "graduation_tte"
    ],
    "train_n": 731899,
    "val_n": 157188
  }
}
```

## Modal usage

```json
{
  "app": "trencher-baselines-v1",
  "volume": "trencher-world-model-v1",
  "artifacts_volume": "trencher-baseline-artifacts-v1",
  "feature_build_rows": 889087,
  "feature_build_partitions": 32,
  "feature_build_runtime_sec": 1395.2,
  "family_runtime_sec": {
    "lightgbm": 947.2,
    "xgboost": 460.1,
    "catboost": 465.3,
    "survival": 590.6
  },
  "gpu": {
    "lightgbm": "attempted_with_cpu_fallback",
    "xgboost": "A10G",
    "catboost": "A10G",
    "survival": "CPU"
  },
  "runs": {
    "build_features": "https://modal.com/apps/ronithworks/main/ap-9sYGVbSmvFI87qaXMLSxB1",
    "lightgbm": "https://modal.com/apps/ronithworks/main/ap-yqT2zDOnakMobwUgixMn6n",
    "xgboost": "https://modal.com/apps/ronithworks/main/ap-EFVcWGDJjP6WsTi4OZWqYJ",
    "survival": "https://modal.com/apps/ronithworks/main/ap-fwtGSSwvQy5FZFA1aJh7Qg",
    "catboost": "https://modal.com/apps/ronithworks/main/ap-StPPuL4ThG3P1BUMPXanXb"
  },
  "note": "32 mint-partition Modal population (~889k feature rows / ~732k train / ~157k val); permanent TEST never accessed"
}
```
