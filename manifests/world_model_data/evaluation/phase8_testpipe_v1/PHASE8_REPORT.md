# PHASE8_REPORT

## Summary

1. Frozen 10 models under scratch/test_models_v2/; all val_reproduction_max_abs_diff ≤ 1e-6.
2. Full val pipeline runtime_sec=2226.7, peak_rss_mb=6873.0.
3. Reproduction table: 30 pass / 0 fail (tol 1e-6).
4. TEST population (metadata only): mints=116027 obs=1295766 launch=[1783459141.407781, 1784041302.759724].
5. V2 prereg SHA=3299ee8d9b05286bbac9f9e1a7dc085e8d0de349. q85_test_accessed: false. --split test not run.

## 1. Frozen models

Manifest: `manifests/world_model_data/evaluation/TEST_MODEL_MANIFEST_V2.json`

| model_id | sha256 | fixed_iters | val_max_abs_diff |
|---|---|---:|---:|
| age_curve_xgb_explosion_300s | `2a36252eee37…` | 50 | 0.0 |
| age_curve_xgb_explosion_60s | `0b86849855c1…` | 50 | 0.0 |
| lgbm_tuned_explosion_300s | `52a15c8dfa2e…` | 50 | 0.0 |
| lgbm_tuned_explosion_60s | `19b086538bb2…` | 50 | 0.0 |
| logistic_l2_explosion_300s | `3456b6bdf373…` | None | 0.0 |
| logistic_l2_explosion_60s | `2dd9dfe421f1…` | None | 0.0 |
| stage_b_broad | `ccf27af79bf1…` | None | 0.0 |
| xgb_champion_explosion_300s | `fa58b263c02a…` | 50 | 0.0 |
| xgb_champion_explosion_60s | `5dadc6d1202b…` | 50 | 0.0 |
| xgb_champion_twox_reaches_plus_100pct | `5fc369cc7240…` | 64 | 0.0 |

## 2. Val pipeline reproduction (tol 1e-6)

runtime_sec=2226.7074179649353 peak_rss_mb=6873.0

| metric | phase-report value | dry-run value | abs diff | pass |
|---|---:|---:|---:|:---:|
| explosion_60s/xgb_champion/roc_auc | 0.8734024630446618 | 0.8734024630446618 | 0.0 | PASS |
| explosion_60s/xgb_champion/pr_auc | 0.0919034170516391 | 0.0919034170516391 | 0.0 | PASS |
| explosion_60s/logistic_l2/roc_auc | 0.8577210181555615 | 0.8577210181555615 | 0.0 | PASS |
| explosion_60s/logistic_l2/pr_auc | 0.07590959440935077 | 0.07590959440935077 | 0.0 | PASS |
| explosion_60s/lgbm_tuned/roc_auc | 0.8734519873901027 | 0.8734519873901027 | 0.0 | PASS |
| explosion_60s/lgbm_tuned/pr_auc | 0.08981779108446239 | 0.08981779108446239 | 0.0 | PASS |
| explosion_60s/age_curve_xgb/roc_auc | 0.8540384384127677 | 0.8540384384127677 | 0.0 | PASS |
| explosion_60s/age_curve_xgb/pr_auc | 0.07556788685180386 | 0.07556788685180386 | 0.0 | PASS |
| explosion_60s/stage_b/roc_auc | 0.858235262339151 | 0.858235262339151 | 0.0 | PASS |
| explosion_60s/stage_b/pr_auc | 0.07725719290647641 | 0.07725719290647641 | 0.0 | PASS |
| explosion_60s/strata/fixed_age/n | 908916.0 | 908916.0 | 0.0 | PASS |
| explosion_60s/strata/fixed_age/xgb_champion/roc_auc | 0.923903576144711 | 0.923903576144711 | 0.0 | PASS |
| explosion_300s/xgb_champion/roc_auc | 0.8576585189473002 | 0.8576585189473002 | 0.0 | PASS |
| explosion_300s/xgb_champion/pr_auc | 0.09828454105205248 | 0.09828454105205248 | 0.0 | PASS |
| explosion_300s/logistic_l2/roc_auc | 0.8338884158684785 | 0.8338884158684785 | 0.0 | PASS |
| explosion_300s/logistic_l2/pr_auc | 0.07846196000902864 | 0.07846196000902864 | 0.0 | PASS |
| explosion_300s/lgbm_tuned/roc_auc | 0.8560427464629519 | 0.8560427464629519 | 0.0 | PASS |
| explosion_300s/lgbm_tuned/pr_auc | 0.09513737493211791 | 0.09513737493211791 | 0.0 | PASS |
| explosion_300s/age_curve_xgb/roc_auc | 0.8363519358731858 | 0.8363519358731858 | 0.0 | PASS |
| explosion_300s/age_curve_xgb/pr_auc | 0.08018750684665794 | 0.08018750684665794 | 0.0 | PASS |
| explosion_300s/stage_b/roc_auc | 0.8410435909186866 | 0.8410435909186866 | 0.0 | PASS |
| explosion_300s/stage_b/pr_auc | 0.0818620459666531 | 0.0818620459666531 | 0.0 | PASS |
| explosion_300s/strata/fixed_age/n | 908130.0 | 908130.0 | 0.0 | PASS |
| explosion_300s/strata/fixed_age/xgb_champion/roc_auc | 0.909562133275878 | 0.909562133275878 | 0.0 | PASS |
| twox/n_pos | 6867.0 | 6867.0 | 0.0 | PASS |
| twox/roc_auc | 0.8532102769799242 | 0.8532102769799242 | 0.0 | PASS |
| twox/pr_auc | 0.022619123676620015 | 0.022619123676620015 | 0.0 | PASS |
| twox/alert_lift | 2.919221187111096 | 2.919221187111096 | 0.0 | PASS |
| twox/pump_recall_top1pct | 0.037392939210361395 | 0.037392939210361395 | 0.0 | PASS |
| explosion_300s/alert_lift | 6.195696628932291 | 6.195696628932291 | 0.0 | PASS |

**All pass:** True (30/30)

## 3. TEST population (metadata only)

- n_test_mints: 116027
- n_test_observations: 1295766
- launch_time_range_unix: {'min': 1783459141.407781, 'max': 1784041302.759724}
- q85_unix: 1783459140.3081825

Source: token_assignments.parquet + observation_assignments_broad_stats.json + cutoffs.json

## Notes

- q85_test_accessed: false
- `--split test` was not run
- TEST_PROTOCOL_V2.md was not edited

