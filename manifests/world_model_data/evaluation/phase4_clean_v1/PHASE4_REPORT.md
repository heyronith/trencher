# PHASE4_REPORT

## Summary (10 lines)
1. Stage B ROC@60 − LGBM = -0.0108 (95% CI [-0.0122, -0.0094]); SB=0.8582 LGBM=0.8690 XGB=0.8734.
2. Leak-free micro not rebuilt (no RPC). Clean rich proxy A+graph+wallet ROC@60=0.8473087812416166.
3. Stage-B broad infer: rows=1265314, elapsed_s=8397.627229213715.
4. Committed Stage-B metrics buggy `_safe_auc`? False — stage_b_permanent_val_eval.json written under git 5f895e1 before world_model.evaluation.metrics existed; Colab scorer, n
5. Tree reproduction within ±0.005: {'barrier_u100_l30_1h': {'lgb_within': True, 'xgb_within': True, 'lgb_delta': 5.363565147575855e-11, 'xgb_delta': 0.0022563194249451968}, 'barrier_u25_l15_5m': {'lgb_within': True, 'xgb_within': True, 'lgb_delta': 0.0006210988102470782, 'xgb_delta': -0.001607495219085986}, 'explosion_300s': {'lgb_within': True, 'xgb_within': False, 'lgb_delta': 0.003098009870789009, 'xgb_delta': 0.009495837633115789}, 'explosion_60s': {'lgb_within': True, 'xgb_within': False, 'lgb_delta': 0.0034829672046456928, 'xgb_delta': 0.008609036929455804}}.
6. Stage-C mismatch root cause: Not different rows, labels, masks, dropout, or checkpoint. Identical observation_id_hash order and explosion targets/masks; same Stage-C SHA 11bf593e10e5dcc08915fc0fa2300ce988592f660b3608a5024c4312105
7. Causal trade selection: `select_causal_trades_before` + leak flag on stratified.
8. RPC: configured=False; unique sigs=54144.
9. Quarantine: `manifests/world_model_data/LEAK_QUARANTINE_V1.md`.
10. `q85_test_accessed: false` on all phase-4 outputs.

## 1. Stage B vs LightGBM / XGBoost (broad permanent val)

| head | SB ROC | LGBM ROC | XGB ROC | SB−LGBM ROC CI | SB−XGB ROC CI | Ens−LGBM ROC CI |
|---|---:|---:|---:|---:|---:|---:|
| explosion_60s | 0.8582 | 0.8690 | 0.8734 | -0.0108 [-0.0122, -0.0094] | -0.0152 [-0.0165, -0.0138] | -0.0019 [-0.0026, -0.0012] |
| explosion_300s | 0.8410 | 0.8536 | 0.8577 | -0.0126 [-0.0143, -0.0109] | -0.0166 [-0.0182, -0.0151] | -0.0026 [-0.0035, -0.0018] |
| barrier_u25_l15_5m | 0.5749 | 0.5967 | 0.6038 | -0.0219 [-0.0245, -0.0193] | -0.0290 [-0.0317, -0.0263] | -0.0023 [-0.0036, -0.0010] |
| barrier_u100_l30_1h | 0.5296 | 0.6345 | 0.6341 | -0.1049 [-0.1124, -0.0975] | -0.1044 [-0.1118, -0.0972] | -0.0295 [-0.0333, -0.0258] |

### Precision at top-k (explosion_60s)
| model | 0.1% | 0.5% | 1% | 2% | 5% |
|---|---:|---:|---:|---:|---:|
| stage_b | 0.0986 | 0.1095 | 0.1055 | 0.0988 | 0.0819 |
| lightgbm | 0.1299 | 0.1206 | 0.1278 | 0.1205 | 0.1062 |
| xgboost | 0.1235 | 0.1230 | 0.1260 | 0.1345 | 0.1090 |
| ensemble_stage_b_lgbm | 0.1139 | 0.1212 | 0.1123 | 0.1032 | 0.1020 |

## 2. Clean rich tree baselines

| head | arm | ROC-AUC | PR-AUC | ΔROC vs A (CI) |
|---|---|---:|---:|---|
| explosion_300s | A | 0.8242 | 0.0699 | n/a |
| explosion_300s | A_plus_graph_wallet | 0.7962 | 0.0527 | -0.0280 [-0.0492, -0.0040] |
| explosion_300s | A_plus_graph_wallet_n_trades | 0.7962 | 0.0527 | -0.0280 [-0.0492, -0.0040] |
| explosion_300s | A_plus_n_trades | 0.8242 | 0.0699 | +0.0000 [+0.0000, +0.0000] |
| explosion_60s | A | 0.8348 | 0.0744 | n/a |
| explosion_60s | A_plus_graph_wallet | 0.8473 | 0.0706 | +0.0126 [-0.0074, +0.0329] |
| explosion_60s | A_plus_graph_wallet_n_trades | 0.8473 | 0.0706 | +0.0126 [-0.0074, +0.0329] |
| explosion_60s | A_plus_n_trades | 0.8348 | 0.0744 | +0.0000 [+0.0000, +0.0000] |

## 3. Stage-C 0.850 vs 0.8945 mismatch

Not different rows, labels, masks, dropout, or checkpoint. Identical observation_id_hash order and explosion targets/masks; same Stage-C SHA 11bf593e10e5dcc08915fc0fa2300ce988592f660b3608a5024c43121051d839. Diagnose-masks baseline used CPU fp32 step17 inference (ROC@60=0.8500); frozen preds used Colab CUDA bf16 autocast (ROC@60=0.8945). Logits diverge systematically (corr≈0.95); gap is largest on rows with missing micro/ownership. Secondary: diagnose path git includes empty-sequence pooling fix 9af5548 absent from frozen runtime_git_sha b7abd1de. The 0.850 vs 0.8945 headline is this device/precision (+minor code) gap on the same population, not a population or eval-mode mismatch.

## 4–6. Code fix / RPC / quarantine

- Causal selection + regression test landed.
- RPC section: `{"status": "dry_run_estimate", "n_unique_signatures": null, "n_unique_signatures_estimate": 54144, "rpc_configured": false}`
- See `manifests/world_model_data/LEAK_QUARANTINE_V1.md`.

q85_test_accessed: false
## Lead-time variants

Skipped: Stage-B broad prediction artifact lacks `explosion_already_started` / onset TTE fields, and full `tte_overlays_broad` join was too heavy to complete in this pass. Main permanent-val comparison above remains the primary result.
