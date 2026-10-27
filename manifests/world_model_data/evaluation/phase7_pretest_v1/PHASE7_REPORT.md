# PHASE7_REPORT

## Summary

1. 2× (within_300s ∧ reaches_plus_100pct): train n_pos=14951 prev=0.00270; val n_pos=6867 prev=0.00555; XGB ROC=0.8532 PR=0.0226 P@1%=0.0266.
2. Phase-6's 269 came from training_cache remaining_upside (~97% null on 300s positives); reach vs upside≥1 disagree_rate=0.211 (onset magnitude vs from-T upside — confirmed).
3. Alert-feed lift=2.9192; pump recall mag≥1 top1% caught=0.0374 top2%=0.0769 (n_events=4787).
4. Age+curve baseline (300s): champ−ageΔROC=0.0213 CI=[0.019807054703146053, 0.02284983490942159]; ΔPR=0.0181 CI=[0.01588406645629677, 0.020444073630892387].
5. Dry-run-on-val matches Phase-6 ROC/PR within 1e-9: True.
6. TEST_PROTOCOL_V2 preregistration SHA=3299ee8d9b05286bbac9f9e1a7dc085e8d0de349. q85_test_accessed: false.

## 1. Fixed 2× target

**Definition:** `explosion_within_300s==True AND explosion_reaches_plus_100pct==True`

| split | n_elig | n_pos | prevalence |
|---|---:|---:|---:|
| train | 5534014 | 14951 | 0.002702 |
| val | 1236188 | 6867 | 0.005555 |

**Why Phase-6 had 269:** Phase-6 filled remaining_upside from training_cache_v1, where the column is ~97% null on explosion_300s positives (many mint partitions entirely null). Positives with null upside were excluded, leaving only ~269. Broad observation_targets has upside defined whenever price_target_available=True and explosion_reaches_plus_100pct on the order of thousands.

**reach vs upside≥1:** disagree_rate=0.2113; explosion_reaches_plus_100pct is magnitude from explosion onset (peak/onset_price − 1 ≥ 1.0). remaining_upside_to_explosion_peak is peak/reference_price_at_T − 1, i.e. from observation time T. When T is after some of the run-up, upside-from-T can be < 1 while onset-magnitude still ≥ 100%, and vice versa if reference_price differs from onset_price — so ~20% disagreement is expected.

Fixed iterations (median WF fold 1/2): `64`

### Permanent-val XGB champion on 2×

| metric | value |
|---|---:|
| ROC-AUC | 0.853210 |
| PR-AUC | 0.022619 |
| P@top_0.001 | 0.021036 |
| P@top_0.005 | 0.026533 |
| P@top_0.01 | 0.026614 |
| P@top_0.02 | 0.028191 |

Alert feed: lift=2.9192, hours=165, empty=54.

Pump-level recall (primary, magnitude ≥ +100%, n=4787): top1% caught=0.0374, top2%=0.0769.

## 2. Stratified evaluation (permanent val)

### explosion_300s

#### By trigger

| trigger | n | n_pos | prev | model | ROC | PR | P@1% |
|---|---:|---:|---:|---|---:|---:|---:|
| event_count | 338260 | 12215 | 0.03611 | logistic_l2 | 0.6789 | 0.0665 | 0.0987 |
| event_count | 338260 | 12215 | 0.03611 | stage_b | 0.6990 | 0.0792 | 0.1108 |
| event_count | 338260 | 12215 | 0.03611 | xgb_champion | 0.7098 | 0.0880 | 0.1649 |
| fixed_age | 908130 | 14064 | 0.01549 | logistic_l2 | 0.8882 | 0.0914 | 0.1362 |
| fixed_age | 908130 | 14064 | 0.01549 | stage_b | 0.8906 | 0.0809 | 0.0808 |
| fixed_age | 908130 | 14064 | 0.01549 | xgb_champion | 0.9096 | 0.1120 | 0.1711 |

#### By age bucket

| bucket | n | n_pos | prev | model | ROC | PR | P@1% |
|---|---:|---:|---:|---|---:|---:|---:|
| >=1h | 174907 | 0 | 0.00000 | logistic_l2 | n/a | n/a | n/a |
| >=1h | 174907 | 0 | 0.00000 | stage_b | n/a | n/a | n/a |
| >=1h | 174907 | 0 | 0.00000 | xgb_champion | n/a | n/a | n/a |
| [0,30s) | 435228 | 23038 | 0.05293 | logistic_l2 | 0.6415 | 0.0846 | 0.1291 |
| [0,30s) | 435228 | 23038 | 0.05293 | stage_b | 0.6589 | 0.0870 | 0.1121 |
| [0,30s) | 435228 | 23038 | 0.05293 | xgb_champion | 0.6853 | 0.1045 | 0.1544 |
| [10m,1h) | 185467 | 57 | 0.00031 | logistic_l2 | 0.8744 | 0.0077 | 0.0081 |
| [10m,1h) | 185467 | 57 | 0.00031 | stage_b | 0.9566 | 0.0080 | 0.0113 |
| [10m,1h) | 185467 | 57 | 0.00031 | xgb_champion | 0.9435 | 0.0139 | 0.0124 |
| [2m,10m) | 208483 | 768 | 0.00368 | logistic_l2 | 0.7489 | 0.0140 | 0.0259 |
| [2m,10m) | 208483 | 768 | 0.00368 | stage_b | 0.8341 | 0.0294 | 0.0489 |
| [2m,10m) | 208483 | 768 | 0.00368 | xgb_champion | 0.8650 | 0.0325 | 0.0532 |
| [30s,2m) | 242305 | 2416 | 0.00997 | logistic_l2 | 0.6588 | 0.0207 | 0.0466 |
| [30s,2m) | 242305 | 2416 | 0.00997 | stage_b | 0.7265 | 0.0346 | 0.0648 |
| [30s,2m) | 242305 | 2416 | 0.00997 | xgb_champion | 0.7825 | 0.0413 | 0.0735 |

#### Active-only

| stratum | n | n_pos | prev | model | ROC | PR | P@1% |
|---|---:|---:|---:|---|---:|---:|---:|
| ge1_trade_last_60s_approx | 619283 | 15909 | 0.02569 | logistic_l2 | 0.7269 | 0.0601 | 0.0961 |
| ge1_trade_last_60s_approx | 619283 | 15909 | 0.02569 | stage_b | 0.7477 | 0.0729 | 0.1130 |
| ge1_trade_last_60s_approx | 619283 | 15909 | 0.02569 | xgb_champion | 0.7631 | 0.0808 | 0.1452 |
| ge5_trades_last_60s_approx | 232712 | 5083 | 0.02184 | logistic_l2 | 0.7122 | 0.0474 | 0.0743 |
| ge5_trades_last_60s_approx | 232712 | 5083 | 0.02184 | stage_b | 0.7280 | 0.0588 | 0.0898 |
| ge5_trades_last_60s_approx | 232712 | 5083 | 0.02184 | xgb_champion | 0.7546 | 0.0657 | 0.1139 |

#### Age + curve baseline (explosion_300s)

Features: `['token_age_seconds', 'token_age_log1p', 'latest_v_tok', 'latest_v_sol', 'latest_price', 'latest_curve_pct']`; fixed_iterations=`50`

| metric | age+curve | champ − age+curve | CI95 |
|---|---:|---:|---|
| ROC | 0.8364 | 0.0213 | [0.019807054703146053, 0.02284983490942159] |
| PR | 0.0802 | 0.0181 | [0.01588406645629677, 0.020444073630892387] |

### explosion_60s

#### By trigger

| trigger | n | n_pos | prev | model | ROC | PR | P@1% |
|---|---:|---:|---:|---|---:|---:|---:|
| event_count | 338538 | 10279 | 0.03036 | logistic_l2 | 0.7101 | 0.0621 | 0.0919 |
| event_count | 338538 | 10279 | 0.03036 | stage_b | 0.7191 | 0.0734 | 0.1016 |
| event_count | 338538 | 10279 | 0.03036 | xgb_champion | 0.7260 | 0.0780 | 0.1223 |
| fixed_age | 908916 | 11942 | 0.01314 | logistic_l2 | 0.9089 | 0.0906 | 0.1314 |
| fixed_age | 908916 | 11942 | 0.01314 | stage_b | 0.9059 | 0.0770 | 0.0732 |
| fixed_age | 908916 | 11942 | 0.01314 | xgb_champion | 0.9239 | 0.1091 | 0.1571 |

#### By age bucket

| bucket | n | n_pos | prev | model | ROC | PR | P@1% |
|---|---:|---:|---:|---|---:|---:|---:|
| >=1h | 175053 | 0 | 0.00000 | logistic_l2 | n/a | n/a | n/a |
| >=1h | 175053 | 0 | 0.00000 | stage_b | n/a | n/a | n/a |
| >=1h | 175053 | 0 | 0.00000 | xgb_champion | n/a | n/a | n/a |
| [0,30s) | 435559 | 20589 | 0.04727 | logistic_l2 | 0.6593 | 0.0796 | 0.1173 |
| [0,30s) | 435559 | 20589 | 0.04727 | stage_b | 0.6707 | 0.0809 | 0.1088 |
| [0,30s) | 435559 | 20589 | 0.04727 | xgb_champion | 0.6951 | 0.0958 | 0.1219 |
| [10m,1h) | 185660 | 15 | 0.00008 | logistic_l2 | 0.9122 | 0.0192 | 0.0038 |
| [10m,1h) | 185660 | 15 | 0.00008 | stage_b | 0.9758 | 0.0025 | 0.0027 |
| [10m,1h) | 185660 | 15 | 0.00008 | xgb_champion | 0.9492 | 0.0097 | 0.0038 |
| [2m,10m) | 208670 | 243 | 0.00116 | logistic_l2 | 0.7333 | 0.0041 | 0.0096 |
| [2m,10m) | 208670 | 243 | 0.00116 | stage_b | 0.8440 | 0.0109 | 0.0153 |
| [2m,10m) | 208670 | 243 | 0.00116 | xgb_champion | 0.8690 | 0.0114 | 0.0187 |
| [30s,2m) | 242512 | 1374 | 0.00567 | logistic_l2 | 0.6806 | 0.0128 | 0.0243 |
| [30s,2m) | 242512 | 1374 | 0.00567 | stage_b | 0.7352 | 0.0206 | 0.0392 |
| [30s,2m) | 242512 | 1374 | 0.00567 | xgb_champion | 0.7919 | 0.0251 | 0.0412 |

#### Active-only

| stratum | n | n_pos | prev | model | ROC | PR | P@1% |
|---|---:|---:|---:|---|---:|---:|---:|
| ge1_trade_last_60s_approx | 619800 | 12753 | 0.02058 | logistic_l2 | 0.7687 | 0.0568 | 0.0884 |
| ge1_trade_last_60s_approx | 619800 | 12753 | 0.02058 | stage_b | 0.7752 | 0.0678 | 0.1097 |
| ge1_trade_last_60s_approx | 619800 | 12753 | 0.02058 | xgb_champion | 0.7869 | 0.0722 | 0.1207 |
| ge5_trades_last_60s_approx | 232909 | 3676 | 0.01578 | logistic_l2 | 0.7674 | 0.0436 | 0.0666 |
| ge5_trades_last_60s_approx | 232909 | 3676 | 0.01578 | stage_b | 0.7611 | 0.0534 | 0.0889 |
| ge5_trades_last_60s_approx | 232909 | 3676 | 0.01578 | xgb_champion | 0.7853 | 0.0588 | 0.0962 |

#### Age + curve baseline (explosion_60s)

Features: `['token_age_seconds', 'token_age_log1p', 'latest_v_tok', 'latest_v_sol', 'latest_price', 'latest_curve_pct']`; fixed_iterations=`50`

| metric | age+curve | champ − age+curve | CI95 |
|---|---:|---:|---|
| ROC | 0.8540 | 0.0194 | [0.01791030222101956, 0.020806919580253783] |
| PR | 0.0756 | 0.0163 | [0.01438921109369592, 0.01842857183470496] |

## 3. TEST_PROTOCOL_V2

Preregistration SHA: `3299ee8d9b05286bbac9f9e1a7dc085e8d0de349`

Protocol path: `/Users/ronny/Desktop/trencher/manifests/world_model_data/evaluation/TEST_PROTOCOL_V2.md`

## 4. Scorer dry-run on permanent val

matches_phase6_within_1e-9: **True**

```json
{
  "explosion_60s": {
    "ref_roc_auc": 0.8734024630446618,
    "ref_pr_auc": 0.0919034170516391,
    "got_roc_auc": 0.8734024630446618,
    "got_pr_auc": 0.0919034170516391,
    "abs_delta_roc": 0.0,
    "abs_delta_pr": 0.0,
    "within_1e-9_roc": true,
    "within_1e-9_pr": true
  },
  "explosion_300s": {
    "ref_roc_auc": 0.8576585189473002,
    "ref_pr_auc": 0.09828454105205248,
    "got_roc_auc": 0.8576585189473002,
    "got_pr_auc": 0.09828454105205248,
    "abs_delta_roc": 0.0,
    "abs_delta_pr": 0.0,
    "within_1e-9_roc": true,
    "within_1e-9_pr": true
  }
}
```

## Notes

- q85_test_accessed: false — scorer never read TEST rows.
- All models trained on train only; not refit on train+val.

