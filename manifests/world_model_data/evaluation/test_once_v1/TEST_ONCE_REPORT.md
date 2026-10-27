# TEST_ONCE_REPORT

## Provenance

- q85_test_accessed: **true**
- preregistration_sha: `3299ee8d9b05286bbac9f9e1a7dc085e8d0de349`
- head_sha: `9d46c26dae9a3fcad7c2d6bf90d5a76870df391c`
- protocol_match: head_blob_with_sha_annotation
- runtime_sec: 11837.488940954208
- peak_rss_mb: 6607.390625
- bugfix_rerun: **true**
- bugfix_note: batch_adapter TEST refuse (7f0f56a) then OpenMP SIGSEGV on full step17; streaming explosion-only OMP1 infer (9d46c26)

## Model hashes

| model_id | sha256 |
|---|---|
| age_curve_xgb_explosion_300s | `2a36252eee37cc2ad0b3ae6b686924223a17961dc0816e6c9b89a63bc958c089` |
| age_curve_xgb_explosion_60s | `0b86849855c15a87a704c9afce95c12e22e57d8ea1b93f8f4425be8cc429df04` |
| lgbm_tuned_explosion_300s | `52a15c8dfa2ecfb144fe7cad49a10c9999738c27e0e1afdf9919c6b60ae0f92d` |
| lgbm_tuned_explosion_60s | `19b086538bb2b12d0f41d2d0586bd101d4a00f66637b95d10609547a8c0e687d` |
| logistic_l2_explosion_300s | `3456b6bdf373c878e6555fee4de294698b574b426e24d3edc1d555c9e52a712c` |
| logistic_l2_explosion_60s | `2dd9dfe421f1ecd534e2d634b2084b2709d0c54a2d321050e267f46d4f008dbd` |
| stage_b_broad | `ccf27af79bf178f095c70a84e48ad2e573bae8d5cfcfdde9d32fbb0a1f7aa208` |
| xgb_champion_explosion_300s | `fa58b263c02aecc39c1c9131efed36712b94d81f1a0aff55271a17bad8e25c01` |
| xgb_champion_explosion_60s | `5dadc6d1202b6331c9484964cdcde144da98d35fc2e3f4597e3c384c48c85090` |
| xgb_champion_twox_reaches_plus_100pct | `5fc369cc72405da5a452c1a1b5a5177dd1b2162441e3e3b8244a130a450d9b02` |

## Heads (all variants)

### explosion_60s

event_window: [1783459140.3081825, 1784041302.7597249]

#### variant `all` (n=1273441, n_pos=29212)

| model | ROC-AUC | PR-AUC | P@1% |
|---|---:|---:|---:|
| xgb_champion | 0.8658989172436831 | 0.11230646976660458 | 0.1455159415737396 |
| lgbm_tuned | 0.8660020329082475 | 0.10801388689093779 | 0.13185173551122978 |
| logistic_l2 | 0.8703517234050128 | 0.1095201493381997 | 0.1454374116538401 |
| age_curve_xgb | 0.8423291008960705 | 0.09265876471015899 | 0.10177477618972829 |
| stage_b | 0.8517913276353581 | 0.09477551005224841 | 0.1257264017590702 |

Mint-bootstrap Δ (champion − other):

- champion_minus_logistic_l2 ROC: -0.004453 [-0.006777, -0.002060]; PR: 0.002786 [0.000061, 0.005762]
- champion_minus_stage_b ROC: 0.014108 [0.012744, 0.015460]; PR: 0.017531 [0.015650, 0.019589]
- champion_minus_age_curve_xgb ROC: 0.023570 [0.022210, 0.024902]; PR: 0.019648 [0.017524, 0.021855]

#### variant `a_exclude_already_started` (n=1096710, n_pos=29212)

| model | ROC-AUC | PR-AUC | P@1% |
|---|---:|---:|---:|
| xgb_champion | 0.8652460622558931 | 0.1339165172062863 | 0.1990517005562141 |
| lgbm_tuned | 0.8657288723168732 | 0.12909984980191508 | 0.18482720889942555 |
| logistic_l2 | 0.86856405158594 | 0.12727558071114756 | 0.18820096653597154 |
| age_curve_xgb | 0.8442008123955432 | 0.10855170727579139 | 0.14853651864684964 |
| stage_b | 0.8528150340149439 | 0.11307362546360988 | 0.16449348044132397 |

#### variant `c_onset_ge_5s` (n=1084889, n_pos=17391)

| model | ROC-AUC | PR-AUC | P@1% |
|---|---:|---:|---:|
| xgb_champion | 0.8478331855858503 | 0.06930407341364926 | 0.10415706516729653 |
| lgbm_tuned | 0.8488801524301561 | 0.0680377941393018 | 0.09890312471195502 |
| logistic_l2 | 0.8508083892927145 | 0.06928886076362661 | 0.10056226380311549 |
| age_curve_xgb | 0.82723400185878 | 0.05871909601220177 | 0.0765047469812886 |
| stage_b | 0.8379655121146778 | 0.061603905142988205 | 0.08728915107383169 |

#### Strata (champion + baselines)

- trigger=fixed_age: n=914583 champ_roc=0.9205458156968503
- trigger=event_count: n=358858 champ_roc=0.7099488902652181
- age=[0,30s): n=434610 champ_roc=0.6763060035669218
- age=[30s,2m): n=256034 champ_roc=0.7684799344053758
- age=[2m,10m): n=215367 champ_roc=0.8283847351783826
- age=[10m,1h): n=188125 champ_roc=0.8662618743796966
- age=>=1h: n=179305 champ_roc=None
- active=ge1_trade_last_60s_approx: n=644678 champ_roc=0.7681114065270762
- active=ge5_trades_last_60s_approx: n=250692 champ_roc=0.7720829259753981

Pump-level recall (primary): {"n_events": 12225, "event_filter": "onset_unix in broad permanent_val [q70,q85)", "events_source": "/Users/ronny/Desktop/trencher/data/external/world_model/broad/broad_targets_v1/explosion_events_v1.parquet", "by_top_frac": {"top_0.01": {"threshold": 0.10520406812429428, "n_caught": 1033, "fraction_caught": 0.08449897750511247, "lead_time_sec_percentiles": {"p10": 0.5952991962432861, "p25": 2.062152862548828, "p50": 5.239603042602539, "p75": 17.667893171310425, "p90": 64.4514548301698}, "n_aler

### explosion_300s

event_window: [1783459140.3081825, 1784041302.7597249]

#### variant `all` (n=1272439, n_pos=34783)

| model | ROC-AUC | PR-AUC | P@1% |
|---|---:|---:|---:|
| xgb_champion | 0.8446737461436481 | 0.1169995138622853 | 0.16166299905690035 |
| lgbm_tuned | 0.8439616869376361 | 0.11557731802527153 | 0.16182018233259982 |
| logistic_l2 | 0.8446178027011131 | 0.11366193684540153 | 0.16071989940270356 |
| age_curve_xgb | 0.8184799797066795 | 0.09670208174122207 | 0.1102640679031751 |
| stage_b | 0.8293778416098078 | 0.09896874350902446 | 0.1339987425337944 |

Mint-bootstrap Δ (champion − other):

- champion_minus_logistic_l2 ROC: 0.000056 [-0.002299, 0.002458]; PR: 0.003338 [0.000807, 0.005994]
- champion_minus_stage_b ROC: 0.015296 [0.013818, 0.016737]; PR: 0.018031 [0.016247, 0.019953]
- champion_minus_age_curve_xgb ROC: 0.026194 [0.024794, 0.027628]; PR: 0.020297 [0.018296, 0.022413]

#### variant `a_exclude_already_started` (n=1095708, n_pos=34783)

| model | ROC-AUC | PR-AUC | P@1% |
|---|---:|---:|---:|
| xgb_champion | 0.8450994873794533 | 0.1384146859397766 | 0.20909007940129598 |
| lgbm_tuned | 0.8447630966119254 | 0.1376279484301471 | 0.2142009674180889 |
| logistic_l2 | 0.8443552094629033 | 0.13335121997722782 | 0.20553071096102948 |
| age_curve_xgb | 0.8211391704719114 | 0.11345026349308071 | 0.15670347722916858 |
| stage_b | 0.8314129254856355 | 0.11810236344368504 | 0.1752304462900429 |

#### variant `c_onset_ge_5s` (n=1083887, n_pos=22962)

| model | ROC-AUC | PR-AUC | P@1% |
|---|---:|---:|---:|
| xgb_champion | 0.8230576436215423 | 0.0779649872687731 | 0.11855337208229541 |
| lgbm_tuned | 0.8224936026484194 | 0.07813985043877782 | 0.12021404188578282 |
| logistic_l2 | 0.82001415252117 | 0.07649187824505779 | 0.11716948057938924 |
| age_curve_xgb | 0.798242137634819 | 0.06594171112038806 | 0.09004520712242826 |
| stage_b | 0.8118687755441623 | 0.06967235101596013 | 0.10111633914567765 |

#### Strata (champion + baselines)

- trigger=fixed_age: n=913875 champ_roc=0.9023594623475449
- trigger=event_count: n=358564 champ_roc=0.6840442853098342
- age=[0,30s): n=434227 champ_roc=0.6636565876698218
- age=[30s,2m): n=255839 champ_roc=0.7203779833188384
- age=[2m,10m): n=215166 champ_roc=0.8141361262179566
- age=[10m,1h): n=187981 champ_roc=0.9005687525441938
- age=>=1h: n=179226 champ_roc=None
- active=ge1_trade_last_60s_approx: n=644125 champ_roc=0.7312233796826271
- active=ge5_trades_last_60s_approx: n=250473 champ_roc=0.7191169070349643

Alert feed: lift_vs_base=5.566857997847524 mean_frac=0.15217391304347827 base=None hours=169

Pump-level recall (primary): {"n_events": 12225, "event_filter": "onset_unix in broad permanent_val [q70,q85)", "events_source": "/Users/ronny/Desktop/trencher/data/external/world_model/broad/broad_targets_v1/explosion_events_v1.parquet", "by_top_frac": {"top_0.01": {"threshold": 0.1136180964112281, "n_caught": 1262, "fraction_caught": 0.10323108384458078, "lead_time_sec_percentiles": {"p10": 0.8985690116882326, "p25": 2.219486117362976, "p50": 5.047865986824036, "p75": 17.697536766529083, "p90": 68.21410171985634}, "n_aler

## 2× head (`within_300s` ∧ `reaches_plus_100pct`)

- n_elig=1258849 n_pos=11419 prevalence=0.009070984685216416
- ROC=0.8379119809412134 PR=0.032999343689317215
- alert lift=3.594835266927356
- n_primary_events_mag_ge_1=7501

## Val vs TEST (reporting only; no re-selection)

| head | model | val ROC | val PR | TEST ROC | TEST PR |
|---|---|---:|---:|---:|---:|
| explosion_60s | xgb_champion | 0.8734024630446618 | 0.0919034170516391 | 0.8658989172436831 | 0.11230646976660458 |
| explosion_60s | lgbm_tuned | 0.8734519873901027 | 0.08981779108446239 | 0.8660020329082475 | 0.10801388689093779 |
| explosion_60s | logistic_l2 | 0.8577210181555615 | 0.07590959440935077 | 0.8703517234050128 | 0.1095201493381997 |
| explosion_60s | stage_b | 0.858235262339151 | 0.07725719290647641 | 0.8517913276353581 | 0.09477551005224841 |
| explosion_60s | age_curve_xgb | 0.8540384384127677 | 0.07556788685180386 | 0.8423291008960705 | 0.09265876471015899 |
| explosion_300s | xgb_champion | 0.8576585189473002 | 0.09828454105205248 | 0.8446737461436481 | 0.1169995138622853 |
| explosion_300s | lgbm_tuned | 0.8560427464629519 | 0.09513737493211791 | 0.8439616869376361 | 0.11557731802527153 |
| explosion_300s | logistic_l2 | 0.8338884158684785 | 0.07846196000902864 | 0.8446178027011131 | 0.11366193684540153 |
| explosion_300s | stage_b | 0.8410435909186866 | 0.0818620459666531 | 0.8293778416098078 | 0.09896874350902446 |
| explosion_300s | age_curve_xgb | 0.8363519358731858 | 0.08018750684665794 | 0.8184799797066795 | 0.09670208174122207 |

## Scratch pred artifacts (gitignored)

| key | path | sha256 | n |
|---|---|---|---:|
| explosion_300s/age_curve_xgb | `scratch/phase9_test_once_v1/preds/test_explosion_300s_age_curve_xgb.npy` | `299647018457a02859290ae211ffb251f3979929fb6d196591ccf0b5c9a69216` | 1295766 |
| explosion_300s/lgbm_tuned | `scratch/phase9_test_once_v1/preds/test_explosion_300s_lgbm_tuned.npy` | `12e42750bdb26ed5558194a24dd02dfee37768962fc7a04fce5e75d5c4f12131` | 1295766 |
| explosion_300s/logistic_l2 | `scratch/phase9_test_once_v1/preds/test_explosion_300s_logistic_l2.npy` | `5d90ffabd28822d5ebd0a9b0dfda18e704ed56d86ad77d9e9c0ecfd87c463e4a` | 1295766 |
| explosion_300s/stage_b | `scratch/phase9_test_once_v1/preds/test_explosion_300s_stage_b.npy` | `9a97e0e2b924e1e98718545d096cb05c4b2425f209f66d7772f8f9e254adf6e4` | 1295766 |
| explosion_300s/xgb_champion | `scratch/phase9_test_once_v1/preds/test_explosion_300s_xgb_champion.npy` | `79d29ba7487d014ecf38161635c0f8d9bdc53fd500146d2031c78e2814c4afb3` | 1295766 |
| explosion_60s/age_curve_xgb | `scratch/phase9_test_once_v1/preds/test_explosion_60s_age_curve_xgb.npy` | `04a5f92b59f69e40b4b02d2c1ad4a84653f04c4c6c6ac77272ae047931dd04a5` | 1295766 |
| explosion_60s/lgbm_tuned | `scratch/phase9_test_once_v1/preds/test_explosion_60s_lgbm_tuned.npy` | `5e2d4e54d5f131e8116521a0a493182b1180a04af4da1fcba0e8015b8e2df42c` | 1295766 |
| explosion_60s/logistic_l2 | `scratch/phase9_test_once_v1/preds/test_explosion_60s_logistic_l2.npy` | `dc3714718f5d7dc0ab2c32386e775c7de158e935ccdca4b17dd6b28d99ca61aa` | 1295766 |
| explosion_60s/stage_b | `scratch/phase9_test_once_v1/preds/test_explosion_60s_stage_b.npy` | `0ab361736ee263485ee358d919503a58c43361e94c54acba4700e0e0a17e2d31` | 1295766 |
| explosion_60s/xgb_champion | `scratch/phase9_test_once_v1/preds/test_explosion_60s_xgb_champion.npy` | `f7e4ef67b3388621a7ea33a516a35184760480b1cbfa3bc06d0240a18e5f1271` | 1295766 |
| twox/xgb_champion | `scratch/phase9_test_once_v1/preds/test_twox_xgb.npy` | `11d322d731459648c1629685e1e8c4542799770a7ac84f8b3c7584caa2671664` | 1295766 |

