# Phase 2 Diagnostics Report

`q85_test_accessed: false` · branch `review/phase2-v1` · CPU only

## Summary

1. Stage-B init weights match Stage-B `latest.pt` tensors exactly; file SHA differs (`5e442388e9de…` vs `ccf27af7…`). Rich-val explosion_60s ROC: Stage B 0.8702, Stage C 0.8945, LGBM-B 0.9168, LGBM-A 0.8348.
2. Mint-bootstrap ΔAUROC Stage C−B (60s): 0.0243 95% CI [-0.0075, 0.0542].
3. Dropping rich group **microstructure** hurts arm-B explosion_60s most (ROC 0.8811 vs full 0.9168).
4. Lead-time variant (a) exclude already-started: explosion_60s n_pos=114.
5. Pump-level top-2% recall (LGBM-B, 50 primary val events): 32.00% caught.
6. Survival positives require genuine Pump-trade activity at/after T+H (`compute_activity_survival`, line 334).
7. Activity vs graduation: among graduated rich-train mints with events, 2 have last trade after migration; 0 before.
8. Rich subset: stratified unbiased draw capped at 4000 (`select_unbiased_rich_subset`); 3623 train+val; fusion_complete=3132.
9. Fusion cost plan is ownership-backfill scoped; no unit costs for N=25k/100k/all-broad.
10. LOADER_AUDIT rewriter: `test_broad_loader_audit_sample` via `uv run pytest tests/test_world_model_frozen_v4.py::test_broad_loader_audit_sample -q`.

## 1. Stage B vs Stage C on rich val (CPU)

stage_c_init_150000.pt is a phase-transition copy of Stage-B latest.pt: all 396 model tensors are byte-identical, but the checkpoint file SHA differs because non-weight fields differ (phase broad→rich, extra.phase_transition, optimizer/rng/scheduler/telemetry). Stage-B latest SHA is ccf27af79bf178f095c70a84e48ad2e573bae8d5cfcfdde9d32fbb0a1f7aa208; init SHA is 5e442388e9dee7f874b1367d8c0a52884d6c7a1233e030b634ea5ae9d73fc2fc.

- `--allow-dirty` used: **False** (tree was clean; scratch gitignored).
- Scoring note: ROC-AUC recomputed with trencher.baselines.metrics.binary_metrics on sigmoid horizon logits. step17 summarize_prediction_artifact AUROC is unreliable due to a rank-indexing bug in world_model.evaluation.metrics._safe_auc (uses y_ord against original-order ranks).

### explosion_60s

| model | ROC-AUC | PR-AUC | P@1% | P@2% | P@5% | n_pos |
|---|---:|---:|---:|---:|---:|---:|
| lgbm_a | 0.8348 | 0.0744 | 0.1500 | 0.1304 | 0.0698 | 114 |
| lgbm_b | 0.9168 | 0.1143 | 0.2125 | 0.1615 | 0.1147 | 114 |
| stage_b | 0.8702 | 0.0693 | 0.0625 | 0.0807 | 0.0873 | 114 |
| stage_c | 0.8945 | 0.0896 | 0.1250 | 0.1056 | 0.0948 | 114 |
| stage_c_periodic_174000 | 0.8578 | 0.0714 | 0.1375 | 0.1180 | 0.0798 | 114 |

Bootstrap (mint, 2000, seed 0):

| contrast | metric | delta | 95% CI |
|---|---|---:|---|
| stage_b_minus_lgbm_a_pr | pr_auc | -0.0050 | [-0.0576, 0.0338] |
| stage_b_minus_lgbm_a_roc | roc_auc | 0.0355 | [-0.0022, 0.0727] |
| stage_c_minus_stage_b_pr | pr_auc | 0.0203 | [-0.0161, 0.0626] |
| stage_c_minus_stage_b_roc | roc_auc | 0.0243 | [-0.0075, 0.0542] |

### explosion_300s

| model | ROC-AUC | PR-AUC | P@1% | P@2% | P@5% | n_pos |
|---|---:|---:|---:|---:|---:|---:|
| lgbm_a | 0.8242 | 0.0699 | 0.1375 | 0.0875 | 0.0773 | 133 |
| lgbm_b | 0.9035 | 0.1101 | 0.1500 | 0.1313 | 0.1172 | 133 |
| stage_b | 0.8272 | 0.0643 | 0.0625 | 0.0813 | 0.0823 | 133 |
| stage_c | 0.8750 | 0.0891 | 0.1250 | 0.1062 | 0.1122 | 133 |
| stage_c_periodic_174000 | 0.8451 | 0.0715 | 0.1375 | 0.1250 | 0.0848 | 133 |

Bootstrap (mint, 2000, seed 0):

| contrast | metric | delta | 95% CI |
|---|---|---:|---|
| stage_b_minus_lgbm_a_pr | pr_auc | -0.0056 | [-0.0411, 0.0204] |
| stage_b_minus_lgbm_a_roc | roc_auc | 0.0030 | [-0.0304, 0.0417] |
| stage_c_minus_stage_b_pr | pr_auc | 0.0248 | [-0.0062, 0.0625] |
| stage_c_minus_stage_b_roc | roc_auc | 0.0478 | [0.0110, 0.0833] |

## 2. Rich feature group ablations (LightGBM arm B)

### Column → group map

| column | group |
|---|---|
| `graph_buy_edges` | graph |
| `graph_n` | graph |
| `graph_sell_edges` | graph |
| `graph_sol_sum` | graph |
| `graph_token_sum` | graph |
| `mask_graph` | graph |
| `mask_micro` | microstructure |
| `mask_ownership` | ownership |
| `mask_wallet_prior` | wallet_prior |
| `micro_account_key_mean` | microstructure |
| `micro_cu_mean` | microstructure |
| `micro_fee_mean` | microstructure |
| `micro_n` | microstructure |
| `micro_priority_fee_mean` | microstructure |
| `micro_success_rate` | microstructure |
| `own_delta_abs_mean` | ownership |
| `own_delta_sum` | ownership |
| `own_n` | ownership |
| `own_value_sum` | ownership |
| `wallet_prior_n` | wallet_prior |
| `wallet_prior_sol_sum` | wallet_prior |
| `wallet_prior_unique_approx` | wallet_prior |

### explosion_300s
Full B ROC=0.9035 (committed 0.9035; within ±0.005: True).

| ablation | ROC-AUC | PR-AUC | ΔROC vs full (drop−full) CI |
|---|---:|---:|---|
| full_B | 0.9035 | 0.1101 | — |
| drop_graph | 0.8964 | 0.1002 | -0.0071 [-0.0133, 0.0005] |
| drop_microstructure | 0.8439 | 0.0814 | -0.0596 [-0.0876, -0.0293] |
| drop_ownership | 0.8826 | 0.0805 | -0.0209 [-0.0322, -0.0103] |
| drop_wallet_prior | 0.8939 | 0.1016 | -0.0095 [-0.0261, 0.0056] |
| broad_only | 0.8242 | 0.0699 | — |
| broad_plus_graph | 0.8087 | 0.0574 | — |
| broad_plus_microstructure | 0.8993 | 0.0948 | — |
| broad_plus_ownership | 0.8585 | 0.0803 | — |
| broad_plus_wallet_prior | 0.7966 | 0.0565 | — |

Top-30 gain importance (full B):

| rank | feature | group | gain |
|---:|---|---|---:|
| 1 | `micro_n` | microstructure | 4546.1709 |
| 2 | `token_age_seconds` | broad | 1450.4634 |
| 3 | `latest_v_tok` | broad | 1137.6476 |
| 4 | `global_launches_1h` | broad | 1055.6175 |
| 5 | `global_bucket_trades_sum` | broad | 1054.1789 |
| 6 | `wallet_prior_sol_sum` | wallet_prior | 1016.9230 |
| 7 | `global_bucket_buy_sol_sum` | broad | 962.6052 |
| 8 | `micro_cu_mean` | microstructure | 807.2715 |
| 9 | `global_bucket_sell_sol_sum` | broad | 773.7317 |
| 10 | `global_launches_24h` | broad | 751.0993 |
| 11 | `micro_fee_mean` | microstructure | 727.1550 |
| 12 | `creator_prior_rank` | broad | 600.4329 |
| 13 | `global_bucket_imbalance` | broad | 571.9156 |
| 14 | `own_value_sum` | ownership | 557.1923 |
| 15 | `sol_std_wfull` | broad | 529.3088 |
| 16 | `micro_account_key_mean` | microstructure | 355.8152 |
| 17 | `sol_mean_w30` | broad | 339.9565 |
| 18 | `own_delta_abs_mean` | ownership | 319.0168 |
| 19 | `net_sol_w30` | broad | 282.8475 |
| 20 | `token_age_log1p` | broad | 273.9172 |
| 21 | `wallet_prior_n` | wallet_prior | 265.3897 |
| 22 | `daily_network_buy_volume_sol` | broad | 215.6203 |
| 23 | `mask_creator` | broad | 175.4034 |
| 24 | `daily_network_trades` | broad | 174.7657 |
| 25 | `graph_sol_sum` | graph | 165.2967 |
| 26 | `daily_network_launches` | broad | 153.6664 |
| 27 | `net_sol_w120` | broad | 145.5298 |
| 28 | `creator_history_n` | broad | 142.4319 |
| 29 | `flow_imbalance_w120` | broad | 136.5847 |
| 30 | `latest_price` | broad | 134.8512 |

### explosion_60s
Full B ROC=0.9168 (committed 0.9168; within ±0.005: True).

| ablation | ROC-AUC | PR-AUC | ΔROC vs full (drop−full) CI |
|---|---:|---:|---|
| full_B | 0.9168 | 0.1143 | — |
| drop_graph | 0.9151 | 0.1071 | -0.0018 [-0.0105, 0.0059] |
| drop_microstructure | 0.8811 | 0.1083 | -0.0357 [-0.0613, -0.0093] |
| drop_ownership | 0.9099 | 0.1089 | -0.0070 [-0.0161, 0.0014] |
| drop_wallet_prior | 0.9050 | 0.0912 | -0.0119 [-0.0267, 0.0006] |
| broad_only | 0.8348 | 0.0744 | — |
| broad_plus_graph | 0.8276 | 0.0531 | — |
| broad_plus_microstructure | 0.9001 | 0.0964 | — |
| broad_plus_ownership | 0.8753 | 0.0902 | — |
| broad_plus_wallet_prior | 0.8334 | 0.0638 | — |

Top-30 gain importance (full B):

| rank | feature | group | gain |
|---:|---|---|---:|
| 1 | `micro_n` | microstructure | 4394.6579 |
| 2 | `token_age_seconds` | broad | 1489.1258 |
| 3 | `global_launches_1h` | broad | 1177.5097 |
| 4 | `sol_mean_w30` | broad | 1066.5675 |
| 5 | `global_bucket_trades_sum` | broad | 1054.4010 |
| 6 | `latest_v_tok` | broad | 1046.7231 |
| 7 | `wallet_prior_sol_sum` | wallet_prior | 1004.3315 |
| 8 | `global_bucket_buy_sol_sum` | broad | 870.0619 |
| 9 | `global_launches_24h` | broad | 729.6048 |
| 10 | `micro_cu_mean` | microstructure | 717.2856 |
| 11 | `creator_prior_rank` | broad | 682.3666 |
| 12 | `global_bucket_sell_sol_sum` | broad | 619.8218 |
| 13 | `global_bucket_imbalance` | broad | 541.3868 |
| 14 | `own_value_sum` | ownership | 529.4989 |
| 15 | `micro_account_key_mean` | microstructure | 458.4161 |
| 16 | `micro_fee_mean` | microstructure | 458.2050 |
| 17 | `net_sol_w30` | broad | 433.1397 |
| 18 | `own_delta_abs_mean` | ownership | 418.9629 |
| 19 | `token_age_log1p` | broad | 277.2290 |
| 20 | `wallet_prior_n` | wallet_prior | 229.5761 |
| 21 | `daily_network_buy_volume_sol` | broad | 226.4755 |
| 22 | `mask_creator` | broad | 194.9355 |
| 23 | `creator_history_n` | broad | 188.7750 |
| 24 | `curve_velocity_wfull` | broad | 176.4722 |
| 25 | `latest_v_sol` | broad | 171.4590 |
| 26 | `daily_network_trades` | broad | 169.9806 |
| 27 | `interarrival_mean_w30` | broad | 161.2825 |
| 28 | `sol_max_w30` | broad | 158.6723 |
| 29 | `latest_price` | broad | 145.5609 |
| 30 | `daily_network_launches` | broad | 129.5478 |

## 3. Lead-time-aware pump evaluation

### explosion_60s

| variant | n | n_pos | stage_b ROC | stage_c ROC | lgbm_a ROC | lgbm_b ROC |
|---|---:|---:|---:|---:|---:|---:|
| a_exclude_already_started | 9322 | 114 | 0.8640 | 0.8984 | 0.8388 | 0.9129 |
| b_onset_ge_2s | 9299 | 91 | 0.8585 | 0.8914 | 0.8300 | 0.9036 |
| c_onset_ge_5s | 9273 | 65 | 0.8383 | 0.8715 | 0.8247 | 0.8933 |

### explosion_300s

| variant | n | n_pos | stage_b ROC | stage_c ROC | lgbm_a ROC | lgbm_b ROC |
|---|---:|---:|---:|---:|---:|---:|
| a_exclude_already_started | 9322 | 133 | 0.8099 | 0.8777 | 0.8227 | 0.8961 |
| b_onset_ge_2s | 9299 | 110 | 0.7954 | 0.8672 | 0.8146 | 0.8865 |
| c_onset_ge_5s | 9273 | 84 | 0.7609 | 0.8439 | 0.8030 | 0.8812 |

### Pump-level recall (50 primary val events, top 2% on explosion_60s scores)

| model | fraction caught | n_caught | lead p50 (s) |
|---|---:|---:|---:|
| stage_b | 0.200 | 10 | 3.026015043258667 |
| stage_c | 0.260 | 13 | 7.4944751262664795 |
| lgbm_a | 0.260 | 13 | 5.003175973892212 |
| lgbm_b | 0.320 | 16 | 3.063483476638794 |

Remaining peak gain: **could_not_complete** — Could not compute remaining peak gain: need price at T and peak price; latest_price_col=None, peak_mfe_from_onset present but is onset-relative.

## 4. Survival label investigation

- **Code:** `src/trencher/data/world_model/targets/compute.py` :: `compute_activity_survival` (line 334); positive rule at line 365.
- Positive: any genuine future trade with event_time >= T+H (regime==0 trades)
- Negative: no such trade AND pump_trades source covers [T+H, T+H+SURVIVAL_NEGATIVE_CONFIRMATION_SEC] with no unresolved gap

Broad train graduation rate: 0.00213152398591403 (1190/558286).

Last trade vs migration (graduated rich-train with events, n=3): before=0, near=1, after=2.

Sample of 20 train mints is in `PHASE2_REPORT.json` §section4.sample_20_train_mints.

## 5. Data coverage for expansion

Stratified sample by launch day × mayhem × holder tertile × buy tertile using only pre-T launch fields; graduation is post-hoc composition only; Jul 3 outage day excluded; capped at max_mints=4000; deterministic mint rank.

Rich train+val graduation rate: 0.0036; explosion_60s prevalence: {'column': 'explosion_within_60s', 'n': 36362, 'n_pos': 463, 'prevalence': 0.012733072988284473}.

### Cost plan vs N=25k / 100k / all-broad

- **N_25k**: could_not_complete — FUSION_EXPANSION_COST_PLAN covers ownership recovery for the existing rich subset gap (2,846 ownership-only mints → ~3,132 fusion overlap), not per-group cost/time to build microstructure/ownership/graph/wallet_prior for N=25k new mints.
- **N_100k**: could_not_complete — Same as N_25k: plan has no per-mint unit costs for full rich groups.
- **N_all_broad_train_val**: could_not_complete — No frozen unit-cost model for expanding all rich groups to the full broad train+val mint set; plan is ownership-backfill scoped.

Plan actuals: RPC unlocked None, final overlap None.

## Could not complete

- step17 metrics JSON AUROC unreliable (_safe_auc bug); recomputed with binary_metrics.
- Fusion plan lacks unit costs for rich groups at N=25k/100k/all-broad train+val.
- Remaining peak gain after T: Could not compute remaining peak gain: need price at T and peak price; latest_price_col=None, peak_mfe_from_onset present but is onset-relative.

## LOADER_AUDIT rewrite

`tests/test_world_model_frozen_v4.py` / `test_broad_loader_audit_sample` — run `uv run pytest tests/test_world_model_frozen_v4.py::test_broad_loader_audit_sample -q`.
