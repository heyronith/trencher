# PHASE6_REPORT

## Summary
1. **Champion clearly beats every simple baseline** on explosion_300s (and 60s): XGB frozen V2 PR ahead of logistic/MLP/rules with mint-bootstrap ΔPR CIs excluding 0 (e.g. vs logistic ΔPR≈+0.016).
2. **Broad primary pumps (onset in val):** **n=9182** (not Phase5’s rich-50). XGB@60s top-1% catches **11.3%** (lead p50≈5.2s); top-2% catches **27.8%** (lead p50≈4.9s); ~8–10 false alerts per caught event.
3. **2× target** (explosion@300s ∧ upside≥1): rare (n_pos=269, prev≈0.022%); ROC **0.807**, PR **0.00065**. Holds across days: daily XGB@300s ROC **0.841–0.895**.
4. Alert feed: XGB top-10/hour ≈**13.1%** explode@300s vs **2.1%** base (**6.2×**); 165 hours in window, 54 empty.
5. `reaches_plus_100pct` vs `upside≥1` agree only ~79% where both defined; peak-search cap hits: 0 rows with onset_to_peak≥1799s in obs targets.
6. Entry at T+2s/T+5s: **not available** (no next-trade-after-delay columns).
7. Preregistration: `TEST_PROTOCOL_V1.md` commit `fde6ced` (before any TEST).
8. `q85_test_accessed: false`.

## 1. Fixed pump-level evaluation (broad)

Event counts (onset in [q70,q85)): `{'aggressive': 14561, 'conservative': 5051, 'primary': 9182}`

Phase5 bug note: `{'champion_preds': '/Users/ronny/Desktop/trencher/scratch/phase4_clean_v1/preds/trees_broad_permanent_val.parquet', 'phase5_bug': 'Phase5 filtered explosion_events via token_assignments base_split==val on the RICH target_dataset_v3 table (50 primary events). Phase6 uses broad_targets_v1/explosion_events_v1 with onset_unix in [q70,q85).', 'phase5_n_primary': 50, 'phase6_n_primary_onset_in_window': 9182}`

### Pump recall (XGB)

| defn | head | top | fraction_caught | lead_p50 | FA/caught |
|---|---|---|---:|---:|---:|
| aggressive | explosion_300s | top_0.01 | 0.09807018748712314 | 6.059627056121826 | 6.98109243697479 |
| aggressive | explosion_300s | top_0.02 | 0.22731955222855574 | 5.360332608222961 | 6.013595166163142 |
| conservative | explosion_300s | top_0.01 | 0.12433181548208276 | 5.078984498977661 | 18.073248407643312 |
| conservative | explosion_300s | top_0.02 | 0.28825975054444664 | 4.651036024093628 | 15.60782967032967 |
| primary | explosion_300s | top_0.01 | 0.11119581790459594 | 5.85971999168396 | 10.471106758080314 |
| primary | explosion_300s | top_0.02 | 0.25539098235678503 | 5.063386917114258 | 9.130063965884862 |
| aggressive | explosion_60s | top_0.01 | 0.10054254515486574 | 5.470337629318237 | 6.853825136612022 |
| aggressive | explosion_60s | top_0.02 | 0.24538149852345306 | 5.205343961715698 | 5.532605653512454 |
| conservative | explosion_60s | top_0.01 | 0.12967729162542072 | 4.525562047958374 | 17.485496183206106 |
| conservative | explosion_60s | top_0.02 | 0.32647000593941794 | 4.444913864135742 | 13.734990903577925 |
| primary | explosion_60s | top_0.01 | 0.11261163145284252 | 5.209194540977478 | 10.448742746615087 |
| primary | explosion_60s | top_0.02 | 0.2780439991287301 | 4.901883125305176 | 8.352134743439091 |

### Alert feed

| model | mean_frac | base | lift | hours | empty |
|---|---:|---:|---:|---:|---:|
| lgb | 0.12612612612612611 | 0.02108409085438747 | 5.982051917589799 | 165 | 54 |
| xgb | 0.1306306306306306 | 0.02108409085438747 | 6.195696628932291 | 165 | 54 |

## 2. Simple baselines vs XGB champion

### explosion_300s

| model | ROC | PR | P@1% | ΔPR champ−base CI | alert lift |
|---|---:|---:|---:|---|---:|
| best_single_feature | 0.6640640340600652 | 0.03322359192671852 | 0.02880295250320924 | 0.06506094912533339 [0.06195094137941348, 0.06856147643475577] | 1.6664287484714444 |
| curve_progress_rate | 0.634369043541138 | 0.03881350860647132 | 0.06306161745827984 | 0.0594710324455807 [0.0563036340592085, 0.06312199604710453] | 3.3755864392113875 |
| logistic_l2 | 0.8338884158684785 | 0.07846196000902864 | 0.12379653401797176 | 0.019822581043022605 [0.016826820385067314, 0.022962329761427414] | 4.48653893819235 |
| mlp_2x256 | 0.8088645321494792 | 0.07087449968842648 | 0.10558408215661104 | 0.02741004136362593 [0.024305058394180765, 0.030621061342695443] | 5.298388841293822 |
| net_inflow | 0.5429903723042799 | 0.024938070152515666 | 0.00970795892169448 | 0.07334647089953623 [0.07025672137181085, 0.07704965994728459] | 0.12818682680549573 |
| volume_spike_z | 0.6168009389246786 | 0.0276784658418664 | 0.015324133504492939 | 0.07060607521018557 [0.06769550958021707, 0.07403028864503192] | 0.29910259587949 |
| xgb_champion_frozen_v2 | 0.8576585189473002 | 0.09828454105205248 | 0.14232991014120666 | None None | 6.195696628932293 |

Group ablation / TreeSHAP: see JSON.

Top-10 SHAP:

- token_age_seconds: 0.553017
- latest_v_tok: 0.523914
- token_age_log1p: 0.403025
- latest_price: 0.102131
- sol_mean_w120: 0.055884
- latest_v_sol: 0.050159
- creator_prior_rank: 0.038137
- buy_sol_w30: 0.037450
- sol_mean_w30: 0.033181
- latest_curve_pct: 0.029795

### explosion_60s

| model | ROC | PR | P@1% | ΔPR champ−base CI | alert lift |
|---|---:|---:|---:|---|---:|
| best_single_feature | 0.6831516994627473 | 0.029739352045271405 | 0.026052104208416832 | 0.062164065006367515 [0.05919845859164734, 0.06553299640567986] | 1.6664287484714444 |
| curve_progress_rate | 0.651267693387166 | 0.036242068802591676 | 0.06172344689378757 | 0.05566134824904727 [0.05255763348972397, 0.05908786430269356] | 3.3755864392113875 |
| logistic_l2 | 0.8577210181555615 | 0.07590959440935077 | 0.11959919839679359 | 0.015993822642287908 [0.013013500533448075, 0.019003352718285963] | 4.78564153407184 |
| mlp_2x256 | 0.8434550987393048 | 0.06841157591000072 | 0.08897795591182364 | 0.023491841141638062 [0.0209206039935928, 0.026204857444786175] | 4.913828360877335 |
| net_inflow | 0.5438632617158253 | 0.0205759220190629 | 0.005691382765531062 | 0.07132749503257611 [0.06829151025439288, 0.07471302743813067] | 0.12818682680549573 |
| volume_spike_z | 0.6147768812917566 | 0.022806382160759777 | 0.007615230460921844 | 0.06909703489087916 [0.0662212263164319, 0.07230992747248519] | 0.29910259587949 |
| xgb_champion_frozen_v2 | 0.8734024630446618 | 0.0919034170516391 | 0.1260120240480962 | None None | 5.725678263978809 |

Group ablation / TreeSHAP: see JSON.

Top-10 SHAP:

- token_age_log1p: 0.564934
- token_age_seconds: 0.541391
- latest_v_tok: 0.486755
- latest_price: 0.096699
- buy_sol_w30: 0.049775
- latest_curve_pct: 0.039372
- latest_v_sol: 0.035417
- sol_mean_w120: 0.033887
- global_migrations_24h: 0.029666
- global_launches_1h: 0.027866


## 3. Label sensitivity / 2× target

Data check: `{
  "by_split": {
    "train": {
      "agreement_reach_vs_upside_ge1": {
        "agree_rate": 0.8001187315512607,
        "n_agree": 48520,
        "n_both_defined": 60641,
        "n_disagree": 12121
      },
      "explosion_reaches_plus_100pct": {
        "n_true": 18668,
        "prevalence": 0.0031753359336343222
      },
      "n_onset_to_peak_ge_1799s": 4,
      "n_rows": 5879063,
      "peak_search_cap_sec": 1800.0,
      "remaining_upside_ge_1": {
        "n_defined": 60641,
        "n_true": 12115,
        "prevalence_among_all_rows": 0.0020607025303181815,
        "prevalence_among_defined": 0.1997823254893554
      }
    },
    "val": {
      "agreement_reach_vs_upside_ge1": {
        "agree_rate": 0.7937924454172981,
        "n_agree": 14143,
        "n_both_defined": 17817,
        "n_disagree": 3674
      },
      "explosion_reaches_plus_100pct": {
        "n_true": 7479,
        "prevalence": 0.005865330321852061
      },
      "n_onset_to_peak_ge_1799s": 0,
      "n_rows": 1275120,
      "peak_search_cap_sec": 1800.0,
      "remaining_upside_ge_1": {
        "n_defined": 17817,
        "n_true": 5361,
        "prevalence_among_all_rows": 0.004204310182571052,
        "prevalence_among_defined": 0.30089240612897794
      }
    }
  },
  "note": "train+val only; q85+ TEST never read",
  "q85_test_accessed": false
}`

| target | n_pos | prevalence | ROC | PR |
|---|---:|---:|---:|---:|
| explosion_300s_and_upside_ge_1 | 269 | 0.00022032312966958082 | 0.8067382263769107 | 0.0006450512436539965 |
| explosion_300s_primary_retrain | 26279 | 0.02108409085438747 | 0.8576585189473002 | 0.09828454105205248 |
| explosion_60s_aggressive | 35953 | 0.02819577765229939 | 0.876981036266232 | 0.13788224760389575 |
| explosion_60s_conservative | 12291 | 0.009639092791266705 | 0.8782811158707475 | 0.05441340145975791 |
| explosion_60s_primary_retrain | 22221 | 0.01781308168477555 | 0.8734024630446618 | 0.0919034170516391 |

## 4. Stability

### explosion_300s
- permanent_val: ROC=0.8576585189473002 PR=0.09828454105205248 n_pos=26279
- wf_fold_1: ROC=0.8865084913049953 PR=0.08628269445308373 n_pos=18009
- wf_fold_2: ROC=0.876240447584036 PR=0.06489132759801854 n_pos=16142

### explosion_60s
- permanent_val: ROC=0.8734024630446618 PR=0.0919034170516391 n_pos=22221
- wf_fold_1: ROC=0.8971705069110606 PR=0.08377933972181453 n_pos=15986
- wf_fold_2: ROC=0.8854737185182384 PR=0.05949584784867184 n_pos=13841

Daily permanent-val metrics: see JSON `permanent_val_by_day`.

## 5. Economic context (descriptive)

```json
{
  "all_eligible": {
    "h300_mae": {
      "mean": -0.2645400974693399,
      "median": -0.15735603754378602,
      "n": 516383
    },
    "h300_mfe": {
      "mean": 0.37362474756139763,
      "median": 0.01060294988585131,
      "n": 516383
    },
    "h300_simple_return": {
      "mean": -0.08926381735331855,
      "median": -0.09130812148111134,
      "n": 516383
    }
  },
  "disclaimer": "Descriptive only. Not a trading strategy, PnL, or capacity claim.",
  "entry_delay": {
    "available_proxies": {
      "frac_alerts_with_future_mark_time_after_T": 0.7896510228640192,
      "h300_future_mark_summary_alerts": {
        "mean": 4.991720979214748e-08,
        "median": 2.8333022697552017e-08,
        "n": 9843
      },
      "reference_price_summary_alerts": {
        "mean": null,
        "median": null,
        "n": 0
      }
    },
    "reason": "No next-trade-after-delay price columns exist in training_cache. Closest fields are reference_price (last trade \u2264 T) and h300_future_mark / h300_future_mark_time (last mark in the 300s horizon, not T+2s/T+5s entry). Delayed entry at T+2s / T+5s cannot be computed from available freeze artifacts.",
    "status": "not_available"
  },
  "n_alert": 12465,
  "n_all_eligible": 1246390,
  "q85_test_accessed": false,
  "top1_alerts": {
    "h300_mae": {
      "mean": -0.36808034439815085,
      "median": -0.41243883772503376,
      "n": 9843
    },
    "h300_mfe": {
      "mean": 0.5970624044211622,
      "median": 0.31453759010856164,
      "n": 9843
    },
    "h300_simple_return": {
      "mean": -0.10753118784352485,
      "median": -0.4019351385573704,
      "n": 9843
    }
  },
  "top1_threshold": 0.10637331008911133
}
```

## 6. Test protocol

See `manifests/world_model_data/evaluation/TEST_PROTOCOL_V1.md` (preregistration; no TEST executed).

---
q85_test_accessed: false

