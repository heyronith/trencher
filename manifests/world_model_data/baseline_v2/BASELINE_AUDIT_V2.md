# BASELINE_AUDIT_V2

Generated: `2026-09-06T20:04:54.349832+00:00`

- Label: **final_full_population_validation_control**
- V1 preserved as: `provisional_32_partition_development_run`
- Feature dims: `{'broad': 124, 'rich_additive': 22, 'broad_plus_rich': 146}`
- Feature rows: **7154183** partitions=256

## Family statuses

```json
{
  "lightgbm": {
    "status": "complete",
    "partial": false,
    "inferred": true
  },
  "xgboost": {
    "status": "complete",
    "partial": false,
    "inferred": true
  },
  "catboost": {
    "status": "incomplete_compute_cost_stop",
    "partial": true,
    "reason": "unacceptable_compute_efficiency",
    "operator_requested_compute_stop": true,
    "operator_reported_approx_cost_usd": 20,
    "operator_reported_cost_note": "Approximate operator-reported spend on this CatBoost run; not a machine-measured Modal invoice line item.",
    "partial_artifacts_retained": true,
    "app_id": "ap-o4Ntw6HOYbp2iYCnOsFoon",
    "git_sha_at_stop": "7917b7ef392e3ff7671ea2af27ef1ff64739406a",
    "generated_at": "2026-09-06T18:37:10.877819+00:00",
    "do_not_treat_as_completed_family": true,
    "scientifically_failed": false,
    "fold_inventory": {
      "explosion_300s": {
        "wf_fold_2": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 2.8470520973205566,
          "best_iteration": 22
        },
        "wf_fold_1": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 2.6641745567321777,
          "best_iteration": 27
        },
        "wf_fold_3": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 3.1828131675720215,
          "best_iteration": 24
        },
        "final_fit": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 2.248924493789673,
          "best_iteration": 2
        }
      },
      "barrier_u25_l15_5m": {
        "wf_fold_1": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 4.712094783782959,
          "best_iteration": 350
        },
        "wf_fold_3": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 1.8261127471923828,
          "best_iteration": 68
        },
        "wf_fold_2": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 4.069011688232422,
          "best_iteration": 280
        },
        "final_fit": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 2.727273464202881,
          "best_iteration": 138
        }
      },
      "barrier_u100_l30_1h": {
        "wf_fold_3": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 1.3768823146820068,
          "best_iteration": 38
        },
        "wf_fold_2": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 2.225168466567993,
          "best_iteration": 136
        },
        "wf_fold_1": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 4.1439056396484375,
          "best_iteration": 323
        },
        "final_fit": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 1.5524804592132568,
          "best_iteration": 41
        }
      },
      "return_5m": {
        "wf_fold_2": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 0.8510632514953613,
          "best_iteration": 89
        },
        "wf_fold_1": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 0.7434451580047607,
          "best_iteration": 73
        },
        "wf_fold_3": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 1.037597417831421,
          "best_iteration": 74
        },
        "final_fit": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 1.2763123512268066,
          "best_iteration": 93
        }
      },
      "return_1h": {
        "wf_fold_3": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 0.6174020767211914,
          "best_iteration": 4
        },
        "wf_fold_2": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 0.3407275676727295,
          "best_iteration": 0
        },
        "wf_fold_1": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 0.44290637969970703,
          "best_iteration": 29
        },
        "final_fit": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 0.7994065284729004,
          "best_iteration": 0
        }
      },
      "survival_30m": {
        "wf_fold_2": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 4.326028823852539,
          "best_iteration": 65
        },
        "wf_fold_1": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 3.62764310836792,
          "best_iteration": 55
        },
        "wf_fold_3": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 2.781670331954956,
          "best_iteration": 14
        },
        "final_fit": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 2.6844191551208496,
          "best_iteration": 14
        }
      },
      "survival_24h": {
        "wf_fold_1": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 2.0709331035614014,
          "best_iteration": 5
        },
        "wf_fold_2": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 5.8762640953063965,
          "best_iteration": 395
        },
        "wf_fold_3": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 6.351945877075195,
          "best_iteration": 399
        },
        "final_fit": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 5.846815824508667,
          "best_iteration": 399
        }
      },
      "explosion_60s": {
        "wf_fold_2": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 3.9275972843170166,
          "best_iteration": 55
        },
        "wf_fold_3": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 3.731835126876831,
          "best_iteration": 39
        },
        "final_fit": {
          "progress": 100,
          "message": "COMPLETE",
          "elapsed_sec": 2.5970165729522705,
          "best_iteration": 11
        },
        "wf_fold_1": {
          "progress": 5,
          "message": "gpu fit (Logloss; tree progress)",
          "elapsed_sec": 1.7227566242218018,
          "best_iteration": null
        }
      }
    },
    "note": "CatBoost was stopped to avoid further GPU burn. LightGBM and XGBoost remain the completed tree controls. Partial fold artifacts/progress are preserved for audit only."
  },
  "survival": {
    "status": "complete",
    "partial": false,
    "method": "discrete_hazard",
    "generated_at": "2026-09-06T20:03:36.124027+00:00",
    "prior_lifelines_cox_fullpop": {
      "stopped": true,
      "reason": "pathological_fullpop_lifelines_cox_worker_restarts",
      "app_id": "ap-LAAgokVVzmuQUN6T6G2ZdN"
    }
  }
}
```

## Permanent validation evaluation-only proof

```json
{
  "protocol_version": "v2",
  "label": "final_full_population_validation_control",
  "selection_folds": [
    "wf_fold_1",
    "wf_fold_2"
  ],
  "excluded_from_selection": [
    "wf_fold_3",
    "permanent_val_q70_q85"
  ],
  "permanent_val_used_for": [
    "final_score_once",
    "trading_tables",
    "slice_reports"
  ],
  "permanent_val_forbidden_for": [
    "early_stopping",
    "best_iteration_selection",
    "hyperparameter_selection",
    "threshold_selection",
    "calibration_fitting",
    "model_family_selection"
  ],
  "calibration": "platt_on_chronological_train_holdout_only",
  "final_fit": "fixed_iterations_from_wf1_wf2_median_no_es_on_permanent_val",
  "permanent_test_q85_plus": "untouched"
}
```

## Results summary

```json
{
  "lightgbm": {
    "status": "complete",
    "partial": false,
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
    "train_n": 5879063,
    "val_n": 1275120,
    "protocol_version": "v2",
    "metrics_scope": null
  },
  "xgboost": {
    "status": "complete",
    "partial": false,
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
    "train_n": 5879063,
    "val_n": 1275120,
    "protocol_version": "v2",
    "metrics_scope": null
  },
  "catboost": {
    "status": "incomplete_compute_cost_stop",
    "partial": true,
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
    "val_n": 157188,
    "protocol_version": null,
    "metrics_scope": "partial_or_incomplete"
  },
  "survival": {
    "status": "complete",
    "partial": false,
    "heads": [
      "explosion_tte",
      "graduation_tte"
    ],
    "train_n": 5879063,
    "val_n": 1275120,
    "protocol_version": "v2",
    "metrics_scope": null
  }
}
```

## Modal usage

```json
{
  "app": "trencher-baselines-v1",
  "volume": "trencher-world-model-v1",
  "artifacts_volume": "trencher-baseline-artifacts-v1",
  "label": "final_full_population_validation_control",
  "git_sha": "fd209b8568c38a9a93651c07095c98588e594a55",
  "v1_preserved_as": "provisional_32_partition_development_run"
}
```


## Survival status clarification

- Status: `complete_subsample_train_full_population_validation`
- Train matrix population: 5,879,063
- Max individuals used for person-period hazard fit: 80,000
- Validation population (full): 1,275,120
- Method: discrete-time LightGBM hazard (`discrete_hazard`)
- Do **not** describe survival as full-population-trained.
