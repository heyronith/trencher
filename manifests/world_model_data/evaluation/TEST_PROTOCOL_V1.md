# TEST_PROTOCOL_V1 — preregistered q85+ TEST scoring

**Status:** written and committed **before** any q85+ TEST access.  
**Branch:** `review/phase6-paper-v1`  
**This document is the preregistration.** The git commit SHA that first introduces this file is the binding preregistration ID.

## Scope

Permanent TEST is the observation / mint population with launch time in `[q85, +∞)` per `data/external/world_model/splits/split_v1/cutoffs.json`.  
**Until this protocol is executed in a later phase, no code path may read TEST rows.**

## Models (locked)

Score each of the following **once** on TEST (no retuning, no threshold refit on TEST):

1. **Champion:** frozen Baseline V2 **XGBoost** broad (fixed iterations from `BASELINE_CORE_METRICS_V2.json`, Platt calibration on chronological train holdout only). Predictions reconstructed with the Phase-4 / Phase-6 frozen protocol.
2. **Secondary tree:** Phase-5 **tuned LightGBM** (WF-selected config without `scale_pos_weight`, fixed iterations = median WF best-iteration).
3. **Stage B:** `stage_b_broad/latest.pt` (hash `ccf27af7…`) in `permanent_val_eval` mode — broad population only.
4. **Best simple baseline:** the Phase-6 winner among {volume spike z-score, net inflow, curve-progress rate, best single feature, L2 logistic, MLP 2×256} by permanent-val PR-AUC on `explosion_300s` variant `all` (recorded in `PHASE6_REPORT.json`). If ties, prefer logistic over unfitted rules, then MLP.

## Heads and metrics

**Heads**

- `explosion_60s`, `explosion_300s` (primary definition)
- Lead-time variants: (all), (a) exclude already-started, (c) onset ≥ 5 s
- Optional reporting only (not selection): 2× target from Phase 6 if labels exist on TEST

**Metrics (report all, regardless of outcome)**

- ROC-AUC, PR-AUC (`trencher.baselines.metrics.binary_metrics`)
- Precision / recall at top 0.1%, 0.5%, 1%, 2%
- Alert-feed: per UTC hour, top-10 scores deduped by mint; mean fraction with explosion within 300 s vs base rate
- Pump-level recall on primary-definition events with `onset_unix ∈ [q85, launch_max]` (TEST window), top 1% / 2%, lead-time percentiles, false alerts per caught event

## Bootstrap

- Cluster: **mint**
- Replicates: **2000**
- Seed: **0**
- Paired Δ metrics: champion − baseline (and champion − Stage B) for ROC-AUC and PR-AUC

## Exact TEST scoring command (to be run once, later)

```bash
# From repo root, after confirming this file's commit SHA is the preregistration tip:
uv run python -m scripts.review.phase6.score_test_once \
  --protocol manifests/world_model_data/evaluation/TEST_PROTOCOL_V1.md \
  --preregistration-sha <COMMIT_SHA_OF_THIS_FILE> \
  --out manifests/world_model_data/evaluation/test_once_v1/
```

`score_test_once` must:

1. Verify `--preregistration-sha` matches `git rev-parse HEAD` at protocol freeze (or an ancestor that introduced this file unchanged).
2. Refuse to run if any prior `test_once_v1` metrics artifact exists.
3. Read TEST rows **only** inside that module.
4. Write `TEST_ONCE_REPORT.{md,json}` with `q85_test_accessed: true` and the preregistration SHA.

## Outcome policy

Results will be reported **regardless of outcome** (positive, null, or negative). No post-hoc model swaps, threshold fishing, or silent omission of heads.

## Non-goals

- No hyperparameter search on TEST
- No calibration refit on TEST
- No merging of TRAIN/VAL into TEST scoring
- No claims beyond the preregistered metric list
