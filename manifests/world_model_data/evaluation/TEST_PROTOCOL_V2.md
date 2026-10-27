# TEST_PROTOCOL_V2 — preregistered q85+ TEST scoring (amendment)

**Status:** written and committed **before** any q85+ TEST access and **before** implementing `score_test_once`.  
**Branch:** `review/phase7-pretest-v1`  
**This document is the V2 preregistration.** The git commit SHA that first introduces this file is the binding preregistration ID **Preregistration SHA:** `3299ee8d9b05286bbac9f9e1a7dc085e8d0de349`.  
**Supersedes:** `TEST_PROTOCOL_V1.md` for the eventual one-shot TEST run. V1 remains historical.

**Freeze reference:** all scoring uses artifacts under `world_model_freeze_v1` / local freeze tree `data/external/world_model/` as documented in `manifests/world_model_data/DATA_FREEZE.md` and `world_model_freeze_v1.json`.

## Scope

Permanent TEST is the observation / mint population with launch time in `[q85, +∞)` per `data/external/world_model/splits/split_v1/cutoffs.json`.  
**Until this protocol is executed in a later phase, no code path may read TEST rows.**

## Training rule (locked)

All models are trained on **train only**, with hyperparameters / iterations / calibration frozen as already done on walk-forward folds and chronological train holdouts.  
**Models are not refit on train+val.** Permanent val was used only for one-shot reporting; TEST is a second one-shot report under the same frozen weights.

## Models (locked) — full list

Score each of the following **once** on TEST (no retuning, no threshold refit on TEST):

1. **XGBoost champion:** frozen Baseline V2 XGBoost broad (fixed iterations from `BASELINE_CORE_METRICS_V2.json`, Platt on chronological train holdout only). Phase-4 / Phase-6 frozen protocol.
2. **Tuned LightGBM:** Phase-5 WF-selected config (no `scale_pos_weight`; fixed iterations = median WF best-iteration).
3. **Stage B:** `stage_b_broad/latest.pt` (hash `ccf27af7…`) in `permanent_val_eval` mode — broad population only.
4. **logistic_l2:** Phase-6 best simple baseline by permanent-val PR-AUC on `explosion_300s` variant `all` (explicitly named; C chosen on wf_fold_1/2 only).
5. **Age + curve XGBoost:** features restricted to `token_age_*`, `latest_v_tok`, `latest_v_sol`, `latest_price`, `latest_curve_pct`; frozen V2 XGB params; fixed iterations = median best on wf_fold_1/2 for this feature subset. Age-matched baseline for “is it just age × curve state?”

## Heads and metrics

**Heads**

- `explosion_60s`, `explosion_300s` (primary definition)
- Lead-time variants: (all), (a) exclude already-started, (c) onset ≥ 5 s
- **Fixed 2× target (reported head):** `explosion_within_300s == True` AND `explosion_reaches_plus_100pct == True` (pump magnitude ≥ +100% from onset). Not the Phase-6 `remaining_upside ≥ 1` construction.

**Strata (report ROC, PR, n, n_pos, prevalence, P@1% within each)**

1. By trigger: `fixed_age` vs `event_count` (`observation_trigger`).
2. By age bucket: [0, 30 s), [30 s, 2 m), [2 m, 10 m), [10 m, 1 h), ≥ 1 h.
3. Active-only: ≥ 1 trade in the 60 s before T; also ≥ 5 trades in the last 60 s (documented feature proxy if `event_count_w60` is absent).

**Metrics (report all, regardless of outcome)**

- ROC-AUC, PR-AUC (`trencher.baselines.metrics.binary_metrics`)
- Precision / recall at top 0.1%, 0.5%, 1%, 2%
- Alert-feed: per UTC hour, top-10 scores deduped by mint; mean fraction with explosion within 300 s vs base rate
- Pump-level recall on primary-definition events with `onset_unix` in the TEST window; for the 2× head, restrict to events with magnitude ≥ +100%
- Champion Δ vs logistic_l2, Stage B, and age+curve XGB with mint-bootstrap CIs

## Bootstrap

- Cluster: **mint**
- Replicates: **2000**
- Seed: **0**
- Paired Δ metrics as above

## Exact TEST scoring command (to be run once, later)

```bash
# From repo root, after confirming this file's commit SHA is the preregistration tip:
uv run python -m scripts.review.phase7.score_test_once \
  --protocol manifests/world_model_data/evaluation/TEST_PROTOCOL_V2.md \
  --preregistration-sha <COMMIT_SHA_OF_THIS_FILE> \
  --out manifests/world_model_data/evaluation/test_once_v1/
```

`score_test_once` must:

1. Verify `--preregistration-sha` is an ancestor of `HEAD`, and the protocol file is byte-identical to its content at that SHA.
2. Refuse on a dirty git tree.
3. Refuse if `manifests/world_model_data/evaluation/test_once_v1/` already contains any metrics file, or if `TEST_ACCESSED.lock` exists.
4. Write `TEST_ACCESSED.lock` **before** reading TEST rows (timestamp, HEAD SHA, hostname).
5. Load only frozen / locked models — no fitting, calibration refit, or threshold search on TEST.
6. Compute every metric in this V2 document with the same functions used for permanent val.
7. Write `TEST_ONCE_REPORT.{md,json}` with `q85_test_accessed: true`, the preregistration SHA, HEAD SHA, and model hashes.
8. Support `--dry-run-on-val` (permanent val only; no TEST) for reproduction checks.

## Outcome policy

Results will be reported **regardless of outcome** (positive, null, or negative). No post-hoc model swaps, threshold fishing, or silent omission of heads or strata.

## Non-goals

- No hyperparameter search on TEST
- No calibration refit on TEST
- No merging of TRAIN/VAL into TEST scoring
- No claims beyond the preregistered metric list
- No train+val refit of any locked model
