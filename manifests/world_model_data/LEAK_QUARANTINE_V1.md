# LEAK quarantine V1

Status: **active**. Scientific claims that depend on the items below are invalid until rebuilt under a causal protocol.

## Microstructure and ownership are LEAK

In `world_model_freeze_v1`, microstructure and ownership features are **LEAK**.

**Cause:** Signature selection for RPC `getTransaction` used lifetime-stratified picks (`select_stratified_trades_for_mint` in `src/trencher/data/world_model/acquire/local_corpus.py`). The selected set can depend on post-T extremes (largest trade, curve extremes, last trade). At observation time T, `micro_n` / ownership tallies only count txs with time &lt; T, but *which* txs were fetched (and therefore which pre-T txs appear in the feature) depends on the full mint lifetime.

**Code location:** `extract_signature_targets` → `select_stratified_trades_for_mint` (leaky; now requires `allow_lifetime_selection=True` for frozen reproduction). Causal replacement: `select_causal_trades_before(trades, t, k)`.

Graph and `wallet_prior` modalities remain **CLEAN** under the freeze audit (phase 3).

## Stage-C checkpoint and metrics are invalid for scientific claims

Stage C was trained with leaky microstructure and ownership in the rich phase. The Stage-C checkpoint (`stage_c_rich/latest.pt`, step 175000) and any metrics / predictions derived from it are **invalid for scientific claims** about generalization. They may still be used as forensic / debugging artifacts.

## Step-17 ablation plan is superseded

`manifests/world_model_data/evaluation/STEP17_SCIENTIFIC_ABLATION_REPORT.*` and the paired rich Stage-C vs Stage-B plan assumed Stage-C modalities were causal. That plan is **superseded**. Clean deep-model comparison on broad data uses Stage B only. Rich clean trees may use arm A plus CLEAN modalities (graph, wallet_prior) and/or causal refetch micro/ownership.

## `_safe_auc` bug (pre-`8aba499`)

`world_model.evaluation.metrics._safe_auc` had a rank-indexing bug (positive ranks selected with ordered labels against original-order ranks). Any AUROC produced through `summarize_prediction_artifact` / `_safe_auc` **before commit `8aba499`** is unreliable when score ties are present.

**Affected committed artifacts (AUROC fields):**

| Artifact | Notes |
|---|---|
| Step-17 / phase-2 / phase-3 metrics JSON written via `run_permanent_val_inference` → `summarize_prediction_artifact` before `8aba499` | e.g. phase-3 `infer_masks/*/_*metrics.json` showed ~0.52 AUROC while `binary_metrics` on the same preds is ~0.85 |
| Any other `*_metrics.json` whose AUROC came from `world_model.evaluation.metrics` before `8aba499` | Recompute with fixed `_safe_auc` or `trencher.baselines.metrics.binary_metrics` |

**Not produced by buggy `_safe_auc`:**

| Artifact | Notes |
|---|---|
| `stage_b_permanent_val_eval.json` (SHA `55b305fe…`) | Written under git `5f895e1` before `world_model.evaluation.metrics` existed; Colab eval path, not Step-17 `_safe_auc` |
| `BASELINE_CORE_METRICS_V2.json` | Tree baselines use `trencher.baselines.metrics.binary_metrics` |
| Frozen Stage-C Colab `permanent_val_metrics_175000.json` | Custom Colab scorer; AUROC matches `binary_metrics` on the frozen preds |

Always prefer `binary_metrics` / fixed `_safe_auc` for review reports.
