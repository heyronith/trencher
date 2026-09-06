# World Model Readiness Audit V2

**Status:** `ready_for_first_stage_a_gpu_run` (implementation complete; production GPU not launched in this task)

## What is ready

- Real frozen V4 broad + rich DataLoaders (`FrozenV4IterableDataset` → `ProductionExampleBuilder` → `batch_adapter`)
- Exact frozen target tensors (explosion 10–300s, survival without 10s, 3-way barriers, per-horizon quantiles, execution gross/entry/curve with masks)
- Probabilistic path Student-t NLL (mean/scale/df trained)
- Causal temporal-neighborhood graph (not falsely labeled as streaming TGN memory)
- Production trainer telemetry + health gates + phase weights from planned steps
- Staging manifest V2 over actual frozen stores (~28.6GB enumerated in local discover)
- Colab launch path calls `python -m trencher.world_model.launch` (no scaffold stub message)
- Baseline V2 small artifacts frozen into `manifests/world_model_data/baseline_v2/` with survival status `complete_subsample_train_full_population_validation`
- CPU/real-data tests: loader audit, batch parity, causal reload invariance, forward/backward, path sample, checkpoint resume

## Explicit non-claims

- Survival baseline is **not** full-population-trained (80k person-period fit / full val)
- CatBoost remains `incomplete_compute_cost_stop`
- No Stage A/B/C Colab GPU quota consumed by this task

## First Stage-A command (after approval only)

See `WORLD_MODEL_COLAB_RUNBOOK_V2.md`.
