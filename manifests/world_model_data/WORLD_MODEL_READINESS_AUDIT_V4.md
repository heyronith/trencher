# World Model Readiness Audit V4

**Status:** `ready_for_first_stage_a_gpu_run` (bounded 500-step Stage-A segment)

## Finite-input gate (this pass)

| Check | Result |
|-------|--------|
| Deep inputs finite after tensorization (broad 10k) | pass (`0` non-finite) |
| Deep inputs finite after tensorization (rich 1k) | pass |
| Observations with original missing values exercised | broad 10000 / rich 1000 |
| Per-objective SSL masks | wired |
| Recon excludes originally missing cells | pass |
| `TRENCHER_WM_RESUME=latest` durable resolve (mirror + GCS path) | pass |
| Two-worker interleave non-determinism documented | yes |

## First GPU command

Bounded `TRENCHER_WM_MAX_STEPS=500` Stage-A SSL — see runbook. Do not execute in editor.
