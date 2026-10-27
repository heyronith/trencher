# Phase-3 look-ahead leak audit

CPU-only analysis on `review/phase3-leak-v1`.

## Regenerate

```bash
# Stage-C latest for mask diagnostics (scratch only)
GCS=gs://trencher-world-model-v1-ronithworks-20260907/world_model_freeze_v1
mkdir -p scratch/phase3_leak_v1/ckpts
gsutil cp "$GCS/runs/stage_c_rich/latest.pt" scratch/phase3_leak_v1/ckpts/stage_c_latest.pt

git checkout -- manifests/world_model_data/LOADER_AUDIT_BROAD_SAMPLE_V2.json
OMP_NUM_THREADS=1 uv run python scripts/review/phase3/run_phase3.py
# Resume from scratch/phase3_leak_v1/cache/:
OMP_NUM_THREADS=1 uv run python scripts/review/phase3/run_phase3.py --resume
```

Outputs: `manifests/world_model_data/evaluation/phase3_leak_v1/`.
