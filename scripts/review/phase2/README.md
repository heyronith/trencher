# Phase-2 diagnostics

CPU-only read-only analysis on `review/phase2-v1`.

## Prerequisites

1. Local freeze under `data/external/world_model/` (train + permanent_val only).
2. Lead-intake LGBM models under `scratch/lead_review_v1/lgbm/` (or retrain via lead intake).
3. Stage-C permanent-val predictions in `scratch/lead_review_v1/stage_c_val/` (hash `2d97c92e…`).
4. Download checkpoints once:

```bash
GCS=gs://trencher-world-model-v1-ronithworks-20260907/world_model_freeze_v1
mkdir -p scratch/phase2_v1/ckpts
gsutil cp "$GCS/runs/stage_c_rich/stage_c_init_150000.pt"{,.sha256,.meta.json} scratch/phase2_v1/ckpts/
gsutil cp "$GCS/runs/stage_c_rich/periodic.pt"{,.meta.json} scratch/phase2_v1/ckpts/
gsutil cp "$GCS/runs/stage_b_broad/latest.pt" scratch/phase2_v1/ckpts/stage_b_latest.pt
```

5. Run rich permanent_val CPU inference (clean tree; no `--allow-dirty` needed if only scratch is dirty/ignored):

```bash
INIT_SHA=$(awk '{print $1}' scratch/phase2_v1/ckpts/stage_c_init_150000.pt.sha256)
OMP_NUM_THREADS=1 uv run python -m trencher.world_model.evaluation.step17 infer \
  --checkpoint scratch/phase2_v1/ckpts/stage_c_init_150000.pt \
  --expected-checkpoint-sha256 "$INIT_SHA" \
  --config configs/world_model/world_model_candidate_v2.yaml \
  --world-root data/external/world_model \
  --split-root data/external/world_model/splits/split_v1 \
  --population rich --phase rich --mode permanent_val_eval \
  --device cpu --run-id stage_b_rich_from_init_150000 \
  --out-dir scratch/phase2_v1/infer/stage_b_rich_init --repo-root .

PERIODIC_SHA=$(shasum -a 256 scratch/phase2_v1/ckpts/periodic.pt | awk '{print $1}')
OMP_NUM_THREADS=1 uv run python -m trencher.world_model.evaluation.step17 infer \
  --checkpoint scratch/phase2_v1/ckpts/periodic.pt \
  --expected-checkpoint-sha256 "$PERIODIC_SHA" \
  --config configs/world_model/world_model_candidate_v2.yaml \
  --world-root data/external/world_model \
  --split-root data/external/world_model/splits/split_v1 \
  --population rich --phase rich --mode permanent_val_eval \
  --device cpu --run-id stage_c_periodic_174000 \
  --out-dir scratch/phase2_v1/infer/stage_c_periodic --repo-root .
```

## Regenerate report

```bash
git checkout -- manifests/world_model_data/LOADER_AUDIT_BROAD_SAMPLE_V2.json
OMP_NUM_THREADS=1 uv run python scripts/review/phase2/run_phase2.py
uv run ruff check .
uv run pytest
```

Outputs: `manifests/world_model_data/evaluation/phase2_v1/`.
