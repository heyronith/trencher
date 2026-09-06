# World Model Colab Runbook V2

## Goal

Stage the **minimal frozen V4 training dependency set** to Colab local SSD and run
**Stage A SSL only** on A100/H100 — then stop for review before Stage B/C.

## Preconditions

1. Approved Git SHA checked out on the runtime.
2. `configs/world_model/world_model_candidate_v2.yaml` present.
3. Staging manifest: `manifests/world_model_data/WORLD_MODEL_STAGING_MANIFEST_V2.json`.
4. Permanent TEST (q85+) remains locked.

## Environment

```bash
export TRENCHER_GCS_URI=gs://YOUR_BUCKET/world_model_freeze_v1
export TRENCHER_LOCAL_STAGE=/content/trencher_stage
export TRENCHER_WORLD_ROOT=$TRENCHER_LOCAL_STAGE/world_model
export TRENCHER_SPLIT_ROOT=$TRENCHER_WORLD_ROOT/splits/split_v1
export TRENCHER_WM_CONFIG=$TRENCHER_LOCAL_STAGE/configs/world_model/world_model_candidate_v2.yaml
export TRENCHER_WM_DEVICE=cuda
export TRENCHER_WM_PHASE=ssl
export TRENCHER_WM_MODE=final_train
export TRENCHER_WM_POPULATION=broad
export TRENCHER_GIT_SHA=$(git rev-parse HEAD)
```

Authenticate via Colab Google auth / ADC. Never commit credentials.

## Operator steps

1. `python scripts/colab/sync_manifest.py --world-root data/external/world_model` (from a machine that has the freeze)
2. On Colab: `python scripts/colab/stage_from_gcs.py` (or `--local-source` mirror)
3. Verify hashes / fail closed on required mismatches
4. `bash scripts/colab/launch_training.sh`  
   This calls `python -m trencher.world_model.launch` over **real** frozen DataLoaders.

## First approved command (DO NOT run until review)

```bash
export TRENCHER_GCS_URI=gs://YOUR_BUCKET/world_model_freeze_v1
export TRENCHER_LOCAL_STAGE=/content/trencher_stage
export TRENCHER_WORLD_ROOT=$TRENCHER_LOCAL_STAGE/world_model
export TRENCHER_SPLIT_ROOT=$TRENCHER_WORLD_ROOT/splits/split_v1
export TRENCHER_WM_CONFIG=$TRENCHER_LOCAL_STAGE/configs/world_model/world_model_candidate_v2.yaml
export TRENCHER_WM_DEVICE=cuda
export TRENCHER_WM_PHASE=ssl
export TRENCHER_WM_MODE=final_train
export TRENCHER_WM_POPULATION=broad
export TRENCHER_GIT_SHA=<APPROVED_SHA>
python scripts/colab/stage_from_gcs.py
bash scripts/colab/launch_training.sh
```

Stage B/C are separate launches from approved checkpoints (`TRENCHER_WM_PHASE=broad|rich` + `TRENCHER_WM_RESUME=...`).
