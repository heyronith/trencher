# World Model Colab Runbook V2

## Goal

Stage the **minimal frozen V4 training dependency set** to Colab local SSD and run
**Stage A SSL only** on A100/H100 — then stop for review before Stage B/C.

## Path semantics

| Variable | Meaning |
|----------|---------|
| `TRENCHER_LOCAL_STAGE` | Colab SSD staging parent (`/content/trencher_stage`) |
| `TRENCHER_WORLD_ROOT` | Frozen world-model data root (`$TRENCHER_LOCAL_STAGE/world_model`) |
| `TRENCHER_SPLIT_ROOT` | Split root (`$TRENCHER_WORLD_ROOT/splits/split_v1`) |
| `TRENCHER_WM_CONFIG` | Config from the **checked-out Git SHA** (repo path), not a staged copy |

`WORLD_MODEL_STAGING_MANIFEST_V2.json` paths are relative to the frozen world-model
root (`splits/...`, `broad/...`, …). `stage_from_gcs.py` defaults `--dest` to
`$TRENCHER_WORLD_ROOT` (or `$TRENCHER_LOCAL_STAGE/world_model`).

## Preconditions

1. Approved Git SHA checked out on the runtime.
2. `configs/world_model/world_model_candidate_v2.yaml` present in that checkout.
3. Staging manifest: `manifests/world_model_data/WORLD_MODEL_STAGING_MANIFEST_V2.json`.
4. Permanent TEST (q85+) remains locked.

## Environment

```bash
export TRENCHER_GCS_URI=gs://YOUR_BUCKET/world_model_freeze_v1
export TRENCHER_LOCAL_STAGE=/content/trencher_stage
export TRENCHER_WORLD_ROOT=$TRENCHER_LOCAL_STAGE/world_model
export TRENCHER_SPLIT_ROOT=$TRENCHER_WORLD_ROOT/splits/split_v1
export TRENCHER_WM_CONFIG="$(pwd)/configs/world_model/world_model_candidate_v2.yaml"
export TRENCHER_WM_DEVICE=cuda
export TRENCHER_WM_PHASE=ssl
export TRENCHER_WM_MODE=final_train
export TRENCHER_WM_POPULATION=broad
export TRENCHER_WM_NUM_WORKERS=2
export TRENCHER_WM_RUN_ID=stage_a_ssl
export TRENCHER_GIT_SHA=$(git rev-parse HEAD)
```

Authenticate via Colab Google auth / ADC. Never commit credentials.

Durable checkpoints sync under `$TRENCHER_GCS_URI/runs/$TRENCHER_WM_RUN_ID/`
(`latest.pt`, periodic checkpoints, `progress.json`, `phase_summary.json`).

## Operator steps

1. `python scripts/colab/sync_manifest.py --world-root data/external/world_model` (from a machine that has the freeze)
2. On Colab: `python scripts/colab/stage_from_gcs.py` (or `--local-source` mirror) → writes under `$TRENCHER_WORLD_ROOT`
3. Verify hashes / fail closed on required mismatches
4. `bash scripts/colab/launch_training.sh`  
   This calls `python -m trencher.world_model.launch` over **real** frozen DataLoaders.

## First approved command (DO NOT run until review)

```bash
export TRENCHER_GCS_URI=gs://YOUR_BUCKET/world_model_freeze_v1
export TRENCHER_LOCAL_STAGE=/content/trencher_stage
export TRENCHER_WORLD_ROOT=$TRENCHER_LOCAL_STAGE/world_model
export TRENCHER_SPLIT_ROOT=$TRENCHER_WORLD_ROOT/splits/split_v1
export TRENCHER_WM_CONFIG="$(pwd)/configs/world_model/world_model_candidate_v2.yaml"
export TRENCHER_WM_DEVICE=cuda
export TRENCHER_WM_PHASE=ssl
export TRENCHER_WM_MODE=final_train
export TRENCHER_WM_POPULATION=broad
export TRENCHER_WM_NUM_WORKERS=2
export TRENCHER_WM_RUN_ID=stage_a_ssl
export TRENCHER_GIT_SHA=<APPROVED_SHA>
python scripts/colab/stage_from_gcs.py
bash scripts/colab/launch_training.sh
```

Stage B/C are separate launches from approved checkpoints (`TRENCHER_WM_PHASE=broad|rich` + `TRENCHER_WM_RESUME=...`).
