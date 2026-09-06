# World Model Colab Runbook V1

## Goal

Stage frozen V4 training artifacts and run the three-phase deep world model
(`ssl` → `broad` → `rich`) on an A100-class runtime without embedding
credentials in the repository.

## Preconditions

1. `TRAINING_LOADER_CONTRACT_V4` cache / overlays available in GCS or a local mirror.
2. Candidate config: `configs/world_model/world_model_candidate_v1.yaml`.
3. Permanent TEST (q85+) remains locked — trainer refuses `split=test`.
4. Runtime has Python 3.12+, `torch`, and `pip install -e ".[torch]"`.

## Environment

```bash
export TRENCHER_GCS_URI=gs://YOUR_BUCKET/world_model_freeze_v1
export TRENCHER_LOCAL_STAGE=/content/trencher_stage
export TRENCHER_WM_CONFIG=$TRENCHER_LOCAL_STAGE/configs/world_model/world_model_candidate_v1.yaml
export TRENCHER_WM_DEVICE=cuda
export TRENCHER_WM_PHASE=all
```

Authenticate via Colab's Google auth / `gcloud auth application-default login`.
Never commit service-account JSON.

## Steps

1. `python scripts/colab/sync_manifest.py`
2. `python scripts/colab/stage_from_gcs.py`
3. Verify staging digests via `trencher.world_model.staging.verify_staging_manifest`
4. Review `scripts/colab/launch_training.sh`, then run it
5. Monitor `[world_model_train] k/100` progress lines
6. Resume from atomic checkpoints if preempted (`checkpoint.resume_from_checkpoint`)

## First real production launch (AFTER review — do not run in this task)

```bash
export TRENCHER_GCS_URI=gs://YOUR_BUCKET/world_model_freeze_v1
export TRENCHER_LOCAL_STAGE=/content/trencher_stage
export TRENCHER_WM_CONFIG=$TRENCHER_LOCAL_STAGE/configs/world_model/world_model_candidate_v1.yaml
export TRENCHER_WM_DEVICE=cuda
export TRENCHER_WM_PHASE=all
python scripts/colab/sync_manifest.py
python scripts/colab/stage_from_gcs.py
# Only after independent review + operator approval:
bash scripts/colab/launch_training.sh
```

This command must **not** be executed until the implementation is reviewed and GPU quota spend is approved.

