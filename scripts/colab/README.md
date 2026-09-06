# Colab / remote GPU staging for Trencher world-model training

Backend-neutral helpers to stage frozen training artifacts from GCS (or a local
mirror) and launch the three-phase world-model trainer.

## Principles

- **No credentials in repo.** Use Application Default Credentials / workload
  identity / `gcloud auth` in the runtime. Scripts only read env vars.
- **Env-based GCS.** Set `TRENCHER_GCS_URI` (e.g. `gs://bucket/world_model_v1`)
  and optional `TRENCHER_LOCAL_STAGE` (default `/content/trencher_stage`).
- **Do not train until staging verifies.** `sync_manifest.py` +
  `stage_from_gcs.py` check required artifact sha256 digests.
- **Permanent TEST is refused** by the trainer/data layer.

## Required env

| Variable | Purpose |
|----------|---------|
| `TRENCHER_GCS_URI` | Root GCS prefix for frozen artifacts |
| `TRENCHER_LOCAL_STAGE` | Local staging directory |
| `TRENCHER_WM_CONFIG` | Path to YAML config (default: staged candidate config) |
| `TRENCHER_WM_PHASE` | `ssl` \| `broad` \| `rich` \| `all` |
| `TRENCHER_WM_DEVICE` | `cuda` \| `cpu` |

## Typical Colab flow

```bash
# 1. Install package + torch (Colab usually has torch)
pip install -e ".[torch]"

# 2. Authenticate to GCS in the notebook UI / gcloud — never paste keys into git

# 3. Sync manifest + stage
export TRENCHER_GCS_URI=gs://YOUR_BUCKET/world_model_freeze_v1
export TRENCHER_LOCAL_STAGE=/content/trencher_stage
python scripts/colab/sync_manifest.py
python scripts/colab/stage_from_gcs.py

# 4. Launch (review launch_training.sh before running on a paid GPU)
bash scripts/colab/launch_training.sh
```

## Notes

- Staging is backend-neutral: if `TRENCHER_GCS_URI` is a local path, files are
  copied without invoking `gsutil`.
- Smoke tests use synthetic tensors and never require GCS.
