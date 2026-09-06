# Colab / GCS deep-training tooling

Scripts here stage frozen V4 artifacts and launch the ordinary Python trainer.
No ML logic lives only in notebooks.

| Script | Role |
|--------|------|
| `sync_manifest.py` | Build/discover staging manifest; optional upload to GCS or local mirror |
| `stage_from_gcs.py` | Download required shards to Colab SSD and verify |
| `launch_training.sh` | Env-driven wrapper → `python -m trencher.world_model.launch` |

Default first real run: **Stage A SSL only** (`TRENCHER_WM_PHASE=ssl`).

See `manifests/world_model_data/WORLD_MODEL_COLAB_RUNBOOK_V2.md`.
