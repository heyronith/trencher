# Lead review scripts

Read-only analysis helpers for `review/lead-intake-v1`.

## Regenerate lead intake

```bash
# Optional: download Stage-C/B artifacts into scratch/ (not committed)
# Then build rich baseline features (local, train+val only):
uv run python scripts/review/build_rich_features.py

OMP_NUM_THREADS=1 uv run python scripts/review/run_lead_intake.py
# Resume section A from scratch/section_a.json after a partial run:
OMP_NUM_THREADS=1 uv run python scripts/review/run_lead_intake.py --resume-a
```

Outputs land in `manifests/world_model_data/evaluation/lead_review_v1/`.

Hard rules: never read q85+ TEST rows; never write to GCS freeze paths.
