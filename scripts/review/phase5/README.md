# Phase 5 — tree champion + next lift

Branch: `review/phase5-trees-v1`

## Run

```bash
uv run python -m scripts.review.phase5.run_phase5 --only all
# or: sanity | pump | tune | history | rpc | report
```

Outputs: `manifests/world_model_data/evaluation/phase5_trees_v1/`
Scratch: `scratch/phase5_trees_v1/` (gitignored)

Hard rules: no q85+ TEST; no freeze/GCS writes; WF selection only on `wf_fold_1`/`wf_fold_2`; mint bootstrap 2000/seed 0.
