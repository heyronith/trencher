# Phase 4 — clean comparison & leak-free microstructure

Branch: `review/phase4-clean-v1`  
Outputs: `manifests/world_model_data/evaluation/phase4_clean_v1/`  
Scratch (gitignored): `scratch/phase4_clean_v1/`  
Clean refetch root: `data/external/world_model_clean_v1/` (gitignored via `/data/`)

## Hard rules

- Never read q85+ TEST; every artifact records `q85_test_accessed: false`.
- No writes to freeze GCS or `data/external/world_model/` source trees. Broad feature rebuild uses `scratch/phase4_clean_v1/split_overlay/`.
- AUROC via fixed `_safe_auc` / `binary_metrics` / sklearn.
- Mint bootstrap: 2000 reps, seed 0.

## Entrypoints

```bash
# Stage-B broad permanent_val infer (CPU ~2.3h @ ~150 ex/s)
uv run python -m trencher.world_model.evaluation.step17 infer ...

# Broad LGBM/XGB reproduction (fixed iters + Platt)
uv run python scripts/review/phase4/run_trees_only.py

# Compare + PHASE4_REPORT
uv run python scripts/review/phase4/compare_and_report.py

# Causal micro refetch (dry-run if no SOLANA_RPC_URL / HELIUS_API_KEY)
uv run python -m scripts.review.phase4.refetch_micro
```

## Quarantine

See `manifests/world_model_data/LEAK_QUARANTINE_V1.md`.
