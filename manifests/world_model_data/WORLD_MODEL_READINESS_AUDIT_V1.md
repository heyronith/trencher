# World Model Readiness Audit V1

**Status:** `implementation_ready_cpu_validated` — full production training **not** launched.

## Ready

- Importable `trencher.world_model` package with encoders, gated fusion, multi-task heads, SSL, losses, three-phase trainer, atomic checkpoints, staging, Colab scripts
- A100-oriented candidate config (`d_model=256`, 6 event layers, ~15.1M trainable params)
- Explosion horizons: 10/30/60/120/300s; execution slots sized for 0.1–5 SOL masks
- Conditional autoregressive Student-t path head (diffusion-replaceable interface)
- Hawkes-inspired intensity SSL auxiliary (not forced classical Hawkes)
- Permanent TEST split hard-refuse + dataset modes `wf_fold_1|wf_fold_2|final_train|permanent_val_eval`
- V4 deep-loader manifest binder (`v4_loader.py`) — no new target semantics
- Colab/GCS staging scripts (env-based, no credentials committed)
- CPU smoke / tiny overfit / causality mutation / missing-modality / checkpoint mismatch covered by `tests/test_world_model_smoke.py`

## Not yet (blockers for full A100)

1. Production DataLoaders materializing tensors from frozen V4 partitions at scale
2. Full GCS staging digest verification against live sequence stores
3. Measured BF16 step-time / memory on the target A100/H100 SKU
4. Explicit operator approval to spend Colab GPU quota

## Constraints upheld

- Pure on-chain (no social/text/image towers)
- No learned mint/wallet/creator ID embedding tables
- No CI / smoke path launches GPU training jobs
- No full Stage A/B/C production training in this task
