# World Model Readiness Audit V3

**Status:** `ready_for_first_stage_a_gpu_run`

Corrective launch-blocker pass on top of Candidate V2 / frozen V4.

## Gate checklist

| Gate | Result |
|------|--------|
| Exact runbook staging paths (`LOCAL_STAGE` → `WORLD_ROOT`) | pass |
| Config resolves from approved SHA repo path | pass |
| Two-worker loader zero duplicate coverage | pass |
| Production resume restores scheduler + consumed-microbatch cursor | pass |
| Durable checkpoint sync (GCS / local-mirror) | pass |
| Masked reconstruction trained (`ssl_recon_target` / `ssl_recon_mask`) | pass |
| Proper intensity objective (`λΔt − log λ`) | pass |
| Candidate V2 identifies `WORLD_MODEL_TRAINING_CONTRACT_V2` | pass |
| CPU local-mirror launch/resume integration | pass |
| Permanent TEST untouched | pass |

## Explicit non-claims

- Future-latent SSL remains intentionally deferred
- Survival baseline still subsample-trained / full-pop validated (not rerun)
- CatBoost still `incomplete_compute_cost_stop`
- No Stage A/B/C Colab GPU quota consumed by this task
- Multi-worker **coverage** is deterministic (path shard); batch **interleave order**
  across workers is not bit-stable across preempt — exact param resume is validated
  at `num_workers=0`; production default remains 2 after duplication gate

## First Stage-A command (do not run here)

See `WORLD_MODEL_COLAB_RUNBOOK_V2.md`.
