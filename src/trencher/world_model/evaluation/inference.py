"""Permanent-validation inference for world-model checkpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from trencher.data.checksums import sha256_file
from trencher.world_model.checkpoint import load_checkpoint
from trencher.world_model.config import WorldModelConfig, load_world_model_config
from trencher.world_model.evaluation.masking import (
    active_modalities_for_phase,
    mask_batch_modalities,
)
from trencher.world_model.evaluation.metrics import summarize_prediction_artifact
from trencher.world_model.evaluation.modes import (
    assert_no_test_population,
    assert_step17_mode,
    provenance_q85_flag,
    refuse_test_base_splits,
)
from trencher.world_model.evaluation.provenance import attach_provenance, write_json_with_sidecar
from trencher.world_model.model import WorldModel, count_parameters
from trencher.world_model.v4_loader import make_frozen_dataloader

OUTPUT_KEYS = (
    "explosion_logits",
    "explosion_tte_hazard",
    "explosion_phase_logits",
    "graduation_tte_hazard",
    "survival_horizon_logits",
    "survival_tte_hazard",
    "barrier_logits",
    "return_quantiles",
    "mfe_quantiles",
    "mae_quantiles",
    "time_to_mfe",
    "time_to_mae",
    "persistence_pred",
    "path_mean",
    "path_log_scale",
    "path_df",
    "execution",
    "fusion_gates",
)

TARGET_KEYS = (
    "observation_id_hash",
    "explosion_target",
    "explosion_mask",
    "survival_horizon_target",
    "survival_horizon_mask",
    "barrier_target",
    "barrier_mask",
    "return_target",
    "return_mask",
    "mfe_target",
    "mfe_mask",
    "mae_target",
    "mae_mask",
    "persistence_target",
    "persistence_mask",
    "path_target",
    "path_mask",
    "execution_target",
    "execution_mask",
    "causal_ok",
    "wallet_available",
    "micro_available",
    "own_available",
    "graph_available",
)


def run_permanent_val_inference(
    *,
    checkpoint: Path,
    config: Path | WorldModelConfig,
    world_root: Path,
    split_root: Path,
    out_dir: Path,
    population: str = "rich",
    mode: str = "permanent_val_eval",
    phase: str = "rich",
    device: str = "cuda",
    omit_modalities: list[str] | None = None,
    inference_mask_omit: list[str] | None = None,
    run_id: str = "permanent_val",
    max_batches: int | None = None,
    expected_checkpoint_sha256: str | None = None,
    repo_root: Path | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run frozen permanent-val inference and write predictions + metrics artifacts."""
    mode = assert_step17_mode(mode)
    population = assert_no_test_population(population)
    cfg = config if isinstance(config, WorldModelConfig) else load_world_model_config(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(checkpoint)
    ckpt_sha = sha256_file(ckpt_path)
    if expected_checkpoint_sha256 and ckpt_sha != expected_checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint sha256 mismatch: got={ckpt_sha} want={expected_checkpoint_sha256}"
        )

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model = WorldModel(cfg)
    payload = load_checkpoint(
        ckpt_path,
        model=model,
        expected_config=cfg,
        restore_rng=False,
        map_location="cpu",
    )
    model.to(device)
    model.eval()

    omit_train = set(omit_modalities or [])
    active = active_modalities_for_phase(
        phase,
        modalities_broad=cfg.modalities_broad,
        modalities_rich=cfg.modalities_rich,
        omit=omit_train,
    )
    mask_omit = set(inference_mask_omit or [])

    loader = make_frozen_dataloader(
        world_root=Path(world_root),
        split_root=Path(split_root),
        mode=mode,  # type: ignore[arg-type]
        population=population,  # type: ignore[arg-type]
        batch_size=int(cfg.train.batch_size),
        num_workers=0,
        seed=int(cfg.train.seed),
        shuffle=False,
        pin_memory=device == "cuda",
        seq_max_len={"wallet_prior": int(cfg.wallet_prior.max_len)},
    )

    outs: dict[str, list[torch.Tensor]] = {k: [] for k in OUTPUT_KEYS}
    tgts: dict[str, list[torch.Tensor]] = {k: [] for k in TARGET_KEYS}
    processed = 0
    causal_bad = 0
    t0 = time.time()
    use_bf16 = bool(cfg.train.bf16) and device == "cuda"

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, 1):
            if max_batches is not None and batch_idx > max_batches:
                break
            refuse_test_base_splits(
                ["val"] * int(batch["observation_id_hash"].shape[0]),
                context="step17_inference",
            )
            causal_bad += int((batch["causal_ok"] < 0.5).sum().item())
            if causal_bad:
                raise RuntimeError(f"causal validation failure count={causal_bad}")
            if mask_omit:
                batch = mask_batch_modalities(batch, omit=mask_omit)
            gpu_batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                pred = model(gpu_batch, phase=phase, active=active)
            for k in OUTPUT_KEYS:
                if k in pred:
                    outs[k].append(pred[k].detach().float().cpu())
            for k in TARGET_KEYS:
                if k in batch:
                    tgts[k].append(batch[k].detach().cpu())
            processed += int(batch["observation_id_hash"].shape[0])

    if processed <= 0:
        raise RuntimeError("validation loader produced zero rows")

    out_all = {k: torch.cat(v, dim=0) for k, v in outs.items() if v}
    tgt_all = {k: torch.cat(v, dim=0) for k, v in tgts.items() if v}

    metrics = summarize_prediction_artifact(
        out_all,
        tgt_all,
        explosion_horizons=cfg.heads.explosion_horizons,
        survival_horizons=cfg.heads.survival_horizons,
        return_horizons=cfg.heads.return_horizons,
        quantile_taus=cfg.heads.quantile_taus,
        execution_sizes=cfg.heads.execution_sizes,
    )
    elapsed = time.time() - t0
    meta = attach_provenance(
        {
            "artifact": f"{run_id}_permanent_val",
            "run_id": run_id,
            "checkpoint_step": int(payload.get("step", -1)),
            "checkpoint_phase": str(payload.get("phase", "")),
            "checkpoint_sha256": ckpt_sha,
            "mode": mode,
            "population": population,
            "phase": phase,
            "rows_loaded": processed,
            "elapsed_sec": elapsed,
            "examples_per_sec": processed / max(elapsed, 1e-9),
            "causal_failures": causal_bad,
            "parameters": count_parameters(model),
            "active_modalities": sorted(active),
            "omit_modalities": sorted(omit_train),
            "inference_mask_omit": sorted(mask_omit),
            "inference_mask_diagnostic": bool(mask_omit),
        },
        config_hash=cfg.config_hash(),
        source_checkpoint_sha256=ckpt_sha,
        modality_config={
            "active": sorted(active),
            "omit_train": sorted(omit_train),
            "inference_mask_omit": sorted(mask_omit),
        },
        repo_root=repo_root,
        allow_dirty=allow_dirty,
    )
    meta.update(provenance_q85_flag(accessed=False))
    metrics_payload = dict(metrics)
    metrics_payload.update(meta)

    pred_path = out_dir / f"{run_id}_predictions.pt"
    metrics_path = out_dir / f"{run_id}_metrics.json"
    torch.save({"metadata": meta, "outputs": out_all, "targets": tgt_all}, pred_path)
    write_json_with_sidecar(metrics_path, metrics_payload)
    pred_sha = sha256_file(pred_path)
    pred_path.with_suffix(pred_path.suffix + ".sha256").write_text(
        f"{pred_sha}  {pred_path.name}\n", encoding="utf-8"
    )
    summary = {
        "predictions_path": str(pred_path),
        "predictions_sha256": pred_sha,
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "metadata": meta,
    }
    (out_dir / f"{run_id}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
