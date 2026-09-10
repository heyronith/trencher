"""Verified Stage-B / Stage-C durable lineage for Step-17."""

from __future__ import annotations

from typing import Any

from trencher.world_model.evaluation.ablations import (
    CONFIG_HASH,
    RUNTIME_SHA,
    STAGE_B_CKPT_SHA256,
    STAGE_B_CKPT_URI,
    STAGE_C_CKPT_SHA256,
    STAGE_C_CKPT_URI,
)

STAGE_B_PRODUCER_SHA = "5f895e12779979b9140d293b42186a0e4e070ecf"
STAGE_B_METRICS_URI = (
    "gs://trencher-world-model-v1-ronithworks-20260907/world_model_freeze_v1/"
    "runs/stage_b_broad/stage_b_permanent_val_eval.json"
)
STAGE_B_METRICS_SHA256 = (
    "55b305fe602ac7cfeceab436ab87e960717e432229942c26e5a4c203505dfe75"
)
STAGE_C_VAL_ROOT = (
    "gs://trencher-world-model-v1-ronithworks-20260907/world_model_freeze_v1/"
    "runs/stage_c_rich/validation/stage_c_175000_permanent_val"
)
STAGE_C_PRED_SHA256 = "2d97c92e3bf6a9b721355d9ea2537b7a54787594ba0ee659105643e8200def13"
STAGE_C_METRICS_SHA256 = "c4f0edcb35fd6e9cb9fcf465f4a601e3d93756c75dd0c8c2a13b7b983c2be741"


def stage_b_lineage() -> dict[str, Any]:
    return {
        "checkpoint_uri": STAGE_B_CKPT_URI,
        "checkpoint_sha256": STAGE_B_CKPT_SHA256,
        "checkpoint_step": 150000,
        "checkpoint_phase": "broad",
        "checkpoint_phase_step": 100000,
        "consumed_microbatches": 100000,
        "git_sha": STAGE_B_PRODUCER_SHA,
        "config_hash": CONFIG_HASH,
        "permanent_val_metrics_uri": STAGE_B_METRICS_URI,
        "permanent_val_metrics_sha256": STAGE_B_METRICS_SHA256,
        "permanent_val_population": "broad",
        "permanent_val_examples": 1265314,
        "prediction_artifact_uri": None,
        "prediction_artifact_available": False,
        "q85_test_accessed": False,
        "paired_ready_with_stage_c_rich": False,
        "paired_blocker": (
            "Existing Stage-B permanent-val metrics were computed on population=broad "
            "(~1.26M examples). Stage-C permanent-val is population=rich (~10k). "
            "Paired comparison requires Stage-B permanent_val_eval inference on the "
            "rich population using the frozen Stage-B checkpoint; do not compare "
            "mismatched aggregates as paired evidence."
        ),
    }


def stage_c_lineage() -> dict[str, Any]:
    return {
        "checkpoint_uri": STAGE_C_CKPT_URI,
        "checkpoint_sha256": STAGE_C_CKPT_SHA256,
        "checkpoint_step": 175000,
        "checkpoint_phase": "rich",
        "runtime_git_sha": RUNTIME_SHA,
        "config_hash": CONFIG_HASH,
        "validation_root": STAGE_C_VAL_ROOT,
        "prediction_sha256": STAGE_C_PRED_SHA256,
        "metrics_sha256": STAGE_C_METRICS_SHA256,
        "mode": "permanent_val_eval",
        "population": "rich",
        "assignment_val_rows": 10159,
        "rows_loaded": 10014,
        "q85_test_accessed": False,
        "verified": True,
    }


def verify_approved_stage_c_identity(
    *,
    checkpoint_sha256: str,
    prediction_sha256: str,
    metrics_sha256: str,
    config_hash: str,
    runtime_git_sha: str,
) -> None:
    want = stage_c_lineage()
    checks = {
        "checkpoint_sha256": (checkpoint_sha256, want["checkpoint_sha256"]),
        "prediction_sha256": (prediction_sha256, want["prediction_sha256"]),
        "metrics_sha256": (metrics_sha256, want["metrics_sha256"]),
        "config_hash": (config_hash, want["config_hash"]),
        "runtime_git_sha": (runtime_git_sha, want["runtime_git_sha"]),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise RuntimeError(f"Stage-C identity verification failed: {bad}")
