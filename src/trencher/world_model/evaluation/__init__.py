"""Step-17 scientific evaluation, ablations, and reporting for the world model."""

from __future__ import annotations

STEP17_CONTRACT = "STEP17_SCIENTIFIC_EVALUATION_V1"
STEP17_DURABLE_PREFIX = (
    "gs://trencher-world-model-v1-ronithworks-20260907/world_model_freeze_v1/"
    "evaluation/step17"
)
INFERENCE_MASK_DIAGNOSTIC_LABEL = "inference_time_modality_mask_diagnostic"

__all__ = [
    "INFERENCE_MASK_DIAGNOSTIC_LABEL",
    "STEP17_CONTRACT",
    "STEP17_DURABLE_PREFIX",
]
