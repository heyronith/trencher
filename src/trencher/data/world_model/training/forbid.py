"""Forbidden identity / leakage fields for predictive tensors."""

from __future__ import annotations

FORBIDDEN_FEATURE_KEYS: frozenset[str] = frozenset(
    {
        "mint",
        "creator",
        "wallet",
        "wallet_id",
        "base58_address",
        "observation_id",
        "tx_signature",
        "source_freeze_version",
        "pit_engine_version",
        "observation_dataset_version",
        "sequence_store_version",
        "target_dataset_version",
        "label_semantics",
        "base_split",
        "split_assignment",
        "artifact_id",
        "model_state_digest",
        "censor_reason",
        "purge_reason",
        "source_identity",
    }
)


def strip_forbidden(features: dict) -> dict:
    """Drop forbidden keys from a feature dict (non-destructive copy)."""
    return {k: v for k, v in features.items() if k not in FORBIDDEN_FEATURE_KEYS}


def assert_no_forbidden(features: dict) -> None:
    bad = sorted(set(features) & FORBIDDEN_FEATURE_KEYS)
    if bad:
        raise ValueError(f"forbidden predictive features present: {bad}")
