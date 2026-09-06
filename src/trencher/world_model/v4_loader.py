"""Frozen V4 loader integration for deep training (no new target semantics).

This module does not rebuild caches. It only constructs observation iterators /
batch schemas on top of TRAINING_LOADER_CONTRACT_V4 artifacts and refuses
permanent TEST access.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from trencher.world_model.split_guard import (
    PermanentTestAccessError,
    refuse_test_split,
)

DatasetMode = Literal[
    "wf_fold_1",
    "wf_fold_2",
    "final_train",
    "permanent_val_eval",
]

# Modes that may never include q85+ TEST rows.
_ALLOWED_MODES: frozenset[str] = frozenset(
    {"wf_fold_1", "wf_fold_2", "final_train", "permanent_val_eval"}
)


@dataclass(frozen=True)
class FrozenContractRefs:
    freeze_id: str = "world_model_freeze_v1"
    loader_contract: str = "TRAINING_LOADER_CONTRACT_V4"
    split_id: str = "split_v1"
    sequence_dataset: str = "sequence_dataset_v2"
    target_dataset: str = "target_dataset_v3"
    broad_targets: str = "broad_targets_v3"


@dataclass
class DeepBatchSchema:
    """Declared tensor keys expected by WorldModel forward."""

    required_broad: tuple[str, ...] = (
        "event_x",
        "event_times",
        "event_mask",
        "curve_x",
        "curve_times",
        "curve_mask",
        "global_buckets",
        "global_times",
        "global_scalars",
        "global_mask",
    )
    optional_rich: tuple[str, ...] = (
        "micro_x",
        "micro_times",
        "micro_mask",
        "micro_available",
        "own_x",
        "own_times",
        "own_mask",
        "own_available",
        "own_quality",
        "wallet_x",
        "wallet_times",
        "wallet_mask",
        "wallet_available",
        "graph_edge_feat",
        "graph_times",
        "graph_mask",
        "graph_available",
    )
    target_keys: tuple[str, ...] = (
        "explosion_target",
        "explosion_mask",
        "survival_event",
        "survival_time_idx",
        "survival_mask",
        "barrier_target",
        "return_target",
        "mfe_target",
        "mae_target",
        "persistence_target",
        "path_target",
        "execution_target",
        "sample_mask",
    )


def assert_dataset_mode(mode: str) -> DatasetMode:
    mode = str(mode)
    if mode.lower() in {"test", "permanent_test", "q85_test"}:
        raise PermanentTestAccessError(f"dataset mode {mode!r} is forbidden")
    if mode not in _ALLOWED_MODES:
        raise PermanentTestAccessError(f"unknown/forbidden dataset mode {mode!r}")
    return mode  # type: ignore[return-value]


def verify_no_test_rows(base_splits: list[str], *, context: str = "deep_loader") -> None:
    refuse_test_split("train")  # ensure import path active
    bad = [s for s in base_splits if str(s).lower() in {"test", "permanent_test"}]
    if bad:
        raise PermanentTestAccessError(f"{context}: refused {len(bad)} TEST rows")


def content_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_deep_loader_manifest(
    *,
    world_root: Path,
    split_root: Path,
    mode: DatasetMode,
    population: Literal["broad", "rich"] = "broad",
    refs: FrozenContractRefs | None = None,
) -> dict[str, Any]:
    """Machine-readable manifest for a deep-training dataset construction.

    Does not load q85+ TEST. Production runners materialize tensors from the
    frozen stores listed here without changing target semantics.
    """
    mode = assert_dataset_mode(mode)
    refs = refs or FrozenContractRefs()
    world_root = Path(world_root)
    split_root = Path(split_root)
    return {
        "kind": "world_model_deep_loader_manifest_v1",
        "mode": mode,
        "population": population,
        "contracts": refs.__dict__,
        "paths": {
            "world_root": str(world_root),
            "split_root": str(split_root),
            "feature_cache": str(split_root / "baseline_features_v1"),
            "training_cache": str(split_root / "training_cache_v1"),
            "cutoffs": str(split_root / "cutoffs.json"),
        },
        "batch_schema": DeepBatchSchema().__dict__,
        "guards": {
            "permanent_test": "FORBIDDEN",
            "permanent_val_eval_only": mode == "permanent_val_eval",
            "selection_folds": ["wf_fold_1", "wf_fold_2"],
        },
        "note": (
            "Tensor materialization binds to frozen V4 loaders / sequence stores. "
            "This manifest does not authorize cache rebuilds."
        ),
    }
