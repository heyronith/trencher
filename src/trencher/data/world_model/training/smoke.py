"""CPU-only loader smoke benchmark (no model training)."""

from __future__ import annotations

import resource
import time
from pathlib import Path
from typing import Any

from trencher.data.world_model.splits.cutoffs import SPLIT_TRAIN
from trencher.data.world_model.training.loaders import (
    STRATEGY_BROAD_PRETRAIN,
    STRATEGY_RICH_MULTIMODAL,
    iter_loader,
    mixed_loader,
)


def _rss_mb() -> float:
    # macOS ru_maxrss is bytes; Linux is KiB — normalize roughly.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 1e9:  # already bytes-ish
        return rss / (1024 * 1024)
    # Linux KiB
    if rss > 1e7:
        return rss / (1024 * 1024)
    return rss / 1024.0


def run_loader_smoke(
    *,
    split_root: Path,
    batch_size: int = 64,
    max_batches: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, population, strategy in (
        ("broad", "broad", STRATEGY_BROAD_PRETRAIN),
        ("rich", "rich", STRATEGY_RICH_MULTIMODAL),
    ):
        t0 = time.time()
        n_ex = 0
        n_batch = 0
        shapes: dict[str, Any] = {}
        modality: dict[str, float] = {}
        for batch in iter_loader(
            split_root=split_root,
            population=population,
            split=SPLIT_TRAIN,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
            strategy=strategy,
            max_batches=max_batches,
        ):
            n_batch += 1
            n_ex += int(batch["batch_size"])
            shapes = {
                k: list(np_shape(v))
                for k, v in batch["features"].items()
            }
            for mk, mv in batch.get("modality_masks", {}).items():
                modality[mk] = float(sum(mv) / max(len(mv), 1))
        elapsed = max(time.time() - t0, 1e-6)
        results[name] = {
            "examples": n_ex,
            "batches": n_batch,
            "examples_per_sec": n_ex / elapsed,
            "batches_per_sec": n_batch / elapsed,
            "feature_shapes": shapes,
            "modality_availability_mean": modality,
            "elapsed_sec": elapsed,
        }

    t0 = time.time()
    n_mixed = 0
    for _ in mixed_loader(
        split_root=split_root,
        split=SPLIT_TRAIN,
        batch_size=batch_size,
        seed=seed,
        max_batches=max_batches,
    ):
        n_mixed += 1
    elapsed = max(time.time() - t0, 1e-6)
    results["mixed"] = {
        "batches": n_mixed,
        "batches_per_sec": n_mixed / elapsed,
        "elapsed_sec": elapsed,
    }
    results["peak_rss_mb"] = _rss_mb()
    results["batch_size"] = batch_size
    results["max_batches"] = max_batches
    return results


def np_shape(v: Any) -> tuple[int, ...]:
    import numpy as np

    a = np.asarray(v)
    return tuple(a.shape)
