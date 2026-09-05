"""CPU production-loader smoke benchmark against frozen stores."""

from __future__ import annotations

import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from trencher.data.world_model.splits.cutoffs import SPLIT_TRAIN
from trencher.data.world_model.splits.progress import Progress100
from trencher.data.world_model.training.loaders import (
    STRATEGY_BROAD_PRETRAIN,
    STRATEGY_RICH_MULTIMODAL,
    iter_loader,
    mixed_loader,
)


def _rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 1e9:
        return rss / (1024 * 1024)
    if rss > 1e7:
        return rss / (1024 * 1024)
    return rss / 1024.0


def run_production_loader_smoke(
    *,
    world_root: Path,
    split_root: Path,
    batch_size: int = 16,
    max_batches: int = 30,
    seed: int = 0,
    progress: Progress100 | None = None,
) -> dict[str, Any]:
    progress = progress or Progress100()
    results: dict[str, Any] = {"batch_size": batch_size, "max_batches": max_batches}
    for step, (name, population, strategy) in enumerate(
        (
            ("broad", "broad", STRATEGY_BROAD_PRETRAIN),
            ("rich", "rich", STRATEGY_RICH_MULTIMODAL),
        ),
        start=1,
    ):
        progress(10 + step * 30, f"smoke {name}")
        t0 = time.time()
        n_ex = 0
        n_batch = 0
        shapes: dict[str, Any] = {}
        modality_sum: dict[str, float] = {}
        modality_n = 0
        trunc_sum: dict[str, float] = {}
        length_acc: dict[str, list[float]] = {}
        target_examples: dict[str, Any] = {}
        causal_failures = 0
        for batch in iter_loader(
            world_root=world_root,
            split_root=split_root,
            population=population,
            split=SPLIT_TRAIN,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
            strategy=strategy,
            max_batches=max_batches,
            use_production=True,
        ):
            n_batch += 1
            n_ex += int(batch["batch_size"])
            if not batch.get("causal_ok", True):
                causal_failures += 1
            for k, v in (batch.get("sequences") or {}).items():
                if k.endswith("_padding_mask"):
                    continue
                shapes[k] = list(np.asarray(v).shape)
            for k, st in (batch.get("sequence_length_stats") or {}).items():
                length_acc.setdefault(k, []).append(float(st.get("mean") or 0))
            for k, r in (batch.get("truncation_rates") or {}).items():
                trunc_sum[k] = trunc_sum.get(k, 0.0) + float(r)
            for k, v in (batch.get("modality_masks") or {}).items():
                modality_sum[k] = modality_sum.get(k, 0.0) + float(np.mean(v))
            modality_n += 1
            if not target_examples and isinstance(batch.get("targets"), list) and batch["targets"]:
                t0row = batch["targets"][0]
                target_examples = {
                    "available": t0row.get("available"),
                    "returns_h60": {
                        "is_missing": (t0row.get("returns_h60") or {}).get("is_missing"),
                        "frozen_status": (t0row.get("returns_h60") or {}).get(
                            "frozen_status"
                        ),
                    },
                    "graduation_tte": {
                        "effective_event_observed": (
                            t0row.get("graduation_tte") or {}
                        ).get("effective_event_observed"),
                        "split_boundary_censored": (
                            t0row.get("graduation_tte") or {}
                        ).get("split_boundary_censored"),
                        "frozen_tte_sec": (t0row.get("graduation_tte") or {}).get(
                            "frozen_tte_sec"
                        ),
                    },
                    "explosion_tte": {
                        "effective_event_observed": (
                            t0row.get("explosion_tte") or {}
                        ).get("effective_event_observed"),
                        "split_boundary_censored": (
                            t0row.get("explosion_tte") or {}
                        ).get("split_boundary_censored"),
                    },
                }
        elapsed = max(time.time() - t0, 1e-6)
        results[name] = {
            "examples": n_ex,
            "batches": n_batch,
            "examples_per_sec": n_ex / elapsed,
            "batches_per_sec": n_batch / elapsed,
            "elapsed_sec": elapsed,
            "sequence_shapes": shapes,
            "length_means": {k: float(np.mean(v)) for k, v in length_acc.items()},
            "truncation_rate_mean": {
                k: trunc_sum[k] / max(n_batch, 1) for k in trunc_sum
            },
            "modality_availability_mean": {
                k: modality_sum[k] / max(modality_n, 1) for k in modality_sum
            },
            "target_status_example": target_examples,
            "causal_failures": causal_failures,
            "production": True,
        }

    progress(85, "smoke mixed")
    t0 = time.time()
    n_mixed = 0
    for _ in mixed_loader(
        world_root=world_root,
        split_root=split_root,
        split=SPLIT_TRAIN,
        batch_size=batch_size,
        seed=seed,
        max_batches=max(5, max_batches // 3),
        use_production=True,
    ):
        n_mixed += 1
    elapsed = max(time.time() - t0, 1e-6)
    results["mixed"] = {
        "batches": n_mixed,
        "batches_per_sec": n_mixed / elapsed,
        "elapsed_sec": elapsed,
        "production": True,
    }
    results["peak_rss_mb"] = _rss_mb()
    progress(95, "smoke done")
    return results
