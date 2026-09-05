"""CPU production-loader smoke / realistic benchmark against frozen stores."""

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


def _bench_population(
    *,
    world_root: Path,
    split_root: Path,
    population: str,
    strategy: str,
    batch_size: int,
    max_batches: int,
    seed: int,
    use_cache: bool,
    shuffle: bool = True,
) -> dict[str, Any]:
    cold_ex = 0
    cold_t0 = time.time()
    steady_t0 = None
    n_ex = 0
    n_batch = 0
    shapes: dict[str, Any] = {}
    modality_sum: dict[str, float] = {}
    modality_n = 0
    trunc_sum: dict[str, float] = {}
    length_acc: dict[str, list[float]] = {}
    target_examples: dict[str, Any] = {}
    families: list[str] = []
    channel_schemas: dict[str, Any] = {}
    causal_failures = 0
    used_cache = False
    warmup_batches = min(5, max(1, max_batches // 10))

    for batch in iter_loader(
        world_root=world_root,
        split_root=split_root,
        population=population,
        split=SPLIT_TRAIN,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
        strategy=strategy,
        max_batches=max_batches,
        use_production=True,
        use_cache=use_cache,
    ):
        n_batch += 1
        n_ex += int(batch["batch_size"])
        used_cache = bool(batch.get("used_training_cache"))
        if n_batch == warmup_batches:
            cold_ex = n_ex
            cold_elapsed = max(time.time() - cold_t0, 1e-6)
            steady_t0 = time.time()
            steady_ex0 = n_ex
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
            families = list(t0row.get("families_present") or [])
            channel_schemas = batch.get("channel_schemas") or {}
            g = t0row.get("graduation_tte") or {}
            target_examples = {
                "available": t0row.get("available"),
                "families_present": families,
                "returns_h60": {
                    "value": (t0row.get("returns_h60") or {}).get("value"),
                    "is_missing": (t0row.get("returns_h60") or {}).get("is_missing"),
                    "frozen_status": (t0row.get("returns_h60") or {}).get("frozen_status"),
                },
                "graduation_tte": {
                    "duration_sec": g.get("duration_sec"),
                    "event_observed": g.get("event_observed"),
                    "censored": g.get("censored"),
                    "at_risk": g.get("at_risk"),
                    "split_boundary_censored": g.get("split_boundary_censored"),
                },
                "explosion_tte": {
                    "duration_sec": (t0row.get("explosion_tte") or {}).get("duration_sec"),
                    "event_observed": (t0row.get("explosion_tte") or {}).get(
                        "event_observed"
                    ),
                    "censored": (t0row.get("explosion_tte") or {}).get("censored"),
                },
                "barrier_sample": (t0row.get("barrier_u100_l30_h3600") or {}).get("value"),
                "execution_sample": (t0row.get("execution_gross_0p1sol") or {}).get(
                    "is_missing"
                ),
            }

    total_elapsed = max(time.time() - cold_t0, 1e-6)
    if steady_t0 is None:
        cold_ex = n_ex
        cold_elapsed = total_elapsed
        steady_ex = 0
        steady_elapsed = 1e-6
        steady_ex0 = 0
    else:
        cold_elapsed = max(steady_t0 - cold_t0, 1e-6)
        steady_elapsed = max(time.time() - steady_t0, 1e-6)
        steady_ex = max(n_ex - steady_ex0, 0)

    # Epoch estimate from train population sizes (approx)
    train_n = 5_879_063 if population == "broad" else 36_228
    steady_rate = steady_ex / steady_elapsed if steady_ex else n_ex / total_elapsed
    return {
        "batch_size": batch_size,
        "batches": n_batch,
        "examples": n_ex,
        "shuffle": shuffle,
        "used_training_cache": used_cache,
        "cold_start_examples_per_sec": cold_ex / cold_elapsed,
        "steady_state_examples_per_sec": steady_rate,
        "examples_per_sec": n_ex / total_elapsed,
        "batches_per_sec": n_batch / total_elapsed,
        "elapsed_sec": total_elapsed,
        "sequence_shapes": shapes,
        "channel_schemas": channel_schemas,
        "length_means": {k: float(np.mean(v)) for k, v in length_acc.items()},
        "truncation_rate_mean": {
            k: trunc_sum[k] / max(n_batch, 1) for k in trunc_sum
        },
        "modality_availability_mean": {
            k: modality_sum[k] / max(modality_n, 1) for k in modality_sum
        },
        "target_status_example": target_examples,
        "target_families": families,
        "causal_failures": causal_failures,
        "estimated_epoch_wall_clock_sec": train_n / max(steady_rate, 1e-6),
        "production": True,
    }


def run_production_loader_smoke(
    *,
    world_root: Path,
    split_root: Path,
    batch_size: int = 16,
    max_batches: int = 30,
    seed: int = 0,
    progress: Progress100 | None = None,
) -> dict[str, Any]:
    """Backward-compatible light smoke (delegates to realistic with smaller load)."""
    return run_realistic_loader_benchmark(
        world_root=world_root,
        split_root=split_root,
        batch_sizes=(batch_size,),
        max_batches=max_batches,
        seed=seed,
        progress=progress,
        light=True,
    )


def run_realistic_loader_benchmark(
    *,
    world_root: Path,
    split_root: Path,
    batch_sizes: tuple[int, ...] = (64, 128),
    max_batches: int = 200,
    seed: int = 0,
    progress: Progress100 | None = None,
    light: bool = False,
) -> dict[str, Any]:
    progress = progress or Progress100()
    results: dict[str, Any] = {
        "batch_sizes": list(batch_sizes),
        "max_batches": max_batches,
        "light": light,
    }
    step = 10
    for bs in batch_sizes:
        for name, population, strategy in (
            ("broad", "broad", STRATEGY_BROAD_PRETRAIN),
            ("rich", "rich", STRATEGY_RICH_MULTIMODAL),
        ):
            progress(step, f"bench {name} bs={bs}")
            step = min(80, step + (5 if light else 8))
            key = f"{name}_bs{bs}"
            results[key] = _bench_population(
                world_root=world_root,
                split_root=split_root,
                population=population,
                strategy=strategy,
                batch_size=bs,
                max_batches=max_batches if not light else min(max_batches, 20),
                seed=seed,
                use_cache=True,
                shuffle=True,
            )
            # Also publish primary keys for audit convenience
            if bs == batch_sizes[0]:
                results[name] = results[key]

    progress(85, "bench mixed")
    t0 = time.time()
    n_mixed = 0
    bs = batch_sizes[0]
    for _ in mixed_loader(
        world_root=world_root,
        split_root=split_root,
        split=SPLIT_TRAIN,
        batch_size=bs,
        seed=seed,
        max_batches=max(5, (max_batches if not light else 10) // 4),
        use_production=True,
        use_cache=True,
    ):
        n_mixed += 1
    elapsed = max(time.time() - t0, 1e-6)
    results["mixed"] = {
        "batches": n_mixed,
        "batches_per_sec": n_mixed / elapsed,
        "elapsed_sec": elapsed,
        "batch_size": bs,
    }
    results["peak_rss_mb"] = _rss_mb()
    progress(95, "benchmark complete")
    return results
