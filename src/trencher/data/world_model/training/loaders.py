"""Training loader factories for broad / rich / mixed strategies."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from trencher.data.world_model.splits.cutoffs import SPLIT_TRAIN
from trencher.data.world_model.training.collate import collate_examples
from trencher.data.world_model.training.dataset import (
    StreamingSplitDataset,
    synthetic_example_from_assignment,
)

STRATEGY_BROAD_PRETRAIN = "broad_event_curve_pretrain"
STRATEGY_RICH_MULTIMODAL = "rich_multimodal"
STRATEGY_FUSION = "multimodal_fusion"
STRATEGY_MIXED = "mixed_broad_rich"


def resolve_assignment_paths(split_root: Path, population: str) -> list[Path]:
    if population == "broad":
        d = split_root / "observation_assignments_broad"
        paths = sorted(d.glob("part-*.parquet"))
        if not paths:
            raise FileNotFoundError(d)
        return paths
    rich = split_root / "observation_assignments_rich.parquet"
    if rich.exists():
        return [rich]
    d = split_root / "observation_assignments_rich"
    paths = sorted(d.glob("part-*.parquet"))
    if not paths:
        raise FileNotFoundError("rich observation assignments missing")
    return paths


def iter_loader(
    *,
    split_root: Path,
    population: str,
    split: str = SPLIT_TRAIN,
    batch_size: int = 32,
    seed: int = 0,
    shuffle: bool = False,
    strategy: str = STRATEGY_BROAD_PRETRAIN,
    max_batches: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield collated batches reflecting true population distribution (no pos sampling)."""
    paths = resolve_assignment_paths(split_root, population)
    ds = StreamingSplitDataset(
        assignment_paths=paths,
        split=split,
        seed=seed,
        shuffle=shuffle,
        population=population if population != "mixed" else "broad",
    )
    n = 0
    for batch_rows in ds.iter_batches(batch_size, max_batches=max_batches):
        examples = [synthetic_example_from_assignment(r) for r in batch_rows]
        if strategy == STRATEGY_RICH_MULTIMODAL:
            examples = [e for e in examples if e["meta"].get("population") == "rich"]
            if not examples:
                continue
        batch = collate_examples(examples)
        batch["strategy"] = strategy
        batch["population"] = population
        batch["split"] = split
        # Hooks for later GPU training
        batch["prefetch_ready"] = False
        batch["pin_memory"] = False
        yield batch
        n += 1
        if max_batches is not None and n >= max_batches:
            return


def mixed_loader(
    *,
    split_root: Path,
    split: str = SPLIT_TRAIN,
    batch_size: int = 32,
    seed: int = 0,
    max_batches: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Alternate broad/rich batches with availability masks (true mixture, no outcome sampling)."""
    broad = iter_loader(
        split_root=split_root,
        population="broad",
        split=split,
        batch_size=batch_size // 2 or 1,
        seed=seed,
        shuffle=True,
        strategy=STRATEGY_MIXED,
        max_batches=None,
    )
    rich = iter_loader(
        split_root=split_root,
        population="rich",
        split=split,
        batch_size=batch_size // 2 or 1,
        seed=seed + 1,
        shuffle=True,
        strategy=STRATEGY_MIXED,
        max_batches=None,
    )
    n = 0
    while True:
        try:
            b = next(broad)
        except StopIteration:
            break
        try:
            r = next(rich)
        except StopIteration:
            yield b
            n += 1
            if max_batches is not None and n >= max_batches:
                return
            continue
        yield {
            "strategy": STRATEGY_MIXED,
            "broad": b,
            "rich": r,
            "batch_size": b["batch_size"] + r["batch_size"],
            "prefetch_ready": False,
            "pin_memory": False,
        }
        n += 1
        if max_batches is not None and n >= max_batches:
            return
