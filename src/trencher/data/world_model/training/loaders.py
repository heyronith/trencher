"""Training loader factories for broad / rich / mixed strategies."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from trencher.data.world_model.splits.cutoffs import SPLIT_TRAIN
from trencher.data.world_model.training.collate import collate_examples
from trencher.data.world_model.training.dataset import StreamingSplitDataset
from trencher.data.world_model.training.example import ProductionExampleBuilder

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
    world_root: Path | None = None,
    split_root: Path,
    population: str,
    split: str = SPLIT_TRAIN,
    batch_size: int = 32,
    seed: int = 0,
    shuffle: bool = False,
    strategy: str = STRATEGY_BROAD_PRETRAIN,
    max_batches: int | None = None,
    use_production: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield collated batches from frozen stores (true population distribution)."""
    paths = resolve_assignment_paths(split_root, population)
    ds = StreamingSplitDataset(
        assignment_paths=paths,
        split=split,
        seed=seed,
        shuffle=shuffle,
        population=population if population != "mixed" else "broad",
    )
    world_root = Path(world_root or Path(split_root).parents[1])
    # split_root = world_root/splits/split_v1 → parents[1] = world_root
    if Path(split_root).name == "split_v1":
        world_root = Path(split_root).parent.parent

    builder = None
    if use_production:
        builder = ProductionExampleBuilder(
            world_root=world_root,
            split_root=Path(split_root),
            population="rich" if population == "rich" else "broad",
        )

    for n, batch_rows in enumerate(
        ds.iter_batches(batch_size, max_batches=max_batches), start=1
    ):
        if builder is None:
            from trencher.data.world_model.training.dataset import (
                synthetic_example_from_assignment,
            )

            examples = [synthetic_example_from_assignment(r) for r in batch_rows]
        else:
            oids = [str(r["observation_id"]) for r in batch_rows]
            builder.targets.prefetch(oids)
            builder.tte.prefetch(oids)
            examples = [builder.build(r) for r in batch_rows]
        batch = collate_examples(examples)
        batch["strategy"] = strategy
        batch["population"] = population
        batch["split"] = split
        batch["production"] = bool(use_production)
        batch["prefetch_ready"] = False
        batch["pin_memory"] = False
        yield batch
        if max_batches is not None and n >= max_batches:
            return


def mixed_loader(
    *,
    world_root: Path | None = None,
    split_root: Path,
    split: str = SPLIT_TRAIN,
    batch_size: int = 32,
    seed: int = 0,
    max_batches: int | None = None,
    use_production: bool = True,
) -> Iterator[dict[str, Any]]:
    """Alternate broad/rich production batches (no outcome sampling)."""
    broad = iter_loader(
        world_root=world_root,
        split_root=split_root,
        population="broad",
        split=split,
        batch_size=batch_size // 2 or 1,
        seed=seed,
        shuffle=True,
        strategy=STRATEGY_MIXED,
        max_batches=None,
        use_production=use_production,
    )
    rich = iter_loader(
        world_root=world_root,
        split_root=split_root,
        population="rich",
        split=split,
        batch_size=batch_size // 2 or 1,
        seed=seed + 1,
        shuffle=True,
        strategy=STRATEGY_MIXED,
        max_batches=None,
        use_production=use_production,
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
            "production": use_production,
        }
        n += 1
        if max_batches is not None and n >= max_batches:
            return
