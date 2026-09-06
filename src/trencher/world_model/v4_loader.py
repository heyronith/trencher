"""Frozen V4 deep-training Dataset / DataLoader factory (real stores, not manifests-only)."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from trencher.data.world_model.splits.cutoffs import SPLIT_TRAIN, SPLIT_VAL
from trencher.data.world_model.splits.walkforward import fold_split_for_launch
from trencher.data.world_model.training.dataset import StreamingSplitDataset
from trencher.data.world_model.training.example import ProductionExampleBuilder
from trencher.data.world_model.training.loaders import resolve_assignment_paths
from trencher.world_model.batch_adapter import collate_tensor_examples, example_to_tensors
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


def assert_dataset_mode(mode: str) -> DatasetMode:
    mode = str(mode)
    if mode.lower() in {"test", "permanent_test", "q85_test"}:
        raise PermanentTestAccessError(f"dataset mode {mode!r} is forbidden")
    if mode not in _ALLOWED_MODES:
        raise PermanentTestAccessError(f"unknown/forbidden dataset mode {mode!r}")
    return mode  # type: ignore[return-value]


def _mode_to_base_split(mode: DatasetMode) -> str:
    if mode == "permanent_val_eval":
        return SPLIT_VAL
    return SPLIT_TRAIN


def _load_wf_fold(split_root: Path, fold_id: str) -> dict[str, Any]:
    path = Path(split_root) / "walkforward_folds.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    folds = raw.get("folds") if isinstance(raw, dict) else raw
    if not isinstance(folds, list):
        raise KeyError(f"unexpected walkforward schema in {path}")
    for f in folds:
        if isinstance(f, dict) and f.get("fold_id") == fold_id:
            return f
    raise KeyError(f"walk-forward fold {fold_id!r} not found in {path}")


def _row_in_wf_role(row: dict[str, Any], fold: dict[str, Any], role: str) -> bool:
    det = row.get("detected_at_unix")
    if det is None:
        return False
    return fold_split_for_launch(float(det), fold) == role


class FrozenV4IterableDataset(IterableDataset):
    """Stream frozen assignments → production examples → model tensors.

    Permanent TEST (q85+) cannot be selected via DatasetMode.
    """

    def __init__(
        self,
        *,
        world_root: Path,
        split_root: Path,
        mode: DatasetMode,
        population: Literal["broad", "rich"] = "broad",
        seed: int = 0,
        shuffle: bool = True,
        max_examples: int | None = None,
        use_cache: bool = True,
        path_steps: int = 8,
    ) -> None:
        super().__init__()
        self.mode = assert_dataset_mode(mode)
        self.world_root = Path(world_root)
        self.split_root = Path(split_root)
        self.population = population
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.max_examples = max_examples
        self.path_steps = int(path_steps)
        self.base_split = _mode_to_base_split(self.mode)
        refuse_test_split(self.base_split)

        paths = resolve_assignment_paths(self.split_root, population)
        self._ds = StreamingSplitDataset(
            assignment_paths=paths,
            split=self.base_split,
            seed=seed,
            shuffle=shuffle,
            population=population,
        )
        self._fold: dict[str, Any] | None = None
        self._wf_role: str | None = None
        if self.mode in {"wf_fold_1", "wf_fold_2"}:
            self._fold = _load_wf_fold(self.split_root, self.mode)
            # Internal development: train role within fold for fitting batches
            self._wf_role = "train"
        self._builder = ProductionExampleBuilder(
            world_root=self.world_root,
            split_root=self.split_root,
            population=population,
            use_cache=use_cache,
        )

    def _accept(self, row: dict[str, Any]) -> bool:
        refuse_test_split(row.get("base_split"))
        if str(row.get("base_split")).lower() == "test":
            return False
        if self._fold is not None and self._wf_role is not None:
            return _row_in_wf_role(row, self._fold, self._wf_role)
        return True

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        n = 0
        pending: list[dict[str, Any]] = []
        for row in self._ds.iter_rows(max_examples=None):
            if not self._accept(row):
                continue
            pending.append(row)
            if len(pending) >= 32:
                self._builder.prefetch(pending)
                for r in pending:
                    ex = self._builder.build(r)
                    if not ex.get("causal_ok", True):
                        continue
                    yield example_to_tensors(ex, path_steps=self.path_steps)
                    n += 1
                    if self.max_examples is not None and n >= self.max_examples:
                        return
                pending = []
        if pending:
            self._builder.prefetch(pending)
            for r in pending:
                ex = self._builder.build(r)
                if not ex.get("causal_ok", True):
                    continue
                yield example_to_tensors(ex, path_steps=self.path_steps)
                n += 1
                if self.max_examples is not None and n >= self.max_examples:
                    return


class FrozenV4MapDataset(Dataset):
    """Map-style dataset over a materialised list of assignment rows (fixtures / tiny runs)."""

    def __init__(
        self,
        assignments: list[dict[str, Any]],
        *,
        world_root: Path,
        split_root: Path,
        population: Literal["broad", "rich"] = "broad",
        use_cache: bool = True,
        path_steps: int = 8,
    ) -> None:
        for row in assignments:
            refuse_test_split(row.get("base_split"))
            if str(row.get("base_split")).lower() == "test":
                raise PermanentTestAccessError("map dataset contains TEST row")
        self.assignments = list(assignments)
        self.path_steps = path_steps
        self._builder = ProductionExampleBuilder(
            world_root=Path(world_root),
            split_root=Path(split_root),
            population=population,
            use_cache=use_cache,
        )

    def __len__(self) -> int:
        return len(self.assignments)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.assignments[idx]
        refuse_test_split(row.get("base_split"))
        self._builder.prefetch([row])
        ex = self._builder.build(row)
        return example_to_tensors(ex, path_steps=self.path_steps)


def make_frozen_dataloader(
    *,
    world_root: Path,
    split_root: Path,
    mode: DatasetMode,
    population: Literal["broad", "rich"] = "broad",
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 0,
    shuffle: bool = True,
    max_examples: int | None = None,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = 2,
    use_cache: bool = True,
) -> DataLoader:
    """Build a DataLoader over frozen V4 stores for a development/train mode."""
    mode = assert_dataset_mode(mode)
    ds = FrozenV4IterableDataset(
        world_root=world_root,
        split_root=split_root,
        mode=mode,
        population=population,
        seed=seed,
        shuffle=shuffle,
        max_examples=max_examples,
        use_cache=use_cache,
    )
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_tensor_examples,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(ds, **kwargs)


def audit_loader_sample(
    *,
    world_root: Path,
    split_root: Path,
    mode: DatasetMode,
    population: Literal["broad", "rich"] = "broad",
    max_examples: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """CPU loader audit: lengths, modality avail, targets, zero TEST, causal_ok."""
    t0 = time.time()
    ds = FrozenV4IterableDataset(
        world_root=world_root,
        split_root=split_root,
        mode=mode,
        population=population,
        seed=seed,
        shuffle=False,
        max_examples=max_examples,
        use_cache=True,
    )
    n = 0
    test_hits = 0
    causal_viol = 0
    event_lens: list[int] = []
    modality_hits: dict[str, int] = {
        "events": 0,
        "curve": 0,
        "global": 0,
        "microstructure": 0,
        "ownership": 0,
        "wallet_prior": 0,
        "graph": 0,
    }
    target_hits = {
        "explosion": 0,
        "survival": 0,
        "barrier": 0,
        "return": 0,
        "execution": 0,
    }
    for batch_row in ds:
        n += 1
        if float(batch_row.get("causal_ok", torch.tensor(1.0))) < 0.5:
            causal_viol += 1
        event_lens.append(int(batch_row["event_mask"].sum().item()))
        if float(batch_row["event_available"]) > 0:
            modality_hits["events"] += 1
        if float(batch_row["curve_available"]) > 0:
            modality_hits["curve"] += 1
        if float(batch_row["global_available"]) > 0:
            modality_hits["global"] += 1
        if float(batch_row["micro_available"]) > 0:
            modality_hits["microstructure"] += 1
        if float(batch_row["own_available"]) > 0:
            modality_hits["ownership"] += 1
        if float(batch_row["wallet_available"]) > 0:
            modality_hits["wallet_prior"] += 1
        if float(batch_row["graph_available"]) > 0:
            modality_hits["graph"] += 1
        if float(batch_row["explosion_mask"].sum()) > 0:
            target_hits["explosion"] += 1
        if float(batch_row["survival_horizon_mask"].sum()) > 0:
            target_hits["survival"] += 1
        if float(batch_row["barrier_mask"].sum()) > 0:
            target_hits["barrier"] += 1
        if float(batch_row["return_mask"].sum()) > 0:
            target_hits["return"] += 1
        if float(batch_row["execution_mask"].sum()) > 0:
            target_hits["execution"] += 1
    elapsed = max(time.time() - t0, 1e-9)
    lens = np.asarray(event_lens, dtype=np.float64) if event_lens else np.zeros(0)
    return {
        "mode": mode,
        "population": population,
        "observations_loaded": n,
        "q85_test_rows": test_hits,
        "causal_violations": causal_viol,
        "examples_per_sec": float(n / elapsed),
        "event_length": {
            "mean": float(lens.mean()) if lens.size else 0.0,
            "p50": float(np.median(lens)) if lens.size else 0.0,
            "p90": float(np.quantile(lens, 0.9)) if lens.size else 0.0,
            "max": int(lens.max()) if lens.size else 0,
        },
        "modality_availability_count": modality_hits,
        "target_availability_count": target_hits,
        "elapsed_sec": float(elapsed),
        "contracts": FrozenContractRefs().__dict__,
    }


def build_deep_loader_manifest(
    *,
    world_root: Path,
    split_root: Path,
    mode: DatasetMode,
    population: Literal["broad", "rich"] = "broad",
    refs: FrozenContractRefs | None = None,
) -> dict[str, Any]:
    mode = assert_dataset_mode(mode)
    refs = refs or FrozenContractRefs()
    return {
        "kind": "world_model_deep_loader_manifest_v2",
        "mode": mode,
        "population": population,
        "contracts": refs.__dict__,
        "paths": {
            "world_root": str(world_root),
            "split_root": str(split_root),
            "assignments": str(
                split_root
                / (
                    "observation_assignments_rich"
                    if population == "rich"
                    else "observation_assignments_broad"
                )
            ),
            "training_cache": str(split_root / "training_cache_v1"),
            "cutoffs": str(split_root / "cutoffs.json"),
        },
        "guards": {
            "permanent_test": "FORBIDDEN",
            "permanent_val_eval_only": mode == "permanent_val_eval",
            "selection_folds": ["wf_fold_1", "wf_fold_2"],
        },
        "implementation": "FrozenV4IterableDataset+ProductionExampleBuilder",
    }
