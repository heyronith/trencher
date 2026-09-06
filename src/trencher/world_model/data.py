"""Dataset / collator interfaces with permanent-TEST refusal."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch.utils.data import Dataset

from trencher.world_model.split_guard import (
    PermanentTestAccessError,
    assert_allowed_mode,
    assert_allowed_split,
    refuse_test_split,
)


class WorldModelExample(Protocol):
    def as_tensors(self) -> dict[str, torch.Tensor]: ...


@dataclass
class SyntheticExample:
    """In-memory example used by unit tests (not production data)."""

    tensors: dict[str, torch.Tensor]
    split: str = "train"

    def as_tensors(self) -> dict[str, torch.Tensor]:
        return dict(self.tensors)


class WorldModelDataset(Dataset):
    """Thin dataset wrapper that refuses permanent TEST splits."""

    def __init__(
        self,
        examples: Sequence[dict[str, Any] | SyntheticExample],
        *,
        split: str = "train",
        mode: str = "train",
    ) -> None:
        refuse_test_split(split)
        self.split = assert_allowed_split(split)
        self.mode = assert_allowed_mode(mode)
        self.examples: list[dict[str, Any]] = []
        for ex in examples:
            if isinstance(ex, SyntheticExample):
                refuse_test_split(ex.split)
                row = ex.as_tensors()
                row["_split"] = ex.split
            else:
                refuse_test_split(ex.get("split") or ex.get("base_split"))
                row = dict(ex)
            self.examples.append(row)
        splits = [str(r.get("_split", self.split)) for r in self.examples]
        if any(s.lower() == "test" for s in splits):
            raise PermanentTestAccessError("dataset contains permanent-test examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def collate_world_model_batch(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack tensor fields; refuse if any row claims test split."""
    for r in rows:
        refuse_test_split(r.get("_split") or r.get("split") or r.get("base_split"))
    keys = [k for k in rows[0] if torch.is_tensor(rows[0][k])]
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        out[k] = torch.stack([r[k] for r in rows], dim=0)
    return out


class IterableBatchSource:
    """Minimal iterable of pre-collated batches (CPU smoke / overfit)."""

    def __init__(self, batches: Sequence[dict[str, torch.Tensor]], *, split: str = "train") -> None:
        refuse_test_split(split)
        self.split = assert_allowed_split(split)
        self.batches = list(batches)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)
