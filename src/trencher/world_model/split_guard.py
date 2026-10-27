"""Split access guards for deep world-model training."""

from __future__ import annotations

from typing import Any

from trencher.data.world_model.splits.q85_unlock import q85_test_unlocked

ALLOWED_SPLITS = frozenset({"train", "val"})
ALLOWED_SPLITS_UNLOCKED = frozenset({"train", "val", "test"})
ALLOWED_PHASES = frozenset({"ssl", "broad", "rich"})
ALLOWED_MODES = frozenset(
    {
        "train",
        "eval",
        "smoke",
        "overfit",
        "wf_fold_1",
        "wf_fold_2",
        "final_train",
        "permanent_val_eval",
    }
)
ALLOWED_MODES_UNLOCKED = ALLOWED_MODES | frozenset({"permanent_test_eval"})


class PermanentTestAccessError(RuntimeError):
    """Raised when training/eval would touch permanent TEST (q85+)."""


class PhaseError(ValueError):
    """Invalid training phase."""


def refuse_test_split(split: str | None) -> None:
    """Hard refuse permanent TEST access unless q85 unlock is armed."""
    if split is None:
        return
    s = str(split).strip().lower()
    if s in {"test", "permanent_test", "holdout_test"}:
        if q85_test_unlocked():
            return
        raise PermanentTestAccessError(
            "World-model training refuses split=test / permanent TEST (q85+). "
            "Use train for fitting and val ([q70,q85)) for frozen-control evaluation."
        )


def assert_allowed_split(split: str) -> str:
    refuse_test_split(split)
    s = str(split).strip().lower()
    allowed = ALLOWED_SPLITS_UNLOCKED if q85_test_unlocked() else ALLOWED_SPLITS
    if s not in allowed:
        raise ValueError(f"split must be train|val|test, got {split!r}")
    return s


def assert_allowed_phase(phase: str) -> str:
    p = str(phase).strip().lower()
    if p not in ALLOWED_PHASES:
        raise PhaseError(f"phase must be one of {sorted(ALLOWED_PHASES)}, got {phase!r}")
    return p


def assert_allowed_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    allowed = ALLOWED_MODES_UNLOCKED if q85_test_unlocked() else ALLOWED_MODES
    if m not in allowed:
        raise ValueError(f"mode must be one of {sorted(allowed)}, got {mode!r}")
    return m


def assert_no_test_rows(base_splits: list[str] | Any, *, context: str = "world_model") -> None:
    if q85_test_unlocked():
        return
    bad = sum(1 for s in base_splits if str(s).lower() == "test")
    if bad:
        raise PermanentTestAccessError(
            f"{context}: found {bad} permanent-test rows; aborting."
        )
