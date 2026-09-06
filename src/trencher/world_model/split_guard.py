"""Split access guards for deep world-model training."""

from __future__ import annotations

from typing import Any

ALLOWED_SPLITS = frozenset({"train", "val"})
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


class PermanentTestAccessError(RuntimeError):
    """Raised when training/eval would touch permanent TEST (q85+)."""


class PhaseError(ValueError):
    """Invalid training phase."""


def refuse_test_split(split: str | None) -> None:
    """Hard refuse permanent TEST access."""
    if split is None:
        return
    s = str(split).strip().lower()
    if s in {"test", "permanent_test", "holdout_test"}:
        raise PermanentTestAccessError(
            "World-model training refuses split=test / permanent TEST (q85+). "
            "Use train for fitting and val ([q70,q85)) for frozen-control evaluation."
        )


def assert_allowed_split(split: str) -> str:
    refuse_test_split(split)
    s = str(split).strip().lower()
    if s not in ALLOWED_SPLITS:
        raise ValueError(f"split must be train|val, got {split!r}")
    return s


def assert_allowed_phase(phase: str) -> str:
    p = str(phase).strip().lower()
    if p not in ALLOWED_PHASES:
        raise PhaseError(f"phase must be one of {sorted(ALLOWED_PHASES)}, got {phase!r}")
    return p


def assert_allowed_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m not in ALLOWED_MODES:
        raise ValueError(f"mode must be one of {sorted(ALLOWED_MODES)}, got {mode!r}")
    return m


def assert_no_test_rows(base_splits: list[str] | Any, *, context: str = "world_model") -> None:
    bad = sum(1 for s in base_splits if str(s).lower() == "test")
    if bad:
        raise PermanentTestAccessError(
            f"{context}: found {bad} permanent-test rows; aborting."
        )
