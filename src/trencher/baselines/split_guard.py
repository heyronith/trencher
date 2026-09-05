"""Hard guards: baselines must never touch permanent TEST."""

from __future__ import annotations

from typing import Any

from trencher.data.world_model.splits.cutoffs import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL

ALLOWED_SPLITS = frozenset({SPLIT_TRAIN, SPLIT_VAL})


class PermanentTestAccessError(RuntimeError):
    """Raised when a baseline command would touch q85+ TEST."""


def refuse_test_split(split: str | None) -> None:
    if split is None:
        return
    s = str(split).strip().lower()
    if s == SPLIT_TEST or s == "permanent_test":
        raise PermanentTestAccessError(
            "Baseline commands refuse split=test / permanent TEST (q85+). "
            "Use train for fitting and val ([q70,q85)) for frozen-control evaluation."
        )


def assert_no_test_rows(
    base_splits: list[str] | Any,
    *,
    context: str = "baseline",
) -> None:
    """Fail if any row carries base_split=test."""
    refuse_test_split  # noqa: B018 — keep import used for docs
    bad = 0
    for s in base_splits:
        if str(s).lower() == SPLIT_TEST:
            bad += 1
            if bad <= 3:
                pass
    if bad:
        raise PermanentTestAccessError(
            f"{context}: found {bad} permanent-test rows; aborting."
        )


def filter_allowed_splits(split: str) -> str:
    refuse_test_split(split)
    s = str(split).strip().lower()
    if s not in ALLOWED_SPLITS:
        raise ValueError(f"split must be train|val, got {split!r}")
    return s
