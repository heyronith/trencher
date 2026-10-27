"""Evaluation-mode guards: permanent_val only; q85+ TEST fail-closed unless unlocked."""

from __future__ import annotations

from typing import Any

from trencher.data.world_model.splits.q85_unlock import q85_test_unlocked
from trencher.world_model.split_guard import PermanentTestAccessError, refuse_test_split
from trencher.world_model.v4_loader import assert_dataset_mode

FORBIDDEN_EVAL_ALIASES = frozenset(
    {
        "q85_test",
        "test",
        "permanent_test",
        "holdout_test",
        "q85",
        "q85_plus",
        "permanent_test_q85",
    }
)

ALLOWED_STEP17_MODES = frozenset({"permanent_val_eval", "wf_fold_1", "wf_fold_2"})
ALLOWED_STEP17_MODES_UNLOCKED = ALLOWED_STEP17_MODES | frozenset({"permanent_test_eval"})


def assert_step17_mode(mode: str) -> str:
    """Refuse every TEST alias and unknown mode; allow permanent_test_eval when unlocked."""
    m = str(mode).strip().lower()
    if m in FORBIDDEN_EVAL_ALIASES:
        raise PermanentTestAccessError(
            f"Step-17 refuses evaluation mode {mode!r}; q85+ permanent TEST is locked."
        )
    asserted = assert_dataset_mode(m)
    allowed = (
        ALLOWED_STEP17_MODES_UNLOCKED if q85_test_unlocked() else ALLOWED_STEP17_MODES
    )
    if asserted not in allowed:
        raise PermanentTestAccessError(
            f"Step-17 mode {mode!r} is not permitted (allowed={sorted(allowed)})"
        )
    return asserted


def assert_no_test_population(population: str) -> str:
    p = str(population).strip().lower()
    if p in FORBIDDEN_EVAL_ALIASES or p == "test":
        raise PermanentTestAccessError(
            f"Step-17 refuses population={population!r}"
        )
    if p not in {"broad", "rich"}:
        raise ValueError(f"population must be broad|rich, got {population!r}")
    return p


def provenance_q85_flag(*, accessed: bool = False) -> dict[str, Any]:
    if accessed:
        if q85_test_unlocked():
            return {"q85_test_accessed": True}
        raise PermanentTestAccessError(
            "Step-17 abort: q85 TEST rows appeared during evaluation"
        )
    return {"q85_test_accessed": False}


def refuse_test_base_splits(base_splits: list[str] | tuple[str, ...], *, context: str) -> None:
    for s in base_splits:
        refuse_test_split(s)
        if str(s).strip().lower() == "test" and not q85_test_unlocked():
            raise PermanentTestAccessError(f"{context}: permanent TEST row detected")
