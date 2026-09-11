"""Evaluation-mode guards: permanent_val only; q85+ TEST fail-closed."""

from __future__ import annotations

from typing import Any

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


def assert_step17_mode(mode: str) -> str:
    """Refuse every TEST alias and unknown mode; allow permanent_val / wf only."""
    m = str(mode).strip().lower()
    if m in FORBIDDEN_EVAL_ALIASES:
        raise PermanentTestAccessError(
            f"Step-17 refuses evaluation mode {mode!r}; q85+ permanent TEST is locked."
        )
    # Delegate to loader guard (covers test/permanent_test/q85_test + allowlist).
    asserted = assert_dataset_mode(m)
    if asserted not in ALLOWED_STEP17_MODES:
        raise PermanentTestAccessError(
            f"Step-17 mode {mode!r} is not permitted "
            f"(allowed={sorted(ALLOWED_STEP17_MODES)})"
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
        raise PermanentTestAccessError(
            "Step-17 abort: q85 TEST rows appeared during evaluation"
        )
    return {"q85_test_accessed": False}


def refuse_test_base_splits(base_splits: list[str] | tuple[str, ...], *, context: str) -> None:
    for s in base_splits:
        refuse_test_split(s)
        if str(s).strip().lower() == "test":
            raise PermanentTestAccessError(f"{context}: permanent TEST row detected")
