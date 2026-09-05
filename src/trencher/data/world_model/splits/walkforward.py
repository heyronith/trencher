"""Walk-forward fold definitions using global launch quantiles."""

from __future__ import annotations

from typing import Any

from trencher.data.world_model.splits.cutoffs import SPLIT_TEST


def build_walkforward_folds(cutoffs: dict[str, Any]) -> dict[str, Any]:
    """Expanding chronological folds; permanent test never participates."""
    q = cutoffs["quantiles_unix"]
    launch_min = float(cutoffs["launch_min_unix"])
    launch_max = float(cutoffs["launch_max_unix"])
    folds = [
        {
            "fold_id": "wf_fold_1",
            "train": {"start_unix": launch_min, "end_unix_exclusive": q["q40"]},
            "val": {"start_unix": q["q40"], "end_unix_exclusive": q["q55"]},
            "rule": "train=[min,q40), val=[q40,q55)",
        },
        {
            "fold_id": "wf_fold_2",
            "train": {"start_unix": launch_min, "end_unix_exclusive": q["q55"]},
            "val": {"start_unix": q["q55"], "end_unix_exclusive": q["q70"]},
            "rule": "train=[min,q55), val=[q55,q70)",
        },
        {
            "fold_id": "wf_fold_3",
            "train": {"start_unix": launch_min, "end_unix_exclusive": q["q70"]},
            "val": {"start_unix": q["q70"], "end_unix_exclusive": q["q85"]},
            "rule": "train=[min,q70), val=[q70,q85)",
        },
    ]
    return {
        "method": "global_launch_quantiles_expanding",
        "permanent_test": {
            "start_unix": q["q85"],
            "end_unix_exclusive": None,
            "launch_max_unix": launch_max,
            "rule": "[q85, max]",
        },
        "final_test_isolation": {
            "no_normalization": True,
            "no_threshold_selection": True,
            "no_hyperparameter_selection": True,
            "no_early_stopping": True,
            "no_architecture_selection": True,
            "excluded_from_walkforward": True,
        },
        "folds": folds,
        "note": "Quantiles are global (full event_curve mass), not pre-test-only.",
        "test_split_name": SPLIT_TEST,
    }


def fold_split_for_launch(
    detected_at_unix: float, fold: dict[str, Any]
) -> str | None:
    """Return 'train'/'val' for a fold, or None if outside fold coverage (e.g. test)."""
    t = float(detected_at_unix)
    tr = fold["train"]
    va = fold["val"]
    if tr["start_unix"] <= t < tr["end_unix_exclusive"]:
        return "train"
    if va["start_unix"] <= t < va["end_unix_exclusive"]:
        return "val"
    return None
