"""Evaluation-only control masks (must not feed training features)."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.splits.cutoffs import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
from trencher.data.world_model.targets.status import OUTAGE_END_UNIX, OUTAGE_START_UNIX


def build_eval_controls(
    token_table: pa.Table,
    *,
    population: str,
) -> dict[str, Any]:
    """Creator novelty / overlap and launch-regime summaries for a population."""
    rows = token_table.to_pylist()
    if population == "broad":
        rows = [r for r in rows if r.get("in_broad_event_curve")]
    elif population == "rich":
        rows = [r for r in rows if r.get("in_rich_subset")]

    by_split: dict[str, list[dict[str, Any]]] = {
        SPLIT_TRAIN: [],
        SPLIT_VAL: [],
        SPLIT_TEST: [],
    }
    for r in rows:
        by_split[r["base_split"]].append(r)

    train_creators = {
        str(r["creator"]) for r in by_split[SPLIT_TRAIN] if r.get("creator")
    }
    val_creators = {str(r["creator"]) for r in by_split[SPLIT_VAL] if r.get("creator")}
    test_creators = {
        str(r["creator"]) for r in by_split[SPLIT_TEST] if r.get("creator")
    }

    def _creator_flags(split_rows: list[dict[str, Any]]) -> dict[str, int]:
        unseen = 0
        overlap = 0
        for r in split_rows:
            c = r.get("creator")
            if not c:
                continue
            if str(c) in train_creators:
                overlap += 1
            else:
                unseen += 1
        return {
            "tokens": len(split_rows),
            "tokens_unseen_creator_vs_train": unseen,
            "tokens_creator_overlap_vs_train": overlap,
        }

    outage_launches = sum(
        1
        for r in rows
        if OUTAGE_START_UNIX <= float(r["detected_at_unix"]) < OUTAGE_END_UNIX
    )

    return {
        "population": population,
        "train_creators_n": len(train_creators),
        "val_creators_n": len(val_creators),
        "test_creators_n": len(test_creators),
        "val_creators_unseen_vs_train": len(val_creators - train_creators),
        "test_creators_unseen_vs_train": len(test_creators - train_creators),
        "val_creator_overlap_vs_train": len(val_creators & train_creators),
        "test_creator_overlap_vs_train": len(test_creators & train_creators),
        "val_token_creator_flags": _creator_flags(by_split[SPLIT_VAL]),
        "test_token_creator_flags": _creator_flags(by_split[SPLIT_TEST]),
        "tokens_launched_during_jul3_outage": outage_launches,
        "note": (
            "Creator/wallet novelty masks are evaluation controls only; "
            "they must not introduce future information into training features."
        ),
    }


def attach_obs_eval_controls_summary(
    obs_stats: dict[str, Any],
    *,
    population: str,
) -> dict[str, Any]:
    """Summarize launch-age / outage proximity from observation assignment stats."""
    return {
        "population": population,
        "observations_assigned": obs_stats.get("observations_assigned"),
        "observations_purged_late_state": obs_stats.get(
            "observations_purged_late_state"
        ),
        "observations_by_base_split": obs_stats.get("observations_by_base_split"),
    }


def write_creator_eval_table(
    token_table: pa.Table, path: Any
) -> None:
    """Per-token eval flags relative to permanent train creators."""
    rows = token_table.to_pylist()
    train_creators = {
        str(r["creator"])
        for r in rows
        if r.get("base_split") == SPLIT_TRAIN and r.get("creator")
    }
    out = []
    for r in rows:
        c = r.get("creator")
        unseen = bool(c) and str(c) not in train_creators
        overlap = bool(c) and str(c) in train_creators
        if r.get("base_split") == SPLIT_TRAIN:
            unseen = False
            overlap = bool(c)
        out.append(
            {
                "mint": r["mint"],
                "base_split": r["base_split"],
                "creator": c,
                "eval_unseen_creator_vs_train": unseen,
                "eval_creator_overlap_vs_train": overlap,
                "in_broad_event_curve": r.get("in_broad_event_curve"),
                "in_rich_subset": r.get("in_rich_subset"),
            }
        )
    pq.write_table(pa.Table.from_pylist(out), path)
