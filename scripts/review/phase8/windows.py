"""Split-parametrized event / alert windows for val vs TEST."""

from __future__ import annotations

import json
from typing import Literal

import pyarrow.compute as pc
import pyarrow.parquet as pq
from scripts.review.phase8.common import SPLIT

SplitName = Literal["val", "test"]


def cutoffs_unix() -> dict[str, float]:
    c = json.loads((SPLIT / "cutoffs.json").read_text(encoding="utf-8"))
    q = c["quantiles_unix"]
    return {k: float(v) for k, v in q.items()}


def launch_max_unix() -> float:
    """Max detected_at_unix among TEST mints (metadata only)."""
    tok = pq.read_table(
        SPLIT / "token_assignments.parquet",
        columns=["base_split", "detected_at_unix"],
    )
    test = tok.filter(pc.equal(tok["base_split"], "test"))
    return float(pc.max(test["detected_at_unix"]).as_py())


def event_onset_window(split: SplitName) -> tuple[float, float]:
    """Inclusive-exclusive onset window for pump / 2× event filters.

    val → [q70, q85)
    test → [q85, launch_max]  (upper bound inclusive via +eps for < comparisons)
    """
    q = cutoffs_unix()
    if split == "val":
        return q["q70"], q["q85"]
    if split == "test":
        # Use end exclusive = launch_max + tiny epsilon so max onset is included
        # when callers use ``lo <= onset < hi``.
        return q["q85"], launch_max_unix() + 1e-6
    raise ValueError(f"unknown split {split!r}")


def alert_feed_span(split: SplitName) -> tuple[float, float]:
    """Hour-span anchors for alert-feed covering (same cutoffs as event window)."""
    return event_onset_window(split)
