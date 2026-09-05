"""Load optional exec impact overlays from broad_targets_v3 (eval-only join)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

IMPACT_COLS = (
    "observation_id",
    "exec_1p0sol_gross_return",
    "exec_1p0sol_entry_impact_adjusted_return",
    "exec_1p0sol_curve_impact_adjusted_return",
)


def load_exec_impact_overlay(
    world_root: Path, observation_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Map oid → impact fields from frozen broad targets (no label mutation)."""
    want = set(observation_ids)
    if not want:
        return {}
    root = Path(world_root) / "broad" / "broad_targets_v3" / "observation_targets"
    if not root.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("part-*.parquet")):
        schema = set(pq.read_schema(path).names)
        use = [c for c in IMPACT_COLS if c in schema]
        if "observation_id" not in use:
            continue
        for r in pq.read_table(path, columns=use).to_pylist():
            oid = str(r.get("observation_id") or "")
            if oid in want:
                out[oid] = r
                if len(out) >= len(want):
                    return out
    return out


def merge_impact_into_targets(
    targets: list[dict[str, Any]], meta: list[dict[str, Any]], world_root: Path
) -> None:
    oids = [m["observation_id"] for m in meta]
    overlay = load_exec_impact_overlay(world_root, oids)
    for i, oid in enumerate(oids):
        extra = overlay.get(oid)
        if not extra:
            continue
        for k, v in extra.items():
            if k == "observation_id":
                continue
            if k not in targets[i] or targets[i].get(k) is None:
                targets[i][k] = v
