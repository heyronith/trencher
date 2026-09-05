"""Build persisted split-aware TTE overlay tables (do not rewrite frozen targets)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.splits.progress import Progress100
from trencher.data.world_model.splits.tte_overlay import apply_tte_overlays_for_row
from trencher.data.world_model.targets.status import CORPUS_END_UNIX

logger = logging.getLogger(__name__)


def _load_cutoffs(split_root: Path) -> dict[str, Any]:
    return json.loads((split_root / "cutoffs.json").read_text(encoding="utf-8"))


def _build_oid_index(
    assignment_paths: list[Path],
    *,
    progress: Progress100,
    start: int,
    end: int,
) -> dict[str, tuple[str, float]]:
    """observation_id -> (base_split, observation_time_t)."""
    idx: dict[str, tuple[str, float]] = {}
    n = max(len(assignment_paths), 1)
    for i, path in enumerate(assignment_paths):
        progress(
            start + int((end - start) * (i + 1) / n),
            f"tte oid index {i + 1}/{n}",
        )
        table = pq.read_table(
            path, columns=["observation_id", "base_split", "observation_time_t"]
        )
        for oid, split, t in zip(
            table.column("observation_id").to_pylist(),
            table.column("base_split").to_pylist(),
            table.column("observation_time_t").to_pylist(),
            strict=True,
        ):
            if oid is None:
                continue
            idx[str(oid)] = (str(split), float(t))
    return idx


def build_tte_overlays_v1(
    *,
    world_root: Path,
    split_root: Path | None = None,
    progress: Progress100 | None = None,
) -> dict[str, Any]:
    """Materialize TTE overlays for broad and rich from frozen targets + assignments."""
    progress = progress or Progress100()
    world_root = Path(world_root)
    split_root = Path(split_root or world_root / "splits" / "split_v1")
    cutoffs = _load_cutoffs(split_root)
    q70 = float(cutoffs["quantiles_unix"]["q70"])
    q85 = float(cutoffs["quantiles_unix"]["q85"])

    out_broad = ensure_dir(split_root / "tte_overlays_broad")
    out_rich = split_root / "tte_overlays_rich.parquet"

    # --- Broad ---
    progress(5, "index broad assignments for TTE")
    assign_parts = sorted(
        (split_root / "observation_assignments_broad").glob("part-*.parquet")
    )
    oid_index = _build_oid_index(assign_parts, progress=progress, start=5, end=35)

    target_parts = sorted(
        (world_root / "broad" / "broad_targets_v3" / "observation_targets").glob(
            "part-*.parquet"
        )
    )
    cols = [
        "observation_id",
        "mint",
        "time_to_graduation_sec",
        "time_to_graduation_status",
        "graduated_at_or_before_t",
        "time_to_explosion_sec",
        "explosion_event_observed",
        "explosion_already_started_at_t",
        "explosion_right_censored",
    ]
    stats = {
        "broad_rows": 0,
        "broad_grad_admin_censored": 0,
        "broad_expl_admin_censored": 0,
        "rich_rows": 0,
        "rich_grad_admin_censored": 0,
        "rich_expl_admin_censored": 0,
    }
    n_tp = max(len(target_parts), 1)
    for ti, tpath in enumerate(target_parts):
        progress(
            35 + int(40 * (ti + 1) / n_tp),
            f"tte broad targets {ti + 1}/{n_tp}",
        )
        table = pq.read_table(tpath, columns=[c for c in cols if c in pq.read_schema(tpath).names])
        rows_out: list[dict[str, Any]] = []
        for row in table.to_pylist():
            oid = str(row.get("observation_id") or "")
            meta = oid_index.get(oid)
            if meta is None:
                continue
            split, T = meta
            overlay = apply_tte_overlays_for_row(
                row,
                observation_time_t=T,
                base_split=split,
                q70=q70,
                q85=q85,
                corpus_end=CORPUS_END_UNIX,
            )
            stats["broad_rows"] += 1
            if overlay.get("graduation_tte_split_boundary_censored"):
                stats["broad_grad_admin_censored"] += 1
            if overlay.get("explosion_tte_split_boundary_censored"):
                stats["broad_expl_admin_censored"] += 1
            rows_out.append(overlay)
        if rows_out:
            pq.write_table(
                pa.Table.from_pylist(rows_out), out_broad / f"part-{ti:05d}.parquet"
            )

    # --- Rich ---
    progress(78, "tte rich overlays")
    rich_assign = split_root / "observation_assignments_rich.parquet"
    rich_oid = _build_oid_index([rich_assign], progress=progress, start=78, end=82)
    rich_targets = (
        world_root / "targets" / "target_dataset_v3" / "observation_targets_v3.parquet"
    )
    table = pq.read_table(
        rich_targets, columns=[c for c in cols if c in pq.read_schema(rich_targets).names]
    )
    rows_out = []
    for row in table.to_pylist():
        oid = str(row.get("observation_id") or "")
        meta = rich_oid.get(oid)
        if meta is None:
            continue
        split, T = meta
        overlay = apply_tte_overlays_for_row(
            row,
            observation_time_t=T,
            base_split=split,
            q70=q70,
            q85=q85,
            corpus_end=CORPUS_END_UNIX,
        )
        stats["rich_rows"] += 1
        if overlay.get("graduation_tte_split_boundary_censored"):
            stats["rich_grad_admin_censored"] += 1
        if overlay.get("explosion_tte_split_boundary_censored"):
            stats["rich_expl_admin_censored"] += 1
        rows_out.append(overlay)
    pq.write_table(pa.Table.from_pylist(rows_out), out_rich)

    meta = {
        "generated_at": utc_now(),
        "q70": q70,
        "q85": q85,
        "corpus_end_unix": CORPUS_END_UNIX,
        "rule": "event_time < split_end may remain observed; else admin right-censor",
        "never_admin_censor_to_negative": True,
        "stats": stats,
        "artifacts": {
            "broad": str(out_broad),
            "rich": str(out_rich),
        },
    }
    write_json(split_root / "tte_overlays_manifest.json", meta)
    progress(100, "tte overlays complete")
    return meta
