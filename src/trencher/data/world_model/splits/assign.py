"""Token and observation split assignment."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.io import ensure_dir
from trencher.data.world_model.splits.cutoffs import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    assign_base_split,
    split_end_exclusive,
    split_start_inclusive,
)
from trencher.data.world_model.splits.purge import (
    all_eligibility_columns,
    compute_eligibility_flags,
)
from trencher.data.world_model.splits.walkforward import fold_split_for_launch
from trencher.data.world_model.targets.status import (
    OUTAGE_END_UNIX,
    OUTAGE_START_UNIX,
)

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, str], None]


def _to_unix(value: Any) -> float:
    if value is None:
        return float("nan")
    if hasattr(value, "timestamp"):
        return float(value.timestamp())
    return float(value)


def build_token_assignments(
    *,
    broad_population_path: Path,
    rich_subset_path: Path,
    cutoffs: dict[str, Any],
    walkforward: dict[str, Any],
) -> pa.Table:
    """Assign every broad + rich mint to a base split and walk-forward roles."""
    q70 = float(cutoffs["quantiles_unix"]["q70"])
    q85 = float(cutoffs["quantiles_unix"]["q85"])
    float(cutoffs["launch_min_unix"])

    broad = pq.read_table(
        broad_population_path,
        columns=[
            "mint",
            "detected_at_unix",
            "creator",
            "event_curve_population",
            "price_target_population",
            "launch_population",
            "graduation_event_population",
        ],
    ).to_pylist()
    rich_rows = []
    if rich_subset_path.exists():
        rich_rows = pq.read_table(rich_subset_path).to_pylist()
    rich_by_mint = {str(r["mint"]): r for r in rich_rows if r.get("mint")}

    folds = walkforward["folds"]
    out_rows: list[dict[str, Any]] = []
    for r in broad:
        mint = str(r["mint"])
        det = _to_unix(r.get("detected_at_unix"))
        if not np.isfinite(det):
            continue
        base = assign_base_split(det, q70=q70, q85=q85)
        rich = rich_by_mint.get(mint)
        row: dict[str, Any] = {
            "mint": mint,
            "detected_at_unix": det,
            "creator": r.get("creator"),
            "base_split": base,
            "in_broad_event_curve": bool(r.get("event_curve_population")),
            "in_broad_price_target": bool(r.get("price_target_population")),
            "in_broad_launch": bool(r.get("launch_population")),
            "in_broad_graduation_event": bool(r.get("graduation_event_population")),
            "in_rich_subset": rich is not None,
            "launch_regime": (rich or {}).get("launch_regime"),
            "launch_day": str((rich or {}).get("launch_day") or ""),
            "is_mayhem_mode": (rich or {}).get("is_mayhem_mode"),
        }
        for fold in folds:
            role = fold_split_for_launch(det, fold)
            row[f"{fold['fold_id']}_role"] = role
        out_rows.append(row)

    # Rich-only mints not in broad population (should be rare / empty).
    broad_mints = {str(r["mint"]) for r in broad}
    for mint, r in rich_by_mint.items():
        if mint in broad_mints:
            continue
        det = _to_unix(r.get("detected_at"))
        if not np.isfinite(det):
            continue
        base = assign_base_split(det, q70=q70, q85=q85)
        row = {
            "mint": mint,
            "detected_at_unix": det,
            "creator": r.get("creator"),
            "base_split": base,
            "in_broad_event_curve": False,
            "in_broad_price_target": False,
            "in_broad_launch": False,
            "in_broad_graduation_event": False,
            "in_rich_subset": True,
            "launch_regime": r.get("launch_regime"),
            "launch_day": str(r.get("launch_day") or ""),
            "is_mayhem_mode": r.get("is_mayhem_mode"),
        }
        for fold in folds:
            row[f"{fold['fold_id']}_role"] = fold_split_for_launch(det, fold)
        out_rows.append(row)

    return pa.Table.from_pylist(out_rows)


def actual_split_percentages(token_table: pa.Table, *, population_flag: str) -> dict[str, Any]:
    rows = token_table.to_pylist()
    subset = [r for r in rows if r.get(population_flag)]
    n = len(subset) or 1
    counts = {SPLIT_TRAIN: 0, SPLIT_VAL: 0, SPLIT_TEST: 0}
    for r in subset:
        counts[r["base_split"]] += 1
    return {
        "n": len(subset),
        "counts": counts,
        "fractions": {k: counts[k] / n for k in counts},
    }


def stream_observation_assignments(
    *,
    observation_parts: list[Path],
    mint_to_meta: dict[str, dict[str, Any]],
    cutoffs: dict[str, Any],
    out_dir: Path,
    progress: ProgressFn | None = None,
    progress_start: int = 0,
    progress_end: int = 50,
    label: str = "broad_obs",
) -> dict[str, Any]:
    """Stream observation index parts → assignments + head-specific eligibility."""
    ensure_dir(out_dir)
    q70 = float(cutoffs["quantiles_unix"]["q70"])
    q85 = float(cutoffs["quantiles_unix"]["q85"])
    launch_min = float(cutoffs["launch_min_unix"])
    # Open test end: beyond max launch + 24h + confirmation so test purge is permissive.
    test_open_end = float(cutoffs["launch_max_unix"]) + 86400.0 + 1800.0 + 1.0

    elig_cols = all_eligibility_columns()
    n_parts = max(len(observation_parts), 1)
    total_obs = 0
    in_interval = 0
    purged_late_obs = 0
    by_split = {SPLIT_TRAIN: 0, SPLIT_VAL: 0, SPLIT_TEST: 0}
    # Count purged flags for returns_h86400 and survival_h86400 as headline metrics.
    purge_counts: dict[str, int] = {c: 0 for c in elig_cols}
    eligible_counts: dict[str, int] = {c: 0 for c in elig_cols}

    for pi, path in enumerate(observation_parts):
        if progress:
            frac = (pi + 1) / n_parts
            step = progress_start + int((progress_end - progress_start) * frac)
            progress(min(step, progress_end), f"{label} part {pi + 1}/{n_parts}")

        schema_names = set(pq.read_schema(path).names)
        cols = [
            c
            for c in (
                "mint",
                "observation_id",
                "observation_time_t",
                "detected_at_unix",
                "token_age_seconds",
                "mint_partition",
            )
            if c in schema_names
        ]
        table = pq.read_table(path, columns=cols)
        rows_out: list[dict[str, Any]] = []
        for row in table.to_pylist():
            mint = str(row["mint"])
            meta = mint_to_meta.get(mint)
            if meta is None:
                continue
            base = meta["base_split"]
            T = float(row["observation_time_t"])
            split_start = split_start_inclusive(
                base, launch_min=launch_min, q70=q70, q85=q85
            )
            split_end = split_end_exclusive(
                base, q70=q70, q85=q85, test_open_end=test_open_end
            )
            flags = compute_eligibility_flags(
                T, split_start=split_start, split_end=split_end
            )
            total_obs += 1
            by_split[base] += 1
            if flags["obs_in_split_interval"]:
                in_interval += 1
            else:
                purged_late_obs += 1

            for c in elig_cols:
                if flags.get(c):
                    eligible_counts[c] += 1
                else:
                    purge_counts[c] += 1

            det = float(meta["detected_at_unix"])
            age = row.get("token_age_seconds")
            if age is None:
                age = T - det
            outage_dist = min(abs(T - OUTAGE_START_UNIX), abs(T - OUTAGE_END_UNIX))
            overlaps_outage = T < OUTAGE_END_UNIX and T >= OUTAGE_START_UNIX

            rec = {
                "observation_id": row["observation_id"],
                "mint": mint,
                "observation_time_t": T,
                "detected_at_unix": det,
                "token_age_seconds": float(age) if age is not None else None,
                "base_split": base,
                "creator": meta.get("creator"),
                "in_rich_subset": bool(meta.get("in_rich_subset")),
                "obs_in_split_interval": bool(flags["obs_in_split_interval"]),
                "outage_proximity_sec": float(outage_dist),
                "observation_during_jul3_outage": bool(overlaps_outage),
                "mint_partition": row.get("mint_partition"),
            }
            for c in elig_cols:
                rec[c] = bool(flags.get(c, False))
            rows_out.append(rec)

        if rows_out:
            out_path = out_dir / f"part-{pi:05d}.parquet"
            pq.write_table(pa.Table.from_pylist(rows_out), out_path)

    return {
        "observations_assigned": total_obs,
        "observations_in_split_interval": in_interval,
        "observations_purged_late_state": purged_late_obs,
        "observations_by_base_split": by_split,
        "eligible_counts": eligible_counts,
        "purged_or_ineligible_counts": purge_counts,
        "parts_written": len(list(out_dir.glob("part-*.parquet"))),
    }
