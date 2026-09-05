"""Train-only transforms fit on rows legally reachable by TRAIN observations.

Order (required): clip raw values to train p01/p99 → zscore with train mean/std.

Fit constraints:
  * event_time strictly before q70
  * no val/test rows (train mints / train observation row ranges only)
  * no post-boundary continuation of train mints

Prefer union of underlying row ranges referenced by train observations so
overlapping observations count each physical sequence row once.

Never z-score masks, binary flags, categorical IDs, or graph node indices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.training.schemas import (
    BOUNDED_ORDINAL_EXEMPT,
    RELATIVE_TIME_EXEMPT,
    continuous_channel_names,
)


def fit_numeric_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 1.0, "n": 0.0, "p01": 0.0, "p99": 0.0}
    p01 = float(np.quantile(arr, 0.01))
    p99 = float(np.quantile(arr, 0.99))
    clipped = np.clip(arr, p01, p99)
    std = float(clipped.std())
    return {
        "mean": float(clipped.mean()),
        "std": std if std > 1e-12 else 1.0,
        "n": float(arr.size),
        "p01": p01,
        "p99": p99,
        "order": "clip_raw_p01_p99_then_zscore",
    }


def apply_clip(values: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.clip(arr, stats["p01"], stats["p99"])


def apply_zscore(values: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return (arr - stats["mean"]) / stats["std"]


def apply_clip_then_zscore(values: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    return apply_zscore(apply_clip(values, stats), stats)


def apply_numeric_channel(
    values: np.ndarray, stats: dict[str, float] | None
) -> np.ndarray:
    if not stats:
        return np.asarray(values, dtype=np.float64)
    return apply_clip_then_zscore(values, stats)


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    out = [ordered[0]]
    for lo, hi in ordered[1:]:
        plo, phi = out[-1]
        if lo <= phi:
            out[-1] = (plo, max(phi, hi))
        else:
            out.append((lo, hi))
    return out


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return merge_intervals(intervals)


def collect_train_row_ranges(
    assignment_parts: list[Path],
    *,
    start_col: str,
    end_col: str,
    partition_col: str = "mint_partition",
    split_col: str = "base_split",
    max_parts: int | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Union of [start, end) index ranges from TRAIN observations, per partition."""
    parts = assignment_parts
    if max_parts is not None:
        parts = parts[:max_parts]
    by_part: dict[str, list[tuple[int, int]]] = {}
    for path in parts:
        schema = set(pq.read_schema(path).names)
        need = [split_col, start_col, end_col]
        if not all(c in schema for c in need):
            continue
        cols = need + ([partition_col] if partition_col in schema else [])
        table = pq.read_table(path, columns=cols)
        splits = table.column(split_col).to_pylist()
        starts = table.column(start_col).to_pylist()
        ends = table.column(end_col).to_pylist()
        if partition_col in table.column_names:
            parts_ids = [str(x) for x in table.column(partition_col).to_pylist()]
        else:
            parts_ids = ["_"] * table.num_rows
        for sp, lo, hi, mp in zip(splits, starts, ends, parts_ids, strict=True):
            if sp != "train" or lo is None or hi is None:
                continue
            lo_i, hi_i = int(lo), int(hi)
            if hi_i <= lo_i:
                continue
            by_part.setdefault(mp, []).append((lo_i, hi_i))
    return {mp: _merge_intervals(ivs) for mp, ivs in by_part.items()}


def fit_sequence_transforms_from_unique_rows(
    *,
    sequence_parts: list[Path],
    train_mints: set[str],
    numeric_columns: list[str],
    mint_column: str = "mint",
    time_column: str = "event_time_unix",
    max_event_time: float | None = None,
    max_rows_per_part: int | None = None,
    row_ranges_by_partition: dict[str, list[tuple[int, int]]] | None = None,
    partition_from_path: Any | None = None,
) -> dict[str, Any]:
    """Fit normalization on each eligible training sequence row at most once.

    Eligibility: mint in train_mints AND (if max_event_time set) event_time < q70.
    When row_ranges_by_partition is provided, only rows in the union of train
    observation index ranges are used (each physical row once via merge).
    """
    accum: dict[str, list[np.ndarray]] = {c: [] for c in numeric_columns}
    rows_seen = 0
    for path in sequence_parts:
        schema = set(pq.read_schema(path).names)
        cols = [c for c in [mint_column, time_column, *numeric_columns] if c in schema]
        if mint_column not in cols:
            continue
        table = pq.read_table(path, columns=cols)
        n = table.num_rows
        mask = np.zeros(n, dtype=bool)

        if row_ranges_by_partition is not None:
            mp = None
            if partition_from_path is not None:
                mp = partition_from_path(path)
            else:
                # Heuristic: .../mint_partition=XX/... or parent name
                for part in Path(path).parts:
                    if part.startswith("mint_partition="):
                        mp = part.split("=", 1)[1]
                        break
                if mp is None:
                    mp = Path(path).parent.name
            ranges = row_ranges_by_partition.get(str(mp), [])
            for lo, hi in ranges:
                lo2 = max(0, lo)
                hi2 = min(n, hi)
                if hi2 > lo2:
                    mask[lo2:hi2] = True
        else:
            mask[:] = True
            if max_rows_per_part is not None and n > max_rows_per_part:
                # Only when not using explicit ranges
                mask[max_rows_per_part:] = False

        mints = np.asarray([str(m) for m in table.column(mint_column).to_pylist()])
        mask &= np.array([m in train_mints for m in mints], dtype=bool)

        if max_event_time is not None and time_column in table.column_names:
            times = np.asarray(table.column(time_column).to_pylist(), dtype=np.float64)
            mask &= np.isfinite(times) & (times < float(max_event_time))

        if not mask.any():
            continue
        rows_seen += int(mask.sum())
        for c in numeric_columns:
            if c not in table.column_names:
                continue
            vals = np.asarray(table.column(c).to_pylist(), dtype=np.float64)
            accum[c].append(vals[mask])

    stats = {
        c: fit_numeric_stats(np.concatenate(chunks) if chunks else np.array([]))
        for c, chunks in accum.items()
    }
    return {
        "generated_at": utc_now(),
        "fit_scope": "train_obs_unique_rows_pre_q70",
        "transform_order": "clip_raw_p01_p99_then_zscore",
        "max_event_time": max_event_time,
        "train_mints_n": len(train_mints),
        "rows_seen": rows_seen,
        "columns": stats,
        "relative_time_policy": "raw_exempt_not_zscored",
        "bounded_ordinal_exempt": list(BOUNDED_ORDINAL_EXEMPT),
        "relative_time_exempt": list(RELATIVE_TIME_EXEMPT),
    }


def fit_column_stats_from_table(
    path: Path,
    *,
    numeric_columns: list[str],
    time_column: str | None = None,
    max_event_time: float | None = None,
    second_column: str | None = None,
) -> dict[str, dict[str, float]]:
    """Fit stats from a single table with optional time filter."""
    if not path.exists():
        return {c: fit_numeric_stats(np.array([])) for c in numeric_columns}
    schema = set(pq.read_schema(path).names)
    cols = [c for c in numeric_columns if c in schema]
    extra = []
    if time_column and time_column in schema:
        extra.append(time_column)
    if second_column and second_column in schema:
        extra.append(second_column)
    table = pq.read_table(path, columns=list(dict.fromkeys([*cols, *extra])))
    mask = np.ones(table.num_rows, dtype=bool)
    if max_event_time is not None:
        if time_column and time_column in table.column_names:
            times = np.asarray(table.column(time_column).to_pylist(), dtype=np.float64)
            mask &= np.isfinite(times) & (times < float(max_event_time))
        elif second_column and second_column in table.column_names:
            secs = np.asarray(table.column(second_column).to_pylist(), dtype=np.float64)
            mask &= np.isfinite(secs) & (secs < float(max_event_time))
    out: dict[str, dict[str, float]] = {}
    for c in numeric_columns:
        if c not in table.column_names:
            out[c] = fit_numeric_stats(np.array([]))
            continue
        vals = np.asarray(table.column(c).to_pylist(), dtype=np.float64)
        out[c] = fit_numeric_stats(vals[mask] if mask.any() else np.array([]))
    return out


def transform_coverage_manifest(fit_payload: dict[str, Any]) -> dict[str, Any]:
    """Publish exact transform coverage + intentional exemptions."""
    cols = fit_payload.get("columns") or {}
    continuous = continuous_channel_names()
    covered = sorted(c for c in continuous if c in cols and (cols[c].get("n") or 0) > 0)
    empty = sorted(c for c in continuous if c in cols and (cols[c].get("n") or 0) == 0)
    missing = sorted(c for c in continuous if c not in cols)
    return {
        "transform_order": "clip_raw_p01_p99_then_zscore",
        "fit_scope": fit_payload.get("fit_scope"),
        "max_event_time": fit_payload.get("max_event_time"),
        "transformed_continuous_channels": covered,
        "empty_train_fit_identity_passthrough": empty,
        "missing_or_empty_fit": sorted(set(empty) | set(missing)),
        "intentionally_untransformed": {
            "relative_time_raw": list(RELATIVE_TIME_EXEMPT),
            "bounded_ordinals": list(BOUNDED_ORDINAL_EXEMPT),
            "never_normalize": [
                "masks",
                "binary_flags",
                "categorical_ids",
                "graph_node_ids",
            ],
            "empty_train_channels_note": (
                "Channels with n=0 in train fit pass through unscaled "
                "(e.g. priority_fee_lamports often unavailable)."
            ),
        },
        "rows_seen": fit_payload.get("rows_seen"),
        "column_stats_n": {k: v.get("n") for k, v in cols.items()},
    }


def persist_transforms(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    write_json(path, payload)


def load_transforms(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
