"""Train-only transforms fit on unique underlying training sequence rows.

Order (required): clip raw values to train p01/p99 → zscore with train mean/std.
Never z-score masks, binary flags, categorical IDs, or graph node indices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from trencher.data.world_model.io import ensure_dir, utc_now, write_json


def fit_numeric_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 1.0, "n": 0.0, "p01": 0.0, "p99": 0.0}
    p01 = float(np.quantile(arr, 0.01))
    p99 = float(np.quantile(arr, 0.99))
    # Stats after the same clip applied at train time (clip → zscore).
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
    """Canonical transform: clip raw → zscore with train-only (clipped) mean/std."""
    return apply_zscore(apply_clip(values, stats), stats)


def apply_numeric_channel(
    values: np.ndarray, stats: dict[str, float] | None
) -> np.ndarray:
    if not stats:
        return np.asarray(values, dtype=np.float64)
    return apply_clip_then_zscore(values, stats)


def fit_sequence_transforms_from_unique_rows(
    *,
    sequence_parts: list[Path],
    train_mints: set[str],
    numeric_columns: list[str],
    mint_column: str = "mint",
    max_rows_per_part: int | None = None,
) -> dict[str, Any]:
    """Fit normalization on each training sequence row at most once.

    Dedupes by scanning underlying stores filtered to training mints — does not
    replay the same historical events through overlapping observations.
    """
    accum: dict[str, list[np.ndarray]] = {c: [] for c in numeric_columns}
    rows_seen = 0
    for path in sequence_parts:
        schema = set(pq.read_schema(path).names)
        cols = [c for c in [mint_column, *numeric_columns] if c in schema]
        if mint_column not in cols:
            continue
        table = pq.read_table(path, columns=cols)
        if max_rows_per_part is not None and table.num_rows > max_rows_per_part:
            table = table.slice(0, max_rows_per_part)
        mints = [str(m) for m in table.column(mint_column).to_pylist()]
        mask = np.array([m in train_mints for m in mints], dtype=bool)
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
        "fit_scope": "unique_underlying_training_sequence_rows",
        "transform_order": "clip_raw_p01_p99_then_zscore",
        "train_mints_n": len(train_mints),
        "rows_seen": rows_seen,
        "columns": stats,
    }


def persist_transforms(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    write_json(path, payload)


def load_transforms(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
