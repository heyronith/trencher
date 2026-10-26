"""Memory-conscious broad train/val matrices for Phase-4 tree reproduction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from trencher.baselines.feature_cache import feature_cache_root, population_feature_dir
from trencher.baselines.feature_spec import broad_feature_names
from trencher.baselines.targets import CORE_TARGET_SUITE, TargetSpec
from trencher.data.world_model.training.cache_build import cache_root


def _column_f32(table, name: str, n: int) -> np.ndarray:  # noqa: ANN001
    if name not in table.column_names:
        return np.zeros(n, dtype=np.float32)
    py = table.column(name).to_numpy(zero_copy_only=False)
    if np.issubdtype(getattr(py, "dtype", np.dtype("O")), np.number):
        return np.nan_to_num(py.astype(np.float32, copy=False), nan=0.0)
    out = np.zeros(n, dtype=np.float32)
    for i, v in enumerate(py.tolist()):
        if v is None:
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            out[i] = 0.0
    return out


def _elig_map(split_root: Path, elig_col: str, split: str) -> dict[str, bool]:
    """Stream assignment files: oid -> eligible (train/val only)."""
    out: dict[str, bool] = {}
    assign_dir = split_root / "observation_assignments_broad"
    cols = ["observation_id", "base_split", elig_col]
    for ap in sorted(assign_dir.glob("part-*.parquet")):
        at = pq.read_table(ap, columns=cols)
        at = at.filter(pc.equal(at["base_split"], split))
        if at.num_rows == 0:
            continue
        for oid, ev in zip(
            at["observation_id"].to_pylist(),
            at[elig_col].to_pylist(),
            strict=True,
        ):
            out[str(oid)] = True if ev is None else bool(ev)
    return out


def _load_targets_for_mp(cache: Path, mp: str) -> dict[str, dict[str, Any]]:
    path = cache / "broad" / "by_mint_partition" / mp / "joined.parquet"
    if not path.exists():
        return {}
    # Read only label/status columns needed later — full to_pylist is heavy but
    # scoped per mint_partition (~30k rows).
    t = pq.read_table(path)
    if "observation_id" not in t.column_names:
        return {}
    return {
        str(oid): row
        for oid, row in zip(t["observation_id"].to_pylist(), t.to_pylist(), strict=True)
    }


def load_broad_head_arrays(
    *,
    split_root: Path,
    split: str,
    spec: TargetSpec,
) -> dict[str, Any]:
    """Return X, y, elig, times, observation_id, mint for one head (train or val)."""
    feat_root = feature_cache_root(split_root=split_root)
    parts_dir = population_feature_dir(feat_root, "broad")
    cache = cache_root(split_root)
    names = broad_feature_names()

    elig_col = spec.eligible_col
    label_col = spec.label_col
    status_col = spec.status_col
    elig_by_oid = _elig_map(split_root, elig_col, split) if elig_col else {}

    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    elig_chunks: list[np.ndarray] = []
    time_chunks: list[np.ndarray] = []
    oid_chunks: list[list[str]] = []
    mint_chunks: list[list[str]] = []

    for feat_path in sorted(parts_dir.glob("*/features.parquet")):
        mp = feat_path.parent.name
        schema = set(pq.read_schema(feat_path).names)
        cols = [
            c
            for c in ["observation_id", "mint", "base_split", "observation_time_t", *names]
            if c in schema
        ]
        table = pq.read_table(feat_path, columns=cols)
        table = table.filter(pc.equal(table["base_split"], split))
        n = table.num_rows
        if n == 0:
            continue

        oids = [str(x) for x in table["observation_id"].to_pylist()]
        tgt_by = _load_targets_for_mp(cache, mp)
        y = np.zeros(n, dtype=np.float64)
        elig = np.zeros(n, dtype=bool)
        for i, oid in enumerate(oids):
            if elig_col and not elig_by_oid.get(oid, True):
                continue
            row = tgt_by.get(oid) or {}
            if status_col:
                st = row.get(status_col)
                if st is not None and st not in ("valid_positive", "valid_negative"):
                    continue
            if spec.kind == "binary":
                val = row.get(label_col) if label_col else None
                if val is None:
                    continue
                label = 1.0 if bool(val) else 0.0
                st = row.get(status_col) if status_col else None
                if st == "valid_positive":
                    label = 1.0
                elif st == "valid_negative":
                    label = 0.0
                y[i] = label
                elig[i] = True
            elif spec.kind == "multiclass":
                mapping = spec.outcome_map or {}
                raw = row.get(label_col) if label_col else None
                if raw is None or str(raw) not in mapping:
                    continue
                y[i] = float(mapping[str(raw)])
                elig[i] = True

        feat_cols = [_column_f32(table, name, n) for name in names]
        X_chunks.append(np.column_stack(feat_cols).astype(np.float32, copy=False))
        y_chunks.append(y)
        elig_chunks.append(elig)
        times_raw = table["observation_time_t"].to_pylist()
        time_chunks.append(
            np.array([float(x or 0.0) for x in times_raw], dtype=np.float64)
        )
        oid_chunks.append(oids)
        mint_chunks.append([str(x) for x in table["mint"].to_pylist()])
        del table, tgt_by

    if not X_chunks:
        raise RuntimeError(f"no broad rows for split={split} head={spec.head_id}")

    return {
        "X": np.concatenate(X_chunks, axis=0),
        "y": np.concatenate(y_chunks, axis=0),
        "elig": np.concatenate(elig_chunks, axis=0),
        "times": np.concatenate(time_chunks, axis=0),
        "observation_id": np.asarray(sum(oid_chunks, []), dtype=object),
        "mint": np.asarray(sum(mint_chunks, []), dtype=object),
        "feature_names": names,
        "spec": spec.head_id,
        "split": split,
        "q85_test_accessed": False,
    }


def head_spec(head_id: str) -> TargetSpec:
    return next(s for s in CORE_TARGET_SUITE if s.head_id == head_id)
