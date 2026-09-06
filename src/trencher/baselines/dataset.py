"""Load joined baseline feature + target matrices (train/val only)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from trencher.baselines.feature_cache import feature_cache_root, population_feature_dir
from trencher.baselines.feature_spec import all_feature_names
from trencher.baselines.split_guard import (
    assert_no_test_rows,
    filter_allowed_splits,
    refuse_test_split,
)
from trencher.baselines.targets import TargetSpec
from trencher.data.world_model.training.cache_build import cache_root

Population = Literal["broad", "rich"]

ProgressCb = Callable[[int, int, str], None]

_META_COLS = (
    "observation_id",
    "mint",
    "observation_time_t",
    "detected_at_unix",
    "token_age_seconds",
    "in_rich_subset",
    "outage_proximity_sec",
    "observation_during_jul3_outage",
    "creator_bucket",
    "mask_events",
    "mask_global",
    "mask_creator",
    "mask_micro",
    "base_split",
)


def load_cutoffs(split_root: Path) -> dict[str, Any]:
    return json.loads((split_root / "cutoffs.json").read_text(encoding="utf-8"))


def iter_feature_partitions(
    feat_root: Path, *, population: Population = "broad"
) -> list[Path]:
    return sorted(population_feature_dir(feat_root, population).glob("*/features.parquet"))


def _float_keep_zero(v: Any) -> float:
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _column_to_float32(table: pa.Table, name: str, n: int) -> np.ndarray:
    if name not in table.column_names:
        return np.zeros(n, dtype=np.float32)
    col = table.column(name)
    # Fill nulls with 0 to match prior to_pylist + _float_keep_zero semantics.
    arr = pc.fill_null(col, 0).to_numpy(zero_copy_only=False)
    return np.asarray(arr, dtype=np.float32)


def _normalize_tte_fields(trow: dict[str, Any]) -> dict[str, Any]:
    for prefix in ("explosion_tte", "graduation_tte"):
        if f"{prefix}_duration_sec" not in trow and f"{prefix}_effective_tte_sec" in trow:
            trow[f"{prefix}_duration_sec"] = trow.get(f"{prefix}_effective_tte_sec")
        if (
            f"{prefix}_event_observed" not in trow
            and f"{prefix}_effective_event_observed" in trow
        ):
            trow[f"{prefix}_event_observed"] = trow.get(f"{prefix}_effective_event_observed")
        if f"{prefix}_at_risk" not in trow:
            st = trow.get(f"{prefix}_effective_status") or trow.get(
                f"{prefix}_effective_censoring_status"
            )
            trow[f"{prefix}_at_risk"] = st not in (
                None,
                "left_truncated",
                "modality_unavailable",
                "source_boundary_unavailable",
            ) and trow.get(f"{prefix}_duration_sec") is not None
    return trow


def _load_rich_targets(cache: Path) -> dict[str, dict[str, Any]]:
    path = cache / "rich" / "joined.parquet"
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for tr in pq.read_table(path).to_pylist():
        oid = str(tr.get("observation_id") or "")
        if oid:
            out[oid] = tr
    return out


def _load_partition_targets(cache: Path, mint_partition: str) -> dict[str, dict[str, Any]]:
    tgt_path = cache / "broad" / "by_mint_partition" / mint_partition / "joined.parquet"
    if not tgt_path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for tr in pq.read_table(tgt_path).to_pylist():
        oid = str(tr.get("observation_id") or "")
        if oid:
            out[oid] = tr
    return out


def load_joined_matrices(
    *,
    world_root: Path,
    split_root: Path | None = None,
    split: str,
    feature_names: list[str] | None = None,
    max_partitions: int | None = None,
    include_rich_features: bool = False,
    population: Population | None = None,
    progress_cb: ProgressCb | None = None,
) -> dict[str, Any]:
    """Return X, meta rows, and slim target dicts for an allowed split.

    Feature matrix construction is columnar (pyarrow → numpy) for scale.
    Target join remains oid-keyed for V4 field normalization compatibility.
    """
    split = filter_allowed_splits(split)
    world_root = Path(world_root)
    split_root = Path(split_root or world_root / "splits" / "split_v1")
    feat_root = feature_cache_root(split_root=split_root)
    cache = cache_root(split_root)
    if population is None:
        population = "rich" if include_rich_features else "broad"
    names = feature_names or all_feature_names(include_rich=(population == "rich"))

    parts = iter_feature_partitions(feat_root, population=population)
    if max_partitions is not None:
        parts = parts[:max_partitions]

    rich_targets = _load_rich_targets(cache) if population == "rich" else None

    X_chunks: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    n_parts = len(parts)

    for pi, path in enumerate(parts):
        mp = path.parent.name
        if progress_cb is not None:
            progress_cb(pi, n_parts, f"load {split} features part {pi + 1}/{n_parts}")

        # Read only columns we need when possible.
        schema_names = set(pq.read_schema(path).names)
        cols = [c for c in names if c in schema_names]
        meta_want = [c for c in _META_COLS if c in schema_names]
        want = list(dict.fromkeys([*meta_want, *cols, "base_split"]))
        table = pq.read_table(path, columns=want)
        if "base_split" not in table.column_names:
            continue
        mask = pc.equal(table.column("base_split"), split)
        table = table.filter(mask)
        n = table.num_rows
        if n == 0:
            continue
        splits_present = [str(x) for x in table.column("base_split").to_pylist()]
        assert_no_test_rows(splits_present, context="features")

        feat_cols = []
        for name in names:
            feat_cols.append(_column_to_float32(table, name, n))
        X_part = np.column_stack(feat_cols).astype(np.float32, copy=False)

        oids = [str(x) for x in table.column("observation_id").to_pylist()]
        if rich_targets is not None:
            tgt_by_oid = rich_targets
        else:
            tgt_by_oid = _load_partition_targets(cache, mp)

        # Build meta via columnar extracts where present.
        col_cache: dict[str, list[Any]] = {}
        for c in _META_COLS:
            if c == "observation_id":
                continue
            if c in table.column_names:
                col_cache[c] = table.column(c).to_pylist()
            else:
                col_cache[c] = [None] * n

        for i, oid in enumerate(oids):
            meta.append(
                {
                    "observation_id": oid,
                    "mint": col_cache["mint"][i],
                    "mint_partition": mp,
                    "base_split": split,
                    "observation_time_t": col_cache["observation_time_t"][i],
                    "detected_at_unix": col_cache["detected_at_unix"][i],
                    "token_age_seconds": col_cache["token_age_seconds"][i],
                    "in_rich_subset": col_cache["in_rich_subset"][i],
                    "outage_proximity_sec": col_cache["outage_proximity_sec"][i],
                    "observation_during_jul3_outage": col_cache[
                        "observation_during_jul3_outage"
                    ][i],
                    "creator_bucket": col_cache["creator_bucket"][i],
                    "mask_events": col_cache["mask_events"][i],
                    "mask_global": col_cache["mask_global"][i],
                    "mask_creator": col_cache["mask_creator"][i],
                    "mask_micro": col_cache["mask_micro"][i],
                }
            )
            targets.append(_normalize_tte_fields(dict(tgt_by_oid.get(oid) or {})))

        X_chunks.append(X_part)

    if progress_cb is not None:
        progress_cb(n_parts, n_parts, f"load {split} features done")

    X = np.concatenate(X_chunks, axis=0) if X_chunks else np.zeros((0, len(names)), np.float32)
    return {
        "X": X,
        "feature_names": names,
        "meta": meta,
        "targets": targets,
        "split": split,
        "n": int(X.shape[0]),
        "population": population,
    }


def load_assignment_eligibility(
    split_root: Path, observation_ids: list[str], *, population: Population = "broad"
) -> dict[str, dict[str, Any]]:
    """Optional: map oid -> eligibility flags (sampled parts)."""
    want = set(observation_ids)
    out: dict[str, dict[str, Any]] = {}
    assign = (
        "observation_assignments_rich"
        if population == "rich"
        else "observation_assignments_broad"
    )
    for path in sorted((split_root / assign).glob("part-*.parquet")):
        t = pq.read_table(path)
        for r in t.to_pylist():
            oid = str(r.get("observation_id") or "")
            if oid in want:
                out[oid] = r
        if len(out) >= len(want):
            break
    return out


def merge_eligibility_into_targets(
    bundle: dict[str, Any], split_root: Path, spec: TargetSpec
) -> None:
    if not spec.eligible_col:
        return
    population = bundle.get("population") or "broad"
    oids = [m["observation_id"] for m in bundle["meta"]]
    elig = load_assignment_eligibility(split_root, oids, population=population)
    for i, oid in enumerate(oids):
        row = elig.get(oid)
        if row is not None:
            bundle["targets"][i][spec.eligible_col] = row.get(spec.eligible_col, True)
        else:
            bundle["targets"][i].setdefault(spec.eligible_col, True)


def walkforward_mask(
    meta: list[dict[str, Any]], fold: dict[str, Any], role: str
) -> np.ndarray:
    """Mask rows by detected_at_unix against walk-forward fold train/val window."""
    from trencher.data.world_model.splits.walkforward import fold_split_for_launch

    refuse_test_split(role)
    out = np.zeros(len(meta), dtype=bool)
    for i, m in enumerate(meta):
        det = m.get("detected_at_unix")
        if det is None:
            continue
        got = fold_split_for_launch(float(det), fold)
        out[i] = got == role
    return out
