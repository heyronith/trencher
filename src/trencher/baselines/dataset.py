"""Load joined baseline feature + target matrices (train/val only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from trencher.baselines.feature_cache import feature_cache_root
from trencher.baselines.feature_spec import all_feature_names
from trencher.baselines.split_guard import (
    assert_no_test_rows,
    filter_allowed_splits,
    refuse_test_split,
)
from trencher.baselines.targets import TargetSpec
from trencher.data.world_model.training.cache_build import cache_root


def load_cutoffs(split_root: Path) -> dict[str, Any]:
    return json.loads((split_root / "cutoffs.json").read_text(encoding="utf-8"))


def iter_feature_partitions(feat_root: Path) -> list[Path]:
    return sorted((feat_root / "broad" / "by_mint_partition").glob("*/features.parquet"))


def load_joined_matrices(
    *,
    world_root: Path,
    split_root: Path | None = None,
    split: str,
    feature_names: list[str] | None = None,
    max_partitions: int | None = None,
    include_rich_features: bool = False,
) -> dict[str, Any]:
    """Return X, meta rows, and slim target dicts for an allowed split."""
    split = filter_allowed_splits(split)
    world_root = Path(world_root)
    split_root = Path(split_root or world_root / "splits" / "split_v1")
    feat_root = feature_cache_root(split_root=split_root)
    cache = cache_root(split_root)
    names = feature_names or all_feature_names(include_rich=include_rich_features)

    parts = iter_feature_partitions(feat_root)
    if max_partitions is not None:
        parts = parts[:max_partitions]

    X_chunks: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []

    # Preload assignment eligibility from assignments (optional join)
    # Prefer columns already on feature rows + joined cache targets
    for path in parts:
        mp = path.parent.name
        ft = pq.read_table(path)
        rows = ft.to_pylist()
        # filter split
        rows = [r for r in rows if str(r.get("base_split")) == split]
        if not rows:
            continue
        assert_no_test_rows([r.get("base_split") for r in rows], context="features")

        tgt_path = cache / "broad" / "by_mint_partition" / mp / "joined.parquet"
        tgt_by_oid: dict[str, dict[str, Any]] = {}
        if tgt_path.exists():
            for tr in pq.read_table(tgt_path).to_pylist():
                oid = str(tr.get("observation_id") or "")
                if oid:
                    tgt_by_oid[oid] = tr

        # eligibility from assignments part if needed — join via oid from assignment files is heavy;
        # use joined cache + default True when missing
        xs = []
        for r in rows:
            oid = str(r["observation_id"])
            vec = np.array([float(r.get(n) or 0.0) for n in names], dtype=np.float32)
            xs.append(vec)
            meta.append(
                {
                    "observation_id": oid,
                    "mint": r.get("mint"),
                    "mint_partition": mp,
                    "base_split": split,
                    "observation_time_t": r.get("observation_time_t"),
                    "detected_at_unix": r.get("detected_at_unix"),
                    "token_age_seconds": r.get("token_age_seconds"),
                    "in_rich_subset": r.get("in_rich_subset"),
                    "outage_proximity_sec": r.get("outage_proximity_sec"),
                    "observation_during_jul3_outage": r.get(
                        "observation_during_jul3_outage"
                    ),
                    "creator_bucket": r.get("creator_bucket"),
                    "mask_events": r.get("mask_events"),
                    "mask_global": r.get("mask_global"),
                    "mask_creator": r.get("mask_creator"),
                    "mask_micro": r.get("mask_micro"),
                }
            )
            trow = dict(tgt_by_oid.get(oid) or {})
            # normalize TTE column names from cache overlays
            for prefix in ("explosion_tte", "graduation_tte"):
                if f"{prefix}_duration_sec" not in trow and f"{prefix}_effective_tte_sec" in trow:
                    trow[f"{prefix}_duration_sec"] = trow.get(f"{prefix}_effective_tte_sec")
                if f"{prefix}_event_observed" not in trow and f"{prefix}_effective_event_observed" in trow:
                    trow[f"{prefix}_event_observed"] = trow.get(
                        f"{prefix}_effective_event_observed"
                    )
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
            targets.append(trow)
        if xs:
            X_chunks.append(np.stack(xs, axis=0))

    X = np.concatenate(X_chunks, axis=0) if X_chunks else np.zeros((0, len(names)), np.float32)
    return {
        "X": X,
        "feature_names": names,
        "meta": meta,
        "targets": targets,
        "split": split,
        "n": int(X.shape[0]),
    }


def load_assignment_eligibility(
    split_root: Path, observation_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Optional: map oid -> eligibility flags (sampled parts)."""
    want = set(observation_ids)
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((split_root / "observation_assignments_broad").glob("part-*.parquet")):
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
    oids = [m["observation_id"] for m in bundle["meta"]]
    elig = load_assignment_eligibility(split_root, oids)
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
