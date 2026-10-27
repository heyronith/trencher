"""Broad train/val/test matrix loaders for Phase-5 (overlay features + WF meta)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scripts.review.phase4.broad_matrices import head_spec, load_broad_head_arrays
from scripts.review.phase5.common import Q85, REPO, SCRATCH, SPLIT

from trencher.baselines.split_guard import refuse_test_split

OVERLAY = REPO / "scratch" / "phase4_clean_v1" / "split_overlay"


def broad_split_root() -> Path:
    """Prefer explicit phase8/9 overlay, else phase4 overlay, else freeze."""
    env = os.environ.get("TRENCHER_PHASE8_SPLIT_ROOT")
    if env:
        return Path(env)
    feat = OVERLAY / "baseline_features_v1" / "broad" / "by_mint_partition"
    if len(list(feat.glob("*/features.parquet"))) >= 200:
        return OVERLAY
    return SPLIT


def load_detected_at_map(split: str) -> dict[str, float]:
    """oid -> detected_at_unix for train/val/test assignments."""
    refuse_test_split(split)
    out: dict[str, float] = {}
    # Prefer overlay assignments if present, else freeze
    for root in (broad_split_root(), SPLIT):
        assign = root / "observation_assignments_broad"
        if not assign.exists():
            continue
        for ap in sorted(assign.glob("part-*.parquet")):
            t = pq.read_table(
                ap, columns=["observation_id", "base_split", "detected_at_unix"]
            )
            for oid, sp, det in zip(
                t["observation_id"].to_pylist(),
                t["base_split"].to_pylist(),
                t["detected_at_unix"].to_pylist(),
                strict=True,
            ):
                if sp != split:
                    continue
                if det is None:
                    continue
                out[str(oid)] = float(det)
        if out:
            return out
    return out


def load_head_bundle(head_id: str, split: str) -> dict[str, Any]:
    """Load X/y/elig/times/oids/mints + detected_at for one head."""
    refuse_test_split(split)
    # Cache TEST matrices under phase9 scratch to avoid colliding with val caches.
    if split == "test":
        cache_dir = (
            REPO / "scratch" / "phase9_test_once_v1" / "matrices" / head_id
        )
    else:
        cache_dir = SCRATCH / "matrices" / head_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f"{split}_meta.json"
    x_path = cache_dir / f"{split}_X.npy"
    y_path = cache_dir / f"{split}_y.npy"
    elig_path = cache_dir / f"{split}_elig.npy"
    times_path = cache_dir / f"{split}_times.npy"
    oid_path = cache_dir / f"{split}_oids.npy"
    mint_path = cache_dir / f"{split}_mints.npy"
    det_path = cache_dir / f"{split}_detected.npy"

    if x_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "X": np.load(x_path, mmap_mode="r"),
            "y": np.load(y_path),
            "elig": np.load(elig_path),
            "times": np.load(times_path),
            "observation_id": np.load(oid_path, allow_pickle=True),
            "mint": np.load(mint_path, allow_pickle=True),
            "detected_at": np.load(det_path),
            "feature_names": meta["feature_names"],
            "split": split,
            "head_id": head_id,
            **Q85,
            **({"q85_test_accessed": True} if split == "test" else {}),
        }

    root = broad_split_root()
    print(f"loading broad {split}/{head_id} from {root}...", flush=True)
    arr = load_broad_head_arrays(split_root=root, split=split, spec=head_spec(head_id))
    det_map = load_detected_at_map(split)
    detected = np.array(
        [det_map.get(str(o), float("nan")) for o in arr["observation_id"]],
        dtype=np.float64,
    )
    np.save(x_path, np.asarray(arr["X"], dtype=np.float32))
    np.save(y_path, np.asarray(arr["y"], dtype=np.float64))
    np.save(elig_path, np.asarray(arr["elig"], dtype=bool))
    np.save(times_path, np.asarray(arr["times"], dtype=np.float64))
    np.save(oid_path, arr["observation_id"])
    np.save(mint_path, arr["mint"])
    np.save(det_path, detected)
    meta = {
        "feature_names": list(arr["feature_names"]),
        "n": int(arr["X"].shape[0]),
        "n_features": int(arr["X"].shape[1]),
        "split_root": str(root),
        **Q85,
        **({"q85_test_accessed": True} if split == "test" else {}),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {
        "X": np.load(x_path, mmap_mode="r"),
        "y": np.load(y_path),
        "elig": np.load(elig_path),
        "times": np.load(times_path),
        "observation_id": np.load(oid_path, allow_pickle=True),
        "mint": np.load(mint_path, allow_pickle=True),
        "detected_at": detected,
        "feature_names": meta["feature_names"],
        "split": split,
        "head_id": head_id,
        **Q85,
        **({"q85_test_accessed": True} if split == "test" else {}),
    }


def walkforward_masks_from_detected(
    detected: np.ndarray, folds: list[dict[str, Any]]
) -> dict[str, dict[str, np.ndarray]]:
    """fold_id -> {train: mask, val: mask} using detected_at_unix (vectorized)."""
    det = np.asarray(detected, dtype=np.float64)
    finite = np.isfinite(det)
    out: dict[str, dict[str, np.ndarray]] = {}
    for fold in folds:
        fid = fold["fold_id"]
        tr_lo = float(fold["train"]["start_unix"])
        tr_hi = float(fold["train"]["end_unix_exclusive"])
        va_lo = float(fold["val"]["start_unix"])
        va_hi = float(fold["val"]["end_unix_exclusive"])
        tr = finite & (det >= tr_lo) & (det < tr_hi)
        va = finite & (det >= va_lo) & (det < va_hi)
        out[fid] = {"train": tr, "val": va}
    return out
