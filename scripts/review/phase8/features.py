"""Split-agnostic broad feature loading via the baseline_features_v1 code path."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from scripts.review.phase5.matrices import load_head_bundle
from scripts.review.phase8.common import Q85, REPO, SPLIT, WORLD, write_json

from trencher.baselines.feature_cache import build_baseline_features_v1
from trencher.baselines.feature_spec import broad_feature_names
from trencher.data.world_model.splits.q85_unlock import q85_test_unlocked

SplitName = Literal["val", "test"]

# TEST features live under phase9 scratch; never written into freeze train+val cache.
TEST_FEATURE_ROOT = REPO / "scratch" / "phase9_test_once_v1" / "baseline_features_v1"
TEST_SPLIT_OVERLAY = REPO / "scratch" / "phase9_test_once_v1" / "split_overlay"


class FeatureBuildRefusal(RuntimeError):
    """Hard refuse TEST feature IO without q85 unlock."""


def _ensure_test_overlay() -> Path:
    """Overlay: TEST features + symlinks to freeze assignments / training_cache / cutoffs."""
    overlay = TEST_SPLIT_OVERLAY
    overlay.mkdir(parents=True, exist_ok=True)
    feat_dst = overlay / "baseline_features_v1"
    if feat_dst.exists() or feat_dst.is_symlink():
        if feat_dst.is_symlink() or feat_dst.is_dir():
            pass
        else:
            raise RuntimeError(f"unexpected path {feat_dst}")
    else:
        # Features are written directly into TEST_FEATURE_ROOT; symlink into overlay.
        TEST_FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
        os.symlink(TEST_FEATURE_ROOT, feat_dst)

    for name in (
        "training_cache_v1",
        "observation_assignments_broad",
        "cutoffs.json",
        "walkforward_folds.json",
    ):
        dst = overlay / name
        src = SPLIT / name
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(src, dst)
    return overlay


def ensure_broad_features(*, split: SplitName, resume: bool = True) -> dict[str, Any]:
    """Ensure broad features exist using ``build_baseline_features_v1``.

    val: resume existing train+val cache (phase4 overlay preferred).
    test: requires q85 unlock; builds TEST-only features into scratch out_root.
    """
    if split == "test":
        if not q85_test_unlocked():
            raise FeatureBuildRefusal(
                "TEST feature build requires TRENCHER_Q85_TEST_UNLOCK "
                "(TEST_ACCESSED.lock must be written first)"
            )
        print(
            "building TEST broad features via build_baseline_features_v1 "
            f"→ {TEST_FEATURE_ROOT} ...",
            flush=True,
        )
        built = build_baseline_features_v1(
            world_root=WORLD,
            split_root=SPLIT,
            population="broad",
            resume=resume,
            base_splits=("test",),
            out_root=TEST_FEATURE_ROOT,
        )
        overlay = _ensure_test_overlay()
        names = broad_feature_names()
        disk = json.loads((TEST_FEATURE_ROOT / "feature_names.json").read_text())
        if disk != names:
            raise RuntimeError("TEST feature_names.json order differs from broad_feature_names()")
        return {
            **Q85,
            "q85_test_accessed": True,
            "split": "test",
            "split_root": str(overlay),
            "feature_root": str(TEST_FEATURE_ROOT),
            "builder": "build_baseline_features_v1",
            "base_splits": ["test"],
            "build_meta": built,
            "feature_names": names,
            "n_features": len(names),
        }

    # --- val ---
    overlay = REPO / "scratch/phase4_clean_v1/split_overlay"
    feat_overlay = overlay / "baseline_features_v1" / "broad" / "by_mint_partition"
    if overlay.exists() and len(list(feat_overlay.glob("*/features.parquet"))) >= 200:
        split_root = overlay
        source = "phase4_split_overlay"
        meta: dict[str, Any] = {
            "resumed_existing": True,
            "split_root": str(split_root),
            "source": source,
            "builder": "build_baseline_features_v1",
            **Q85,
        }
    else:
        split_root = SPLIT
        source = "freeze_split_v1"
        print("ensuring broad features via build_baseline_features_v1...", flush=True)
        built = build_baseline_features_v1(
            world_root=WORLD,
            split_root=split_root,
            population="broad",
            resume=resume,
            base_splits=("train", "val"),
        )
        meta = {
            "resumed_existing": False,
            "split_root": str(split_root),
            "source": source,
            "builder": "build_baseline_features_v1",
            "build_meta": built,
            **Q85,
        }

    names = broad_feature_names()
    names_path = Path(split_root) / "baseline_features_v1" / "feature_names.json"
    if names_path.exists():
        disk = json.loads(names_path.read_text(encoding="utf-8"))
        if disk != names:
            raise RuntimeError("feature_names.json order differs from broad_feature_names()")
    meta["feature_names"] = names
    meta["n_features"] = len(names)
    meta["split"] = "val"
    return meta


def load_split_bundle(head_id: str, split: SplitName) -> dict[str, Any]:
    """Load X/y/elig for a head after ensure_broad_features(split)."""
    if split == "test" and not q85_test_unlocked():
        raise FeatureBuildRefusal(
            "TEST matrix load requires TRENCHER_Q85_TEST_UNLOCK "
            "(TEST_ACCESSED.lock must be written first)"
        )
    meta = ensure_broad_features(split=split)
    # Point load_head_bundle at the correct feature root via env consumed below.
    if split == "test":
        os.environ["TRENCHER_PHASE8_SPLIT_ROOT"] = str(TEST_SPLIT_OVERLAY)
        os.environ["TRENCHER_PHASE8_FORCE_SPLIT"] = "test"
    else:
        os.environ.pop("TRENCHER_PHASE8_FORCE_SPLIT", None)
        # Prefer overlay used by ensure
        os.environ["TRENCHER_PHASE8_SPLIT_ROOT"] = str(meta["split_root"])
    try:
        return load_head_bundle(head_id, split)
    finally:
        os.environ.pop("TRENCHER_PHASE8_FORCE_SPLIT", None)


def write_feature_meta(out_dir: Path, meta: dict[str, Any]) -> None:
    write_json(Path(out_dir) / "feature_build_meta.json", meta)
