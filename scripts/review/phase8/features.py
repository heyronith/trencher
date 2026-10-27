"""Split-agnostic broad feature loading via the baseline_features_v1 code path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from scripts.review.phase5.matrices import load_head_bundle
from scripts.review.phase8.common import Q85, REPO, SPLIT, WORLD, write_json

from trencher.baselines.feature_cache import build_baseline_features_v1
from trencher.baselines.feature_spec import broad_feature_names

SplitName = Literal["val", "test"]


class FeatureBuildRefusal(RuntimeError):
    """Hard refuse TEST feature materialization in Phase-8."""


def ensure_broad_features(*, split: SplitName, resume: bool = True) -> dict[str, Any]:
    """Ensure broad features exist using ``build_baseline_features_v1``.

    For ``val``: invoke the same builder (train+val only; never TEST). Existing
    partitions are resumed. Features are then loaded via ``load_head_bundle``.

    For ``test``: refuse in Phase-8 (TEST locked). The call site for TEST must
    unlock this in a later phase without changing TEST_PROTOCOL_V2.md.
    """
    if split == "test":
        raise FeatureBuildRefusal(
            "Phase-8 refuses to build or read q85+ TEST features. "
            "Unlock only in a dedicated later phase after val proof."
        )

    # Prefer phase4 overlay (full 256 partitions) when present — same builder.
    overlay = REPO / "scratch/phase4_clean_v1/split_overlay"
    feat_overlay = overlay / "baseline_features_v1" / "broad" / "by_mint_partition"
    if overlay.exists() and len(list(feat_overlay.glob("*/features.parquet"))) >= 200:
        split_root = overlay
        source = "phase4_split_overlay"
        # Features already built by build_baseline_features_v1 into overlay.
        meta = {
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
    # Confirm on-disk feature_names match contract order
    names_path = Path(split_root) / "baseline_features_v1" / "feature_names.json"
    if names_path.exists():
        disk = json.loads(names_path.read_text(encoding="utf-8"))
        if disk != names:
            raise RuntimeError("feature_names.json order differs from broad_feature_names()")
    meta["feature_names"] = names
    meta["n_features"] = len(names)
    return meta


def load_split_bundle(head_id: str, split: SplitName) -> dict[str, Any]:
    """Load X/y/elig for a head after ensure_broad_features(split)."""
    if split == "test":
        raise FeatureBuildRefusal("Phase-8 refuses TEST matrix load")
    ensure_broad_features(split=split)
    return load_head_bundle(head_id, split)


def write_feature_meta(out_dir: Path, meta: dict[str, Any]) -> None:
    write_json(Path(out_dir) / "feature_build_meta.json", meta)
