#!/usr/bin/env python3
"""Build rich baseline_features_v1 into the local freeze (train+val only)."""

from __future__ import annotations

from pathlib import Path

from trencher.baselines.feature_cache import build_baseline_features_v1
from trencher.baselines.progress import ProgressEmitter


def main() -> None:
    world_root = Path("data/external/world_model")
    meta = build_baseline_features_v1(
        world_root=world_root,
        population="rich",
        progress=ProgressEmitter("rich features"),
    )
    print("DONE", meta)


if __name__ == "__main__":
    main()
