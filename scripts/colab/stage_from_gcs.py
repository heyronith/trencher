#!/usr/bin/env python3
"""Stage world-model artifacts from GCS or a local mirror (no credentials in code)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REQUIRED_RELATIVE = (
    "configs/world_model/world_model_candidate_v1.yaml",
    "manifests/world_model_data/WORLD_MODEL_TRAINING_CONTRACT_V1.json",
    "manifests/world_model_data/WORLD_MODEL_CANDIDATE_V1.json",
)


def _is_gcs(uri: str) -> bool:
    return uri.startswith("gs://")


def _copy_one(src_root: str, rel: str, dest_root: Path) -> None:
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _is_gcs(src_root):
        src = f"{src_root.rstrip('/')}/{rel}"
        # Prefer gsutil when available; fall back to gcloud storage.
        cmds = [
            ["gsutil", "-q", "cp", src, str(dest)],
            ["gcloud", "storage", "cp", src, str(dest)],
        ]
        last_err: Exception | None = None
        for cmd in cmds:
            try:
                subprocess.run(cmd, check=True)
                return
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                last_err = exc
        raise RuntimeError(f"failed to copy {src}: {last_err}")
    src_path = Path(src_root) / rel
    if not src_path.exists():
        # Also allow staging from the git checkout itself
        repo = Path(__file__).resolve().parents[2]
        alt = repo / rel
        if alt.exists():
            src_path = alt
        else:
            raise FileNotFoundError(src_path)
    shutil.copy2(src_path, dest)


def main() -> int:
    src = os.environ.get("TRENCHER_GCS_URI", "").strip()
    stage = Path(os.environ.get("TRENCHER_LOCAL_STAGE", "/content/trencher_stage"))
    stage.mkdir(parents=True, exist_ok=True)

    if not src:
        # Default: stage from the repository checkout (offline / Colab-mounted drive)
        src = str(Path(__file__).resolve().parents[2])
        print(f"TRENCHER_GCS_URI unset; staging from repo root {src}")

    for rel in REQUIRED_RELATIVE:
        print(f"staging {rel}")
        _copy_one(src, rel, stage)

    print(f"staged into {stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
