#!/usr/bin/env python3
"""Sync / print the world-model staging manifest (env-based, no credentials)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    gcs = os.environ.get("TRENCHER_GCS_URI", "").rstrip("/")
    stage = Path(os.environ.get("TRENCHER_LOCAL_STAGE", "/content/trencher_stage"))
    stage.mkdir(parents=True, exist_ok=True)
    manifest_name = os.environ.get(
        "TRENCHER_STAGING_MANIFEST", "WORLD_MODEL_STAGING_MANIFEST_V1.json"
    )

    print(f"TRENCHER_GCS_URI={gcs or '(unset)'}")
    print(f"TRENCHER_LOCAL_STAGE={stage}")

    local_manifest = stage / manifest_name
    repo_manifest = (
        Path(__file__).resolve().parents[2]
        / "manifests"
        / "world_model_data"
        / "WORLD_MODEL_CANDIDATE_V1.json"
    )

    if local_manifest.exists():
        data = json.loads(local_manifest.read_text(encoding="utf-8"))
        print(f"found local staging manifest: {local_manifest}")
        print(json.dumps({"version": data.get("version"), "n": len(data.get("artifacts", []))}))
        return 0

    if repo_manifest.exists():
        print(f"repo candidate manifest present: {repo_manifest}")
        print("Run stage_from_gcs.py to materialize artifacts under TRENCHER_LOCAL_STAGE.")
        return 0

    print("No staging manifest found yet.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
