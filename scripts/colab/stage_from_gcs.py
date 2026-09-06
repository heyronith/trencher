#!/usr/bin/env python3
"""Stage required training artifacts from GCS (or local mirror) to Colab SSD and verify."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from trencher.world_model.staging import verify_staging_manifest


def _stage_from_local(src: Path, dest: Path, manifest: dict) -> dict:
    copied = skipped = verified = 0
    for art in manifest.get("artifacts") or []:
        rel = art["path"]
        s = src / rel
        d = dest / rel
        if not s.exists():
            if art.get("required"):
                raise FileNotFoundError(f"required missing in source: {rel}")
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.exists() and d.stat().st_size == s.stat().st_size:
            skipped += 1
        else:
            shutil.copy2(s, d)
            copied += 1
        verified += 1
    return {"copied": copied, "skipped": skipped, "verified": verified}


def _default_dest() -> str:
    """Frozen world-model root: WORLD_ROOT or LOCAL_STAGE/world_model."""
    world = os.environ.get("TRENCHER_WORLD_ROOT")
    if world:
        return world
    stage = os.environ.get("TRENCHER_LOCAL_STAGE", "/content/trencher_stage")
    return str(Path(stage) / "world_model")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default=os.environ.get(
            "TRENCHER_STAGING_MANIFEST",
            "manifests/world_model_data/WORLD_MODEL_STAGING_MANIFEST_V2.json",
        ),
    )
    ap.add_argument("--dest", default=_default_dest())
    ap.add_argument("--gcs-uri", default=os.environ.get("TRENCHER_GCS_URI", ""))
    ap.add_argument("--local-source", default=os.environ.get("TRENCHER_LOCAL_MIRROR", ""))
    args = ap.parse_args()

    man_path = Path(args.manifest)
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(
        f"staging files={manifest.get('total_files')} bytes={manifest.get('total_bytes')} -> {dest}"
    )

    if args.local_source:
        stats = _stage_from_local(Path(args.local_source), dest, manifest)
        print(json.dumps({"backend": "local", **stats}))
    elif args.gcs_uri:
        try:
            from google.cloud import storage  # type: ignore
        except ImportError as exc:
            raise SystemExit("google-cloud-storage required for GCS staging") from exc
        uri = args.gcs_uri.replace("gs://", "")
        bucket_name, _, prefix = uri.partition("/")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        copied = skipped = 0
        for art in manifest.get("artifacts") or []:
            rel = art["path"]
            blob_path = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
            blob = bucket.blob(blob_path)
            d = dest / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            if d.exists() and blob.exists() and blob.size == d.stat().st_size:
                skipped += 1
                continue
            if not blob.exists():
                if art.get("required"):
                    raise FileNotFoundError(blob_path)
                continue
            blob.download_to_filename(str(d))
            copied += 1
        print(json.dumps({"backend": "gcs", "copied": copied, "skipped": skipped}))
    else:
        raise SystemExit("Provide --local-source or --gcs-uri / TRENCHER_GCS_URI")

    # Rewrite manifest root to dest for verification
    manifest["root"] = str(dest)
    staged_man = dest / "WORLD_MODEL_STAGING_MANIFEST_V2.json"
    staged_man.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = verify_staging_manifest(staged_man, fail_closed=True)
    print(json.dumps({"verify": result}))
    if not result.get("ok"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
