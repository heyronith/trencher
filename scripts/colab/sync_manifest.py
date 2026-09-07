#!/usr/bin/env python3
"""Sync required frozen training artifacts to GCS (resumable, hash-aware).

Never commits credentials. Uses google-cloud-storage if available, else local mirror.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from trencher.world_model.staging import (
    assert_staging_manifest_complete,
    discover_frozen_training_manifest,
    write_staging_manifest,
)


def _upload_local_mirror(src_root: Path, dest_root: Path, manifest: dict) -> dict:
    copied = skipped = 0
    bytes_copied = 0
    for art in manifest["artifacts"]:
        if art.get("sha256") == "" and art.get("required"):
            continue
        rel = art["path"]
        src = src_root / rel
        dst = dest_root / rel
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
        bytes_copied += int(src.stat().st_size)
    return {"copied": copied, "skipped": skipped, "bytes_copied": bytes_copied}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-root", default="data/external/world_model")
    ap.add_argument(
        "--out-manifest",
        default="manifests/world_model_data/WORLD_MODEL_STAGING_MANIFEST_V2.json",
    )
    ap.add_argument("--gcs-uri", default=os.environ.get("TRENCHER_GCS_URI", ""))
    ap.add_argument("--local-mirror", default=os.environ.get("TRENCHER_LOCAL_MIRROR", ""))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    world = Path(args.world_root)
    man = discover_frozen_training_manifest(world)
    try:
        assert_staging_manifest_complete(man)
    except FileNotFoundError as exc:
        write_staging_manifest(man, args.out_manifest)
        raise SystemExit(f"refusing incomplete Stage-A staging manifest: {exc}") from exc
    out = write_staging_manifest(man, args.out_manifest)
    print(f"manifest={out} files={man.total_files} bytes={man.total_bytes}")

    if args.dry_run:
        return

    payload = man.to_dict()
    if args.local_mirror:
        dest = Path(args.local_mirror)
        stats = _upload_local_mirror(world, dest, payload)
        print(json.dumps({"backend": "local_mirror", **stats}))
        return

    if not args.gcs_uri:
        print("No TRENCHER_GCS_URI / --gcs-uri; wrote manifest only.")
        return

    try:
        from google.cloud import storage  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "google-cloud-storage not installed; set --local-mirror or install the extra"
        ) from exc

    # gs://bucket/prefix
    uri = args.gcs_uri.replace("gs://", "")
    bucket_name, _, prefix = uri.partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    copied = skipped = 0
    for art in payload["artifacts"]:
        src = world / art["path"]
        if not src.exists() or not src.is_file():
            continue
        blob_path = f"{prefix.rstrip('/')}/{art['path']}" if prefix else art["path"]
        blob = bucket.blob(blob_path)
        if blob.exists() and blob.size == src.stat().st_size:
            skipped += 1
            continue
        blob.upload_from_filename(str(src))
        copied += 1
    print(json.dumps({"backend": "gcs", "copied": copied, "skipped": skipped}))


if __name__ == "__main__":
    main()
