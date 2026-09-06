#!/usr/bin/env python3
"""Bulk-upload broad sequence stores to Modal via Volume.batch_upload.

Streams the three complete local trees (events / creator / refs) with
force=True so partial prior uploads are overwritten harmlessly.
No local tar/copy staging.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

VOL_NAME = "trencher-world-model-v1"
WORLD = Path("data/external/world_model")

BROAD_TREES = (
    ("broad/broad_sequence_events_v1", "broad_sequence_events_v1"),
    ("broad/broad_sequence_creator_v1", "broad_sequence_creator_v1"),
    ("broad/broad_observation_sequence_refs_v1", "broad_observation_sequence_refs_v1"),
)

RICH_FILES = (
    ("graph/edges_wallet_token.parquet", "/graph/edges_wallet_token.parquet"),
    ("graph/edges_cobuy_rich.parquet", "/graph/edges_cobuy_rich.parquet"),
    ("bigquery/token_transfers_rich.parquet", "/bigquery/token_transfers_rich.parquet"),
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _count_local_mps(sub: str) -> int:
    root = WORLD / "broad" / sub
    return sum(1 for p in root.iterdir() if p.is_dir() and (p / "part.parquet").exists())


def verify_remote_mps(vol: modal.Volume, remote_sub: str, expect: int = 256) -> list[str]:
    entries = vol.listdir(f"/broad/{remote_sub}", recursive=False)
    names = []
    for e in entries:
        path = getattr(e, "path", None) or str(e)
        base = Path(path).name
        if base and base not in (".", ".."):
            names.append(base)
    names = sorted(set(names))
    missing = [f"{i:02x}" for i in range(256) if f"{i:02x}" not in names]
    _log(
        f"VERIFY /broad/{remote_sub}: found={len(names)} expect={expect} missing={len(missing)}"
    )
    if missing[:10]:
        _log(f"  missing sample: {missing[:10]}")
    return missing


def main() -> int:
    for _local_rel, sub in BROAD_TREES:
        n = _count_local_mps(sub)
        _log(f"LOCAL {sub}: {n} mint partitions")
        if n < 256:
            print(f"ERROR: expected 256 local partitions for {sub}, got {n}", file=sys.stderr)
            return 1

    vol = modal.Volume.from_name(VOL_NAME)
    t0 = time.time()
    _log(f"BATCH UPLOAD start force=True → {VOL_NAME}")
    with vol.batch_upload(force=True) as batch:
        for _local_rel, sub in BROAD_TREES:
            local = WORLD / "broad" / sub
            remote = f"/broad/{sub}"
            _log(f"  put_directory {local} → {remote}")
            batch.put_directory(str(local), remote)
    _log(f"BATCH UPLOAD broad trees done in {time.time() - t0:.1f}s")

    all_missing: dict[str, list[str]] = {}
    for _local_rel, sub in BROAD_TREES:
        miss = verify_remote_mps(vol, sub, expect=256)
        all_missing[sub] = miss

    if any(all_missing.values()):
        print("ERROR: remote partition verification failed", file=sys.stderr)
        for k, v in all_missing.items():
            if v:
                print(f"  {k}: missing {len(v)}", file=sys.stderr)
        return 2

    _log("RICH deps upload…")
    t1 = time.time()
    with vol.batch_upload(force=True) as batch:
        for local_rel, remote in RICH_FILES:
            local = WORLD / local_rel
            if not local.exists():
                _log(f"  SKIP missing {local}")
                continue
            _log(f"  put_file {local} → {remote}")
            batch.put_file(str(local), remote)
    _log(f"RICH deps done in {time.time() - t1:.1f}s")
    _log("ALL SYNC OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
