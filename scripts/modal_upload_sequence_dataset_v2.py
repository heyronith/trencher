#!/usr/bin/env python3
"""Bulk-upload frozen sequence_dataset_v2 stores to Modal (no broad re-upload).

Uploads one file per ``batch_upload`` context so large blobs (wallet prior)
do not hit expired multipart tokens from long multi-file batches.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

VOL_NAME = "trencher-world-model-v1"
WORLD = Path("data/external/world_model")
SEQ = WORLD / "sequences" / "sequence_dataset_v2"

REQUIRED = (
    "sequence_microstructure_v2.parquet",
    "sequence_microstructure_v2_entity_index.parquet",
    "sequence_ownership_v2.parquet",
    "sequence_ownership_v2_entity_index.parquet",
    "sequence_graph_token_v2.parquet",
    "sequence_graph_token_v2_entity_index.parquet",
    "sequence_graph_wallet_prior_v2_entity_index.parquet",
    "observation_sequence_refs_v2.parquet",
    "MODEL_INPUT_CONTRACT_V2.json",
    "sequence_dataset_manifest.json",
    # largest last
    "sequence_graph_wallet_prior_v2.parquet",
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    if not SEQ.is_dir():
        print(f"ERROR: missing {SEQ}", file=sys.stderr)
        return 1
    missing = [n for n in REQUIRED if not (SEQ / n).exists()]
    if missing:
        print(f"ERROR: missing sequence files: {missing}", file=sys.stderr)
        return 1

    vol = modal.Volume.from_name(VOL_NAME)
    remote = "/sequences/sequence_dataset_v2"
    t0 = time.time()
    _log(f"UPLOAD sequence_dataset_v2 → {VOL_NAME}:{remote} (one file / batch, force=True)")

    for name in REQUIRED:
        local = SEQ / name
        size_mb = local.stat().st_size / 1e6
        _log(f"  put_file {name} ({size_mb:.1f} MB)…")
        t1 = time.time()
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                with vol.batch_upload(force=True) as batch:
                    batch.put_file(str(local), f"{remote}/{name}")
                _log(f"    OK in {time.time() - t1:.1f}s")
                break
            except Exception as exc:  # noqa: BLE001
                _log(f"    attempt {attempt}/{retries} failed: {type(exc).__name__}: {exc}")
                if attempt >= retries:
                    print(f"ERROR: failed to upload {name}", file=sys.stderr)
                    return 1
                time.sleep(5 * attempt)

    _log(f"SEQUENCE UPLOAD done in {time.time() - t0:.1f}s")

    entries = list(vol.listdir(remote, recursive=False))
    names = {Path(getattr(e, "path", str(e))).name for e in entries}
    still = [n for n in REQUIRED if n not in names]
    if still:
        print(f"ERROR: remote missing after upload: {still}", file=sys.stderr)
        return 2
    _log(f"VERIFY OK: {len(REQUIRED)} sequence files present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
