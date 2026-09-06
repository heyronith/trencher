#!/usr/bin/env python3
"""Bulk-upload frozen sequence_dataset_v2 stores to Modal (no broad re-upload)."""

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
    "sequence_graph_wallet_prior_v2.parquet",
    "sequence_graph_wallet_prior_v2_entity_index.parquet",
    "observation_sequence_refs_v2.parquet",
    "MODEL_INPUT_CONTRACT_V2.json",
    "sequence_dataset_manifest.json",
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
    _log(f"BATCH UPLOAD sequence_dataset_v2 → {VOL_NAME}:{remote} force=True")
    with vol.batch_upload(force=True) as batch:
        for name in REQUIRED:
            local = SEQ / name
            _log(f"  put_file {local} ({local.stat().st_size / 1e6:.1f} MB)")
            batch.put_file(str(local), f"{remote}/{name}")
    _log(f"SEQUENCE UPLOAD done in {time.time() - t0:.1f}s")

    # Verify listing
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
