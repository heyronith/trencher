"""Paths and constants for external historical datasets."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_EXTERNAL_ROOT = Path("data/external")
DEFAULT_MANIFEST_ROOT = Path("manifests/external")

SOURCES = {
    "pumpfun_memecoin_corpus": {
        "kind": "huggingface",
        "repo_id": "Slinky21/Pumpfun_Memecoin_Corpus",
        "repo_type": "dataset",
        "url": "https://huggingface.co/datasets/Slinky21/Pumpfun_Memecoin_Corpus",
        "license": "mit",
        "dir_name": "pumpfun_memecoin_corpus",
        "requires_disk_check": True,
    },
    "red_pump_2026": {
        "kind": "zenodo",
        "record_id": "21383616",
        "url": "https://zenodo.org/records/21383616",
        "doi": "10.5281/zenodo.21383616",
        "license": "cc-by-4.0",
        "dir_name": "red_pump_2026",
        "requires_disk_check": False,
    },
    "red_cohort_2026": {
        "kind": "zenodo",
        "record_id": "21765387",
        "url": "https://zenodo.org/records/21765387",
        "doi": "10.5281/zenodo.21765387",
        "license": "cc-by-4.0",
        "dir_name": "red_cohort_2026",
        "requires_disk_check": False,
        "extract_archives": True,
    },
    "solarchive": {
        "kind": "huggingface_light",
        "repo_id": "solarchive/solarchive",
        "repo_type": "dataset",
        "url": "https://huggingface.co/datasets/solarchive/solarchive",
        "license": "unknown",
        "dir_name": "solarchive",
        "requires_disk_check": False,
    },
}

JOIN_KEY_CANDIDATES = (
    "mint",
    "token_mint",
    "token_mint_address",
    "base_mint",
    "wallet",
    "wallet_address",
    "trader",
    "trader_wallet",
    "creator",
    "creator_wallet",
    "user",
    "cohort_id",
    "signature",
    "tx_id",
)

TIMESTAMP_CANDIDATES = (
    "block_time",
    "timestamp",
    "ts",
    "time",
    "created_at",
    "launch_time",
    "event_time",
    "first_seen",
    "last_seen",
    "graduated_at",
    "migrated_at",
)


def resolve_external_root(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get("TRENCHER_EXTERNAL_ROOT", DEFAULT_EXTERNAL_ROOT))


def resolve_manifest_root(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get("TRENCHER_MANIFEST_ROOT", DEFAULT_MANIFEST_ROOT))


def source_data_dir(external_root: Path, source_id: str) -> Path:
    return external_root / SOURCES[source_id]["dir_name"]


def source_manifest_dir(manifest_root: Path, source_id: str) -> Path:
    return manifest_root / source_id
