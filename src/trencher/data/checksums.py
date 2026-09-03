"""Checksum helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksum_field(value: str) -> tuple[str, str]:
    """Parse Zenodo-style ``md5:<hex>`` or bare hex (assumed sha256 if len 64)."""
    text = value.strip().lower()
    if ":" in text:
        alg, digest = text.split(":", 1)
        return alg, digest
    if len(text) == 32:
        return "md5", text
    if len(text) == 64:
        return "sha256", text
    raise ValueError(f"Unrecognized checksum format: {value!r}")


def verify_checksum(path: Path, algorithm: str, expected: str) -> str:
    alg = algorithm.lower().removeprefix("sha-")
    expected_norm = expected.lower().strip()
    if alg in {"sha256", "sha_256"}:
        actual = sha256_file(path)
    elif alg == "md5":
        actual = md5_file(path)
    else:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
    if actual.lower() != expected_norm:
        raise ValueError(
            f"Checksum mismatch for {path}: expected {alg}:{expected_norm}, got {actual}"
        )
    return actual


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse GNU ``SHA256SUMS`` content into ``{filename: hex}``."""
    mapping: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        digest = parts[0].lower()
        name = parts[1].lstrip("*")
        mapping[name] = digest
    return mapping
