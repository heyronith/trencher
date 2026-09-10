"""Artifact hashing, SHA256SUMS manifests, and lineage helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from trencher.data.checksums import parse_sha256sums, sha256_file
from trencher.data.world_model.io import utc_now, write_json
from trencher.world_model.evaluation.modes import provenance_q85_flag


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(raw.encode("utf-8"))


def write_sha256sums(directory: Path, files: list[Path]) -> Path:
    """Write GNU SHA256SUMS for ``files`` relative to ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for path in files:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        rel = path.relative_to(directory) if path.is_relative_to(directory) else path.name
        lines.append(f"{sha256_file(path)}  {rel.as_posix()}")
    out = directory / "SHA256SUMS"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def verify_sha256sums(directory: Path, *, sums_name: str = "SHA256SUMS") -> dict[str, str]:
    directory = Path(directory)
    text = (directory / sums_name).read_text(encoding="utf-8")
    mapping = parse_sha256sums(text)
    for name, digest in mapping.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"manifest entry missing: {path}")
        got = sha256_file(path)
        if got != digest:
            raise ValueError(f"sha256 mismatch for {name}: got={got} want={digest}")
    return mapping


def attach_provenance(
    payload: dict[str, Any],
    *,
    git_sha: str,
    config_hash: str,
    source_checkpoint_sha256: str | None = None,
    modality_config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(payload)
    out.update(provenance_q85_flag(accessed=False))
    out["git_sha"] = git_sha
    out["config_hash"] = config_hash
    out["written_at"] = utc_now()
    if source_checkpoint_sha256 is not None:
        out["source_checkpoint_sha256"] = source_checkpoint_sha256
    if modality_config is not None:
        out["modality_config"] = modality_config
    if extra:
        out.update(extra)
    return out


def write_json_with_sidecar(path: Path, payload: dict[str, Any]) -> dict[str, str]:
    path = Path(path)
    write_json(path, payload)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return {"path": str(path), "sha256": digest}
