"""Required-artifact staging manifests with sha256 digests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_ARTIFACT_KEYS: tuple[str, ...] = (
    "config",
    "training_contract",
    "candidate_manifest",
    "freeze",
    "split_audit",
    "loader_audit",
)


@dataclass
class ArtifactEntry:
    key: str
    path: str
    sha256: str
    bytes: int
    required: bool = True


@dataclass
class StagingManifest:
    version: str = "WORLD_MODEL_STAGING_MANIFEST_V1"
    root: str = ""
    artifacts: list[ArtifactEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "root": self.root,
            "artifacts": [
                {
                    "key": a.key,
                    "path": a.path,
                    "sha256": a.sha256,
                    "bytes": a.bytes,
                    "required": a.required,
                }
                for a in self.artifacts
            ],
        }


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_staging_manifest(
    root: str | Path,
    artifacts: dict[str, str | Path],
    *,
    required_keys: Iterable[str] = REQUIRED_ARTIFACT_KEYS,
) -> StagingManifest:
    """Build a manifest of relative paths under ``root`` with sha256 digests."""
    root = Path(root).resolve()
    required = set(required_keys)
    entries: list[ArtifactEntry] = []
    missing_required: list[str] = []
    for key, rel in artifacts.items():
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            if key in required:
                missing_required.append(key)
            continue
        digest = sha256_file(path)
        try:
            rel_s = str(path.resolve().relative_to(root))
        except ValueError:
            rel_s = str(path)
        entries.append(
            ArtifactEntry(
                key=key,
                path=rel_s,
                sha256=digest,
                bytes=path.stat().st_size,
                required=key in required,
            )
        )
    if missing_required:
        raise FileNotFoundError(f"missing required staging artifacts: {missing_required}")
    for key in required:
        if not any(e.key == key for e in entries):
            raise FileNotFoundError(f"required artifact key not staged: {key}")
    return StagingManifest(root=str(root), artifacts=entries)


def write_staging_manifest(manifest: StagingManifest, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def verify_staging_manifest(
    manifest_path: str | Path,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify all artifacts exist and sha256 matches; raise on mismatch."""
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = Path(root) if root is not None else Path(data.get("root") or manifest_path.parent)
    errors: list[str] = []
    verified: list[dict[str, Any]] = []
    for art in data.get("artifacts", []):
        path = base / art["path"]
        if not path.exists():
            errors.append(f"missing: {art['key']} -> {path}")
            continue
        actual = sha256_file(path)
        if actual != art["sha256"]:
            errors.append(
                f"sha256 mismatch for {art['key']}: expected {art['sha256']}, got {actual}"
            )
            continue
        verified.append({"key": art["key"], "path": str(path), "sha256": actual})
    if errors:
        raise ValueError("staging manifest verification failed:\n" + "\n".join(errors))
    return {
        "ok": True,
        "verified_n": len(verified),
        "verified": verified,
        "manifest": str(manifest_path),
    }
