"""Training-artifact staging manifests (V2) over frozen V4 dependency trees."""

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

# Relative paths under world_root / split_root that Stage A/B/C may need.
FROZEN_STORE_GLOBS: tuple[tuple[str, str, bool], ...] = (
    # (semantic, relative_glob_from_world_or_split, required)
    ("cutoffs", "splits/split_v1/cutoffs.json", True),
    ("walkforward", "splits/split_v1/walkforward_folds.json", True),
    ("assignments_broad", "splits/split_v1/observation_assignments_broad/part-*.parquet", True),
    ("assignments_rich", "splits/split_v1/observation_assignments_rich.parquet", False),
    ("training_cache_manifest", "splits/split_v1/training_cache_v1/manifest.json", False),
    ("training_cache_rich", "splits/split_v1/training_cache_v1/rich/joined.parquet", False),
    ("sequence_refs", "sequences/sequence_dataset_v2/observation_sequence_refs_v2.parquet", True),
    ("sequence_events", "sequences/sequence_dataset_v2/sequence_events_v2.parquet", True),
    ("sequence_curve", "sequences/sequence_dataset_v2/sequence_curve_v2.parquet", True),
    ("sequence_micro", "sequences/sequence_dataset_v2/sequence_microstructure_v2.parquet", False),
    ("sequence_ownership", "sequences/sequence_dataset_v2/sequence_ownership_v2.parquet", False),
    ("sequence_graph", "sequences/sequence_dataset_v2/sequence_graph_token_v2.parquet", False),
    ("sequence_wallet_prior",
     "sequences/sequence_dataset_v2/sequence_graph_wallet_prior_v2.parquet", False),
    ("targets_rich", "targets/target_dataset_v3/observation_targets_v3.parquet", False),
    ("broad_targets", "broad/broad_targets_v3/observation_targets/part-*.parquet", True),
    ("broad_seq_events", "broad/broad_sequence_events_v1/**/*.parquet", True),
    ("broad_seq_curve", "broad/broad_sequence_curve_v1/**/*.parquet", True),
    ("loader_contract", "manifests_placeholder", False),
)


@dataclass
class ArtifactEntry:
    key: str
    path: str
    sha256: str
    bytes: int
    required: bool = True
    semantic_version: str = ""
    population: str = ""


@dataclass
class StagingManifest:
    version: str = "WORLD_MODEL_STAGING_MANIFEST_V2"
    root: str = ""
    artifacts: list[ArtifactEntry] = field(default_factory=list)
    total_bytes: int = 0
    total_files: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "root": self.root,
            "total_bytes": self.total_bytes,
            "total_files": self.total_files,
            "artifacts": [
                {
                    "key": a.key,
                    "path": a.path,
                    "sha256": a.sha256,
                    "bytes": a.bytes,
                    "required": a.required,
                    "semantic_version": a.semantic_version,
                    "population": a.population,
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
        entries.append(
            ArtifactEntry(
                key=key,
                path=str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                sha256=digest,
                bytes=int(path.stat().st_size),
                required=key in required,
            )
        )
    if missing_required:
        raise FileNotFoundError(f"missing required staging artifacts: {missing_required}")
    total = sum(e.bytes for e in entries)
    return StagingManifest(
        root=str(root), artifacts=entries, total_bytes=total, total_files=len(entries)
    )


def discover_frozen_training_manifest(
    world_root: Path,
    *,
    max_files_per_glob: int = 50_000,
) -> StagingManifest:
    """Enumerate minimal frozen stores required for deep training (shard-aware)."""
    world_root = Path(world_root).resolve()
    entries: list[ArtifactEntry] = []
    for semantic, rel, required in FROZEN_STORE_GLOBS:
        if "manifests_placeholder" in rel:
            continue
        pattern = rel
        matches = sorted(world_root.glob(pattern))
        if not matches and required:
            # Still record missing required as empty with required flag for fail-closed verify
            entries.append(
                ArtifactEntry(
                    key=f"{semantic}:MISSING",
                    path=rel,
                    sha256="",
                    bytes=0,
                    required=True,
                    semantic_version=semantic,
                )
            )
            continue
        for i, path in enumerate(matches):
            if i >= max_files_per_glob:
                break
            if not path.is_file():
                continue
            # Skip hashing very large files in editor smoke — record size; hash on sync
            size = int(path.stat().st_size)
            digest = ""
            digest = sha256_file(path) if size < 64 * 1024 * 1024 else f"size_deferred:{size}"
            entries.append(
                ArtifactEntry(
                    key=f"{semantic}:{path.name}",
                    path=str(path.relative_to(world_root)),
                    sha256=digest,
                    bytes=size,
                    required=required,
                    semantic_version=semantic,
                )
            )
    total = sum(e.bytes for e in entries)
    return StagingManifest(
        version="WORLD_MODEL_STAGING_MANIFEST_V2",
        root=str(world_root),
        artifacts=entries,
        total_bytes=total,
        total_files=len(entries),
    )


def write_staging_manifest(manifest: StagingManifest, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def verify_staging_manifest(path: str | Path, *, fail_closed: bool = True) -> dict[str, Any]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = Path(raw.get("root") or path.parent)
    ok = True
    errors: list[str] = []
    verified = 0
    for art in raw.get("artifacts") or []:
        p = root / art["path"]
        if not p.exists():
            if art.get("required", True):
                ok = False
                errors.append(f"missing required {art['path']}")
            continue
        if str(art.get("sha256") or "").startswith("size_deferred:"):
            verified += 1
            continue
        if art.get("sha256"):
            got = sha256_file(p)
            if got != art["sha256"]:
                ok = False
                errors.append(f"hash mismatch {art['path']}")
            else:
                verified += 1
    if fail_closed and not ok:
        return {"ok": False, "verified": verified, "errors": errors}
    return {"ok": ok, "verified": verified, "errors": errors}
