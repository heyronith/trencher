"""Artifact hashing, SHA256SUMS manifests, and evaluation-runtime provenance."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from trencher.data.checksums import parse_sha256sums, sha256_file
from trencher.data.world_model.io import utc_now, write_json
from trencher.world_model.evaluation.modes import provenance_q85_flag


class DirtyRepositoryError(RuntimeError):
    """Raised when an official Step-17 run requires a clean Git tree."""


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


def discover_evaluation_runtime(
    *,
    repo_root: Path | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Discover the Git commit actually executing Step-17 evaluation.

    Official scientific runs fail closed on a dirty tree unless ``allow_dirty``
    is explicitly enabled (development/test only; never default).
    """
    # Explicit env override for unit tests only (must be clearly labeled).
    override = os.environ.get("TRENCHER_STEP17_EVAL_RUNTIME_SHA_OVERRIDE")
    dirty_override = os.environ.get("TRENCHER_STEP17_EVAL_RUNTIME_DIRTY_OVERRIDE")
    if override:
        dirty = dirty_override in {"1", "true", "True"}
        if dirty and not allow_dirty:
            raise DirtyRepositoryError(
                "Step-17 official run refuses dirty evaluation runtime "
                "(TRENCHER_STEP17_EVAL_RUNTIME_DIRTY_OVERRIDE=1)"
            )
        return {
            "git_sha": str(override).strip(),
            "git_dirty": dirty,
            "discovery": "env_override_test_only",
            "allow_dirty": allow_dirty,
        }

    root = Path(repo_root) if repo_root is not None else Path.cwd()
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"unable to discover evaluation runtime Git SHA under {root}: {exc}"
        ) from exc
    dirty = bool(porcelain.strip())
    if dirty and not allow_dirty:
        raise DirtyRepositoryError(
            "Step-17 official run requires a clean Git working tree; "
            "repository is dirty. Re-run with an explicit development "
            "--allow-dirty override only for non-scientific tests."
        )
    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "discovery": "git_rev_parse_head",
        "allow_dirty": allow_dirty,
        "repo_root": str(root),
    }


def evaluation_runtime_fields(runtime: dict[str, Any]) -> dict[str, Any]:
    """Flatten + nest evaluation-runtime provenance for artifacts."""
    return {
        "evaluation_runtime_git_sha": runtime["git_sha"],
        "evaluation_runtime_git_dirty": bool(runtime["git_dirty"]),
        "evaluation_runtime": {
            "git_sha": runtime["git_sha"],
            "git_dirty": bool(runtime["git_dirty"]),
            "discovery": runtime.get("discovery"),
            "allow_dirty": bool(runtime.get("allow_dirty", False)),
        },
    }


def attach_provenance(
    payload: dict[str, Any],
    *,
    config_hash: str,
    source_checkpoint_sha256: str | None = None,
    modality_config: dict[str, Any] | None = None,
    evaluation_runtime: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    allow_dirty: bool = False,
    extra: dict[str, Any] | None = None,
    # Deprecated: caller-supplied SHA is ignored unless tests pass override env.
    git_sha: str | None = None,
) -> dict[str, Any]:
    """Attach q85 + evaluation-runtime provenance (never source-training SHAs)."""
    del git_sha  # historical callers; runtime is always discovered
    runtime = evaluation_runtime or discover_evaluation_runtime(
        repo_root=repo_root, allow_dirty=allow_dirty
    )
    out = dict(payload)
    out.update(provenance_q85_flag(accessed=False))
    out.update(evaluation_runtime_fields(runtime))
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
