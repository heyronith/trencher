"""Hugging Face dataset downloads."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from trencher.data.checksums import sha256_file, verify_checksum
from trencher.data.download import DownloadError, ensure_free_disk

logger = logging.getLogger(__name__)

SOLARCHIVE_LIGHT_NAMES = {
    "readme.md",
    "license",
    "license.md",
    "citation.cff",
    "dataset_infos.json",
    ".gitattributes",
    "index.json",
}


def list_hf_files(repo_id: str, *, repo_type: str = "dataset") -> list[dict[str, Any]]:
    api = HfApi()
    files: list[dict[str, Any]] = []
    for entry in api.list_repo_tree(repo_id, repo_type=repo_type, recursive=True):
        path = getattr(entry, "path", None)
        if path is None:
            continue
        # Skip folders.
        if entry.__class__.__name__ == "RepoFolder" or not hasattr(entry, "size"):
            continue
        size = int(getattr(entry, "size", 0) or 0)
        lfs = getattr(entry, "lfs", None)
        sha = None
        if lfs is not None:
            sha = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
            if isinstance(lfs, dict):
                sha = lfs.get("sha256") or lfs.get("oid")
                size = int(lfs.get("size") or size)
            else:
                size = int(getattr(lfs, "size", size) or size)
        files.append({"path": str(path), "size": size, "sha256": sha})
    return files


def is_solarchive_light_file(path: str, size: int) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    lower = path.lower()
    if size >= 50_000_000:
        return False
    if name in SOLARCHIVE_LIGHT_NAMES:
        return True
    if lower.endswith((".md", ".json", ".yml", ".yaml", ".txt", ".cff")) and (
        "schema" in lower or "manifest" in lower or "index" in lower or "/" not in path
    ):
        return True
    if name == "index.json":
        return True
    return "schemas/" in lower and lower.endswith(".json")


def select_solarchive_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [f for f in files if is_solarchive_light_file(f["path"], int(f.get("size") or 0))]
    if not any(f["path"] == "README.md" for f in selected):
        raise DownloadError("solarchive README.md not found in repository listing")
    if not any(f["path"].startswith("schemas/") for f in selected):
        raise DownloadError("solarchive schemas/ not found in repository listing")
    return selected


@retry(
    retry=retry_if_exception_type((OSError, DownloadError, ConnectionError, TimeoutError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _hf_download_one(repo_id: str, repo_type: str, filename: str, cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            local_dir=str(cache_dir),
            force_download=False,
        )
    )


def download_hf_dataset(
    *,
    source_id: str,
    repo_id: str,
    dest_dir: Path,
    repo_type: str = "dataset",
    files: list[dict[str, Any]] | None = None,
    force: bool = False,
    disk_check: bool = False,
    light_only: bool = False,
) -> list[dict[str, Any]]:
    """Download HF files into ``dest_dir`` preserving relative filenames."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    listing = files if files is not None else list_hf_files(repo_id, repo_type=repo_type)
    if light_only:
        listing = select_solarchive_files(listing)

    expected_total = sum(int(f.get("size") or 0) for f in listing)
    if disk_check:
        # Require payload + 15% + 2 GiB headroom.
        required = int(expected_total * 1.15) + 2 * 1024**3
        ensure_free_disk(dest_dir, required, label=source_id)

    results: list[dict[str, Any]] = []
    for item in listing:
        rel = item["path"]
        if rel.endswith("/") or rel in {".", ".."}:
            continue
        target = dest_dir / rel
        expected_size = int(item.get("size") or 0) or None
        publisher_sha = item.get("sha256")

        if target.exists() and not force:
            size = target.stat().st_size
            if expected_size is not None and size != expected_size:
                logger.warning("Size mismatch for %s; re-downloading", rel)
                target.unlink()
            elif publisher_sha:
                try:
                    actual = verify_checksum(target, "sha256", str(publisher_sha))
                    results.append(
                        {
                            "path": rel,
                            "url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel}",
                            "size": size,
                            "sha256": actual,
                            "publisher_checksum": {"alg": "sha256", "value": publisher_sha},
                            "verified": True,
                            "status": "skipped_verified",
                        }
                    )
                    continue
                except ValueError:
                    logger.warning("Checksum mismatch for %s; re-downloading", rel)
                    target.unlink()
            else:
                results.append(
                    {
                        "path": rel,
                        "url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel}",
                        "size": size,
                        "sha256": sha256_file(target),
                        "publisher_checksum": None,
                        "verified": False,
                        "status": "skipped_existing",
                    }
                )
                continue

        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded = _hf_download_one(repo_id, repo_type, rel, dest_dir)
        # hf_hub_download with local_dir should place file at dest_dir/rel.
        if downloaded.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded, target)

        size = target.stat().st_size
        if expected_size is not None and size != expected_size:
            target.unlink(missing_ok=True)
            raise DownloadError(
                f"Incomplete HF download {rel}: expected {expected_size} bytes, got {size}"
            )

        sha = sha256_file(target)
        verified = False
        publisher = None
        if publisher_sha:
            verify_checksum(target, "sha256", str(publisher_sha))
            verified = True
            publisher = {"alg": "sha256", "value": str(publisher_sha)}

        results.append(
            {
                "path": rel,
                "url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel}",
                "size": size,
                "sha256": sha,
                "publisher_checksum": publisher,
                "verified": verified,
                "status": "downloaded",
            }
        )
        logger.info("HF %s: %s (%s bytes)", source_id, rel, size)

    return results
