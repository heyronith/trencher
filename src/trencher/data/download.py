"""Resumable HTTP download helpers."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from trencher.data.checksums import verify_checksum

logger = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    pass


def free_disk_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def ensure_free_disk(path: Path, required_bytes: int, *, label: str) -> None:
    free = free_disk_bytes(path)
    if free < required_bytes:
        raise DownloadError(
            f"Insufficient free disk for {label}: need "
            f"{required_bytes / (1024**3):.2f} GiB, have {free / (1024**3):.2f} GiB under {path}"
        )


@retry(
    retry=retry_if_exception_type((requests.RequestException, DownloadError, OSError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def download_http(
    url: str,
    dest: Path,
    *,
    expected_size: int | None = None,
    checksum: tuple[str, str] | None = None,
    session: requests.Session | None = None,
    timeout: int = 120,
) -> dict[str, object]:
    """Download ``url`` to ``dest`` with resume support and optional checksum verify."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    http = session or requests.Session()

    if dest.exists() and dest.stat().st_size > 0:
        if expected_size is not None and dest.stat().st_size != expected_size:
            dest.unlink()
        elif checksum is not None:
            try:
                actual = verify_checksum(dest, checksum[0], checksum[1])
                return {
                    "path": dest,
                    "status": "skipped_verified",
                    "size": dest.stat().st_size,
                    "sha256": actual if checksum[0].lower().startswith("sha") else None,
                    "checksum_ok": True,
                }
            except ValueError:
                logger.warning("Existing file failed checksum; re-downloading %s", dest)
                dest.unlink()
        else:
            return {
                "path": dest,
                "status": "skipped_existing",
                "size": dest.stat().st_size,
                "sha256": None,
                "checksum_ok": None,
            }

    existing = partial.stat().st_size if partial.exists() else 0
    headers: dict[str, str] = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    with http.get(url, stream=True, headers=headers, timeout=timeout) as response:
        if response.status_code == 416:
            # Partial already complete according to server.
            if partial.exists():
                partial.replace(dest)
            elif not dest.exists():
                raise DownloadError(f"Server returned 416 for {url} with no usable local file")
        else:
            if existing and response.status_code == 200:
                # Server ignored Range; restart.
                existing = 0
                partial.unlink(missing_ok=True)
            if response.status_code not in {200, 206}:
                raise DownloadError(f"HTTP {response.status_code} downloading {url}")
            mode = "ab" if response.status_code == 206 and existing > 0 else "wb"
            if mode == "wb":
                partial.unlink(missing_ok=True)
            with partial.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            partial.replace(dest)

    size = dest.stat().st_size
    if expected_size is not None and size != expected_size:
        dest.unlink(missing_ok=True)
        raise DownloadError(
            f"Incomplete download for {dest.name}: expected {expected_size} bytes, got {size}"
        )

    sha256 = None
    checksum_ok = None
    if checksum is not None:
        actual = verify_checksum(dest, checksum[0], checksum[1])
        checksum_ok = True
        if checksum[0].lower().startswith("sha"):
            sha256 = actual

    return {
        "path": dest,
        "status": "downloaded",
        "size": size,
        "sha256": sha256,
        "checksum_ok": checksum_ok,
    }
