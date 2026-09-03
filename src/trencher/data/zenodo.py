"""Zenodo record downloads."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any

import requests

from trencher.data.checksums import (
    parse_checksum_field,
    parse_sha256sums,
    sha256_file,
    verify_checksum,
)
from trencher.data.download import DownloadError, download_http

logger = logging.getLogger(__name__)


def fetch_zenodo_record(
    record_id: str,
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    http = session or requests.Session()
    url = f"https://zenodo.org/api/records/{record_id}"
    response = http.get(url, timeout=60)
    if response.status_code != 200:
        raise DownloadError(
            f"Zenodo record {record_id} fetch failed: HTTP {response.status_code}"
        )
    return response.json()


def _file_download_url(record: dict[str, Any], file_meta: dict[str, Any]) -> str:
    links = file_meta.get("links") or {}
    if "content" in links:
        return str(links["content"])
    key = file_meta["key"]
    return f"https://zenodo.org/api/records/{record['id']}/files/{key}/content"


def download_zenodo_record(
    *,
    source_id: str,
    record_id: str,
    dest_dir: Path,
    force: bool = False,
    extract_archives: bool = False,
    session: requests.Session | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Download all files for a Zenodo record; optionally extract zip archives."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    http = session or requests.Session()
    record = fetch_zenodo_record(record_id, session=http)
    files = list(record.get("files") or [])
    if not files:
        raise DownloadError(f"Zenodo record {record_id} has no files")

    # First pass: download everything.
    downloaded: list[dict[str, Any]] = []
    local_paths: dict[str, Path] = {}
    for file_meta in files:
        key = str(file_meta["key"])
        size = int(file_meta.get("size") or 0)
        checksum_raw = file_meta.get("checksum")
        checksum = parse_checksum_field(str(checksum_raw)) if checksum_raw else None
        url = _file_download_url(record, file_meta)
        dest = dest_dir / key
        if force and dest.exists():
            dest.unlink()
        result = download_http(
            url,
            dest,
            expected_size=size or None,
            checksum=checksum,
            session=http,
        )
        path = Path(result["path"])  # type: ignore[arg-type]
        local_paths[key] = path
        sha = result.get("sha256") or sha256_file(path)
        downloaded.append(
            {
                "path": key,
                "url": url,
                "size": int(result["size"]),  # type: ignore[arg-type]
                "sha256": sha,
                "publisher_checksum": (
                    {"alg": checksum[0], "value": checksum[1]} if checksum else None
                ),
                "verified": bool(result.get("checksum_ok")),
                "status": result["status"],
            }
        )
        logger.info("Zenodo %s: %s (%s)", source_id, key, result["status"])

    # Second pass: if SHA256SUMS present, verify all listed files.
    sums_path = local_paths.get("SHA256SUMS")
    if sums_path and sums_path.exists():
        mapping = parse_sha256sums(sums_path.read_text())
        for name, digest in mapping.items():
            target = dest_dir / name
            if not target.exists():
                raise DownloadError(f"SHA256SUMS lists missing file: {name}")
            verify_checksum(target, "sha256", digest)
            for item in downloaded:
                if item["path"] == name:
                    item["sha256"] = digest
                    item["verified"] = True
                    item["publisher_checksum"] = {"alg": "sha256", "value": digest}
                    break
            else:
                downloaded.append(
                    {
                        "path": name,
                        "url": None,
                        "size": target.stat().st_size,
                        "sha256": digest,
                        "publisher_checksum": {"alg": "sha256", "value": digest},
                        "verified": True,
                        "status": "verified_via_sha256sums",
                    }
                )

    extracted_files: list[dict[str, Any]] = []
    if extract_archives:
        for key, path in local_paths.items():
            if not key.lower().endswith(".zip"):
                continue
            extract_root = dest_dir / "extracted"
            extract_root.mkdir(parents=True, exist_ok=True)
            marker = extract_root / f".extracted_{path.stem}"
            if marker.exists() and not force:
                logger.info("Zenodo %s: archive already extracted (%s)", source_id, key)
            else:
                with zipfile.ZipFile(path) as zf:
                    # Fail loudly on bad zip.
                    bad = zf.testzip()
                    if bad is not None:
                        raise DownloadError(f"Corrupt zip member {bad} in {path}")
                    zf.extractall(extract_root)
                marker.write_text(f"source={key}\n", encoding="utf-8")
            for extracted in sorted(extract_root.rglob("*")):
                if not extracted.is_file() or extracted.name.startswith(".extracted_"):
                    continue
                rel = extracted.relative_to(dest_dir).as_posix()
                extracted_files.append(
                    {
                        "path": rel,
                        "url": None,
                        "size": extracted.stat().st_size,
                        "sha256": sha256_file(extracted),
                        "publisher_checksum": None,
                        "verified": False,
                        "status": "extracted",
                    }
                )

    meta = {
        "title": (record.get("metadata") or {}).get("title"),
        "version": (record.get("metadata") or {}).get("version"),
        "doi": (record.get("metadata") or {}).get("doi") or record.get("doi"),
        "license": ((record.get("metadata") or {}).get("license") or {}).get("id"),
        "record_id": str(record.get("id") or record_id),
        "url": f"https://zenodo.org/records/{record_id}",
    }
    return meta, downloaded + extracted_files
