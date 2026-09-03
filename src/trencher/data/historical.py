"""Orchestrate historical external dataset downloads and local registration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trencher.data.download import DownloadError
from trencher.data.hf import download_hf_dataset, list_hf_files
from trencher.data.inventory import build_source_inventory
from trencher.data.paths import (
    SOURCES,
    resolve_external_root,
    resolve_manifest_root,
    source_data_dir,
    source_manifest_dir,
)
from trencher.data.zenodo import download_zenodo_record

logger = logging.getLogger(__name__)


@dataclass
class HistoricalDownloadResult:
    ok: bool
    external_root: Path
    manifest_root: Path
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    inventory: dict[str, Any] = field(default_factory=dict)
    registry: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _provenance(
    *,
    source_id: str,
    spec: dict[str, Any],
    meta: dict[str, Any],
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_name": meta.get("title") or source_id,
        "kind": spec["kind"],
        "url": meta.get("url") or spec.get("url"),
        "repo_id": spec.get("repo_id"),
        "record_id": spec.get("record_id") or meta.get("record_id"),
        "doi": meta.get("doi") or spec.get("doi"),
        "version": meta.get("version"),
        "license": meta.get("license") or spec.get("license"),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "file_count": len(files),
        "total_bytes": sum(int(f.get("size") or 0) for f in files),
        "files": files,
    }


def _download_source(
    source_id: str,
    *,
    external_root: Path,
    force: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    if source_id not in SOURCES:
        raise KeyError(f"Unknown source {source_id!r}")
    spec = SOURCES[source_id]
    data_dir = source_data_dir(external_root, source_id)
    data_dir.mkdir(parents=True, exist_ok=True)

    if spec["kind"] == "huggingface":
        files = download_hf_dataset(
            source_id=source_id,
            repo_id=str(spec["repo_id"]),
            dest_dir=data_dir,
            repo_type=str(spec.get("repo_type", "dataset")),
            force=force,
            disk_check=bool(spec.get("requires_disk_check")),
        )
        # Capture revision from listing if possible.
        listing = list_hf_files(str(spec["repo_id"]))
        meta = {
            "title": "Pumpfun Memecoin Corpus",
            "url": spec["url"],
            "version": None,
            "license": spec.get("license"),
            "listed_files": len(listing),
        }
        return meta, files, data_dir

    if spec["kind"] == "huggingface_light":
        files = download_hf_dataset(
            source_id=source_id,
            repo_id=str(spec["repo_id"]),
            dest_dir=data_dir,
            repo_type=str(spec.get("repo_type", "dataset")),
            force=force,
            disk_check=False,
            light_only=True,
        )
        meta = {
            "title": "Solarchive metadata/index cache",
            "url": spec["url"],
            "version": None,
            "license": spec.get("license"),
            "note": "Downloaded README/schemas/indexes only; full chain archive excluded.",
        }
        return meta, files, data_dir

    if spec["kind"] == "zenodo":
        meta, files = download_zenodo_record(
            source_id=source_id,
            record_id=str(spec["record_id"]),
            dest_dir=data_dir,
            force=force,
            extract_archives=bool(spec.get("extract_archives")),
        )
        return meta, files, data_dir

    raise DownloadError(f"Unsupported source kind for {source_id}: {spec['kind']}")


def download_historical(
    *,
    sources: list[str] | None = None,
    external_root: Path | None = None,
    manifest_root: Path | None = None,
    force: bool = False,
) -> HistoricalDownloadResult:
    """Download, verify, inventory, and register finalized public historical datasets."""
    root = resolve_external_root(external_root)
    manifests = resolve_manifest_root(manifest_root)
    root.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    selected = list(sources) if sources else list(SOURCES)
    result = HistoricalDownloadResult(ok=True, external_root=root, manifest_root=manifests)
    inventory_sources: dict[str, Any] = {}

    for source_id in selected:
        try:
            logger.info("Downloading historical source %s", source_id)
            meta, files, data_dir = _download_source(
                source_id, external_root=root, force=force
            )
            # Fail loudly if any file is missing/unverified when publisher hash exists.
            for item in files:
                path = data_dir / str(item["path"])
                if item.get("status") == "extracted":
                    if not path.exists():
                        raise DownloadError(f"Extracted file missing: {path}")
                    continue
                if not path.exists():
                    raise DownloadError(f"Downloaded file missing: {path}")
                if item.get("publisher_checksum") and not item.get("verified"):
                    raise DownloadError(
                        f"Publisher checksum present but unverified for {item['path']}"
                    )

            provenance = _provenance(
                source_id=source_id,
                spec=SOURCES[source_id],
                meta=meta,
                files=files,
            )
            inventory = build_source_inventory(source_id, data_dir, files)
            unopenable = [f["path"] for f in inventory["files"] if not f.get("openable")]
            if unopenable:
                raise DownloadError(
                    f"{source_id}: files failed open/inventory checks: {unopenable}"
                )

            # Local copies under data tree + committed manifest copies.
            _write_json(data_dir / "provenance.json", provenance)
            _write_json(data_dir / "inventory.json", inventory)
            mdir = source_manifest_dir(manifests, source_id)
            _write_json(mdir / "provenance.json", provenance)
            _write_json(mdir / "inventory.json", inventory)

            result.sources[source_id] = {
                "data_dir": str(data_dir),
                "provenance": provenance,
                "inventory_summary": {
                    "file_count": inventory["file_count"],
                    "total_bytes": inventory["total_bytes"],
                    "tables": [
                        {
                            "path": f["path"],
                            "format": f["format"],
                            "rows": f.get("row_count"),
                            "join_keys": f.get("join_keys"),
                        }
                        for f in inventory["files"]
                        if f.get("format")
                        in {"parquet", "csv", "csv.gz", "jsonl", "jsonl.gz", "json"}
                    ],
                },
            }
            inventory_sources[source_id] = inventory
            logger.info(
                "Registered %s: %s files, %.2f GiB",
                source_id,
                inventory["file_count"],
                inventory["total_bytes"] / (1024**3),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed historical source %s", source_id)
            result.ok = False
            result.errors.append(f"{source_id}: {exc}")

    inventory_doc = {
        "generated_at": datetime.now(UTC).isoformat(),
        "external_root": str(root),
        "sources": inventory_sources,
        "total_bytes": sum(int(s.get("total_bytes") or 0) for s in inventory_sources.values()),
        "errors": list(result.errors),
    }
    registry_doc = {
        "generated_at": datetime.now(UTC).isoformat(),
        "external_root": str(root),
        "manifest_root": str(manifests),
        "datasets": [
            {
                "source_id": source_id,
                "url": SOURCES[source_id].get("url"),
                "license": SOURCES[source_id].get("license"),
                "data_dir": str(source_data_dir(root, source_id)),
                "file_count": result.sources.get(source_id, {})
                .get("inventory_summary", {})
                .get("file_count"),
                "total_bytes": result.sources.get(source_id, {})
                .get("inventory_summary", {})
                .get("total_bytes"),
                "status": "registered" if source_id in result.sources else "failed",
            }
            for source_id in selected
        ],
        "ok": result.ok,
        "errors": list(result.errors),
    }
    _write_json(manifests / "inventory.json", inventory_doc)
    _write_json(manifests / "registry.json", registry_doc)
    _write_json(root / "inventory.json", inventory_doc)
    _write_json(root / "registry.json", registry_doc)

    result.inventory = inventory_doc
    result.registry = registry_doc
    return result
