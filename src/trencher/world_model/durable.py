"""Durable checkpoint / progress sync for disposable Colab runtimes.

Local-mirror backend is used in CI; GCS backend is used in production.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DurableSyncError(RuntimeError):
    """Raised when a durable upload/verification fails (never silent)."""


def _resume_alias(relative: str) -> str | None:
    """Return the durable resume alias advanced by a periodic artifact."""
    normalized = relative.lstrip("/")
    if normalized == "periodic.pt":
        return "latest.pt"
    if normalized == "periodic.pt.meta.json":
        return "latest.pt.meta.json"
    return None


class DurableStore(ABC):
    """Abstraction over durable run artifact storage."""

    @abstractmethod
    def uri_for(self, relative: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def upload_file(self, local_path: Path, relative: str) -> str:
        """Atomically publish ``local_path`` as ``relative``; return durable URI."""
        raise NotImplementedError

    def download_file(self, relative: str, dest: Path) -> Path:
        raise NotImplementedError("download not implemented for this backend")


class LocalMirrorStore(DurableStore):
    """Filesystem mirror used by CI / local-mirror Stage-A dry runs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def uri_for(self, relative: str) -> str:
        return str((self.root / relative).resolve())

    def _publish_copy(self, local_path: Path, relative: str) -> Path:
        dest = self.root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
        try:
            shutil.copy2(local_path, tmp)
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                with contextlib.suppress(OSError):
                    tmp.unlink()
        if not dest.is_file() or dest.stat().st_size != local_path.stat().st_size:
            raise DurableSyncError(f"durable verify failed for {dest}")
        return dest

    def upload_file(self, local_path: Path, relative: str) -> str:
        local_path = Path(local_path)
        if not local_path.is_file():
            raise DurableSyncError(f"missing local checkpoint for durable sync: {local_path}")
        self._publish_copy(local_path, relative)
        alias = _resume_alias(relative)
        if alias is not None:
            self._publish_copy(local_path, alias)
        return self.uri_for(relative)

    def download_file(self, relative: str, dest: Path) -> Path:
        src = self.root / relative
        if not src.is_file():
            raise DurableSyncError(f"durable object missing: {src}")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest


class GCSStore(DurableStore):
    """Upload small training artifacts under ``gs://bucket/prefix/runs/<run_id>/``."""

    def __init__(self, gcs_uri: str, run_id: str) -> None:
        uri = gcs_uri.replace("gs://", "").rstrip("/")
        bucket_name, _, prefix = uri.partition("/")
        self.bucket_name = bucket_name
        self.prefix = "/".join(p for p in (prefix, "runs", run_id) if p)
        self.run_id = run_id
        try:
            from google.cloud import storage  # type: ignore
        except ImportError as exc:
            raise DurableSyncError(
                "google-cloud-storage required for GCS durable checkpoints"
            ) from exc
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)

    def uri_for(self, relative: str) -> str:
        return f"gs://{self.bucket_name}/{self.prefix}/{relative.lstrip('/')}"

    def _publish_blob(self, local_path: Path, relative: str) -> None:
        blob_path = f"{self.prefix}/{relative.lstrip('/')}"
        blob = self._bucket.blob(blob_path)
        try:
            blob.upload_from_filename(str(local_path))
            blob.reload()
        except Exception as exc:  # noqa: BLE001
            raise DurableSyncError(f"GCS upload failed for {blob_path}: {exc}") from exc
        if blob.size != local_path.stat().st_size:
            raise DurableSyncError(
                f"GCS size mismatch for {blob_path}: "
                f"remote={blob.size} local={local_path.stat().st_size}"
            )

    def upload_file(self, local_path: Path, relative: str) -> str:
        local_path = Path(local_path)
        if not local_path.is_file():
            raise DurableSyncError(f"missing local checkpoint for durable sync: {local_path}")
        self._publish_blob(local_path, relative)
        alias = _resume_alias(relative)
        if alias is not None:
            self._publish_blob(local_path, alias)
        return self.uri_for(relative)

    def download_file(self, relative: str, dest: Path) -> Path:
        blob_path = f"{self.prefix}/{relative.lstrip('/')}"
        blob = self._bucket.blob(blob_path)
        if not blob.exists():
            raise DurableSyncError(f"GCS object missing: gs://{self.bucket_name}/{blob_path}")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        return dest


def build_durable_store_from_env(*, run_id: str | None = None) -> DurableStore | None:
    """Resolve durable backend from env. Returns None if durable sync disabled."""
    mirror = os.environ.get("TRENCHER_DURABLE_MIRROR") or os.environ.get(
        "TRENCHER_WM_DURABLE_MIRROR"
    )
    run_id = run_id or os.environ.get("TRENCHER_WM_RUN_ID") or "default_run"
    if mirror:
        return LocalMirrorStore(Path(mirror) / run_id)
    gcs = os.environ.get("TRENCHER_GCS_URI") or ""
    if (
        gcs
        and os.environ.get("TRENCHER_WM_DURABLE", "1") not in {"0", "false", "False"}
        and (os.environ.get("TRENCHER_WM_PHASE") or os.environ.get("TRENCHER_WM_DURABLE_FORCE"))
    ):
        return GCSStore(gcs, run_id)
    return None


def write_progress_sidecar(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path
