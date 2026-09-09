"""Durable checkpoint / progress sync for disposable Colab runtimes.

Local-mirror backend is used in CI; GCS backend is used in production.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
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
    """Durable GCS storage with a fail-closed ``gcloud storage`` fallback.

    ``TRENCHER_GCS_TRANSPORT`` accepts ``auto`` (default), ``python``, or
    ``gcloud``. ``auto`` starts with google-cloud-storage when available and
    permanently switches this store instance to ``gcloud storage`` after the
    first Python transport failure. This protects disposable Colab runs from
    ADC / billing-context failures where the authenticated gcloud CLI can still
    access the same bucket.
    """

    _VALID_TRANSPORTS = {"auto", "python", "gcloud"}

    def __init__(
        self,
        gcs_uri: str,
        run_id: str,
        *,
        storage_client: Any | None = None,
        transport: str | None = None,
    ) -> None:
        uri = gcs_uri.replace("gs://", "").rstrip("/")
        bucket_name, _, prefix = uri.partition("/")
        if not bucket_name:
            raise DurableSyncError(f"invalid GCS URI: {gcs_uri}")

        self.bucket_name = bucket_name
        self.prefix = "/".join(p for p in (prefix, "runs", run_id) if p)
        self.run_id = run_id
        self.transport = (
            transport or os.environ.get("TRENCHER_GCS_TRANSPORT") or "auto"
        ).strip().lower()
        if self.transport not in self._VALID_TRANSPORTS:
            raise DurableSyncError(
                "TRENCHER_GCS_TRANSPORT must be one of auto, python, gcloud; "
                f"got {self.transport!r}"
            )

        self._client: Any | None = None
        self._bucket: Any | None = None
        self._python_disabled = self.transport == "gcloud"
        self._fallback_reported = False

        if storage_client is not None:
            self._client = storage_client
            self._bucket = storage_client.bucket(bucket_name)
            self._python_disabled = self.transport == "gcloud"
        elif self.transport != "gcloud":
            try:
                from google.cloud import storage  # type: ignore
            except ImportError as exc:
                if self.transport == "python":
                    raise DurableSyncError(
                        "google-cloud-storage required for Python GCS transport"
                    ) from exc
                self._python_disabled = True
            else:
                self._client = storage.Client()
                self._bucket = self._client.bucket(bucket_name)

        if self.transport == "gcloud":
            self._require_gcloud()

    @property
    def active_transport(self) -> str:
        """Return the transport this store will use for the next operation."""
        return "gcloud" if self._python_disabled else "python"

    def uri_for(self, relative: str) -> str:
        return f"gs://{self.bucket_name}/{self.prefix}/{relative.lstrip('/')}"

    def _require_gcloud(self) -> str:
        executable = shutil.which("gcloud")
        if not executable:
            raise DurableSyncError(
                "gcloud CLI is required for GCS durable fallback but was not found in PATH"
            )
        return executable

    def _run_gcloud(self, *args: str) -> subprocess.CompletedProcess[str]:
        executable = self._require_gcloud()
        proc = subprocess.run(
            [executable, "storage", *args],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            detail = stderr or stdout or f"exit={proc.returncode}"
            raise DurableSyncError(
                f"gcloud storage {' '.join(args[:2])} failed: {detail}"
            )
        return proc

    def _remote_size_via_gcloud(self, relative: str) -> int:
        proc = self._run_gcloud(
            "objects",
            "describe",
            self.uri_for(relative),
            "--format=value(size)",
        )
        raw = (proc.stdout or "").strip()
        try:
            return int(raw)
        except ValueError as exc:
            raise DurableSyncError(
                f"unable to parse gcloud object size for {self.uri_for(relative)}: {raw!r}"
            ) from exc

    def _publish_blob_python(self, local_path: Path, relative: str) -> None:
        if self._bucket is None:
            raise DurableSyncError("Python GCS client unavailable")
        blob_path = f"{self.prefix}/{relative.lstrip('/')}"
        blob = self._bucket.blob(blob_path)
        blob.upload_from_filename(str(local_path))
        blob.reload()
        if blob.size != local_path.stat().st_size:
            raise DurableSyncError(
                f"GCS size mismatch for {blob_path}: "
                f"remote={blob.size} local={local_path.stat().st_size}"
            )

    def _publish_blob_gcloud(self, local_path: Path, relative: str) -> None:
        self._run_gcloud("cp", str(local_path), self.uri_for(relative))
        remote_size = self._remote_size_via_gcloud(relative)
        local_size = local_path.stat().st_size
        if remote_size != local_size:
            raise DurableSyncError(
                f"gcloud GCS size mismatch for {self.uri_for(relative)}: "
                f"remote={remote_size} local={local_size}"
            )

    def _download_python(self, relative: str, dest: Path) -> Path:
        if self._bucket is None:
            raise DurableSyncError("Python GCS client unavailable")
        blob_path = f"{self.prefix}/{relative.lstrip('/')}"
        blob = self._bucket.blob(blob_path)
        if not blob.exists():
            raise DurableSyncError(
                f"GCS object missing: gs://{self.bucket_name}/{blob_path}"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        return dest

    def _download_gcloud(self, relative: str, dest: Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        remote_size = self._remote_size_via_gcloud(relative)
        tmp = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
        try:
            with contextlib.suppress(OSError):
                tmp.unlink()
            self._run_gcloud("cp", self.uri_for(relative), str(tmp))
            if not tmp.is_file():
                raise DurableSyncError(f"gcloud download missing temporary file: {tmp}")
            local_size = tmp.stat().st_size
            if local_size != remote_size:
                raise DurableSyncError(
                    f"gcloud download size mismatch for {self.uri_for(relative)}: "
                    f"remote={remote_size} local={local_size}"
                )
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                with contextlib.suppress(OSError):
                    tmp.unlink()
        return dest

    def _switch_to_gcloud(self, operation: str, exc: Exception) -> None:
        if self.transport == "python":
            raise DurableSyncError(
                f"Python GCS {operation} failed for run {self.run_id}: {exc}"
            ) from exc
        self._require_gcloud()
        self._python_disabled = True
        if not self._fallback_reported:
            print(
                "[durable] Python GCS transport failed; switching this run to "
                f"gcloud storage ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
            self._fallback_reported = True

    def _publish_blob(self, local_path: Path, relative: str) -> None:
        if self._python_disabled:
            self._publish_blob_gcloud(local_path, relative)
            return
        try:
            self._publish_blob_python(local_path, relative)
            return
        except Exception as exc:  # noqa: BLE001
            self._switch_to_gcloud("upload", exc)
        try:
            self._publish_blob_gcloud(local_path, relative)
        except Exception as cli_exc:  # noqa: BLE001
            raise DurableSyncError(
                f"GCS upload failed after Python→gcloud fallback for {relative}: {cli_exc}"
            ) from cli_exc

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
        dest = Path(dest)
        if self._python_disabled:
            return self._download_gcloud(relative, dest)
        try:
            return self._download_python(relative, dest)
        except Exception as exc:  # noqa: BLE001
            self._switch_to_gcloud("download", exc)
        try:
            return self._download_gcloud(relative, dest)
        except Exception as cli_exc:  # noqa: BLE001
            raise DurableSyncError(
                f"GCS download failed after Python→gcloud fallback for {relative}: {cli_exc}"
            ) from cli_exc


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
