from __future__ import annotations

from pathlib import Path

from trencher.world_model.durable import GCSStore


class _FailingBlob:
    def __init__(self, state: dict[str, int]) -> None:
        self.state = state

    def exists(self) -> bool:
        self.state["exists"] += 1
        raise RuntimeError("billing account disabled")

    def download_to_filename(self, _path: str) -> None:
        raise AssertionError("download should not run after exists failure")

    def upload_from_filename(self, _path: str) -> None:
        self.state["uploads"] += 1
        raise RuntimeError("billing account disabled")

    def reload(self) -> None:
        raise AssertionError("reload should not run after upload failure")


class _FailingBucket:
    def __init__(self, state: dict[str, int]) -> None:
        self.state = state

    def blob(self, _path: str) -> _FailingBlob:
        return _FailingBlob(self.state)


class _FailingClient:
    def __init__(self, state: dict[str, int]) -> None:
        self.state = state

    def bucket(self, _name: str) -> _FailingBucket:
        return _FailingBucket(self.state)


def test_auto_download_sticks_to_gcloud_after_python_failure(
    tmp_path: Path, monkeypatch
) -> None:
    state = {"exists": 0, "uploads": 0}
    store = GCSStore(
        "gs://bucket/prefix",
        "stage_b_broad",
        storage_client=_FailingClient(state),
        transport="auto",
    )

    gcloud_calls: list[str] = []

    def fake_gcloud_download(relative: str, dest: Path) -> Path:
        gcloud_calls.append(relative)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"checkpoint")
        return dest

    monkeypatch.setattr(store, "_require_gcloud", lambda: "/usr/bin/gcloud")
    monkeypatch.setattr(store, "_download_gcloud", fake_gcloud_download)

    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"

    assert store.download_file("latest.pt", first) == first
    assert first.read_bytes() == b"checkpoint"
    assert store.active_transport == "gcloud"

    assert store.download_file("latest.pt", second) == second
    assert second.read_bytes() == b"checkpoint"

    assert state["exists"] == 1
    assert gcloud_calls == ["latest.pt", "latest.pt"]


def test_auto_upload_fallback_preserves_periodic_latest_alias(
    tmp_path: Path, monkeypatch
) -> None:
    state = {"exists": 0, "uploads": 0}
    store = GCSStore(
        "gs://bucket/prefix",
        "stage_b_broad",
        storage_client=_FailingClient(state),
        transport="auto",
    )

    published: list[tuple[str, bytes]] = []

    def fake_gcloud_publish(local_path: Path, relative: str) -> None:
        published.append((relative, local_path.read_bytes()))

    monkeypatch.setattr(store, "_require_gcloud", lambda: "/usr/bin/gcloud")
    monkeypatch.setattr(store, "_publish_blob_gcloud", fake_gcloud_publish)

    checkpoint = tmp_path / "periodic.pt"
    checkpoint.write_bytes(b"global-60000")

    uri = store.upload_file(checkpoint, "periodic.pt")

    assert uri == "gs://bucket/prefix/runs/stage_b_broad/periodic.pt"
    assert store.active_transport == "gcloud"
    assert state["uploads"] == 1
    assert published == [
        ("periodic.pt", b"global-60000"),
        ("latest.pt", b"global-60000"),
    ]


def test_explicit_python_transport_fails_closed_without_cli_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    state = {"exists": 0, "uploads": 0}
    store = GCSStore(
        "gs://bucket/prefix",
        "stage_b_broad",
        storage_client=_FailingClient(state),
        transport="python",
    )

    called = False

    def fake_gcloud_download(relative: str, dest: Path) -> Path:
        nonlocal called
        called = True
        return dest

    monkeypatch.setattr(store, "_download_gcloud", fake_gcloud_download)

    try:
        store.download_file("latest.pt", tmp_path / "latest.pt")
    except Exception as exc:
        assert "Python GCS download failed" in str(exc)
    else:
        raise AssertionError("explicit python transport should fail closed")

    assert called is False
    assert store.active_transport == "python"
