from __future__ import annotations

from pathlib import Path

from trencher.world_model.durable import LocalMirrorStore
from trencher.world_model.launch import _maybe_fetch_durable_resume


def test_periodic_checkpoint_advances_durable_latest(tmp_path: Path) -> None:
    mirror_root = tmp_path / "durable" / "run"
    store = LocalMirrorStore(mirror_root)

    first = tmp_path / "first.pt"
    first.write_bytes(b"checkpoint-v1")

    uri = store.upload_file(first, "periodic.pt")

    assert uri == str((mirror_root / "periodic.pt").resolve())
    assert (mirror_root / "periodic.pt").read_bytes() == b"checkpoint-v1"
    assert (mirror_root / "latest.pt").read_bytes() == b"checkpoint-v1"

    second = tmp_path / "second.pt"
    second.write_bytes(b"checkpoint-v2-newer")

    store.upload_file(second, "periodic.pt")

    assert (mirror_root / "periodic.pt").read_bytes() == b"checkpoint-v2-newer"
    assert (mirror_root / "latest.pt").read_bytes() == b"checkpoint-v2-newer"


def test_periodic_metadata_advances_latest_metadata(tmp_path: Path) -> None:
    mirror_root = tmp_path / "durable" / "run"
    store = LocalMirrorStore(mirror_root)

    meta = tmp_path / "periodic.pt.meta.json"
    meta.write_text('{"step": 54000}\n', encoding="utf-8")

    store.upload_file(meta, "periodic.pt.meta.json")

    assert (mirror_root / "periodic.pt.meta.json").read_text(encoding="utf-8") == (
        '{"step": 54000}\n'
    )
    assert (mirror_root / "latest.pt.meta.json").read_text(encoding="utf-8") == (
        '{"step": 54000}\n'
    )


def test_durable_resume_latest_reads_newest_periodic_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    durable_root = tmp_path / "durable"
    run_id = "stage_b_broad"
    store = LocalMirrorStore(durable_root / run_id)

    accepted = tmp_path / "accepted.pt"
    accepted.write_bytes(b"global-52000")
    store.upload_file(accepted, "latest.pt")

    periodic = tmp_path / "periodic.pt"
    periodic.write_bytes(b"global-54000")
    store.upload_file(periodic, "periodic.pt")

    monkeypatch.setenv("TRENCHER_WM_RUN_ID", run_id)
    monkeypatch.setenv("TRENCHER_DURABLE_MIRROR", str(durable_root))
    monkeypatch.delenv("TRENCHER_WM_DURABLE_MIRROR", raising=False)
    monkeypatch.delenv("TRENCHER_GCS_URI", raising=False)

    ckpt_dir = tmp_path / "restore"
    restored = Path(_maybe_fetch_durable_resume("latest", ckpt_dir))

    assert restored == ckpt_dir / "latest.pt"
    assert restored.read_bytes() == b"global-54000"


def test_final_latest_save_still_overrides_resume_pointer(tmp_path: Path) -> None:
    mirror_root = tmp_path / "durable" / "run"
    store = LocalMirrorStore(mirror_root)

    periodic = tmp_path / "periodic.pt"
    periodic.write_bytes(b"global-54000")
    store.upload_file(periodic, "periodic.pt")

    final = tmp_path / "latest.pt"
    final.write_bytes(b"global-150000")
    store.upload_file(final, "latest.pt")

    assert (mirror_root / "periodic.pt").read_bytes() == b"global-54000"
    assert (mirror_root / "latest.pt").read_bytes() == b"global-150000"
