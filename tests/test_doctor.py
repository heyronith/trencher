"""Doctor command tests."""

from __future__ import annotations

from trencher.dune.doctor import run_doctor


def test_doctor_passes_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DUNE_API_KEY", raising=False)
    report = run_doctor()
    assert report.ok
    assert any("DUNE_API_KEY is not set" in w for w in report.warnings)


def test_doctor_lists_all_datasets() -> None:
    report = run_doctor()
    assert report.ok
    assert any("token_lifecycle" in c for c in report.checks)
