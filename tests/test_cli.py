"""CLI parsing and smoke tests."""

from __future__ import annotations

from click.testing import CliRunner

from trencher.cli import main


def test_cli_version() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_doctor() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["dune", "doctor"])
    assert result.exit_code == 0
    assert "token_lifecycle" in result.output or "schema token_lifecycle" in result.output


def test_cli_backfill_dry_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRENCHER_DATA_ROOT", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "dune",
            "backfill",
            "--dataset",
            "market_context",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-01",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "dry-run" in result.output


def test_cli_backfill_unknown_dataset() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "dune",
            "backfill",
            "--dataset",
            "nonexistent",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-01",
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
