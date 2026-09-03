"""CLI entrypoint for trencher."""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import click

from trencher import __version__
from trencher.dune.backfill import backfill_range
from trencher.dune.client import LiveDuneRunner
from trencher.dune.doctor import run_doctor


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@click.group()
@click.version_option(__version__, prog_name="trencher")
def main() -> None:
    """Trencher — on-chain raw data extraction."""


@main.group()
def dune() -> None:
    """Dune Analytics extraction commands."""


@dune.command("doctor")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--data-root", type=click.Path(path_type=Path), default=None)
def dune_doctor(config_path: Path | None, data_root: Path | None) -> None:
    """Validate environment and dataset configuration without spending Dune credits."""
    report = run_doctor(config_path=config_path, data_root=data_root)
    for check in report.checks:
        click.echo(f"OK  {check}")
    for warn in report.warnings:
        click.echo(f"WARN {warn}")
    for err in report.errors:
        click.echo(f"ERR {err}", err=True)
    if not report.ok:
        sys.exit(1)


@dune.command("backfill")
@click.option("--dataset", required=True, help="Dataset name from datasets.yaml")
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--chunk-days", default=1, show_default=True, type=int)
@click.option("--force", is_flag=True, help="Re-query Dune even if partition exists")
@click.option("--dry-run", is_flag=True, help="Plan partitions without querying Dune")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--data-root", type=click.Path(path_type=Path), default=None)
@click.option("--verbose", "-v", is_flag=True)
def dune_backfill(
    dataset: str,
    start,
    end,
    chunk_days: int,
    force: bool,
    dry_run: bool,
    config_path: Path | None,
    data_root: Path | None,
    verbose: bool,
) -> None:
    """Backfill a dataset over a date range with resumable partitions."""
    _configure_logging(verbose)
    start_date = date(start.year, start.month, start.day)
    end_date = date(end.year, end.month, end.day)

    from trencher.dune.client import MockDuneRunner

    runner = MockDuneRunner() if dry_run else LiveDuneRunner()

    results = backfill_range(
        runner,
        dataset,
        start_date,
        end_date,
        chunk_days=chunk_days,
        data_root=data_root,
        force=force,
        dry_run=dry_run,
        config_path=config_path,
    )
    skipped = sum(1 for r in results if r.skipped)
    written = sum(1 for r in results if not r.skipped and not r.dry_run)
    planned = sum(1 for r in results if r.dry_run)
    click.echo(
        f"Done: {len(results)} partitions "
        f"({skipped} skipped, {written} written, {planned} dry-run)"
    )


if __name__ == "__main__":
    main()
