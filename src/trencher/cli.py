"""CLI entrypoint for trencher."""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import click

from trencher import __version__
from trencher.data.historical import download_historical
from trencher.data.paths import SOURCES, resolve_external_root, resolve_manifest_root
from trencher.dune.backfill import backfill_range
from trencher.dune.client import LiveDuneRunner, MockDuneRunner
from trencher.dune.config import resolve_data_root
from trencher.dune.doctor import run_doctor
from trencher.dune.preflight import format_preflight, run_preflight


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


@dune.command("validate-queries")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--data-root", type=click.Path(path_type=Path), default=None)
@click.option("--skip-samples", is_flag=True, help="Only check information_schema catalog")
@click.option("--json-out", is_flag=True, help="Print machine-readable JSON summary")
def dune_validate_queries(
    config_path: Path | None,
    data_root: Path | None,
    skip_samples: bool,
    json_out: bool,
) -> None:
    """Low-cost live check of Dune tables/columns/queries before backfill."""
    root = resolve_data_root(data_root)
    runner = LiveDuneRunner()
    report = run_preflight(
        runner,
        config_path=config_path,
        data_root=root,
        include_samples=not skip_samples,
    )
    if json_out:
        click.echo(format_preflight(report))
    else:
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
@click.option(
    "--chunk-days",
    default=1,
    show_default=True,
    type=int,
    help="Calendar days per Dune query; results are still written as daily partitions",
)
@click.option("--force", is_flag=True, help="Re-query Dune even if partition exists")
@click.option("--dry-run", is_flag=True, help="Plan partitions without querying Dune")
@click.option("--skip-preflight", is_flag=True, help="Skip live catalog validation")
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
    skip_preflight: bool,
    config_path: Path | None,
    data_root: Path | None,
    verbose: bool,
) -> None:
    """Backfill a dataset over a date range with resumable partitions."""
    _configure_logging(verbose)
    start_date = date(start.year, start.month, start.day)
    end_date = date(end.year, end.month, end.day)
    root = resolve_data_root(data_root)

    runner = MockDuneRunner() if dry_run else LiveDuneRunner()
    enabled_tables = None
    enabled_columns = None
    if not dry_run and not skip_preflight:
        preflight = run_preflight(
            runner,
            config_path=config_path,
            data_root=root,
            include_samples=False,
        )
        for warn in preflight.warnings:
            click.echo(f"WARN {warn}")
        if not preflight.ok:
            for err in preflight.errors:
                click.echo(f"ERR {err}", err=True)
            raise SystemExit("Preflight failed; refusing to start backfill")
        enabled_tables = preflight.existing_tables
        enabled_columns = preflight.columns

    results = backfill_range(
        runner,
        dataset,
        start_date,
        end_date,
        chunk_days=chunk_days,
        data_root=root,
        force=force,
        dry_run=dry_run,
        config_path=config_path,
        enabled_tables=enabled_tables,
        enabled_columns=enabled_columns,
    )
    skipped = sum(1 for r in results if r.skipped)
    written = sum(1 for r in results if not r.skipped and not r.dry_run)
    planned = sum(1 for r in results if r.dry_run)
    click.echo(
        f"Done: {len(results)} partitions "
        f"({skipped} skipped, {written} written, {planned} dry-run)"
    )


@main.group()
def data() -> None:
    """Public historical dataset acquisition and local registration."""


@data.command("download-historical")
@click.option(
    "--source",
    "sources",
    multiple=True,
    type=click.Choice(sorted(SOURCES)),
    help="Subset of sources (default: all)",
)
@click.option("--force", is_flag=True, help="Re-download even if verified files exist")
@click.option(
    "--external-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Download root (default: data/external)",
)
@click.option(
    "--manifest-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Committed manifest root (default: manifests/external)",
)
@click.option("--verbose", "-v", is_flag=True)
def data_download_historical(
    sources: tuple[str, ...],
    force: bool,
    external_root: Path | None,
    manifest_root: Path | None,
    verbose: bool,
) -> None:
    """Download, verify, inventory, and register finalized public historical datasets."""
    _configure_logging(verbose)
    result = download_historical(
        sources=list(sources) if sources else None,
        external_root=resolve_external_root(external_root),
        manifest_root=resolve_manifest_root(manifest_root),
        force=force,
    )
    click.echo(json.dumps(result.registry, indent=2))
    if not result.ok:
        for err in result.errors:
            click.echo(f"ERR {err}", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
