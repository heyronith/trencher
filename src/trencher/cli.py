"""CLI entrypoint for trencher."""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import click

from trencher import __version__
from trencher.data.feasibility import run_feasibility_audit
from trencher.data.historical import download_historical
from trencher.data.paths import SOURCES, resolve_external_root, resolve_manifest_root
from trencher.data.world_model import run_source_probe, run_world_model_pipeline
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


@data.command("audit-feasibility")
@click.option(
    "--external-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Downloaded corpora root (default: data/external)",
)
@click.option(
    "--manifest-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Parent of feasibility/ reports (default: manifests)",
)
@click.option(
    "--trade-wallet-batches",
    type=int,
    default=None,
    help="Optional cap on trade batches when collecting wallet sets",
)
@click.option(
    "--skip-heavy-trade-scan",
    is_flag=True,
    help="Skip full trades/snapshots streaming quality scans",
)
@click.option("--verbose", "-v", is_flag=True)
def data_audit_feasibility(
    external_root: Path | None,
    manifest_root: Path | None,
    trade_wallet_batches: int | None,
    skip_heavy_trade_scan: bool,
    verbose: bool,
) -> None:
    """Audit local historical corpora for research-grade model feasibility."""
    _configure_logging(verbose)
    result = run_feasibility_audit(
        external_root=resolve_external_root(external_root),
        manifest_root=manifest_root,
        trade_wallet_max_batches=trade_wallet_batches,
        skip_heavy_trade_scan=skip_heavy_trade_scan,
    )
    if not result.ok:
        for err in result.errors:
            click.echo(f"ERR {err}", err=True)
        raise SystemExit(1)
    click.echo(json.dumps(result.summary.get("verdict", {}), indent=2))
    click.echo(f"Reports written under {result.report_dir}")


@data.command("probe-world-model-sources")
@click.option(
    "--external-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Downloaded corpora root (default: data/external)",
)
@click.option(
    "--manifest-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output dir (default: manifests/world_model_data)",
)
@click.option(
    "--skip-live",
    is_flag=True,
    help="Skip live BigQuery/Gecko calls (CI-safe)",
)
@click.option("--bq-project", default=None, help="GCP project for BigQuery")
@click.option("--verbose", "-v", is_flag=True)
def data_probe_world_model_sources(
    external_root: Path | None,
    manifest_dir: Path | None,
    skip_live: bool,
    bq_project: str | None,
    verbose: bool,
) -> None:
    """Probe BigQuery, Solarchive indexes, and GeckoTerminal for corpus coverage."""
    _configure_logging(verbose)
    root = resolve_external_root(external_root)
    out = Path(manifest_dir) if manifest_dir else Path("manifests") / "world_model_data"
    payload = run_source_probe(
        external_root=root,
        manifest_dir=out,
        skip_live=skip_live,
        bq_project=bq_project,
    )
    click.echo(
        json.dumps(
            {
                "bigquery": payload.get("bigquery", {}).get("decision"),
                "solarchive": payload.get("solarchive", {}).get("decision"),
                "geckoterminal": payload.get("geckoterminal", {}).get("decision"),
            },
            indent=2,
        )
    )
    click.echo(f"Wrote {out / 'source_probe.json'}")


@data.command("recover-world-model-data")
@click.option(
    "--external-root",
    type=click.Path(path_type=Path),
    default=None,
)
@click.option(
    "--manifest-dir",
    type=click.Path(path_type=Path),
    default=None,
)
@click.option(
    "--world-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Raw enrichment root (default: data/external/world_model)",
)
@click.option("--skip-live", is_flag=True, help="CI-safe: no live BQ/Gecko/RPC")
@click.option("--bq-project", default=None)
@click.option("--bq-execute", is_flag=True, help="Execute BQ after successful dry-run")
@click.option("--rpc-max-signatures", type=int, default=24000)
@click.option("--gecko-max-pools", type=int, default=200)
@click.option("--rich-subset-max-mints", type=int, default=4000)
@click.option(
    "--max-trade-batches",
    type=int,
    default=None,
    help="Optional cap on trade batches (tests)",
)
@click.option(
    "--stage",
    "stages",
    multiple=True,
    type=click.Choice(
        [
            "probe",
            "plan",
            "acquire",
            "report",
            "freeze",
            "pit",
            "observations",
            "sequences",
        ]
    ),
    help="Subset of pipeline stages (default: all)",
)
@click.option("--verbose", "-v", is_flag=True)
def data_recover_world_model_data(
    external_root: Path | None,
    manifest_dir: Path | None,
    world_root: Path | None,
    skip_live: bool,
    bq_project: str | None,
    bq_execute: bool,
    rpc_max_signatures: int,
    gecko_max_pools: int,
    rich_subset_max_mints: int,
    max_trade_batches: int | None,
    stages: tuple[str, ...],
    verbose: bool,
) -> None:
    """Probe, plan, acquire, and report world-model historical enrichments."""
    _configure_logging(verbose)
    root = resolve_external_root(external_root)
    result = run_world_model_pipeline(
        external_root=root,
        manifest_dir=Path(manifest_dir)
        if manifest_dir
        else Path("manifests") / "world_model_data",
        world_root=Path(world_root) if world_root else root / "world_model",
        skip_live=skip_live,
        bq_project=bq_project,
        bq_execute=bq_execute,
        rich_subset_max_mints=rich_subset_max_mints,
        rpc_max_signatures=rpc_max_signatures,
        gecko_max_pools=gecko_max_pools,
        max_trade_batches=max_trade_batches,
        stages=list(stages) if stages else None,
    )
    if not result.ok:
        for err in result.errors:
            click.echo(f"ERR {err}", err=True)
        raise SystemExit(1)
    click.echo(json.dumps(result.summary.get("encoders", result.summary), indent=2))
    click.echo(f"Manifests: {result.manifest_dir}")
    click.echo(f"Data: {result.world_root}")


if __name__ == "__main__":
    main()
