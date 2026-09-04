"""Orchestrate probe → plan → acquire → report for world-model data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trencher.data.world_model.acquire import (
    run_bigquery_acquisition,
    run_gecko_acquisition,
    run_local_acquisition,
    run_rpc_acquisition,
)
from trencher.data.world_model.acquire.funding import run_native_sol_funding_acquisition
from trencher.data.world_model.acquire.rpc import SolanaRpc
from trencher.data.world_model.freeze import build_data_freeze
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.pit_audit import build_pit_v3_audit
from trencher.data.world_model.planner import build_extraction_plan, write_plan
from trencher.data.world_model.probe import run_source_probe
from trencher.data.world_model.report import build_coverage_report, write_reports

logger = logging.getLogger(__name__)


@dataclass
class WorldModelResult:
    ok: bool
    manifest_dir: Path
    world_root: Path
    summary: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def run_world_model_pipeline(
    *,
    external_root: Path,
    manifest_dir: Path | None = None,
    world_root: Path | None = None,
    skip_live: bool = False,
    bq_project: str | None = None,
    bq_execute: bool = False,
    rich_subset_max_mints: int = 4000,
    rpc_max_signatures: int = 24_000,
    gecko_max_pools: int = 200,
    max_trade_batches: int | None = None,
    stages: list[str] | None = None,
) -> WorldModelResult:
    """Run selected stages: probe, plan, acquire, report (default: all).

    Report stage also builds the data freeze and a PIT sample.
    Explicit ``freeze`` / ``pit`` stages can run without re-acquiring.
    """
    stages = stages or ["probe", "plan", "acquire", "report"]
    manifest_dir = ensure_dir(manifest_dir or Path("manifests") / "world_model_data")
    world_root = ensure_dir(world_root or external_root / "world_model")
    errors: list[str] = []
    summary: dict[str, Any] = {"generated_at": utc_now(), "stages": stages}

    probe: dict[str, Any] = {}
    if "probe" in stages:
        logger.info("Probing world-model sources")
        sample_pool = None
        mig = external_root / "pumpfun_memecoin_corpus" / "migrations.parquet"
        if mig.exists():
            pools = pq.read_table(mig, columns=["pool_address"]).column(0).to_pylist()
            for p in pools:
                if p and p not in {
                    "synthetic_graduation_queue",
                    "backfilled_from_pumpswap_trade",
                }:
                    sample_pool = p
                    break
        probe = run_source_probe(
            external_root=external_root,
            manifest_dir=manifest_dir,
            skip_live=skip_live,
            bq_project=bq_project,
            sample_pool=sample_pool,
        )
        summary["probe"] = {
            "bigquery": probe.get("bigquery", {}).get("decision"),
            "solarchive": probe.get("solarchive", {}).get("decision"),
            "geckoterminal": probe.get("geckoterminal", {}).get("decision"),
        }
    elif (manifest_dir / "source_probe.json").exists():
        from trencher.data.world_model.io import read_json

        probe = read_json(manifest_dir / "source_probe.json")

    plan: dict[str, Any] = {}
    if "plan" in stages:
        logger.info("Building extraction plan")
        plan = build_extraction_plan(
            external_root=external_root,
            probe=probe or {"bigquery": {}, "solarchive": {}, "geckoterminal": {}},
            rich_subset_max_mints=rich_subset_max_mints,
            rpc_max_signatures=rpc_max_signatures,
            gecko_max_pools=gecko_max_pools,
        )
        write_plan(plan, manifest_dir)
        summary["plan_layers"] = list(plan.get("layers", {}))

    local: dict[str, Any] = {}
    gecko: dict[str, Any] = {}
    rpc: dict[str, Any] = {}
    bq: dict[str, Any] = {}
    if "acquire" in stages:
        logger.info("Acquiring local corpus enrichments")
        local = run_local_acquisition(
            external_root=external_root,
            world_root=world_root,
            rich_subset_max_mints=rich_subset_max_mints,
            rpc_max_signatures=max(rpc_max_signatures, 1),
            max_trade_batches=max_trade_batches,
        )
        pools_path = world_root / "universe" / "graduated_real_pool_mints.parquet"
        pools: list[str] = []
        if pools_path.exists():
            try:
                pools = [
                    p
                    for p in pq.read_table(pools_path, columns=["pool_address"])
                    .column(0)
                    .to_pylist()
                    if p
                ]
            except Exception:  # noqa: BLE001
                pools = []
        logger.info("Acquiring GeckoTerminal data")
        gecko = run_gecko_acquisition(
            world_root=world_root,
            pools=pools,
            max_pools=gecko_max_pools,
            skip_live=skip_live,
        )
        logger.info("Acquiring RPC microstructure")
        rpc_client = None if skip_live else SolanaRpc()
        rpc = run_rpc_acquisition(
            world_root=world_root,
            max_signatures=rpc_max_signatures,
            rpc=rpc_client,
            skip_live=skip_live,
        )
        logger.info("Acquiring native SOL funding (pre-T System Program transfers)")
        funding = run_native_sol_funding_acquisition(
            world_root=world_root,
            rpc=rpc_client or SolanaRpc(min_interval=0),
            skip_live=skip_live,
        )
        logger.info("BigQuery acquisition (dry-run unless enabled)")
        bq = run_bigquery_acquisition(
            world_root=world_root,
            project=bq_project,
            skip_live=skip_live,
            execute=bq_execute,
        )
        summary["acquire"] = {
            "local_ok": bool(local),
            "gecko": gecko.get("skipped") or gecko.get("api_calls"),
            "rpc_found": rpc.get("found"),
            "native_sol_edges": funding.get("native_sol_edges"),
            "bq": bq.get("decision"),
        }

    report: dict[str, Any] = {}
    need_report_inputs = "report" in stages
    if need_report_inputs or "freeze" in stages or "pit" in stages:
        from trencher.data.world_model.io import read_json

        if need_report_inputs:
            logger.info("Writing coverage report")
            if not local and (world_root / "local_acquisition_manifest.json").exists():
                local = read_json(world_root / "local_acquisition_manifest.json")
            if not gecko and (world_root / "gecko" / "gecko_manifest.json").exists():
                gecko = read_json(world_root / "gecko" / "gecko_manifest.json")
            if not rpc and (world_root / "microstructure" / "rpc_manifest.json").exists():
                rpc = read_json(world_root / "microstructure" / "rpc_manifest.json")
            if not bq and (world_root / "bigquery" / "bigquery_manifest.json").exists():
                bq = read_json(world_root / "bigquery" / "bigquery_manifest.json")
            if not plan and (manifest_dir / "extraction_plan.json").exists():
                plan = read_json(manifest_dir / "extraction_plan.json")
            if not probe and (manifest_dir / "source_probe.json").exists():
                probe = read_json(manifest_dir / "source_probe.json")

        if "freeze" in stages:
            # Explicit only — world_model_freeze_v1 stays frozen unless requested.
            logger.info("Rebuilding data freeze manifest (explicit stage)")
            freeze = build_data_freeze(
                world_root=world_root,
                manifest_dir=manifest_dir,
                external_root=external_root,
            )
            summary["freeze_id"] = freeze.get("freeze_id")
            summary["fusion_coverage"] = freeze.get("fusion_coverage")
            summary["ownership_evidence_counts"] = freeze.get(
                "ownership_evidence_counts"
            )

        if "pit" in stages or "report" in stages:
            logger.info("Building PIT v3 audit (derived; freeze acquisition unchanged)")
            audit = build_pit_v3_audit(
                world_root=world_root,
                manifest_dir=manifest_dir,
                pump_root=external_root / "pumpfun_memecoin_corpus",
                max_mints=50,
            )
            summary["pit_v3_audit"] = {
                "future_labels_absent": audit.get(
                    "future_labels_structurally_absent_from_model_state"
                ),
                "leakage_all_passed": (audit.get("leakage_tests") or {}).get(
                    "all_passed"
                ),
                "holder_state_reconstructable_at_t_by_age": audit.get(
                    "holder_state_reconstructable_at_t_by_age"
                ),
            }
            summary["fusion_coverage"] = audit.get("fusion_coverage")
            summary["ownership_evidence_counts"] = audit.get(
                "ownership_evidence_counts"
            )

        if "report" in stages:
            report = build_coverage_report(
                world_root=world_root,
                probe=probe,
                plan=plan,
                local=local,
                gecko=gecko,
                rpc=rpc,
                bq=bq,
            )
            write_reports(manifest_dir, report)
            summary["encoders"] = report.get("encoders")
            summary["world_model_dataset_ready"] = report.get("world_model_dataset_ready")
            summary["fusion_holder_state_reconstructable"] = report.get(
                "metrics", {}
            ).get("fusion_holder_state_reconstructable")

    write_json(manifest_dir / "pipeline_summary.json", summary)
    return WorldModelResult(
        ok=not errors,
        manifest_dir=manifest_dir,
        world_root=world_root,
        summary=summary,
        errors=errors,
    )
