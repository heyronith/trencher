"""Orchestrate broad event/curve supervision build phases."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from trencher.data.world_model.broad.audit import build_broad_supervision_audit_v1
from trencher.data.world_model.broad.audit_v2 import build_broad_supervision_audit_v2
from trencher.data.world_model.broad.audit_v3 import build_broad_supervision_audit_v3
from trencher.data.world_model.broad.graduation_censor import graduation_censor_decision
from trencher.data.world_model.broad.observations import build_broad_observations_v1
from trencher.data.world_model.broad.population import build_broad_population_v1
from trencher.data.world_model.broad.sequences import build_broad_sequences_v1
from trencher.data.world_model.broad.targets import build_broad_targets_v1
from trencher.data.world_model.broad.targets_v2 import build_broad_targets_v2_from_v1
from trencher.data.world_model.broad.targets_v3 import build_broad_targets_v3_from_v2
from trencher.data.world_model.io import ensure_dir, utc_now, write_json

logger = logging.getLogger(__name__)

DEFAULT_PHASES = (
    "population",
    "sequences",
    "observations",
    "targets",
    "audit",
)


def build_broad_supervision_v1(
    *,
    world_root: Path,
    manifest_dir: Path,
    pump_root: Path,
    phases: list[str] | tuple[str, ...] | None = None,
    broad_root: Path | None = None,
) -> dict[str, Any]:
    """Build broad supervision artifacts under ``world_root/broad/``.

    Phases (subset supported for testing): population, sequences,
    observations, targets, audit.
    """
    phases = tuple(phases) if phases else DEFAULT_PHASES
    world_root = Path(world_root)
    pump_root = Path(pump_root)
    manifest_dir = ensure_dir(manifest_dir)
    broad_root = ensure_dir(broad_root or world_root / "broad")
    progress_path = broad_root / "build_progress.jsonl"

    timings: dict[str, float] = {}
    metas: dict[str, Any] = {}
    summary: dict[str, Any] = {
        "generated_at": utc_now(),
        "phases": list(phases),
        "broad_root": str(broad_root),
        "pump_root": str(pump_root),
        "world_root": str(world_root),
    }

    def _log_progress(phase: str, status: str, **extra: Any) -> None:
        rec = {
            "ts": utc_now(),
            "phase": phase,
            "status": status,
            **extra,
        }
        with progress_path.open("a", encoding="utf-8") as fh:
            import json

            fh.write(json.dumps(rec, default=str) + "\n")
        logger.info("broad phase=%s status=%s %s", phase, status, extra)

    population_path = broad_root / "broad_population_v1.parquet"

    if "population" in phases:
        _log_progress("population", "start")
        t0 = time.perf_counter()
        # Always document graduation censor alongside population.
        censor = graduation_censor_decision(pump_root)
        write_json(broad_root / "graduation_censor_decision.json", censor)
        write_json(manifest_dir / "graduation_censor_decision.json", censor)
        metas["population"] = build_broad_population_v1(
            world_root=world_root,
            pump_root=pump_root,
            out_dir=broad_root,
        )
        timings["population"] = time.perf_counter() - t0
        _log_progress(
            "population",
            "done",
            seconds=timings["population"],
            counts=(metas["population"] or {}).get("counts"),
        )

    if "sequences" in phases:
        _log_progress("sequences", "start")
        t0 = time.perf_counter()
        metas["sequences"] = build_broad_sequences_v1(
            world_root=world_root,
            pump_root=pump_root,
            broad_root=broad_root,
            population_path=population_path,
        )
        timings["sequences"] = time.perf_counter() - t0
        _log_progress("sequences", "done", seconds=timings["sequences"])

    if "observations" in phases:
        _log_progress("observations", "start")
        t0 = time.perf_counter()
        metas["observations"] = build_broad_observations_v1(
            world_root=world_root,
            pump_root=pump_root,
            broad_root=broad_root,
            population_path=population_path,
        )
        timings["observations"] = time.perf_counter() - t0
        _log_progress(
            "observations",
            "done",
            seconds=timings["observations"],
            observations_n=(metas["observations"] or {}).get("observations_n"),
        )

    if "targets" in phases:
        _log_progress("targets", "start")
        t0 = time.perf_counter()
        metas["targets"] = build_broad_targets_v1(
            world_root=world_root,
            pump_root=pump_root,
            broad_root=broad_root,
            population_path=population_path,
        )
        timings["targets"] = time.perf_counter() - t0
        _log_progress(
            "targets",
            "done",
            seconds=timings["targets"],
            observations_processed=(metas["targets"] or {}).get(
                "observations_processed"
            ),
        )

    if "targets_v2" in phases:
        _log_progress("targets_v2", "start")
        t0 = time.perf_counter()
        metas["targets_v2"] = build_broad_targets_v2_from_v1(
            world_root=world_root,
            pump_root=pump_root,
            broad_root=broad_root,
        )
        timings["targets_v2"] = time.perf_counter() - t0
        _log_progress(
            "targets_v2",
            "done",
            seconds=timings["targets_v2"],
            observations_processed=(metas["targets_v2"] or {}).get(
                "observations_processed"
            ),
            primary_explosion_events=(metas["targets_v2"] or {}).get(
                "primary_explosion_events"
            ),
        )

    if "audit" in phases:
        _log_progress("audit", "start")
        t0 = time.perf_counter()
        metas["audit"] = build_broad_supervision_audit_v1(
            world_root=world_root,
            manifest_dir=manifest_dir,
            pump_root=pump_root,
            broad_root=broad_root,
            phase_timings=timings,
            phase_metas=metas,
        )
        timings["audit"] = time.perf_counter() - t0
        _log_progress("audit", "done", seconds=timings["audit"])

    if "audit_v2" in phases:
        _log_progress("audit_v2", "start")
        t0 = time.perf_counter()
        metas["audit_v2"] = build_broad_supervision_audit_v2(
            world_root=world_root,
            manifest_dir=manifest_dir,
            pump_root=pump_root,
            broad_root=broad_root,
            phase_timings=timings,
            phase_metas=metas,
        )
        timings["audit_v2"] = time.perf_counter() - t0
        _log_progress("audit_v2", "done", seconds=timings["audit_v2"])

    if "targets_v3" in phases:
        _log_progress("targets_v3", "start")
        t0 = time.perf_counter()
        metas["targets_v3"] = build_broad_targets_v3_from_v2(
            world_root=world_root,
            pump_root=pump_root,
            broad_root=broad_root,
        )
        timings["targets_v3"] = time.perf_counter() - t0
        _log_progress(
            "targets_v3",
            "done",
            seconds=timings["targets_v3"],
            observations_processed=(metas["targets_v3"] or {}).get(
                "observations_processed"
            ),
            primary_explosion_events=(metas["targets_v3"] or {}).get(
                "primary_explosion_events"
            ),
            mints_without_primary=(metas["targets_v3"] or {}).get(
                "mints_without_primary_definition_row"
            ),
        )

    if "audit_v3" in phases:
        _log_progress("audit_v3", "start")
        t0 = time.perf_counter()
        metas["audit_v3"] = build_broad_supervision_audit_v3(
            world_root=world_root,
            manifest_dir=manifest_dir,
            pump_root=pump_root,
            broad_root=broad_root,
            phase_timings=timings,
            phase_metas=metas,
        )
        timings["audit_v3"] = time.perf_counter() - t0
        _log_progress("audit_v3", "done", seconds=timings["audit_v3"])

    summary["timings_sec"] = timings
    summary["metas"] = {
        k: {
            kk: vv
            for kk, vv in (v or {}).items()
            if kk not in {"price_capable_mint_set", "phase_metas"}
        }
        for k, v in metas.items()
    }
    write_json(broad_root / "broad_supervision_build_summary.json", summary)
    write_json(manifest_dir / "broad_supervision_build_summary.json", summary)
    return summary
