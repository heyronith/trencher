"""Build permanent chronological splits for broad + rich populations."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.splits.assign import (
    actual_split_percentages,
    build_token_assignments,
    stream_observation_assignments,
)
from trencher.data.world_model.splits.audit import (
    build_split_contract_v1,
    publish_split_audit_v1,
)
from trencher.data.world_model.splits.cutoffs import compute_launch_cutoffs
from trencher.data.world_model.splits.eval_controls import (
    attach_obs_eval_controls_summary,
    build_eval_controls,
    write_creator_eval_table,
)
from trencher.data.world_model.splits.invariants import run_all_invariants
from trencher.data.world_model.splits.progress import Progress100
from trencher.data.world_model.splits.purge import purge_reason_summary
from trencher.data.world_model.splits.walkforward import build_walkforward_folds

logger = logging.getLogger(__name__)

SPLIT_STORE = "split_v1"


def build_splits_v1(
    *,
    world_root: Path,
    manifest_dir: Path,
    broad_root: Path | None = None,
    progress: Progress100 | None = None,
) -> dict[str, Any]:
    """Materialize leakage-safe chronological splits. Does not alter frozen targets."""
    progress = progress or Progress100()
    world_root = Path(world_root)
    broad_root = Path(broad_root or world_root / "broad")
    manifest_dir = ensure_dir(manifest_dir)
    out_dir = ensure_dir(world_root / "splits" / SPLIT_STORE)

    progress(1, "load broad population for cutoffs")
    pop_path = broad_root / "broad_population_v1.parquet"
    pop = pq.read_table(
        pop_path, columns=["mint", "detected_at_unix", "event_curve_population"]
    ).to_pandas()
    ec = pop[pop["event_curve_population"]]["detected_at_unix"].astype(float).to_numpy()
    cutoffs = compute_launch_cutoffs(ec)
    write_json(out_dir / "cutoffs.json", cutoffs)
    progress(5, "cutoffs computed")

    walkforward = build_walkforward_folds(cutoffs)
    write_json(out_dir / "walkforward_folds.json", walkforward)

    progress(8, "token assignments")
    rich_path = world_root / "universe" / "rich_subset_mints.parquet"
    token_table = build_token_assignments(
        broad_population_path=pop_path,
        rich_subset_path=rich_path,
        cutoffs=cutoffs,
        walkforward=walkforward,
    )
    token_path = out_dir / "token_assignments.parquet"
    pq.write_table(token_table, token_path)
    token_rows = token_table.to_pylist()
    mint_to_meta = {r["mint"]: r for r in token_rows}

    actual_pct = {
        "broad_event_curve": actual_split_percentages(
            token_table, population_flag="in_broad_event_curve"
        ),
        "rich_subset": actual_split_percentages(
            token_table, population_flag="in_rich_subset"
        ),
    }
    write_json(out_dir / "actual_split_percentages.json", actual_pct)
    progress(12, "token assignments written")

    # Broad observation index parts
    obs_root = broad_root / "broad_observation_index_v1"
    # Layout: either partitioned dirs or flat parts
    parts = sorted(obs_root.rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no observation parquet under {obs_root}")
    progress(15, f"streaming {len(parts)} broad obs parts")
    broad_obs_dir = ensure_dir(out_dir / "observation_assignments_broad")
    broad_obs_stats = stream_observation_assignments(
        observation_parts=parts,
        mint_to_meta=mint_to_meta,
        cutoffs=cutoffs,
        out_dir=broad_obs_dir,
        progress=progress,
        progress_start=15,
        progress_end=55,
        label="broad_obs",
    )
    write_json(out_dir / "observation_assignments_broad_stats.json", broad_obs_stats)

    progress(58, "rich observation assignments")
    rich_obs_index = (
        world_root / "observations" / "observation_dataset_v1" / "observation_index_v1.parquet"
    )
    rich_obs_dir = ensure_dir(out_dir / "observation_assignments_rich")
    # Single file — copy into one part via same streamer
    rich_obs_stats = stream_observation_assignments(
        observation_parts=[rich_obs_index],
        mint_to_meta=mint_to_meta,
        cutoffs=cutoffs,
        out_dir=rich_obs_dir,
        progress=progress,
        progress_start=58,
        progress_end=70,
        label="rich_obs",
    )
    # Also write a single consolidated path alias
    rich_parts = sorted(rich_obs_dir.glob("part-*.parquet"))
    rich_obs_path = out_dir / "observation_assignments_rich.parquet"
    if len(rich_parts) == 1:
        shutil.copy2(rich_parts[0], rich_obs_path)
    else:
        rich_obs_path = rich_obs_dir
    write_json(out_dir / "observation_assignments_rich_stats.json", rich_obs_stats)

    progress(72, "eval controls")
    eval_broad = build_eval_controls(token_table, population="broad")
    eval_rich = build_eval_controls(token_table, population="rich")
    eval_broad["observation_summary"] = attach_obs_eval_controls_summary(
        broad_obs_stats, population="broad"
    )
    eval_rich["observation_summary"] = attach_obs_eval_controls_summary(
        rich_obs_stats, population="rich"
    )
    write_json(out_dir / "eval_controls_broad.json", eval_broad)
    write_json(out_dir / "eval_controls_rich.json", eval_rich)
    write_creator_eval_table(token_table, out_dir / "eval_creator_flags.parquet")

    progress(80, "invariants")
    invariants = run_all_invariants(
        token_rows=token_rows,
        cutoffs=cutoffs,
        walkforward=walkforward,
        broad_obs_dir=broad_obs_dir,
        rich_obs_path=rich_obs_path,
    )
    leakage_audit = {
        "generated_at": utc_now(),
        "purge_rules": purge_reason_summary(),
        "invariants": invariants,
        "actual_split_percentages": actual_pct,
    }
    write_json(out_dir / "leakage_purge_audit.json", leakage_audit)

    if not invariants.get("all_passed"):
        raise RuntimeError(f"split invariants failed: {invariants}")

    progress(88, "publish SPLIT_CONTRACT/AUDIT_V1")
    contract = build_split_contract_v1(cutoffs=cutoffs, walkforward=walkforward)
    write_json(manifest_dir / "SPLIT_CONTRACT_V1.json", contract)
    write_json(out_dir / "SPLIT_CONTRACT_V1.json", contract)

    meta = {
        "generated_at": utc_now(),
        "version": SPLIT_STORE,
        "cutoffs": cutoffs,
        "walkforward": walkforward,
        "actual_split_percentages": actual_pct,
        "broad_observation_stats": broad_obs_stats,
        "rich_observation_stats": rich_obs_stats,
        "eval_controls_broad": eval_broad,
        "eval_controls_rich": eval_rich,
        "invariants": invariants,
        "artifacts": {
            "root": str(out_dir),
            "token_assignments": str(token_path),
            "observation_assignments_broad": str(broad_obs_dir),
            "observation_assignments_rich": str(rich_obs_path),
        },
    }
    write_json(out_dir / "split_manifest.json", meta)
    audit = publish_split_audit_v1(
        meta=meta,
        contract=contract,
        leakage_audit=leakage_audit,
        manifest_dir=manifest_dir,
        out_dir=out_dir,
    )
    progress(100, "splits complete")
    meta["audit_id"] = audit.get("audit_id")
    return meta
