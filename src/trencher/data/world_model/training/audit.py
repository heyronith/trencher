"""TRAINING_LOADER_CONTRACT_V1 / AUDIT publishers + fit transforms."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trencher.data.world_model.freeze import FREEZE_ID
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.sequences import WINDOW_PRESETS
from trencher.data.world_model.splits.cutoffs import SPLIT_TRAIN
from trencher.data.world_model.splits.progress import Progress100
from trencher.data.world_model.training.forbid import FORBIDDEN_FEATURE_KEYS
from trencher.data.world_model.training.smoke import run_loader_smoke
from trencher.data.world_model.training.transforms import (
    fit_sequence_transforms_from_unique_rows,
    persist_transforms,
)

logger = logging.getLogger(__name__)

CONTRACT_ID = "TRAINING_LOADER_CONTRACT_V1"
AUDIT_ID = "TRAINING_LOADER_AUDIT_V1"


def build_training_loader_contract_v1() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_ID,
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "depends_on_splits": "SPLIT_CONTRACT_V1",
        "no_model_training": True,
        "runtime": "numpy_pyarrow_streaming",
        "strategies": [
            "broad_event_curve_pretrain",
            "rich_multimodal",
            "multimodal_fusion",
            "mixed_broad_rich",
        ],
        "forbidden_predictive_features": sorted(FORBIDDEN_FEATURE_KEYS),
        "train_only_transforms": {
            "fit_on": "unique_underlying_training_sequence_rows",
            "apply_on": "all_splits_frozen_params",
            "never_fit_on": ["val", "test", "walkforward_val", "final_test"],
        },
        "window_presets": list(WINDOW_PRESETS),
        "target_policy": (
            "load missing/censored explicitly; never fill with numeric zero; "
            "respect head-specific split eligibility masks"
        ),
        "sampling_policy": (
            "default reflects true population distribution; "
            "no positive/outcome-based canonical sampling"
        ),
        "graph_ids_policy": (
            "wallet/node IDs may be used only as graph indexing machinery, "
            "not generic predictive categorical features"
        ),
    }


def build_training_loaders_v1(
    *,
    world_root: Path,
    manifest_dir: Path,
    progress: Progress100 | None = None,
    smoke_max_batches: int = 20,
) -> dict[str, Any]:
    """Fit train transforms + run CPU smoke. Requires split invariants already passed."""
    progress = progress or Progress100()
    world_root = Path(world_root)
    manifest_dir = ensure_dir(manifest_dir)
    split_root = world_root / "splits" / "split_v1"
    inv_path = split_root / "leakage_purge_audit.json"
    if not inv_path.exists():
        raise FileNotFoundError("Part A splits missing; refuse to build loaders")
    import json

    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    if not (inv.get("invariants") or {}).get("all_passed"):
        raise RuntimeError("Part A invariants not passed; refusing Part B")

    progress(5, "load train mints")
    tokens = pq.read_table(split_root / "token_assignments.parquet").to_pylist()
    train_mints = {
        r["mint"]
        for r in tokens
        if r["base_split"] == SPLIT_TRAIN and r.get("in_broad_event_curve")
    }

    progress(15, "fit sequence transforms on unique train rows")
    # Broad events: sample partitions for fit (unique rows within scanned parts)
    events_root = world_root / "broad" / "broad_sequence_events_v1"
    parts = sorted(events_root.rglob("*.parquet"))
    # Prefer entity-bearing trade-like columns if present
    numeric_cols = []
    if parts:
        names = set(pq.read_schema(parts[0]).names)
        for c in (
            "sol_amount",
            "token_amount",
            "v_sol_bonding_curve",
            "v_tokens_bonding_curve",
            "price",
        ):
            if c in names:
                numeric_cols.append(c)
    transform_dir = ensure_dir(split_root / "transforms_train_v1")
    # Cap parts for fit scalability while still unique-row semantics within scanned set
    fit_parts = parts[:: max(1, len(parts) // 32)][:32] if parts else []
    transforms = fit_sequence_transforms_from_unique_rows(
        sequence_parts=fit_parts,
        train_mints=train_mints,
        numeric_columns=numeric_cols or ["price"],
        max_rows_per_part=200_000,
    )
    persist_transforms(transform_dir / "sequence_numeric_stats.json", transforms)
    progress(50, "transforms persisted")

    progress(60, "CPU loader smoke")
    smoke = run_loader_smoke(
        split_root=split_root, batch_size=64, max_batches=smoke_max_batches, seed=0
    )
    write_json(split_root / "loader_smoke_benchmark.json", smoke)
    progress(85, "publish loader contract/audit")

    contract = build_training_loader_contract_v1()
    write_json(manifest_dir / f"{CONTRACT_ID}.json", contract)
    write_json(split_root / f"{CONTRACT_ID}.json", contract)

    audit = {
        "audit_id": AUDIT_ID,
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "contract_version": CONTRACT_ID,
        "gated_on_split_invariants": True,
        "transforms": {
            "path": str(transform_dir / "sequence_numeric_stats.json"),
            "rows_seen": transforms.get("rows_seen"),
            "train_mints_n": transforms.get("train_mints_n"),
            "columns": list((transforms.get("columns") or {}).keys()),
            "fit_scope": transforms.get("fit_scope"),
        },
        "smoke_benchmark": smoke,
        "window_presets": list(WINDOW_PRESETS),
    }
    write_json(manifest_dir / f"{AUDIT_ID}.json", audit)
    write_json(split_root / f"{AUDIT_ID}.json", audit)
    md = render_loader_audit_md(audit)
    (manifest_dir / f"{AUDIT_ID}.md").write_text(md, encoding="utf-8")
    (split_root / f"{AUDIT_ID}.md").write_text(md, encoding="utf-8")
    progress(100, "training loaders complete")
    return {"contract": contract, "audit": audit, "transforms": transforms, "smoke": smoke}


def render_loader_audit_md(audit: dict[str, Any]) -> str:
    smoke = audit.get("smoke_benchmark") or {}
    lines = [
        f"# Training Loader Audit (`{audit['audit_id']}`)",
        "",
        f"Generated at: `{audit['generated_at']}`",
        "",
        "## Transforms",
        "",
        f"- `{audit.get('transforms')}`",
        "",
        "## Smoke benchmark",
        "",
        f"- peak_rss_mb: **{smoke.get('peak_rss_mb')}**",
        f"- broad: `{smoke.get('broad')}`",
        f"- rich: `{smoke.get('rich')}`",
        f"- mixed: `{smoke.get('mixed')}`",
        "",
    ]
    return "\n".join(lines)
