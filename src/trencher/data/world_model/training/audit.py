"""TRAINING_LOADER_CONTRACT/AUDIT V2 + build orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trencher.data.world_model.freeze import FREEZE_ID
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.sequences import WINDOW_PRESETS
from trencher.data.world_model.splits.build_tte import build_tte_overlays_v1
from trencher.data.world_model.splits.cutoffs import SPLIT_TRAIN
from trencher.data.world_model.splits.progress import Progress100
from trencher.data.world_model.training.forbid import FORBIDDEN_FEATURE_KEYS
from trencher.data.world_model.training.smoke import run_production_loader_smoke
from trencher.data.world_model.training.transforms import (
    fit_sequence_transforms_from_unique_rows,
    persist_transforms,
)

logger = logging.getLogger(__name__)

CONTRACT_ID = "TRAINING_LOADER_CONTRACT_V2"
AUDIT_ID = "TRAINING_LOADER_AUDIT_V2"


def build_training_loaders_v1(**kwargs: Any) -> dict[str, Any]:
    """Back-compat alias → v2 production path."""
    return build_training_loaders_v2(**kwargs)


def build_training_loader_contract_v2() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_ID,
        "supersedes": "TRAINING_LOADER_CONTRACT_V1",
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "depends_on_splits": "SPLIT_CONTRACT_V1",
        "no_model_training": True,
        "runtime": "numpy_pyarrow_streaming_production",
        "production_loader": True,
        "synthetic_path": "tests_only",
        "tte_admin_censor": {
            "rule": "event_time < split_end may remain observed; else hide + right_censor",
            "never_admin_censor_to_negative": True,
            "test_bound": "CORPUS_END_UNIX",
        },
        "forbidden_predictive_features": sorted(FORBIDDEN_FEATURE_KEYS),
        "train_only_transforms": {
            "fit_on": "unique_underlying_training_sequence_rows",
            "apply_on": "production_sequence_numeric_channels",
            "never_fit_on": ["val", "test"],
            "no_raw_unix_as_generic_feature": True,
        },
        "window_presets": list(WINDOW_PRESETS),
        "stores": {
            "broad": [
                "broad_sequence_events_v1",
                "broad_sequence_curve_v1",
                "broad_sequence_creator_v1",
                "broad_targets_v3",
                "tte_overlays_broad",
            ],
            "rich": [
                "sequence_dataset_v2",
                "target_dataset_v3",
                "tte_overlays_rich",
            ],
        },
    }


def build_training_loaders_v2(
    *,
    world_root: Path,
    manifest_dir: Path,
    progress: Progress100 | None = None,
    smoke_max_batches: int = 30,
    rebuild_tte: bool = True,
) -> dict[str, Any]:
    progress = progress or Progress100()
    world_root = Path(world_root)
    manifest_dir = ensure_dir(manifest_dir)
    split_root = world_root / "splits" / "split_v1"
    inv = json.loads((split_root / "leakage_purge_audit.json").read_text(encoding="utf-8"))
    if not (inv.get("invariants") or {}).get("all_passed"):
        raise RuntimeError("split invariants not passed")

    if rebuild_tte:
        progress(2, "build TTE overlays")
        tte_meta = build_tte_overlays_v1(
            world_root=world_root, split_root=split_root, progress=progress
        )
    else:
        tte_meta = json.loads(
            (split_root / "tte_overlays_manifest.json").read_text(encoding="utf-8")
        )

    # Patch SPLIT_AUDIT with TTE overlay stats (additive)
    split_audit_path = manifest_dir / "SPLIT_AUDIT_V1.json"
    if split_audit_path.exists():
        audit = json.loads(split_audit_path.read_text(encoding="utf-8"))
        audit["tte_admin_censor_overlays"] = tte_meta
        write_json(split_audit_path, audit)
        write_json(split_root / "SPLIT_AUDIT_V1.json", audit)

    progress(55, "refit train sequence transforms")
    tokens = pq.read_table(split_root / "token_assignments.parquet").to_pylist()
    train_mints = {
        r["mint"]
        for r in tokens
        if r["base_split"] == SPLIT_TRAIN and r.get("in_broad_event_curve")
    }
    events_root = world_root / "broad" / "broad_sequence_events_v1"
    parts = sorted(events_root.rglob("part.parquet"))
    fit_parts = parts[:: max(1, len(parts) // 32)][:32] if parts else []
    numeric_cols = [
        c
        for c in ("sol_amount", "token_amount", "price_sol", "v_sol", "v_tok")
        if fit_parts and c in pq.read_schema(fit_parts[0]).names
    ]
    transforms = fit_sequence_transforms_from_unique_rows(
        sequence_parts=fit_parts,
        train_mints=train_mints,
        numeric_columns=numeric_cols,
        max_rows_per_part=200_000,
    )
    tdir = ensure_dir(split_root / "transforms_train_v1")
    persist_transforms(tdir / "sequence_numeric_stats.json", transforms)

    progress(65, "production loader smoke")
    smoke = run_production_loader_smoke(
        world_root=world_root,
        split_root=split_root,
        batch_size=16,
        max_batches=smoke_max_batches,
        seed=0,
        progress=progress,
    )
    write_json(split_root / "loader_smoke_benchmark_v2.json", smoke)

    progress(92, "publish TRAINING_LOADER_*_V2")
    contract = build_training_loader_contract_v2()
    write_json(manifest_dir / f"{CONTRACT_ID}.json", contract)
    write_json(split_root / f"{CONTRACT_ID}.json", contract)

    audit = {
        "audit_id": AUDIT_ID,
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "contract_version": CONTRACT_ID,
        "supersedes": "TRAINING_LOADER_AUDIT_V1",
        "tte_overlays": tte_meta,
        "transforms": {
            "rows_seen": transforms.get("rows_seen"),
            "train_mints_n": transforms.get("train_mints_n"),
            "columns": list((transforms.get("columns") or {}).keys()),
            "fit_scope": transforms.get("fit_scope"),
        },
        "smoke_benchmark": smoke,
    }
    write_json(manifest_dir / f"{AUDIT_ID}.json", audit)
    write_json(split_root / f"{AUDIT_ID}.json", audit)
    md = (
        f"# Training Loader Audit (`{AUDIT_ID}`)\n\n"
        f"Generated: `{audit['generated_at']}`\n\n"
        f"## TTE overlays\n\n`{tte_meta.get('stats')}`\n\n"
        f"## Smoke\n\n"
        f"- peak_rss_mb: **{smoke.get('peak_rss_mb')}**\n"
        f"- broad: `{smoke.get('broad')}`\n"
        f"- rich: `{smoke.get('rich')}`\n"
    )
    (manifest_dir / f"{AUDIT_ID}.md").write_text(md, encoding="utf-8")
    (split_root / f"{AUDIT_ID}.md").write_text(md, encoding="utf-8")
    progress(100, "training loaders v2 complete")
    return {"contract": contract, "audit": audit, "smoke": smoke, "tte": tte_meta}
