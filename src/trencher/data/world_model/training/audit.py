"""TRAINING_LOADER_CONTRACT/AUDIT V3 + build orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from trencher.data.world_model.freeze import FREEZE_ID
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.sequences import WINDOW_PRESETS
from trencher.data.world_model.splits.progress import Progress100
from trencher.data.world_model.training.cache_build import build_training_cache_v1
from trencher.data.world_model.training.forbid import FORBIDDEN_FEATURE_KEYS
from trencher.data.world_model.training.schemas import modality_schema_manifest
from trencher.data.world_model.training.smoke import run_realistic_loader_benchmark

logger = logging.getLogger(__name__)

CONTRACT_ID = "TRAINING_LOADER_CONTRACT_V3"
AUDIT_ID = "TRAINING_LOADER_AUDIT_V3"


def build_training_loaders_v1(**kwargs: Any) -> dict[str, Any]:
    return build_training_loaders_v3(**kwargs)


def build_training_loaders_v2(**kwargs: Any) -> dict[str, Any]:
    return build_training_loaders_v3(**kwargs)


def build_training_loader_contract_v3() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_ID,
        "supersedes": "TRAINING_LOADER_CONTRACT_V2",
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "depends_on_splits": "SPLIT_CONTRACT_V1",
        "no_model_training": True,
        "runtime": "numpy_pyarrow_streaming_production_cached",
        "production_loader": True,
        "synthetic_path": "tests_only",
        "training_cache": "training_cache_v1",
        "tte_training_representation": {
            "fields": [
                "duration_sec",
                "event_observed",
                "censored",
                "at_risk",
                "effective_status",
                "split_boundary_censored",
            ],
            "rule_event_before_boundary": "duration=event-T; event_observed=1",
            "rule_event_at_or_after_boundary": (
                "duration=split_end-T; event_observed=0; censored=1; never negative"
            ),
            "rule_unresolved": "duration=earliest_bound-T; event_observed=0",
            "rule_already_occurred": "at_risk=0; left_truncated; not a new future event",
            "retain_censor_duration": True,
        },
        "transforms": {
            "order": "clip_raw_p01_p99_then_zscore",
            "fit_on": "unique_underlying_training_sequence_rows",
            "never_fit_on": ["val", "test"],
            "do_not_normalize": [
                "masks",
                "binary_flags",
                "categorical_event_edge_types",
                "graph_node_ids",
            ],
            "no_raw_unix_as_generic_feature": True,
        },
        "modality_schemas": modality_schema_manifest(),
        "target_families": [
            "returns",
            "barriers",
            "mfe_mae",
            "survival",
            "graduation_binary",
            "graduation_tte",
            "explosion_imminence",
            "explosion_tte",
            "explosion_magnitude",
            "persistence",
            "ownership",
            "execution",
            "future_path_refs",
        ],
        "forbidden_predictive_features": sorted(FORBIDDEN_FEATURE_KEYS),
        "window_presets": list(WINDOW_PRESETS),
        "engineering_targets_examples_per_sec": {"broad": 500, "rich": 100},
        "stores": {
            "broad": [
                "broad_sequence_events_v1",
                "broad_sequence_curve_v1",
                "broad_sequence_creator_v1",
                "broad_targets_v3",
                "tte_overlays_broad",
                "training_cache_v1",
            ],
            "rich": [
                "sequence_dataset_v2",
                "target_dataset_v3",
                "tte_overlays_rich",
                "pit/global",
                "training_cache_v1",
            ],
        },
    }


def build_training_loaders_v3(
    *,
    world_root: Path,
    manifest_dir: Path,
    progress: Progress100 | None = None,
    smoke_max_batches: int = 200,
    rebuild_cache: bool = True,
    light_smoke: bool = False,
) -> dict[str, Any]:
    progress = progress or Progress100()
    world_root = Path(world_root)
    manifest_dir = ensure_dir(manifest_dir)
    split_root = world_root / "splits" / "split_v1"
    inv = json.loads((split_root / "leakage_purge_audit.json").read_text(encoding="utf-8"))
    if not (inv.get("invariants") or {}).get("all_passed"):
        raise RuntimeError("split invariants not passed")

    cache_meta: dict[str, Any] = {}
    if rebuild_cache:
        progress(1, "build training_cache_v1")
        cache_meta = build_training_cache_v1(
            world_root=world_root, split_root=split_root, progress=progress
        )
    else:
        mpath = split_root / "training_cache_v1" / "manifest.json"
        if mpath.exists():
            cache_meta = json.loads(mpath.read_text(encoding="utf-8"))

    progress(91, "realistic loader benchmark")
    smoke = run_realistic_loader_benchmark(
        world_root=world_root,
        split_root=split_root,
        batch_sizes=(64, 128) if not light_smoke else (16,),
        max_batches=smoke_max_batches if not light_smoke else 20,
        seed=0,
        progress=progress,
        light=light_smoke,
    )
    write_json(split_root / "loader_smoke_benchmark_v3.json", smoke)

    progress(96, "publish TRAINING_LOADER_*_V3")
    contract = build_training_loader_contract_v3()
    write_json(manifest_dir / f"{CONTRACT_ID}.json", contract)
    write_json(split_root / f"{CONTRACT_ID}.json", contract)

    audit = {
        "audit_id": AUDIT_ID,
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "contract_version": CONTRACT_ID,
        "supersedes": "TRAINING_LOADER_AUDIT_V2",
        "training_cache": cache_meta,
        "modality_schemas": modality_schema_manifest(),
        "smoke_benchmark": smoke,
        "provisional_targets": {"broad_ex_per_sec": 500, "rich_ex_per_sec": 100},
        "notes": [
            "Frozen datasets/targets/cutoffs unchanged",
            "TTE censor durations retained with event_observed=0",
            "Transforms: clip raw p01/p99 then zscore",
            "No model training in this build",
        ],
    }
    write_json(manifest_dir / f"{AUDIT_ID}.json", audit)
    write_json(split_root / f"{AUDIT_ID}.json", audit)
    md_lines = [
        f"# Training Loader Audit (`{AUDIT_ID}`)",
        "",
        f"Generated: `{audit['generated_at']}`",
        "",
        "## Cache",
        "",
        f"`{cache_meta}`",
        "",
        "## Benchmark",
        "",
        f"- peak_rss_mb: **{smoke.get('peak_rss_mb')}**",
        f"- broad: `{smoke.get('broad')}`",
        f"- rich: `{smoke.get('rich')}`",
        f"- mixed: `{smoke.get('mixed')}`",
        "",
    ]
    md = "\n".join(md_lines)
    (manifest_dir / f"{AUDIT_ID}.md").write_text(md, encoding="utf-8")
    (split_root / f"{AUDIT_ID}.md").write_text(md, encoding="utf-8")
    progress(100, "training loaders v3 complete")
    return {"contract": contract, "audit": audit, "smoke": smoke, "cache": cache_meta}
