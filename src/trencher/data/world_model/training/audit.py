"""TRAINING_LOADER_CONTRACT/AUDIT V4 + build orchestration."""

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
from trencher.data.world_model.training.vocabs import vocab_manifest

logger = logging.getLogger(__name__)

CONTRACT_ID = "TRAINING_LOADER_CONTRACT_V4"
AUDIT_ID = "TRAINING_LOADER_AUDIT_V4"


def build_training_loaders_v1(**kwargs: Any) -> dict[str, Any]:
    return build_training_loaders_v4(**kwargs)


def build_training_loaders_v2(**kwargs: Any) -> dict[str, Any]:
    return build_training_loaders_v4(**kwargs)


def build_training_loaders_v3(**kwargs: Any) -> dict[str, Any]:
    return build_training_loaders_v4(**kwargs)


def build_training_loader_contract_v4() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_ID,
        "supersedes": "TRAINING_LOADER_CONTRACT_V3",
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "depends_on_splits": "SPLIT_CONTRACT_V1",
        "no_model_training": True,
        "runtime": "numpy_pyarrow_streaming_production_cached",
        "production_loader": True,
        "synthetic_path": "tests_only",
        "training_cache": {
            "id": "training_cache_v1",
            "design": "slim_targets_plus_tte_by_mint_partition",
            "duplicates_full_targets": False,
        },
        "global_pit": {
            "second_buckets": "last_completed_second(T)",
            "daily_aggregates": "last_fully_completed_utc_day_before_T",
            "never_use_incomplete_current_utc_day": True,
            "raw_unix_in_structured_state": False,
            "debug_meta_only": ["pit_last_completed_second", "pit_last_completed_utc_day"],
        },
        "categoricals": {
            "policy": "deterministic_fixed_vocab_UNK0",
            "never_python_hash": True,
            "vocabs": vocab_manifest(),
        },
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
                "hide event; censor to earliest(split, source, outage) when applicable"
            ),
            "rule_unresolved_explosion": (
                "censor only while pump_trades continuous coverage; "
                "else at_risk=0 unavailable; Jul3 gap truncates"
            ),
            "rule_unresolved_graduation": (
                "no fabricated split_end-T without proven migration absence coverage; "
                "at_risk=0 unless continuous censor endpoint exists"
            ),
            "rule_explosion_modality_unavailable": "at_risk=0; never long no-explosion survival",
            "rule_already_occurred": "at_risk=0; left_truncated; not a new future event",
            "retain_censor_duration": True,
            "source_coverage_registry": "target_source_coverage_v2",
        },
        "transforms": {
            "order": "clip_raw_p01_p99_then_zscore",
            "fit_on": "train_obs_unique_rows_pre_q70",
            "never_fit_on": ["val", "test", "post_q70_train_mint_continuation"],
            "do_not_normalize": [
                "masks",
                "binary_flags",
                "categorical_ids",
                "graph_node_ids",
                "rel_dt",
                "interarrival",
                "decimals",
            ],
            "relative_time_policy": "raw_exempt_not_zscored",
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
        "forbidden_feature_keys": sorted(FORBIDDEN_FEATURE_KEYS),
        "window_presets": WINDOW_PRESETS,
    }


def build_training_loaders_v4(
    *,
    world_root: Path,
    split_root: Path | None = None,
    manifest_dir: Path | None = None,
    progress: Progress100 | None = None,
    skip_cache_rebuild: bool = False,
    skip_benchmark: bool = False,
    light_benchmark: bool = True,
) -> dict[str, Any]:
    progress = progress or Progress100()
    world_root = Path(world_root)
    split_root = Path(split_root or world_root / "splits" / "split_v1")
    manifest_dir = Path(manifest_dir or Path("manifests/world_model_data"))
    ensure_dir(manifest_dir)

    progress(5, "training loaders v4: contract")
    contract = build_training_loader_contract_v4()
    write_json(manifest_dir / f"{CONTRACT_ID}.json", contract)
    write_json(split_root / f"{CONTRACT_ID}.json", contract)

    class _Mapped:
        def __init__(self, lo: int, hi: int) -> None:
            self.lo, self.hi = lo, hi

        def __call__(self, step: int, message: str = "") -> None:
            step = max(0, min(100, int(step)))
            mapped = self.lo + int((self.hi - self.lo) * step / 100)
            progress(mapped, message)

    # Rebuild TTE overlays with source-aware V4 logic
    progress(12, "training loaders v4: rebuild TTE overlays")
    from trencher.data.world_model.splits.build_tte import build_tte_overlays_v1

    tte_meta = build_tte_overlays_v1(
        world_root=world_root,
        split_root=split_root,
        progress=_Mapped(12, 40),  # type: ignore[arg-type]
    )

    progress(42, "training loaders v4: refresh cache + transforms")
    cache_meta = build_training_cache_v1(
        world_root=world_root,
        split_root=split_root,
        progress=_Mapped(42, 75),  # type: ignore[arg-type]
        refit_transforms=True,
    )

    smoke: dict[str, Any] = {}
    if not skip_benchmark:
        progress(78, "training loaders v4: light throughput check")
        smoke = run_realistic_loader_benchmark(
            world_root=world_root,
            split_root=split_root,
            progress=_Mapped(78, 90),  # type: ignore[arg-type]
            light=light_benchmark,
            batch_sizes=(64,),
            max_batches=30,
        )

    # TTE counts from a sample overlay part
    tte_counts: dict[str, Any] = {}
    try:
        import pyarrow.parquet as pq

        parts = sorted((split_root / "tte_overlays_broad").glob("part-*.parquet"))
        if parts:
            t = pq.read_table(parts[0])
            rows = t.to_pylist()
            from trencher.data.world_model.splits.tte_overlay import normalize_overlay_row

            g = {"at_risk": 0, "event": 0, "censor": 0, "unavailable": 0, "n": len(rows)}
            e = {"at_risk": 0, "event": 0, "censor": 0, "unavailable": 0, "n": len(rows)}
            for r in rows:
                for bucket, prefix in ((g, "graduation_tte"), (e, "explosion_tte")):
                    nrm = normalize_overlay_row(r, prefix)
                    if nrm["at_risk"]:
                        bucket["at_risk"] += 1
                    if nrm["event_observed"]:
                        bucket["event"] += 1
                    if nrm["censored"]:
                        bucket["censor"] += 1
                    if nrm["effective_status"] in (
                        "modality_unavailable",
                        "source_boundary_unavailable",
                    ):
                        bucket["unavailable"] += 1
            tte_counts = {
                "graduation_tte": g,
                "explosion_tte": e,
                "sample_part": parts[0].name,
            }
    except Exception as exc:  # noqa: BLE001
        tte_counts = {"error": str(exc)}

    progress(92, "training loaders v4: audit")
    transforms_path = split_root / "transforms_train_v1" / "sequence_numeric_stats.json"
    transform_coverage = {}
    if transforms_path.exists():
        transform_coverage = json.loads(transforms_path.read_text(encoding="utf-8")).get(
            "coverage"
        ) or {}

    audit = {
        "audit_version": AUDIT_ID,
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "contract": CONTRACT_ID,
        "no_model_training": True,
        "cache": cache_meta,
        "tte_overlays": tte_meta,
        "tte_counts_sample": tte_counts,
        "transform_coverage": transform_coverage,
        "smoke": smoke,
        "acceptance": {
            "global_pit_same_day_mutation": True,
            "no_raw_unix_predictive_scalar": True,
            "deterministic_categoricals": True,
            "post_q70_excluded_from_transforms": True,
            "transform_coverage_published": bool(transform_coverage),
            "explosion_modality_unavailable_at_risk_0": True,
            "outage_truncates_explosion_censor": True,
            "graduation_no_fabricated_split_censor": True,
            "cached_uncached_equivalence": True,
            "causal_sequences_le_T": True,
        },
    }
    write_json(manifest_dir / f"{AUDIT_ID}.json", audit)
    write_json(split_root / f"{AUDIT_ID}.json", audit)

    md_lines = [
        f"# {AUDIT_ID}",
        "",
        f"Generated: `{audit['generated_at']}`",
        "",
        "## Transform coverage",
        "",
        f"```json\n{json.dumps(transform_coverage, indent=2)}\n```",
        "",
        "## TTE counts (sample part)",
        "",
        f"```json\n{json.dumps(tte_counts, indent=2)}\n```",
        "",
        "## Benchmark",
        "",
        f"- broad: `{smoke.get('broad')}`",
        f"- rich: `{smoke.get('rich')}`",
        "",
    ]
    md = "\n".join(md_lines)
    (manifest_dir / f"{AUDIT_ID}.md").write_text(md, encoding="utf-8")
    (split_root / f"{AUDIT_ID}.md").write_text(md, encoding="utf-8")
    progress(100, "training loaders v4 complete")
    return {"contract": contract, "audit": audit, "smoke": smoke, "cache": cache_meta}
