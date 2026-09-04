"""Build target_dataset_v2 artifacts, contracts, and audits.

``target_dataset_v1`` artifacts are preserved on disk for audit history and are
not overwritten by this module.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.freeze import FREEZE_ID
from trencher.data.world_model.io import ensure_dir, file_provenance, read_json, utc_now, write_json
from trencher.data.world_model.targets.compute import (
    compute_activity_survival,
    compute_execution,
    compute_explosion_obs,
    compute_graduation,
    compute_ownership_targets,
    compute_persistence,
    compute_return_barrier_mfe,
    future_path_refs_for_obs,
    reference_price_at_t,
)
from trencher.data.world_model.targets.coverage import (
    build_source_coverage_registry,
    write_coverage_manifest,
)
from trencher.data.world_model.targets.explosion import (
    DETECTOR_VERSION,
    PRIMARY_DEFINITION_ID,
    detect_explosions_for_mint,
    detector_config_manifest,
    primary_event,
)
from trencher.data.world_model.targets.future_path import (
    FUTURE_PATH_VERSION,
    build_future_market_path_v1,
    load_migration_unix,
    load_path_arrays,
)
from trencher.data.world_model.targets.status import (
    BARRIER_LOWER_FIRST,
    BARRIER_NEITHER,
    BARRIER_PAIRS,
    BARRIER_UPPER_FIRST,
    EXPLOSION_IMMINENCE_HORIZONS_SEC,
    HORIZONS_SEC,
    POSITION_SIZES_SOL,
    STATUS_KNOWN_SOURCE_OUTAGE,
    STATUS_RIGHT_CENSORED,
    STATUS_SOURCE_BOUNDARY_UNAVAILABLE,
    STATUS_VALID_NEGATIVE,
    STATUS_VALID_POSITIVE,
    SURVIVAL_HORIZONS_SEC,
    status_contract,
)

logger = logging.getLogger(__name__)

TARGET_DATASET_V1 = "target_dataset_v1"
TARGET_DATASET_V2 = "target_dataset_v2"
TARGET_CONTRACT_VERSION = "target_contract_v2"
PIT_ENGINE_VERSION = "pit_v4"
OBSERVATION_DATASET_VERSION = "observation_dataset_v1"
SEQUENCE_DATASET_VERSION = "sequence_dataset_v2"
MODEL_INPUT_CONTRACT_VERSION = "model_input_contract_v2"

INPUT_MODULES_FORBIDDEN_TO_IMPORT_TARGETS = (
    "pit.py",
    "observations.py",
    "sequences.py",
    "pit_audit.py",
    "ownership.py",
)


def build_target_contract_v2() -> dict[str, Any]:
    return {
        "contract_version": TARGET_CONTRACT_VERSION,
        "generated_at": utc_now(),
        "freeze_id": FREEZE_ID,
        "pit_engine_version": PIT_ENGINE_VERSION,
        "observation_dataset_version": OBSERVATION_DATASET_VERSION,
        "sequence_dataset_version": SEQUENCE_DATASET_VERSION,
        "model_input_contract_version": MODEL_INPUT_CONTRACT_VERSION,
        "target_dataset_version": TARGET_DATASET_V2,
        "supersedes": TARGET_DATASET_V1,
        "label_only": True,
        "causal_rule": "everything knowable at T -> outcomes strictly after T (event_time > T)",
        "survival_definition": "last_genuine_future_activity_ge_T_plus_H",
        "absence_policy": (
            "No-event/absence targets use centralized source coverage registry; "
            "never mint last_path_t; Jul 3 outage never creates death labels."
        ),
        "isolation": {
            "pit_input_modules_must_not_import_targets": True,
            "target_fields_never_enter_frozen_input_artifacts": True,
            "target_artifact_ids_not_structured_features": True,
        },
        "status_contract": status_contract(),
        "explosion_detector_version": DETECTOR_VERSION,
        "explosion_primary_definition": PRIMARY_DEFINITION_ID,
        "horizons_sec": list(HORIZONS_SEC),
        "families": [
            "returns",
            "barriers",
            "mfe_mae",
            "survival_inactivity",
            "graduation",
            "activity_persistence",
            "trend_path",
            "execution_adjusted",
            "explosion_ignition",
            "ownership_sparse",
        ],
        "execution_semantics": {
            "status_is_availability_not_return_sign": True,
            "curve_impact_adjusted_return": "round_trip_amm_when_entry_and_exit_reserves",
            "entry_impact_adjusted_return": "entry_amm_only_vs_future_mark",
            "net_return": "unavailable_without_row_level_fee_schedule",
        },
        "chronological_splits": "not_created",
        "training": "not_started",
        "artifacts": [
            "future_market_path_v1.parquet",
            "observation_targets_v2.parquet",
            "future_path_refs_v2.parquet",
            "explosion_events_v1.parquet",
            "explosion_detector_config_v1.json",
            "target_source_coverage_v1.json",
            "TARGET_CONTRACT_V2.json",
            "TARGET_DATASET_V2_AUDIT.json",
            "EXPLOSION_TARGET_AUDIT_V2.json",
        ],
    }


def assert_input_modules_do_not_import_targets(package_dir: Path | None = None) -> dict[str, Any]:
    """Structural proof: PIT/input modules do not import the targets package."""
    root = package_dir or Path(__file__).resolve().parents[1]
    violations: list[str] = []
    checked: list[str] = []
    for name in INPUT_MODULES_FORBIDDEN_TO_IMPORT_TARGETS:
        path = root / name
        if not path.exists():
            continue
        checked.append(str(path))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "world_model.targets" in alias.name or alias.name.endswith(
                        ".targets"
                    ):
                        violations.append(f"{name}: import {alias.name}")
                    if "future_labels" in alias.name:
                        violations.append(f"{name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if "world_model.targets" in mod or mod.endswith(".targets"):
                    violations.append(f"{name}: from {mod} import ...")
                if "future_labels" in mod:
                    # PIT must never import future_labels.
                    violations.append(f"{name}: from {mod} import ...")
    return {
        "checked_modules": checked,
        "violations": violations,
        "ok": len(violations) == 0,
    }


def _quantiles(vals: list[float]) -> dict[str, float | None]:
    if not vals:
        return {"n": 0, "p50": None, "p90": None, "p99": None, "min": None, "max": None}
    s = sorted(vals)
    def q(p: float) -> float:
        idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        return float(s[idx])
    return {
        "n": len(s),
        "p50": q(0.5),
        "p90": q(0.9),
        "p99": q(0.99),
        "min": float(s[0]),
        "max": float(s[-1]),
        "mean": float(statistics.fmean(s)),
    }


def _sufficiency(
    *,
    name: str,
    n_obs: int,
    n_tokens: int,
    n_positive_tokens: int,
    n_independent_events: int,
    prevalence: float | None,
    censor_rate: float | None,
    recommendation: str,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "head": name,
        "observations": n_obs,
        "unique_tokens_with_valid_supervision": n_tokens,
        "distinct_positive_tokens": n_positive_tokens,
        "independent_event_count": n_independent_events,
        "prevalence": prevalence,
        "effective_censoring": censor_rate,
        "recommendation": recommendation,
        "notes": notes,
    }


def build_target_dataset_v1(**_kwargs: Any) -> dict[str, Any]:
    """v1 artifacts are frozen for audit history — use ``build_target_dataset_v2``."""
    raise RuntimeError(
        "target_dataset_v1 is frozen; call build_target_dataset_v2 "
        "(v1 artifacts remain under data/.../target_dataset_v1/)."
    )


def build_target_dataset_v2(
    *,
    world_root: Path,
    manifest_dir: Path,
    pump_root: Path | None = None,
    observation_index_path: Path | None = None,
) -> dict[str, Any]:
    """Materialize label-only targets v2 for all approved observations."""
    pump_root = pump_root or Path("data/external/pumpfun_memecoin_corpus")
    out_dir = ensure_dir(world_root / "targets" / TARGET_DATASET_V2)
    manifest_dir = ensure_dir(manifest_dir)

    obs_path = observation_index_path or (
        world_root
        / "observations"
        / OBSERVATION_DATASET_VERSION
        / "observation_index_v1.parquet"
    )
    observations = pq.read_table(obs_path).to_pylist()
    mints = {r["mint"] for r in observations if r.get("mint")}
    logger.info(
        "Building target_dataset_v2 for %d observations / %d mints",
        len(observations),
        len(mints),
    )

    coverage = build_source_coverage_registry(
        world_root=world_root, pump_root=pump_root
    )
    cov_manifest = write_coverage_manifest(
        coverage, out_dir / "target_source_coverage_v1.json"
    )
    write_json(manifest_dir / "target_source_coverage_v1.json", cov_manifest)

    # 1) Future market path (label-only; schema unchanged)
    path_meta = build_future_market_path_v1(
        world_root=world_root,
        pump_root=pump_root,
        mints=mints,
        out_dir=out_dir,
    )
    path_parquet = out_dir / f"{FUTURE_PATH_VERSION}.parquet"
    entity_index = out_dir / f"{FUTURE_PATH_VERSION}_entity_index.parquet"
    by_mint, entity = load_path_arrays(path_parquet, entity_index)
    migration_unix = load_migration_unix(
        world_root=world_root, pump_root=pump_root, mints=mints
    )

    own_path = world_root / "universe" / "ownership_evidence.parquet"
    own_by_mint: dict[str, dict[str, Any]] = {}
    if own_path.exists():
        for row in pq.read_table(own_path).to_pylist():
            if row.get("mint") in mints:
                own_by_mint[row["mint"]] = row

    logger.info("Detecting explosion events (prespecified detector family)")
    explosion_rows: list[dict[str, Any]] = []
    primary_by_mint: dict[str, dict[str, Any]] = {}
    all_defs_by_mint: dict[str, list[dict[str, Any]]] = {}
    for mint, arr in by_mint.items():
        events = detect_explosions_for_mint(mint, arr)
        all_defs_by_mint[mint] = events
        for ev in events:
            flat = {k: v for k, v in ev.items() if k != "precursor"}
            prec = ev.get("precursor") or {}
            flat["precursor_json"] = json.dumps(prec, default=str)
            for pk, pv in prec.items():
                if isinstance(pv, dict):
                    continue
                flat[f"precursor_{pk}"] = pv
            explosion_rows.append(flat)
        prim = primary_event(events)
        if prim:
            primary_by_mint[mint] = prim

    expl_path = out_dir / "explosion_events_v1.parquet"
    if explosion_rows:
        pq.write_table(pa.Table.from_pylist(explosion_rows), expl_path)
    else:
        pq.write_table(pa.table({"mint": pa.array([], pa.string())}), expl_path)

    det_cfg = detector_config_manifest()
    write_json(out_dir / "explosion_detector_config_v1.json", det_cfg)
    write_json(manifest_dir / "explosion_detector_config_v1.json", det_cfg)

    logger.info("Computing observation-level targets (v2 semantics)")
    target_rows: list[dict[str, Any]] = []
    ref_rows: list[dict[str, Any]] = []

    for n, obs in enumerate(observations):
        if n and n % 5000 == 0:
            logger.info("  targets progress %d / %d", n, len(observations))
        mint = obs["mint"]
        T = float(obs["observation_time_t"])
        oid = obs["observation_id"]
        arr = by_mint.get(mint)
        last_path_t = None
        if mint in entity:
            last_path_t = entity[mint].get("last_event_unix")
        mig = migration_unix.get(mint)

        row: dict[str, Any] = {
            "observation_id": oid,
            "mint": mint,
            "observation_time_t": T,
            "token_age_seconds": obs.get("token_age_seconds"),
            "observation_trigger": obs.get("observation_trigger"),
            "trigger_value": obs.get("trigger_value"),
            "launch_day": obs.get("launch_day"),
            "launch_regime": obs.get("launch_regime"),
            "target_dataset_version": TARGET_DATASET_V2,
            "label_only": True,
        }

        if arr is None:
            ref = {
                "reference_price": None,
                "reference_price_time": None,
                "reference_price_source": None,
                "reference_price_age_sec": None,
                "price_target_available": False,
            }
            row.update(ref)
            empty = {
                "t": np.asarray([], dtype=np.float64),
                "price": np.asarray([], dtype=np.float64),
                "regime": np.asarray([], dtype=np.int8),
                "is_buy": np.asarray([], dtype=np.int8),
                "sol": np.asarray([], dtype=np.float64),
                "token": np.asarray([], dtype=np.float64),
                "wallet": np.asarray([], dtype=object),
                "v_sol": np.asarray([], dtype=np.float64),
                "v_tok": np.asarray([], dtype=np.float64),
                "curve": np.asarray([], dtype=np.float64),
                "price_source": np.asarray([], dtype=object),
                "path_idx": np.asarray([], dtype=np.int64),
                "usable_price": np.asarray([], dtype=bool),
            }
            arr = empty
        else:
            ref = reference_price_at_t(arr, T)
            row.update({k: ref[k] for k in (
                "reference_price",
                "reference_price_time",
                "reference_price_source",
                "reference_price_age_sec",
                "price_target_available",
            )})
            row["reference_regime"] = ref.get("reference_regime")

        row.update(
            compute_return_barrier_mfe(
                arr, T=T, ref=ref, last_path_t=last_path_t
            )
        )
        row.update(
            compute_activity_survival(
                arr, T=T, coverage=coverage, migration_unix=mig
            )
        )
        row.update(
            compute_graduation(T=T, migration_unix=mig, coverage=coverage)
        )
        row.update(compute_persistence(arr, T=T, coverage=coverage))
        row.update(
            compute_explosion_obs(
                T=T,
                primary=primary_by_mint.get(mint),
                all_defs=all_defs_by_mint.get(mint, []),
                coverage=coverage,
                ref=ref,
            )
        )
        row.update(compute_execution(arr, T=T, ref=ref))
        row.update(
            compute_ownership_targets(
                mint=mint, T=T, ownership_meta=own_by_mint.get(mint)
            )
        )

        target_rows.append(row)
        ref_rows.append(
            {
                "observation_id": oid,
                "mint": mint,
                "observation_time_t": T,
                "future_path_store": FUTURE_PATH_VERSION,
                **future_path_refs_for_obs(arr, T=T),
            }
        )

    targets_path = out_dir / "observation_targets_v2.parquet"
    refs_path = out_dir / "future_path_refs_v2.parquet"
    logger.info("Writing observation_targets_v2 (%d rows)", len(target_rows))
    pq.write_table(pa.Table.from_pylist(target_rows), targets_path)
    pq.write_table(pa.Table.from_pylist(ref_rows), refs_path)

    contract = build_target_contract_v2()
    write_json(out_dir / "TARGET_CONTRACT_V2.json", contract)
    write_json(manifest_dir / "TARGET_CONTRACT_V2.json", contract)

    isolation = assert_input_modules_do_not_import_targets()
    audit = _build_audit(
        observations=observations,
        target_rows=target_rows,
        explosion_rows=explosion_rows,
        primary_by_mint=primary_by_mint,
        path_meta=path_meta,
        isolation=isolation,
        out_dir=out_dir,
        targets_path=targets_path,
        refs_path=refs_path,
        expl_path=expl_path,
        coverage_manifest=cov_manifest,
        world_root=world_root,
        manifest_dir=manifest_dir,
    )
    write_json(out_dir / "TARGET_DATASET_V2_AUDIT.json", audit)
    write_json(manifest_dir / "TARGET_DATASET_V2_AUDIT.json", audit)
    write_json(out_dir / "EXPLOSION_TARGET_AUDIT_V2.json", audit["explosion_audit"])
    write_json(
        manifest_dir / "EXPLOSION_TARGET_AUDIT_V2.json", audit["explosion_audit"]
    )
    _write_audit_markdown(manifest_dir / "TARGET_DATASET_V2_AUDIT.md", audit)
    _write_explosion_markdown(
        manifest_dir / "EXPLOSION_TARGET_AUDIT_V2.md", audit["explosion_audit"]
    )
    write_json(
        out_dir / "target_dataset_manifest.json",
        {
            "generated_at": utc_now(),
            "version": TARGET_DATASET_V2,
            "supersedes": TARGET_DATASET_V1,
            "artifacts": {
                "future_market_path": file_provenance(
                    path_parquet, source="targets"
                ),
                "observation_targets": file_provenance(
                    targets_path, source="targets"
                ),
                "future_path_refs": file_provenance(refs_path, source="targets"),
                "explosion_events": file_provenance(expl_path, source="targets"),
                "source_coverage": file_provenance(
                    out_dir / "target_source_coverage_v1.json", source="targets"
                ),
            },
        },
    )
    return audit


def _build_audit(
    *,
    observations: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    explosion_rows: list[dict[str, Any]],
    primary_by_mint: dict[str, dict[str, Any]],
    path_meta: dict[str, Any],
    isolation: dict[str, Any],
    out_dir: Path,
    targets_path: Path,
    refs_path: Path,
    expl_path: Path,
    coverage_manifest: dict[str, Any],
    world_root: Path,
    manifest_dir: Path,
) -> dict[str, Any]:
    n_obs = len(target_rows)
    mints = {r["mint"] for r in target_rows}
    price_cov = sum(1 for r in target_rows if r.get("price_target_available"))

    # Horizon status counts for returns
    horizon_stats: dict[str, Any] = {}
    for h in HORIZONS_SEC:
        key = f"h{h}_status"
        c = Counter(r.get(key) for r in target_rows)
        rets = [
            r[f"h{h}_simple_return"]
            for r in target_rows
            if r.get(f"h{h}_simple_return") is not None
            and math.isfinite(float(r[f"h{h}_simple_return"]))
        ]
        mfes = [
            r[f"h{h}_mfe"]
            for r in target_rows
            if r.get(f"h{h}_mfe") is not None and math.isfinite(float(r[f"h{h}_mfe"]))
        ]
        maes = [
            r[f"h{h}_mae"]
            for r in target_rows
            if r.get(f"h{h}_mae") is not None and math.isfinite(float(r[f"h{h}_mae"]))
        ]
        barrier_counts = {}
        for _, _, bid in BARRIER_PAIRS:
            bc = Counter(r.get(f"h{h}_barrier_{bid}_outcome") for r in target_rows)
            barrier_counts[bid] = dict(bc)
        horizon_stats[str(h)] = {
            "status_counts": dict(c),
            "valid": c.get(STATUS_VALID_POSITIVE, 0) + c.get(STATUS_VALID_NEGATIVE, 0),
            "valid_positive": c.get(STATUS_VALID_POSITIVE, 0),
            "valid_negative": c.get(STATUS_VALID_NEGATIVE, 0),
            "censored": c.get(STATUS_RIGHT_CENSORED, 0),
            "unavailable": n_obs
            - (
                c.get(STATUS_VALID_POSITIVE, 0)
                + c.get(STATUS_VALID_NEGATIVE, 0)
                + c.get(STATUS_RIGHT_CENSORED, 0)
            ),
            "simple_return_quantiles": _quantiles([float(x) for x in rets]),
            "mfe_quantiles": _quantiles([float(x) for x in mfes]),
            "mae_quantiles": _quantiles([float(x) for x in maes]),
            "barriers": barrier_counts,
            "mfe_terminal_consistent_rate": (
                sum(1 for r in target_rows if r.get(f"h{h}_mfe_terminal_consistent"))
                / max(1, sum(1 for r in target_rows if r.get(f"h{h}_mfe") is not None))
            ),
            "mae_terminal_consistent_rate": (
                sum(1 for r in target_rows if r.get(f"h{h}_mae_terminal_consistent"))
                / max(1, sum(1 for r in target_rows if r.get(f"h{h}_mae") is not None))
            ),
        }

    survival_stats: dict[str, Any] = {}
    surv_pos_by_h: list[int] = []
    for h in SURVIVAL_HORIZONS_SEC:
        c = Counter(r.get(f"surv_h{h}_status") for r in target_rows)
        pos = c.get(STATUS_VALID_POSITIVE, 0)
        surv_pos_by_h.append(pos)
        survival_stats[str(h)] = {
            "positive": pos,
            "negative": c.get(STATUS_VALID_NEGATIVE, 0),
            "censored": c.get(STATUS_RIGHT_CENSORED, 0)
            + c.get(STATUS_SOURCE_BOUNDARY_UNAVAILABLE, 0),
            "outage": c.get(STATUS_KNOWN_SOURCE_OUTAGE, 0),
            "status_counts": dict(c),
        }
    # Positives should be monotonically non-increasing with horizon
    # (aside from status/denominator differences — check raw positive counts).
    survival_monotonic = all(
        surv_pos_by_h[i] >= surv_pos_by_h[i + 1]
        for i in range(len(surv_pos_by_h) - 1)
    )

    persistence_stats: dict[str, Any] = {}
    for h in HORIZONS_SEC:
        c = Counter(r.get(f"act_h{h}_status") for r in target_rows)
        zero_as_valid = sum(
            1
            for r in target_rows
            if r.get(f"act_h{h}_status") == STATUS_VALID_NEGATIVE
            and r.get(f"act_h{h}_trade_count") == 0
        )
        null_when_censored = sum(
            1
            for r in target_rows
            if r.get(f"act_h{h}_status")
            not in (STATUS_VALID_POSITIVE, STATUS_VALID_NEGATIVE, None)
            and r.get(f"act_h{h}_trade_count") is None
        )
        persistence_stats[str(h)] = {
            "valid": c.get(STATUS_VALID_POSITIVE, 0) + c.get(STATUS_VALID_NEGATIVE, 0),
            "valid_positive": c.get(STATUS_VALID_POSITIVE, 0),
            "valid_negative_zero_activity": zero_as_valid,
            "censored": c.get(STATUS_RIGHT_CENSORED, 0)
            + c.get(STATUS_SOURCE_BOUNDARY_UNAVAILABLE, 0),
            "outage": c.get(STATUS_KNOWN_SOURCE_OUTAGE, 0),
            "status_counts": dict(c),
            "null_numerics_when_not_valid": null_when_censored,
        }

    grad_stats: dict[str, Any] = {}
    grad_pos_tokens: set[str] = set()
    for h in HORIZONS_SEC:
        pos = [r for r in target_rows if r.get(f"grad_h{h}_graduated") is True]
        for r in pos:
            grad_pos_tokens.add(r["mint"])
        c = Counter(r.get(f"grad_h{h}_status") for r in target_rows)
        grad_stats[str(h)] = {
            "positives": len(pos),
            "positive_tokens": len({r["mint"] for r in pos}),
            "negatives": c.get(STATUS_VALID_NEGATIVE, 0),
            "censored": c.get(STATUS_RIGHT_CENSORED, 0)
            + c.get(STATUS_SOURCE_BOUNDARY_UNAVAILABLE, 0),
            "outage": c.get(STATUS_KNOWN_SOURCE_OUTAGE, 0),
            "status_counts": dict(c),
        }

    # Explosion audit (same family; re-count under v2 observation statuses)
    primary_events = [
        e
        for e in explosion_rows
        if e.get("definition_id") == PRIMARY_DEFINITION_ID and e.get("explosion_exists")
    ]
    tokens_with_expl = {e["mint"] for e in primary_events}
    tokens_no_expl = mints - tokens_with_expl
    pre_expl_obs = [
        r
        for r in target_rows
        if r.get("explosion_event_observed")
        and not r.get("explosion_already_started_at_t")
        and r.get("time_to_explosion_sec") is not None
    ]
    expl_imminence = {}
    for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
        pos = [r for r in target_rows if r.get(f"explosion_within_{h}s") is True]
        valid = [
            r
            for r in target_rows
            if r.get(f"explosion_within_{h}s_status")
            in (STATUS_VALID_POSITIVE, STATUS_VALID_NEGATIVE)
        ]
        expl_imminence[str(h)] = {
            "positives": len(pos),
            "positive_tokens": len({r["mint"] for r in pos}),
            "valid": len(valid),
            "prevalence_among_valid": len(pos) / max(1, len(valid)),
            "prevalence_among_all_obs": len(pos) / max(1, n_obs),
        }

    meaningful = 0
    zero_prec = 0
    act_leads: list[float] = []
    buy_leads: list[float] = []
    curve_leads: list[float] = []
    trades_before_dist: dict[str, list[int]] = defaultdict(list)
    obs_before: dict[str, int] = {str(h): 0 for h in EXPLOSION_IMMINENCE_HORIZONS_SEC}
    obs_by_mint: dict[str, list[float]] = defaultdict(list)
    for r in target_rows:
        obs_by_mint[r["mint"]].append(float(r["observation_time_t"]))
    for mint in obs_by_mint:
        obs_by_mint[mint].sort()

    for e in primary_events:
        prec = {}
        if e.get("precursor_json"):
            try:
                prec = json.loads(e["precursor_json"])
            except json.JSONDecodeError:
                prec = {}
        if prec.get("meaningful_preonset_warning"):
            meaningful += 1
        if prec.get("essentially_zero_measurable_precursor"):
            zero_prec += 1
        for key, bucket in (
            ("time_activity_accel_to_onset_sec", act_leads),
            ("time_buyflow_accel_to_onset_sec", buy_leads),
            ("time_curve_accel_to_onset_sec", curve_leads),
        ):
            v = prec.get(key)
            if v is not None:
                bucket.append(float(v))
        tb = prec.get("trades_before") or {}
        for k, v in tb.items():
            trades_before_dist[k].append(int(v))
        onset = e.get("onset_unix")
        if onset is None:
            continue
        mint_ts = obs_by_mint.get(e["mint"], [])
        for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
            lo = onset - h
            i = int(np.searchsorted(mint_ts, lo, side="right"))
            j = int(np.searchsorted(mint_ts, onset, side="right"))
            if j > i:
                obs_before[str(h)] += 1

    by_mint_defs: dict[str, dict[str, bool]] = defaultdict(dict)
    for e in explosion_rows:
        by_mint_defs[e["mint"]][e.get("definition_label")] = bool(
            e.get("explosion_exists")
        )
    agree_all = sum(
        1
        for m, d in by_mint_defs.items()
        if d.get("conservative") == d.get("primary") == d.get("aggressive")
    )
    agree_prim_cons = sum(
        1
        for m, d in by_mint_defs.items()
        if d.get("conservative") == d.get("primary")
    )

    magnitudes = [
        float(e["explosion_magnitude"])
        for e in primary_events
        if e.get("explosion_magnitude") is not None
    ]
    onset_peak = [
        float(e["onset_to_peak_sec"])
        for e in primary_events
        if e.get("onset_to_peak_sec") is not None
    ]
    tte = [
        float(r["time_to_explosion_sec"])
        for r in pre_expl_obs
        if r.get("time_to_explosion_sec") is not None
    ]

    n_expl = len(primary_events)
    frac_meaningful = meaningful / max(1, n_expl)
    frac_zero = zero_prec / max(1, n_expl)

    ret_valid_tokens = {
        r["mint"]
        for r in target_rows
        if r.get("h300_status") in (STATUS_VALID_POSITIVE, STATUS_VALID_NEGATIVE)
    }
    bar_pos_tokens = {
        r["mint"]
        for r in target_rows
        if r.get("h3600_barrier_u100_l30_outcome") == BARRIER_UPPER_FIRST
    }
    surv_tokens = {
        r["mint"]
        for r in target_rows
        if r.get("surv_h86400_status") in (STATUS_VALID_POSITIVE, STATUS_VALID_NEGATIVE)
    }
    surv_pos_tokens = {
        r["mint"]
        for r in target_rows
        if r.get("surv_h86400_status") == STATUS_VALID_POSITIVE
    }
    expl_pos_tokens_60 = {
        r["mint"] for r in target_rows if r.get("explosion_within_60s") is True
    }
    holder_tokens = {
        r["mint"] for r in target_rows if r.get("holder_state_reconstructable")
    }

    def _rec_explosion() -> str:
        if n_expl >= 500 and obs_before["60"] >= 200:
            return "sufficient_on_rich_4k"
        if n_expl >= 80 and obs_before["60"] >= 40:
            return "marginal"
        return "requires_broader_population"

    def _rec_grad() -> str:
        if len(grad_pos_tokens) >= 200:
            return "sufficient_on_rich_4k"
        if len(grad_pos_tokens) >= 50:
            return "marginal"
        return "requires_broader_population"

    sufficiency = [
        _sufficiency(
            name="returns_h300",
            n_obs=sum(
                1
                for r in target_rows
                if r.get("h300_status")
                in (STATUS_VALID_POSITIVE, STATUS_VALID_NEGATIVE)
            ),
            n_tokens=len(ret_valid_tokens),
            n_positive_tokens=len(
                {
                    r["mint"]
                    for r in target_rows
                    if isinstance(r.get("h300_simple_return"), (int, float))
                    and float(r["h300_simple_return"]) > 0
                }
            ),
            n_independent_events=len(ret_valid_tokens),
            prevalence=None,
            censor_rate=horizon_stats["300"]["censored"] / max(1, n_obs),
            recommendation=(
                "sufficient_on_rich_4k"
                if len(ret_valid_tokens) >= 1000
                else "marginal"
            ),
        ),
        _sufficiency(
            name="barrier_u100_l30_h3600",
            n_obs=sum(
                1
                for r in target_rows
                if r.get("h3600_barrier_u100_l30_outcome")
                in (BARRIER_UPPER_FIRST, BARRIER_LOWER_FIRST, BARRIER_NEITHER)
            ),
            n_tokens=len(
                {
                    r["mint"]
                    for r in target_rows
                    if r.get("h3600_barrier_u100_l30_outcome")
                    in (BARRIER_UPPER_FIRST, BARRIER_LOWER_FIRST, BARRIER_NEITHER)
                }
            ),
            n_positive_tokens=len(bar_pos_tokens),
            n_independent_events=len(bar_pos_tokens),
            prevalence=len(bar_pos_tokens) / max(1, len(mints)),
            censor_rate=None,
            recommendation=(
                "sufficient_on_rich_4k"
                if len(bar_pos_tokens) >= 200
                else (
                    "marginal"
                    if len(bar_pos_tokens) >= 50
                    else "requires_broader_population"
                )
            ),
        ),
        _sufficiency(
            name="survival_24h",
            n_obs=sum(
                1
                for r in target_rows
                if r.get("surv_h86400_status")
                in (STATUS_VALID_POSITIVE, STATUS_VALID_NEGATIVE)
            ),
            n_tokens=len(surv_tokens),
            n_positive_tokens=len(surv_pos_tokens),
            n_independent_events=len(surv_tokens),
            prevalence=len(surv_pos_tokens) / max(1, len(surv_tokens)),
            censor_rate=survival_stats["86400"]["censored"] / max(1, n_obs),
            recommendation=(
                "sufficient_on_rich_4k"
                if len(surv_tokens) >= 1000
                else "marginal"
            ),
        ),
        _sufficiency(
            name="graduation",
            n_obs=sum(1 for r in target_rows if r.get("grad_h86400_graduated") is True),
            n_tokens=len(
                {
                    r["mint"]
                    for r in target_rows
                    if r.get("grad_h86400_status")
                    in (STATUS_VALID_POSITIVE, STATUS_VALID_NEGATIVE)
                }
            ),
            n_positive_tokens=len(grad_pos_tokens),
            n_independent_events=len(grad_pos_tokens),
            prevalence=len(grad_pos_tokens) / max(1, len(mints)),
            censor_rate=None,
            recommendation=_rec_grad(),
            notes=(
                "Negatives use migration-source coverage, not token trade activity."
            ),
        ),
        _sufficiency(
            name="explosion_onset_primary",
            n_obs=len(pre_expl_obs),
            n_tokens=len(tokens_with_expl),
            n_positive_tokens=len(expl_pos_tokens_60),
            n_independent_events=n_expl,
            prevalence=n_expl / max(1, len(mints)),
            censor_rate=None,
            recommendation=_rec_explosion(),
        ),
        _sufficiency(
            name="holder_state_targets",
            n_obs=sum(1 for r in target_rows if r.get("holder_state_reconstructable")),
            n_tokens=len(holder_tokens),
            n_positive_tokens=len(holder_tokens),
            n_independent_events=len(holder_tokens),
            prevalence=len(holder_tokens) / max(1, len(mints)),
            censor_rate=None,
            recommendation="requires_broader_population",
        ),
    ]

    exec_cov = {}
    for sz in POSITION_SIZES_SOL:
        tag = str(sz).replace(".", "p")
        exec_cov[str(sz)] = {
            "gross_available": sum(
                1 for r in target_rows if r.get(f"exec_{tag}sol_gross_available")
            ),
            "entry_impact_available": sum(
                1 for r in target_rows if r.get(f"exec_{tag}sol_entry_impact_available")
            ),
            "curve_impact_available": sum(
                1 for r in target_rows if r.get(f"exec_{tag}sol_curve_impact_available")
            ),
            "net_available": sum(
                1
                for r in target_rows
                if r.get(f"exec_{tag}sol_execution_target_available") == 1
            ),
            "status_available": sum(
                1 for r in target_rows if r.get(f"exec_{tag}sol_status") == "available"
            ),
        }

    disk = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())

    explosion_audit = {
        "detector_version": DETECTOR_VERSION,
        "primary_definition_id": PRIMARY_DEFINITION_ID,
        "detector_unchanged_from_v1": True,
        "distinct_tokens_with_primary_explosion": len(tokens_with_expl),
        "fraction_tokens_with_no_explosion": len(tokens_no_expl) / max(1, len(mints)),
        "independent_explosion_events": n_expl,
        "observations_pre_explosion": len(pre_expl_obs),
        "imminence": expl_imminence,
        "time_to_explosion_quantiles": _quantiles(tte),
        "explosion_magnitude_quantiles": _quantiles(magnitudes),
        "onset_to_peak_sec_quantiles": _quantiles(onset_peak),
        "precursor": {
            "n_with_meaningful_preonset_warning": meaningful,
            "fraction_with_meaningful_preonset_warning": frac_meaningful,
            "n_essentially_zero_measurable_precursor": zero_prec,
            "fraction_apparently_externally_triggered_or_no_precursor": frac_zero,
            "activity_accel_lead_sec": _quantiles(act_leads),
            "buyflow_accel_lead_sec": _quantiles(buy_leads),
            "curve_accel_lead_sec": _quantiles(curve_leads),
            "trades_before_onset": {
                k: _quantiles([float(x) for x in v])
                for k, v in trades_before_dist.items()
            },
            "answer_how_many_explosions_offer_meaningful_preonset_warning": meaningful,
        },
        "observations_before_onset_by_horizon": obs_before,
        "definition_agreement": {
            "mints_scored": len(by_mint_defs),
            "all_three_agree": agree_all,
            "primary_conservative_agree": agree_prim_cons,
            "primary_exists": sum(
                1 for d in by_mint_defs.values() if d.get("primary")
            ),
            "conservative_exists": sum(
                1 for d in by_mint_defs.values() if d.get("conservative")
            ),
            "aggressive_exists": sum(
                1 for d in by_mint_defs.values() if d.get("aggressive")
            ),
        },
        "is_4k_enough_for_explosion_hazard_head": _rec_explosion(),
        "expand_later_without_outcome_selection": [
            "pumpfun_memecoin_corpus/trades (broader mint population)",
            "execution_curve_states / bonding-curve path",
            "migrations.parquet for graduation-linked explosion contexts",
        ],
    }

    v1_compare = _compare_to_v1_audit(manifest_dir, survival_stats, grad_stats)

    return {
        "audit_id": TARGET_DATASET_V2,
        "generated_at": utc_now(),
        "acquisition_freeze_id": FREEZE_ID,
        "pit_engine_version": PIT_ENGINE_VERSION,
        "observation_dataset_version": OBSERVATION_DATASET_VERSION,
        "supersedes": TARGET_DATASET_V1,
        "observations_processed": n_obs,
        "unique_tokens": len(mints),
        "reference_price_coverage": {
            "available": price_cov,
            "unavailable": n_obs - price_cov,
            "rate": price_cov / max(1, n_obs),
        },
        "target_rows": n_obs,
        "future_market_path": path_meta,
        "source_coverage": coverage_manifest,
        "horizons": horizon_stats,
        "survival": survival_stats,
        "survival_positive_counts_monotonic_nonincreasing": survival_monotonic,
        "survival_positive_counts_by_horizon": {
            str(h): n for h, n in zip(SURVIVAL_HORIZONS_SEC, surv_pos_by_h, strict=True)
        },
        "persistence": persistence_stats,
        "graduation": {
            "by_horizon": grad_stats,
            "distinct_graduated_tokens_any_horizon": len(grad_pos_tokens),
        },
        "explosion_audit": explosion_audit,
        "execution_coverage": exec_cov,
        "ownership": {
            "holder_state_reconstructable_tokens": len(holder_tokens),
            "dynamic_holder_targets_materialized": 0,
        },
        "isolation_proof": isolation,
        "supervision_sufficiency": sufficiency,
        "v1_comparison": v1_compare,
        "estimated_disk_bytes": disk,
        "estimated_disk_mb": disk / (1024 * 1024),
        "artifacts": {
            "observation_targets": str(targets_path),
            "future_path_refs": str(refs_path),
            "explosion_events": str(expl_path),
            "out_dir": str(out_dir),
            "v1_preserved": str(world_root / "targets" / TARGET_DATASET_V1),
        },
        "chronological_splits": "not_created",
        "training": "not_started",
    }


def _compare_to_v1_audit(
    manifest_dir: Path,
    survival_v2: dict[str, Any],
    grad_v2: dict[str, Any],
) -> dict[str, Any]:
    v1_path = manifest_dir / "TARGET_DATASET_V1_AUDIT.json"
    if not v1_path.exists():
        return {"available": False}
    v1 = read_json(v1_path)
    surv_cmp = {}
    for h, st in survival_v2.items():
        v1s = (v1.get("survival") or {}).get(h) or {}
        surv_cmp[h] = {
            "v1_positive": v1s.get("positive"),
            "v2_positive": st.get("positive"),
            "v1_negative": v1s.get("negative"),
            "v2_negative": st.get("negative"),
            "v1_censored": v1s.get("censored"),
            "v2_censored": st.get("censored"),
            "v2_outage": st.get("outage"),
        }
    grad_cmp = {}
    for h, st in grad_v2.items():
        v1g = ((v1.get("graduation") or {}).get("by_horizon") or {}).get(h) or {}
        grad_cmp[h] = {
            "v1_positives": v1g.get("positives"),
            "v2_positives": st.get("positives"),
            "v2_negatives": st.get("negatives"),
            "v2_censored": st.get("censored"),
        }
    return {
        "available": True,
        "survival": surv_cmp,
        "graduation": grad_cmp,
        "v1_explosion_events": (v1.get("explosion_audit") or {}).get(
            "independent_explosion_events"
        ),
    }


def _write_audit_markdown(path: Path, audit: dict[str, Any]) -> None:
    expl = audit["explosion_audit"]
    lines = [
        f"# Target Dataset v2 (`{audit['audit_id']}`)",
        "",
        f"Generated at: `{audit['generated_at']}`",
        f"Freeze: `{audit['acquisition_freeze_id']}`",
        f"Supersedes: `{audit.get('supersedes')}`",
        f"Observations: **{audit['observations_processed']}**",
        f"Unique tokens: **{audit['unique_tokens']}**",
        (
            f"Reference-price coverage: "
            f"**{audit['reference_price_coverage']['available']}** "
            f"({audit['reference_price_coverage']['rate']:.3f})"
        ),
        f"Disk MB: **{audit['estimated_disk_mb']:.1f}**",
        "",
        "## Survival (last activity >= T+H)",
        (
            f"- positive counts monotonic nonincreasing: "
            f"**{audit.get('survival_positive_counts_monotonic_nonincreasing')}**"
        ),
        f"- by horizon: `{audit.get('survival_positive_counts_by_horizon')}`",
        "",
        "## Isolation",
        f"- input modules do not import targets: **{audit['isolation_proof']['ok']}**",
        "",
        "## Explosion (primary; detector unchanged)",
        f"- independent events: **{expl['independent_explosion_events']}**",
        f"- tokens with explosion: **{expl['distinct_tokens_with_primary_explosion']}**",
        (
            f"- meaningful pre-onset warning: "
            f"**{expl['precursor']['n_with_meaningful_preonset_warning']}**"
        ),
        f"- 4k sufficiency: **{expl['is_4k_enough_for_explosion_hazard_head']}**",
        "",
        "## Supervision sufficiency",
    ]
    for s in audit["supervision_sufficiency"]:
        lines.append(
            f"- `{s['head']}`: tokens={s['unique_tokens_with_valid_supervision']}, "
            f"pos_tokens={s['distinct_positive_tokens']}, "
            f"events={s['independent_event_count']}, "
            f"**{s['recommendation']}**"
        )
    grad_n = audit["graduation"]["distinct_graduated_tokens_any_horizon"]
    lines += [
        "",
        "## Graduation",
        f"- distinct graduated tokens: **{grad_n}**",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_explosion_markdown(path: Path, expl: dict[str, Any]) -> None:
    lines = [
        "# Explosion Target Audit v1",
        "",
        f"Detector: `{expl['detector_version']}`",
        f"Primary: `{expl['primary_definition_id']}`",
        "",
        f"Independent explosion events: **{expl['independent_explosion_events']}**",
        f"Tokens with primary explosion: **{expl['distinct_tokens_with_primary_explosion']}**",
        f"Fraction with no explosion: **{expl['fraction_tokens_with_no_explosion']:.3f}**",
        f"Pre-explosion observations: **{expl['observations_pre_explosion']}**",
        "",
        "## Imminence positives",
    ]
    for h, st in expl["imminence"].items():
        lines.append(
            f"- {h}s: positives={st['positives']}, "
            f"prevalence_valid={st['prevalence_among_valid']:.4f}"
        )
    lines += [
        "",
        "## Precursors",
        f"- meaningful warning: **{expl['precursor']['n_with_meaningful_preonset_warning']}**",
        (
            f"- no measurable precursor: "
            f"**{expl['precursor']['n_essentially_zero_measurable_precursor']}**"
        ),
        "",
        f"Sufficiency: **{expl['is_4k_enough_for_explosion_hazard_head']}**",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
