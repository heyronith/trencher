"""Materialize ``baseline_features_v1`` from frozen stores (partition-once NumPy).

Two separate caches:
- ``broad/by_mint_partition`` — full train+val population, 124 broad features
- ``rich/by_mint_partition`` — frozen rich observation population, 146 features
  (124 broad + 22 rich-additive from ``sequence_dataset_v2``)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trencher.baselines.feature_spec import (
    FEATURE_CACHE_VERSION,
    all_feature_names,
    feature_contract_manifest,
    rich_additive_feature_names,
)
from trencher.baselines.features import (
    age_features,
    creator_features,
    event_features_from_arrays,
    global_features_from_ctx,
    vectorize_features,
)
from trencher.baselines.progress import ProgressEmitter
from trencher.baselines.rich_index import CausalRichSequenceIndex
from trencher.baselines.split_guard import PermanentTestAccessError, refuse_test_split
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.training.global_batch import GlobalContextReader

Population = Literal["broad", "rich"]


def feature_cache_root(world_root: Path | None = None, split_root: Path | None = None) -> Path:
    split_root = Path(
        split_root
        or (
            Path(world_root) / "splits" / "split_v1"
            if world_root
            else Path("data/external/world_model/splits/split_v1")
        )
    )
    return split_root / FEATURE_CACHE_VERSION


def population_feature_dir(feat_root: Path, population: Population) -> Path:
    return Path(feat_root) / population / "by_mint_partition"


def _as_float_col(table: pa.Table, name: str, n: int) -> np.ndarray:
    if name not in table.column_names:
        return np.zeros(n, dtype=np.float64)
    return np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype=np.float64)


def _load_events_partition(path: Path) -> dict[str, np.ndarray]:
    cols = [
        "event_time_unix",
        "is_buy",
        "sol_amount",
        "price_sol",
        "v_sol",
        "v_tok",
        "curve_pct_depleted",
    ]
    schema = set(pq.read_schema(path).names)
    use = [c for c in cols if c in schema]
    table = pq.read_table(path, columns=use)
    n = table.num_rows
    return {
        "times": _as_float_col(table, "event_time_unix", n),
        "is_buy": _as_float_col(table, "is_buy", n),
        "sol": _as_float_col(table, "sol_amount", n),
        "price": _as_float_col(table, "price_sol", n),
        "v_sol": _as_float_col(table, "v_sol", n),
        "v_tok": _as_float_col(table, "v_tok", n),
        "curve": _as_float_col(table, "curve_pct_depleted", n),
    }


def _load_creator_partition(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    cols = ["event_time_unix", "prior_rank", "prior_is_mayhem_mode"]
    schema = set(pq.read_schema(path).names)
    use = [c for c in cols if c in schema]
    if "event_time_unix" not in use:
        return None
    table = pq.read_table(path, columns=use)
    n = table.num_rows
    return {
        "times": _as_float_col(table, "event_time_unix", n),
        "prior_rank": _as_float_col(table, "prior_rank", n),
        "mayhem": _as_float_col(table, "prior_is_mayhem_mode", n),
    }


def _schema_resume_ok(
    out_path: Path, *, feature_names: list[str], population: Population
) -> tuple[bool, int]:
    """Skip only when schema matches the requested population exactly enough."""
    try:
        pf = pq.ParquetFile(out_path)
        schema_names = set(pf.schema_arrow.names)
        if not all(n in schema_names for n in feature_names):
            return False, 0
        rich_names = set(rich_additive_feature_names())
        if population == "broad" and (schema_names & rich_names):
            # Stale 146-col partitions must not be resumed as broad.
            return False, 0
        return True, int(pf.metadata.num_rows)
    except Exception:  # noqa: BLE001
        return False, 0


def _index_assignments(
    *,
    assign_dir: Path,
    q85: float,
    max_partitions: int | None,
    progress: ProgressEmitter,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    assign_parts = sorted(assign_dir.glob("part-*.parquet"))
    if max_partitions is not None:
        # Keep full assignment scan for correct population counts; max_partitions
        # limits mint partitions after grouping.
        pass
    by_mp: dict[str, list[dict[str, Any]]] = {}
    n_skipped_test = 0
    cols = [
        "observation_id",
        "mint",
        "observation_time_t",
        "detected_at_unix",
        "token_age_seconds",
        "base_split",
        "creator",
        "mint_partition",
        "in_rich_subset",
        "outage_proximity_sec",
        "observation_during_jul3_outage",
    ]
    for i, path in enumerate(assign_parts):
        schema = set(pq.read_schema(path).names)
        use = [c for c in cols if c in schema]
        table = pq.read_table(path, columns=use)
        for row in table.to_pylist():
            split = str(row.get("base_split") or "")
            if split == "test":
                n_skipped_test += 1
                continue
            if split not in ("train", "val"):
                continue
            refuse_test_split(split)
            det = row.get("detected_at_unix")
            if det is not None and float(det) >= q85:
                n_skipped_test += 1
                continue
            mp = str(row.get("mint_partition") or "")
            by_mp.setdefault(mp, []).append(row)
        progress.update(
            i + 1,
            max(len(assign_parts), 1),
            message=f"assign {i + 1}/{len(assign_parts)}",
        )
    if max_partitions is not None:
        keep = sorted(by_mp.keys())[:max_partitions]
        by_mp = {k: by_mp[k] for k in keep}
    return by_mp, n_skipped_test


def _broad_features_for_row(
    *,
    row: dict[str, Any],
    ref: dict[str, Any],
    ev: dict[str, np.ndarray] | None,
    cr: dict[str, np.ndarray] | None,
    global_reader: GlobalContextReader,
) -> dict[str, float]:
    T = float(row["observation_time_t"])
    feat: dict[str, float] = {}
    feat.update(age_features(row.get("token_age_seconds")))

    if ev is not None:
        lo = int(ref.get("events_start_idx") or 0)
        hi = int(ref.get("events_end_idx") or 0)
        if hi > lo:
            sl = slice(lo, hi)
            times = ev["times"][sl]
            ef = event_features_from_arrays(
                times=times,
                is_buy=ev["is_buy"][sl],
                sol=ev["sol"][sl],
                price=ev["price"][sl],
                v_sol=ev["v_sol"][sl],
                curve=ev["curve"][sl],
                T=T,
            )
            hi_t = int(np.searchsorted(times, T, side="right"))
            if hi_t > 0:
                ef["latest_v_tok"] = float(ev["v_tok"][sl][hi_t - 1])
            feat.update(ef)
        else:
            feat.update(
                event_features_from_arrays(
                    times=np.array([]),
                    is_buy=np.array([]),
                    sol=np.array([]),
                    price=np.array([]),
                    v_sol=np.array([]),
                    curve=np.array([]),
                    T=T,
                )
            )
    else:
        feat["mask_events"] = 0.0
        feat["mask_curve"] = 0.0

    if cr is not None:
        clo = int(ref.get("creator_start_idx") or 0)
        chi = int(ref.get("creator_end_idx") or 0)
        if chi > clo:
            cts = cr["times"][clo:chi]
            chi2 = clo + int(np.searchsorted(cts, T, side="right"))
            if chi2 > clo:
                feat.update(
                    creator_features(
                        prior_rank=float(cr["prior_rank"][chi2 - 1]),
                        prior_is_mayhem=float(cr["mayhem"][chi2 - 1]),
                        history_n=float(chi2 - clo),
                        available=True,
                    )
                )
            else:
                feat.update(
                    creator_features(
                        prior_rank=None, prior_is_mayhem=None, history_n=0, available=False
                    )
                )
        else:
            feat.update(
                creator_features(
                    prior_rank=None, prior_is_mayhem=None, history_n=0, available=False
                )
            )
    else:
        feat.update(
            creator_features(
                prior_rank=None, prior_is_mayhem=None, history_n=0, available=False
            )
        )

    gctx = global_reader.load_for_t(T)
    feat.update(global_features_from_ctx(gctx))
    return feat


class FeatureCacheCoverageError(RuntimeError):
    """Raised when assignment→sequence-ref coverage is incomplete."""


def _require_ref(
    refs_by_oid: dict[str, dict[str, Any]],
    oid: str,
) -> dict[str, Any]:
    ref = refs_by_oid.get(oid)
    if ref is None:
        raise FeatureCacheCoverageError(f"missing sequence ref for observation_id={oid}")
    return ref


def _preflight_ref_coverage(
    *,
    by_mp: dict[str, list[dict[str, Any]]],
    refs_root: Path,
    rich_refs: dict[str, dict[str, Any]] | None,
    include_rich: bool,
) -> None:
    """Fail fast with counts/samples — never proceed with silent empty-ref joins."""
    missing_broad: list[str] = []
    missing_rich: list[str] = []
    for mp, rows in by_mp.items():
        refs_by_oid: dict[str, dict[str, Any]] = {}
        ref_path = refs_root / mp / "part.parquet"
        if ref_path.exists():
            for r in pq.read_table(ref_path).to_pylist():
                oid = str(r.get("observation_id") or "")
                if oid:
                    refs_by_oid[oid] = r
        for row in rows:
            oid = str(row["observation_id"])
            if oid not in refs_by_oid:
                missing_broad.append(oid)
            if include_rich and rich_refs is not None and oid not in rich_refs:
                missing_rich.append(oid)
    if missing_broad:
        raise FeatureCacheCoverageError(
            f"missing broad sequence refs: count={len(missing_broad)} "
            f"sample={missing_broad[:20]}"
        )
    if missing_rich:
        raise FeatureCacheCoverageError(
            f"missing sequence_dataset_v2 observation refs: "
            f"count={len(missing_rich)} sample={missing_rich[:20]}"
        )


def build_baseline_features_v1(
    *,
    world_root: Path,
    split_root: Path | None = None,
    populations: tuple[str, ...] | None = None,
    population: Population | None = None,
    max_partitions: int | None = None,
    progress: ProgressEmitter | None = None,
    include_rich: bool | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Build feature cache for train+val only (never test).

    ``population='broad'`` → 124 features for full broad train+val.
    ``population='rich'`` → 146 features for frozen rich observation IDs only.
    """
    world_root = Path(world_root)
    split_root = Path(split_root or world_root / "splits" / "split_v1")

    if population is None:
        if (populations and "rich" in populations and "broad" not in populations) or include_rich:
            population = "rich"
        else:
            population = "broad"
    if population not in ("broad", "rich"):
        raise ValueError(f"population must be broad|rich, got {population!r}")

    include_rich = population == "rich"
    feature_names = all_feature_names(include_rich=include_rich)
    out_root = ensure_dir(feature_cache_root(split_root=split_root))
    progress = progress or ProgressEmitter(f"Baseline feature cache ({population})")
    progress.emit(1, message="start", force=True)

    cutoffs = json.loads((split_root / "cutoffs.json").read_text(encoding="utf-8"))
    q85 = float(cutoffs["quantiles_unix"]["q85"])
    write_json(out_root / "feature_contract.json", feature_contract_manifest())
    pop_meta_dir = ensure_dir(out_root / population)
    (pop_meta_dir / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8"
    )
    # Keep root feature_names aligned with broad for back-compat readers.
    if population == "broad":
        (out_root / "feature_names.json").write_text(
            json.dumps(feature_names, indent=2), encoding="utf-8"
        )

    global_reader = GlobalContextReader(world_root)
    rich_index = CausalRichSequenceIndex(world_root) if include_rich else None

    assign_name = (
        "observation_assignments_rich"
        if population == "rich"
        else "observation_assignments_broad"
    )
    progress.emit(5, message=f"index {assign_name} train+val")
    by_mp, n_skipped_test = _index_assignments(
        assign_dir=split_root / assign_name,
        q85=q85,
        max_partitions=max_partitions,
        progress=progress,
    )

    mps = sorted(by_mp.keys())
    refs_root = world_root / "broad" / "broad_observation_sequence_refs_v1"
    events_root = world_root / "broad" / "broad_sequence_events_v1"
    creator_root = world_root / "broad" / "broad_sequence_creator_v1"
    out_parts = ensure_dir(population_feature_dir(out_root, population))

    rich_refs = rich_index.refs() if rich_index is not None else None
    progress.emit(8, message="preflight sequence-ref coverage")
    _preflight_ref_coverage(
        by_mp=by_mp,
        refs_root=refs_root,
        rich_refs=rich_refs,
        include_rich=include_rich,
    )

    total = max(len(mps), 1)
    rows_written = 0
    rows_resumed = 0
    partitions_skipped = 0
    partitions_built = 0
    seen_oids: set[str] = set()
    expected_rows = sum(len(v) for v in by_mp.values())

    for pi, mp in enumerate(mps):
        obs_rows = by_mp[mp]
        part_dir = out_parts / mp
        out_path = part_dir / "features.parquet"
        if resume and out_path.exists():
            schema_ok, n_exist = _schema_resume_ok(
                out_path, feature_names=feature_names, population=population
            )
            if schema_ok:
                try:
                    oids = (
                        pq.read_table(out_path, columns=["observation_id"])
                        .column(0)
                        .to_pylist()
                    )
                    for oid in oids:
                        seen_oids.add(str(oid))
                except Exception:  # noqa: BLE001
                    pass
                rows_written += int(n_exist)
                rows_resumed += int(n_exist)
                partitions_skipped += 1
                progress.update(
                    pi + 1,
                    total,
                    message=f"skip existing {pi + 1}/{total} mp={mp} rows={rows_written}",
                    rows_written=rows_written,
                    partitions_skipped=partitions_skipped,
                )
                continue

        refs_by_oid: dict[str, dict[str, Any]] = {}
        ref_path = refs_root / mp / "part.parquet"
        if ref_path.exists():
            for r in pq.read_table(ref_path).to_pylist():
                oid = str(r.get("observation_id") or "")
                if oid:
                    refs_by_oid[oid] = r

        ev_path = events_root / mp / "part.parquet"
        ev = _load_events_partition(ev_path) if ev_path.exists() else None
        cr_path = creator_root / mp / "part.parquet"
        cr = _load_creator_partition(cr_path)

        feat_rows: list[dict[str, Any]] = []
        for row in obs_rows:
            oid = str(row["observation_id"])
            T = float(row["observation_time_t"])
            ref = _require_ref(refs_by_oid, oid)
            feat = _broad_features_for_row(
                row=row, ref=ref, ev=ev, cr=cr, global_reader=global_reader
            )
            if rich_index is not None:
                feat.update(rich_index.features_for_observation(oid, T))

            names, vec = vectorize_features(feat, include_rich=include_rich)
            rec: dict[str, Any] = {
                "observation_id": oid,
                "mint": row.get("mint"),
                "mint_partition": mp,
                "base_split": row["base_split"],
                "observation_time_t": T,
                "detected_at_unix": row.get("detected_at_unix"),
                "token_age_seconds": row.get("token_age_seconds"),
                "in_rich_subset": bool(row.get("in_rich_subset", population == "rich")),
                "outage_proximity_sec": row.get("outage_proximity_sec"),
                "observation_during_jul3_outage": bool(
                    row.get("observation_during_jul3_outage")
                ),
                "creator_bucket": (str(row.get("creator") or "")[:8] or None),
            }
            for j, name in enumerate(names):
                rec[name] = float(vec[j])
            feat_rows.append(rec)
            seen_oids.add(oid)

        if feat_rows:
            part_dir = ensure_dir(out_parts / mp)
            pq.write_table(
                pa.Table.from_pylist(feat_rows),
                part_dir / "features.parquet",
                compression="zstd",
            )
            rows_written += len(feat_rows)
            partitions_built += 1

        progress.update(
            pi + 1,
            total,
            message=f"partition {pi + 1}/{total} mp={mp} rows={rows_written}",
            rows_written=rows_written,
            partitions_built=partitions_built,
            partitions_skipped=partitions_skipped,
        )

    if partitions_skipped == 0:
        if len(seen_oids) != expected_rows:
            raise FeatureCacheCoverageError(
                f"unique observation_ids={len(seen_oids)} != assignment rows={expected_rows}"
            )
        if rows_written != expected_rows:
            raise FeatureCacheCoverageError(
                f"rows_written={rows_written} != assignment rows={expected_rows}"
            )
    if len(feature_names) != (146 if include_rich else 124):
        raise FeatureCacheCoverageError(
            f"feature_n={len(feature_names)} expected "
            f"{146 if include_rich else 124}"
        )

    train_n = sum(1 for rows in by_mp.values() for r in rows if r["base_split"] == "train")
    val_n = sum(1 for rows in by_mp.values() for r in rows if r["base_split"] == "val")
    manifest = {
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "population": population,
        "generated_at": utc_now(),
        "rows_written": rows_written,
        "rows_resumed": rows_resumed,
        "train_n": train_n,
        "val_n": val_n,
        "assignment_rows_expected": expected_rows,
        "unique_observation_ids": len(seen_oids),
        "partitions": len(mps),
        "partitions_built": partitions_built,
        "partitions_skipped": partitions_skipped,
        "resume": resume,
        "feature_n": len(feature_names),
        "skipped_test_rows": n_skipped_test,
        "populations": [population],
        "include_rich": include_rich,
        "rich_source": "sequences/sequence_dataset_v2" if include_rich else None,
        "missing_broad_refs": 0,
        "missing_rich_refs": 0,
        "no_permanent_test": True,
        "forbidden_sources": [
            "graph/edges_wallet_token.parquet lifetime aggregates",
        ]
        if include_rich
        else [],
    }
    write_json(pop_meta_dir / "manifest.json", manifest)
    write_json(out_root / "manifest.json", manifest)
    progress.complete(message=f"{population} feature cache complete", rows_written=rows_written)
    return manifest


def assert_features_exclude_identifiers(feature_names: list[str]) -> None:
    banned = {
        "mint",
        "wallet",
        "creator",
        "observation_id",
        "base_split",
        "tx_signature",
        "source_identity",
    }
    bad = sorted(set(feature_names) & banned)
    if bad:
        raise PermanentTestAccessError(f"identifier features present: {bad}")
