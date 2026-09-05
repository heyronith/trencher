"""Materialize ``baseline_features_v1`` from frozen stores (partition-once NumPy)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trencher.baselines.feature_spec import (
    FEATURE_CACHE_VERSION,
    all_feature_names,
    feature_contract_manifest,
)
from trencher.baselines.features import (
    age_features,
    creator_features,
    event_features_from_arrays,
    global_features_from_ctx,
    rich_features_from_arrays,
    vectorize_features,
)
from trencher.baselines.progress import ProgressEmitter
from trencher.baselines.rich_index import (
    causal_slice_arrays,
    load_graph_by_mint,
    load_micro_by_mint,
    load_ownership_by_mint,
    load_wallet_prior_by_mint,
)
from trencher.baselines.split_guard import PermanentTestAccessError, refuse_test_split
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.training.global_batch import GlobalContextReader


def feature_cache_root(world_root: Path | None = None, split_root: Path | None = None) -> Path:
    split_root = Path(
        split_root
        or (Path(world_root) / "splits" / "split_v1" if world_root else Path("data/external/world_model/splits/split_v1"))
    )
    return split_root / FEATURE_CACHE_VERSION


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


def build_baseline_features_v1(
    *,
    world_root: Path,
    split_root: Path | None = None,
    populations: tuple[str, ...] = ("broad",),
    max_partitions: int | None = None,
    progress: ProgressEmitter | None = None,
    include_rich: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Build feature cache for train+val only (never test).

    When ``resume=True``, existing ``by_mint_partition/*/features.parquet``
    files are skipped so completed partitions are not rebuilt.
    """
    world_root = Path(world_root)
    split_root = Path(split_root or world_root / "splits" / "split_v1")
    out_root = ensure_dir(feature_cache_root(split_root=split_root))
    progress = progress or ProgressEmitter("Baseline feature cache")
    progress.emit(1, message="start", force=True)

    cutoffs = json.loads((split_root / "cutoffs.json").read_text(encoding="utf-8"))
    q85 = float(cutoffs["quantiles_unix"]["q85"])
    feature_names = all_feature_names(include_rich=include_rich or "rich" in populations)
    write_json(out_root / "feature_contract.json", feature_contract_manifest())
    (out_root / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8"
    )

    global_reader = GlobalContextReader(world_root)
    micro_by_mint = load_micro_by_mint(world_root) if include_rich else {}
    own_by_mint = load_ownership_by_mint(world_root) if include_rich else {}
    graph_by_mint = load_graph_by_mint(world_root) if include_rich else {}
    wallet_prior_by_mint = load_wallet_prior_by_mint(world_root) if include_rich else {}
    assign_dir = split_root / "observation_assignments_broad"
    assign_parts = sorted(assign_dir.glob("part-*.parquet"))
    if max_partitions is not None:
        assign_parts = assign_parts[: max_partitions]

    # Group observation rows by mint_partition (train+val only)
    progress.emit(5, message="index assignments train+val")
    by_mp: dict[str, list[dict[str, Any]]] = {}
    n_skipped_test = 0
    for i, path in enumerate(assign_parts):
        table = pq.read_table(
            path,
            columns=[
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
            ],
        )
        for row in table.to_pylist():
            split = str(row.get("base_split") or "")
            if split == "test":
                n_skipped_test += 1
                continue
            if split not in ("train", "val"):
                continue
            # Hard guard: observation time must be before permanent test launch boundary
            # for val we allow [q70,q85); still never include base_split=test.
            refuse_test_split(split)
            T = float(row["observation_time_t"])
            # Extra guard: no observation with detected_at >= q85
            det = row.get("detected_at_unix")
            if det is not None and float(det) >= q85:
                n_skipped_test += 1
                continue
            mp = str(row.get("mint_partition") or "")
            by_mp.setdefault(mp, []).append(row)
        progress.update(i + 1, max(len(assign_parts), 1), message=f"assign {i+1}/{len(assign_parts)}")

    mps = sorted(by_mp.keys())
    refs_root = world_root / "broad" / "broad_observation_sequence_refs_v1"
    events_root = world_root / "broad" / "broad_sequence_events_v1"
    creator_root = world_root / "broad" / "broad_sequence_creator_v1"
    out_parts = ensure_dir(out_root / "broad" / "by_mint_partition")

    total = max(len(mps), 1)
    rows_written = 0
    rows_resumed = 0
    partitions_skipped = 0
    partitions_built = 0
    for pi, mp in enumerate(mps):
        obs_rows = by_mp[mp]
        part_dir = out_parts / mp
        out_path = part_dir / "features.parquet"
        if resume and out_path.exists():
            try:
                n_exist = pq.ParquetFile(out_path).metadata.num_rows
            except Exception:  # noqa: BLE001
                n_exist = 0
            rows_written += int(n_exist)
            rows_resumed += int(n_exist)
            partitions_skipped += 1
            progress.update(
                pi + 1,
                total,
                message=f"skip existing {pi+1}/{total} mp={mp} rows={rows_written}",
                rows_written=rows_written,
                partitions_skipped=partitions_skipped,
            )
            continue
        # refs
        ref_path = refs_root / mp / "part.parquet"
        refs_by_oid: dict[str, dict[str, Any]] = {}
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
            ref = refs_by_oid.get(oid) or {}
            feat: dict[str, float] = {}
            feat.update(age_features(row.get("token_age_seconds")))

            if ev is not None:
                lo = int(ref.get("events_start_idx") or 0)
                hi = int(ref.get("events_end_idx") or 0)
                if hi > lo:
                    sl = slice(lo, hi)
                    times = ev["times"][sl]
                    # causal truncate inside event_features
                    ef = event_features_from_arrays(
                        times=times,
                        is_buy=ev["is_buy"][sl],
                        sol=ev["sol"][sl],
                        price=ev["price"][sl],
                        v_sol=ev["v_sol"][sl],
                        curve=ev["curve"][sl],
                        T=T,
                    )
                    # latest v_tok
                    hi_t = int(np.searchsorted(times, T, side="right"))
                    if hi_t > 0:
                        ef["latest_v_tok"] = float(ev["v_tok"][sl][hi_t - 1])
                    feat.update(ef)
                else:
                    feat.update(event_features_from_arrays(
                        times=np.array([]),
                        is_buy=np.array([]),
                        sol=np.array([]),
                        price=np.array([]),
                        v_sol=np.array([]),
                        curve=np.array([]),
                        T=T,
                    ))
            else:
                feat["mask_events"] = 0.0
                feat["mask_curve"] = 0.0

            # creator priors causally before T
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
                        feat.update(creator_features(
                            prior_rank=None, prior_is_mayhem=None, history_n=0, available=False
                        ))
                else:
                    feat.update(creator_features(
                        prior_rank=None, prior_is_mayhem=None, history_n=0, available=False
                    ))
            else:
                feat.update(creator_features(
                    prior_rank=None, prior_is_mayhem=None, history_n=0, available=False
                ))

            gctx = global_reader.load_for_t(T)
            feat.update(global_features_from_ctx(gctx))

            if include_rich:
                mint = str(row.get("mint") or "")
                feat.update(
                    rich_features_from_arrays(
                        micro=causal_slice_arrays(micro_by_mint.get(mint), T),
                        ownership=causal_slice_arrays(own_by_mint.get(mint), T),
                        graph=causal_slice_arrays(graph_by_mint.get(mint), T),
                        wallet_prior=causal_slice_arrays(
                            wallet_prior_by_mint.get(mint), T
                        ),
                    )
                )

            names, vec = vectorize_features(feat, include_rich=include_rich)
            rec: dict[str, Any] = {
                "observation_id": oid,
                "mint": row.get("mint"),
                "mint_partition": mp,
                "base_split": row["base_split"],
                "observation_time_t": T,
                "detected_at_unix": row.get("detected_at_unix"),
                "token_age_seconds": row.get("token_age_seconds"),
                "in_rich_subset": bool(row.get("in_rich_subset")),
                "outage_proximity_sec": row.get("outage_proximity_sec"),
                "observation_during_jul3_outage": bool(
                    row.get("observation_during_jul3_outage")
                ),
                # creator hash for slice analysis only — not a predictive column in X
                "creator_bucket": (str(row.get("creator") or "")[:8] or None),
            }
            for j, name in enumerate(names):
                rec[name] = float(vec[j])
            feat_rows.append(rec)

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
            message=f"partition {pi+1}/{total} mp={mp} rows={rows_written}",
            rows_written=rows_written,
            partitions_built=partitions_built,
            partitions_skipped=partitions_skipped,
        )

    if n_skipped_test == 0:
        # still ok
        pass

    manifest = {
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "generated_at": utc_now(),
        "rows_written": rows_written,
        "rows_resumed": rows_resumed,
        "partitions": len(mps),
        "partitions_built": partitions_built,
        "partitions_skipped": partitions_skipped,
        "resume": resume,
        "feature_n": len(feature_names),
        "skipped_test_rows": n_skipped_test,
        "populations": list(populations),
        "include_rich": include_rich,
        "no_permanent_test": True,
    }
    write_json(out_root / "manifest.json", manifest)
    progress.complete(message="feature cache complete", rows_written=rows_written)
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
