"""Section 1: diagnose Phase-6 2× sparsity and retrain on reaches_plus_100pct."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scripts.review.phase5.common import fit_xgb_deterministic
from scripts.review.phase5.matrices import (
    load_head_bundle,
    walkforward_masks_from_detected,
)
from scripts.review.phase6.common import BROAD_EVENTS, DEFINITIONS, val_window
from scripts.review.phase6.pump_eval import alert_feed_hourly, pump_level_recall
from scripts.review.phase7.common import (
    Q85,
    SCRATCH,
    SEED,
    SPLIT,
    WORLD,
    core_xgb_fixed,
    metrics_block,
    write_json,
)


def _load_reach_and_upside_maps() -> dict[str, Any]:
    """OID -> reaches_plus_100pct / remaining_upside / price_target_available from obs targets."""
    root = WORLD / "broad" / "broad_targets_v1" / "observation_targets"
    cols = [
        "observation_id",
        "explosion_reaches_plus_100pct",
        "remaining_upside_to_explosion_peak",
        "price_target_available",
        "explosion_within_300s",
    ]
    reach: dict[str, bool | None] = {}
    upside: dict[str, float] = {}
    pta: dict[str, bool | None] = {}
    within: dict[str, bool | None] = {}
    for part in sorted(root.glob("part-*.parquet")):
        schema = set(pq.read_schema(part).names)
        use = [c for c in cols if c in schema]
        t = pq.read_table(part, columns=use)
        for row in t.to_pylist():
            oid = str(row["observation_id"])
            r = row.get("explosion_reaches_plus_100pct")
            reach[oid] = None if r is None else bool(r)
            u = row.get("remaining_upside_to_explosion_peak")
            if u is not None:
                upside[oid] = float(u)
            p = row.get("price_target_available")
            pta[oid] = None if p is None else bool(p)
            w = row.get("explosion_within_300s")
            within[oid] = None if w is None else bool(w)
    return {"reach": reach, "upside": upside, "pta": pta, "within": within}


def diagnose_phase6_269() -> dict[str, Any]:
    """Explain why Phase-6 2× used remaining_upside from sparse training_cache."""
    train = load_head_bundle("explosion_300s", "train")
    val = load_head_bundle("explosion_300s", "val")
    maps = _load_reach_and_upside_maps()

    cache_ups: dict[str, float] = {}
    cache = SPLIT / "training_cache_v1" / "broad" / "by_mint_partition"
    for part in sorted(cache.glob("*/joined.parquet")):
        schema = set(pq.read_schema(part).names)
        if "remaining_upside_to_explosion_peak" not in schema:
            continue
        t = pq.read_table(
            part, columns=["observation_id", "remaining_upside_to_explosion_peak"]
        )
        for oid, u in zip(
            t["observation_id"].to_pylist(),
            t["remaining_upside_to_explosion_peak"].to_pylist(),
            strict=True,
        ):
            if u is not None:
                cache_ups[str(oid)] = float(u)

    def stats(oids: np.ndarray, y: np.ndarray, elig: np.ndarray) -> dict[str, Any]:
        pos = np.asarray(elig, dtype=bool) & (np.asarray(y).astype(int) == 1)
        oids_pos = np.asarray(oids)[pos]
        n_pos = int(oids_pos.size)
        n_ot_ups = sum(1 for o in oids_pos if str(o) in maps["upside"])
        n_cache_ups = sum(1 for o in oids_pos if str(o) in cache_ups)
        n_pta_true = sum(1 for o in oids_pos if maps["pta"].get(str(o)) is True)
        n_pta_false = sum(1 for o in oids_pos if maps["pta"].get(str(o)) is False)
        n_reach = sum(1 for o in oids_pos if maps["reach"].get(str(o)) is True)
        n_ot_ge1 = sum(
            1 for o in oids_pos if maps["upside"].get(str(o), float("-inf")) >= 1.0
        )
        n_c_ge1 = sum(
            1 for o in oids_pos if cache_ups.get(str(o), float("-inf")) >= 1.0
        )
        return {
            "n_pos_explosion_300s": n_pos,
            "obs_targets_upside_defined": n_ot_ups,
            "obs_targets_upside_null_frac": (n_pos - n_ot_ups) / n_pos if n_pos else None,
            "obs_targets_upside_ge1": n_ot_ge1,
            "price_target_available_true": n_pta_true,
            "price_target_available_false": n_pta_false,
            "note_pta": (
                "remaining_upside is only set when price_target_available and not "
                "already_started; null upside tracks pta=False on obs_targets."
            ),
            "training_cache_upside_defined": n_cache_ups,
            "training_cache_upside_null_frac": (
                (n_pos - n_cache_ups) / n_pos if n_pos else None
            ),
            "training_cache_upside_ge1": n_c_ge1,
            "obs_targets_reach_plus_100pct_true": n_reach,
        }

    agree = 0
    disagree_onset_vs_t = 0
    n_both = 0
    for oid, r in maps["reach"].items():
        if r is None or oid not in maps["upside"]:
            continue
        n_both += 1
        u_ge1 = maps["upside"][oid] >= 1.0
        if bool(r) == u_ge1:
            agree += 1
        else:
            disagree_onset_vs_t += 1

    return {
        **Q85,
        "phase6_n_pos_was": 269,
        "root_cause": (
            "Phase-6 filled remaining_upside from training_cache_v1, where the column "
            "is ~97% null on explosion_300s positives (many mint partitions entirely "
            "null). Positives with null upside were excluded, leaving only ~269. "
            "Broad observation_targets has upside defined whenever "
            "price_target_available=True and explosion_reaches_plus_100pct on the "
            "order of thousands."
        ),
        "val_stats_on_explosion_300s_positives": stats(
            val["observation_id"], val["y"], val["elig"]
        ),
        "train_stats_on_explosion_300s_positives": stats(
            train["observation_id"], train["y"], train["elig"]
        ),
        "reaches_vs_upside_ge1": {
            "n_both_defined": n_both,
            "n_agree": agree,
            "n_disagree": disagree_onset_vs_t,
            "disagree_rate": disagree_onset_vs_t / n_both if n_both else None,
            "explanation": (
                "explosion_reaches_plus_100pct is magnitude from explosion onset "
                "(peak/onset_price − 1 ≥ 1.0). remaining_upside_to_explosion_peak is "
                "peak/reference_price_at_T − 1, i.e. from observation time T. When T "
                "is after some of the run-up, upside-from-T can be < 1 while "
                "onset-magnitude still ≥ 100%, and vice versa if reference_price "
                "differs from onset_price — so ~20% disagreement is expected."
            ),
            "confirmed_onset_vs_T_measurement": True,
        },
    }


def _build_twox_labels(
    oids: np.ndarray,
    y300: np.ndarray,
    elig300: np.ndarray,
    reach: dict[str, bool | None],
) -> tuple[np.ndarray, np.ndarray]:
    """y=1 iff within_300s and reaches_plus_100pct; exclude if within but reach unknown."""
    y = np.zeros(len(oids), dtype=np.int8)
    elig = np.zeros(len(oids), dtype=bool)
    for i, oid in enumerate(oids):
        if not elig300[i]:
            continue
        r = reach.get(str(oid))
        if int(y300[i]) == 1:
            if r is None:
                continue  # exclude
            y[i] = 1 if r else 0
            elig[i] = True
        else:
            y[i] = 0
            elig[i] = True
    return y, elig


def json_loads_cutoffs() -> dict[str, Any]:
    return json.loads((SPLIT / "cutoffs.json").read_text(encoding="utf-8"))


def _select_fixed_iters(
    X: np.ndarray,
    y: np.ndarray,
    elig: np.ndarray,
    detected: np.ndarray,
) -> int:
    import xgboost as xgb

    from trencher.data.world_model.splits.walkforward import build_walkforward_folds

    cutoffs = json_loads_cutoffs()
    folds = [
        f
        for f in build_walkforward_folds(cutoffs)["folds"]
        if f["fold_id"] in ("wf_fold_1", "wf_fold_2")
    ]
    masks = walkforward_masks_from_detected(detected, folds)
    best_iters: list[int] = []
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "seed": SEED,
        "nthread": 8,
        "verbosity": 0,
    }
    for fold in folds:
        tr = masks[fold["fold_id"]]["train"] & elig
        va = masks[fold["fold_id"]]["val"] & elig
        if tr.sum() < 100 or va.sum() < 50 or y[tr].sum() < 5:
            continue
        dtrain = xgb.DMatrix(X[tr], label=y[tr])
        dval = xgb.DMatrix(X[va], label=y[va])
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=[(dval, "val")],
            early_stopping_rounds=40,
            verbose_eval=False,
        )
        best_iters.append(max(int(booster.best_iteration) + 1, 50))
    if not best_iters:
        return 50
    return int(np.median(best_iters))


def section_twox() -> dict[str, Any]:
    print("diagnosing phase6 2× sparsity...", flush=True)
    diag = diagnose_phase6_269()
    maps = _load_reach_and_upside_maps()

    train = load_head_bundle("explosion_300s", "train")
    val = load_head_bundle("explosion_300s", "val")
    y_tr, elig_tr = _build_twox_labels(
        train["observation_id"], train["y"], train["elig"], maps["reach"]
    )
    y_va, elig_va = _build_twox_labels(
        val["observation_id"], val["y"], val["elig"], maps["reach"]
    )
    counts = {
        "train": {
            "n_elig": int(elig_tr.sum()),
            "n_pos": int(y_tr[elig_tr].sum()),
            "prevalence": float(y_tr[elig_tr].mean()) if elig_tr.any() else None,
        },
        "val": {
            "n_elig": int(elig_va.sum()),
            "n_pos": int(y_va[elig_va].sum()),
            "prevalence": float(y_va[elig_va].mean()) if elig_va.any() else None,
        },
        "definition": (
            "explosion_within_300s==True AND explosion_reaches_plus_100pct==True"
        ),
    }
    print(
        f"  2× counts train n_pos={counts['train']['n_pos']} "
        f"val n_pos={counts['val']['n_pos']}",
        flush=True,
    )

    X_tr = np.asarray(train["X"])
    X_va = np.asarray(val["X"])
    print("  selecting fixed iters on wf_fold_1/2...", flush=True)
    fixed = _select_fixed_iters(X_tr, y_tr, elig_tr, train["detected_at"])
    print(f"  fixed_iterations={fixed}", flush=True)

    preds = fit_xgb_deterministic(
        X_train=X_tr,
        y_train=y_tr.astype(np.float64),
        elig_train=elig_tr,
        times_train=train["times"],
        X_val=X_va,
        fixed_iters=fixed,
        seed=SEED,
        params_override={"nthread": 8, "seed": SEED},
        use_platt=True,
    )["calibrated"]
    yy = y_va[elig_va].astype(int)
    ss = preds[elig_va]
    met = metrics_block(yy, ss)

    alert = alert_feed_hourly(
        scores=preds,
        y300=y_va.astype(int),
        elig300=elig_va,
        mints=val["mint"],
        obs_t=val["times"],
    )

    q70, q85 = val_window()
    ev = pq.read_table(BROAD_EVENTS)
    prim = ev.filter(
        (pc.field("definition_id") == DEFINITIONS["primary"])
        & (pc.field("explosion_exists") == True)  # noqa: E712
    )
    mints_e = prim["mint"].to_pylist()
    onset = prim["onset_unix"].to_pylist()
    mag = prim["explosion_magnitude"].to_pylist()
    events = []
    for m, o, g in zip(mints_e, onset, mag, strict=True):
        if o is None or g is None:
            continue
        if not (q70 <= float(o) < q85):
            continue
        if float(g) < 1.0:
            continue
        events.append({"mint": str(m), "onset_unix": float(o)})
    pump = pump_level_recall(
        scores=preds[elig_va],
        mints=val["mint"][elig_va],
        obs_t=val["times"][elig_va],
        events=events,
    )

    (SCRATCH / "cache").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SCRATCH / "cache" / "twox_val_scores.npz",
        scores=preds,
        y=y_va,
        elig=elig_va,
        observation_id=val["observation_id"],
        mint=val["mint"],
        times=val["times"],
    )

    result = {
        **Q85,
        "diagnosis": diag,
        "counts": counts,
        "fixed_iterations": fixed,
        "core_v2_explosion_300s_fixed_iterations": core_xgb_fixed("explosion_300s"),
        "permanent_val_metrics": met,
        "alert_feed": alert,
        "pump_level_recall_magnitude_ge_1": pump,
        "n_primary_events_mag_ge_1_in_val_window": len(events),
        "permanent_val_configs_scored": [
            {
                "model": "xgb_champion_twox_reaches_plus_100pct",
                "fixed_iterations": fixed,
            }
        ],
    }
    write_json(SCRATCH / "cache" / "section_twox.json", result)
    return result
