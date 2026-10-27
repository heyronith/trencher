"""Section 3: WF-fold grid tune for LGBM/XGB; score best once on permanent val."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from scripts.review.phase4.fast_boot import mint_bootstrap_delta
from scripts.review.phase5.common import (
    CORE_METRICS,
    Q85,
    SCRATCH,
    SEED,
    SPLIT,
    fit_lgbm_deterministic,
    fit_xgb_deterministic,
    lgbm_deterministic_params,
    metrics_block,
    write_json,
    xgb_deterministic_params,
)
from scripts.review.phase5.matrices import (
    load_head_bundle,
    walkforward_masks_from_detected,
)


def _lgbm_grid() -> list[dict[str, Any]]:
    """≤24 curated configs covering leaves, min_child, lr, subsample, spw."""
    configs: list[dict[str, Any]] = []
    # 12 configs: cover each axis without full factorial (≤24 allowed).
    for num_leaves, min_data, lr, ff, bf, use_spw in [
        (63, 100, 0.05, 0.8, 0.8, True),  # V2-like + spw
        (63, 100, 0.05, 0.8, 0.8, False),  # V2-like no spw
        (31, 100, 0.05, 0.8, 0.8, True),
        (127, 100, 0.05, 0.8, 0.8, True),
        (63, 50, 0.05, 0.8, 0.8, True),
        (63, 200, 0.05, 0.8, 0.8, True),
        (63, 100, 0.02, 0.8, 0.8, True),
        (63, 100, 0.1, 0.8, 0.8, True),
        (63, 100, 0.05, 1.0, 1.0, True),
        (63, 100, 0.05, 0.6, 0.8, True),
        (31, 50, 0.02, 0.8, 0.8, True),
        (127, 200, 0.1, 0.8, 0.8, True),
    ]:
        configs.append(
            {
                "num_leaves": num_leaves,
                "min_data_in_leaf": min_data,
                "learning_rate": lr,
                "feature_fraction": ff,
                "bagging_fraction": bf,
                "bagging_freq": 1 if bf < 1.0 else 0,
                "use_scale_pos_weight": use_spw,
            }
        )
    assert len(configs) <= 24
    return configs


def _xgb_grid() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for max_depth, min_child, lr, subsample, colsample, use_spw in [
        (8, 1, 0.05, 0.8, 0.8, True),
        (8, 1, 0.05, 0.8, 0.8, False),
        (6, 1, 0.05, 0.8, 0.8, True),
        (10, 1, 0.05, 0.8, 0.8, True),
        (8, 5, 0.05, 0.8, 0.8, True),
        (8, 20, 0.05, 0.8, 0.8, True),
        (8, 1, 0.02, 0.8, 0.8, True),
        (8, 1, 0.1, 0.8, 0.8, True),
        (8, 1, 0.05, 1.0, 1.0, True),
        (8, 1, 0.05, 0.6, 0.8, True),
        (6, 5, 0.02, 0.8, 0.8, True),
        (10, 20, 0.1, 0.8, 0.8, True),
    ]:
        configs.append(
            {
                "max_depth": max_depth,
                "min_child_weight": min_child,
                "learning_rate": lr,
                "subsample": subsample,
                "colsample_bytree": colsample,
                "use_scale_pos_weight": use_spw,
            }
        )
    assert len(configs) <= 24
    return configs


def _pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    from trencher.baselines.metrics import binary_metrics

    return float(binary_metrics(y.astype(int), s)["pr_auc"])


def _train_lgbm_es(
    *,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cfg: dict[str, Any],
    spw: float | None,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 40,
    num_threads: int = 8,
) -> tuple[float, int]:
    import lightgbm as lgb

    params = lgbm_deterministic_params(
        seed=SEED,
        num_threads=num_threads,
        learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"],
        min_data_in_leaf=cfg["min_data_in_leaf"],
        feature_fraction=cfg["feature_fraction"],
        bagging_fraction=cfg["bagging_fraction"],
        bagging_freq=cfg["bagging_freq"],
        scale_pos_weight=spw if cfg["use_scale_pos_weight"] else None,
    )
    # ES path: allow multi-thread; drop force_row_wise for speed
    params.pop("force_row_wise", None)
    params["deterministic"] = False
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_it = int(booster.best_iteration or booster.current_iteration() or num_boost_round)
    pred = np.asarray(booster.predict(X_va, num_iteration=best_it), dtype=np.float64)
    return _pr_auc(y_va, pred), best_it


def _train_xgb_es(
    *,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cfg: dict[str, Any],
    spw: float | None,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 40,
    nthread: int = 8,
) -> tuple[float, int]:
    import xgboost as xgb

    params = xgb_deterministic_params(
        seed=SEED,
        nthread=nthread,
        learning_rate=cfg["learning_rate"],
        max_depth=cfg["max_depth"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        scale_pos_weight=spw if cfg["use_scale_pos_weight"] else None,
    )
    params["min_child_weight"] = cfg["min_child_weight"]
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    best_it = int(booster.best_iteration) + 1  # 0-indexed -> count
    end = min(best_it, int(booster.num_boosted_rounds()))
    pred = np.asarray(
        booster.predict(dval, iteration_range=(0, end)), dtype=np.float64
    )
    return _pr_auc(y_va, pred), best_it


def _selection_folds() -> list[dict[str, Any]]:
    from trencher.data.world_model.splits.walkforward import build_walkforward_folds

    cutoffs = json.loads((SPLIT / "cutoffs.json").read_text(encoding="utf-8"))
    folds = build_walkforward_folds(cutoffs)["folds"]
    return [f for f in folds if f["fold_id"] in ("wf_fold_1", "wf_fold_2")]


def tune_family_head(
    *,
    family: str,
    head_id: str,
    train: dict[str, Any],
    fold_masks: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    configs = _lgbm_grid() if family == "lightgbm" else _xgb_grid()
    elig = train["elig"].astype(bool)
    y = train["y"]
    X = np.asarray(train["X"])  # may copy from mmap for speed
    results = []
    for ci, cfg in enumerate(configs):
        fold_pr = []
        fold_best = []
        t0 = time.time()
        for _fid, masks in fold_masks.items():
            tr_m = masks["train"] & elig
            va_m = masks["val"] & elig
            X_tr, y_tr = X[tr_m], y[tr_m]
            X_va, y_va = X[va_m], y[va_m]
            if X_tr.shape[0] < 100 or X_va.shape[0] < 50 or y_tr.sum() < 5:
                continue
            pos = float(y_tr.sum())
            neg = float(len(y_tr) - pos)
            spw = neg / max(pos, 1.0)
            if family == "lightgbm":
                pr, bi = _train_lgbm_es(
                    X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, cfg=cfg, spw=spw
                )
            else:
                pr, bi = _train_xgb_es(
                    X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, cfg=cfg, spw=spw
                )
            fold_pr.append(pr)
            fold_best.append(bi)
        mean_pr = float(np.mean(fold_pr)) if fold_pr else float("nan")
        median_iters = int(np.median(fold_best)) if fold_best else 50
        median_iters = max(median_iters, 50)
        rec = {
            "config_id": ci,
            "config": cfg,
            "fold_pr_auc": fold_pr,
            "mean_pr_auc": mean_pr,
            "fold_best_iterations": fold_best,
            "median_best_iterations": median_iters,
            "runtime_sec": float(time.time() - t0),
        }
        results.append(rec)
        print(
            f"  {family}/{head_id} cfg={ci} mean_pr={mean_pr:.5f} "
            f"iters={median_iters} ({rec['runtime_sec']:.1f}s)",
            flush=True,
        )
    results.sort(key=lambda r: (-(r["mean_pr_auc"] if np.isfinite(r["mean_pr_auc"]) else -1),))
    best = results[0]
    return {
        "family": family,
        "head_id": head_id,
        "n_configs": len(configs),
        "selection_folds": ["wf_fold_1", "wf_fold_2"],
        "selection_metric": "mean_pr_auc",
        "all_configs": results,
        "best": best,
        **Q85,
    }


def score_best_on_permanent_val(
    *,
    family: str,
    head_id: str,
    best: dict[str, Any],
    train: dict[str, Any],
    val: dict[str, Any],
) -> dict[str, Any]:
    """Final fit once; log permanent-val scoring."""
    cfg = best["config"]
    fixed = int(best["median_best_iterations"])
    elig_tr = train["elig"].astype(bool)
    y_tr = train["y"]
    times_tr = train["times"]
    X_tr = np.asarray(train["X"])
    X_va = np.asarray(val["X"])
    elig_va = val["elig"].astype(bool)
    y_va = val["y"].astype(int)

    pos = float(y_tr[elig_tr].sum())
    neg = float(elig_tr.sum() - pos)
    spw = (neg / max(pos, 1.0)) if cfg["use_scale_pos_weight"] else None

    out_dir = SCRATCH / "tuned" / family / head_id
    # Multi-thread final fit (selection already locked); bit-determinism not required here.
    if family == "lightgbm":
        override = {
            "learning_rate": cfg["learning_rate"],
            "num_leaves": cfg["num_leaves"],
            "min_data_in_leaf": cfg["min_data_in_leaf"],
            "feature_fraction": cfg["feature_fraction"],
            "bagging_fraction": cfg["bagging_fraction"],
            "bagging_freq": cfg["bagging_freq"],
            "num_threads": 8,
            "deterministic": False,
            "force_row_wise": False,
            "seed": SEED,
            "bagging_seed": SEED,
            "feature_fraction_seed": SEED,
        }
        if spw is not None:
            override["scale_pos_weight"] = spw
        preds = fit_lgbm_deterministic(
            X_train=X_tr,
            y_train=y_tr,
            elig_train=elig_tr,
            times_train=times_tr,
            X_val=X_va,
            fixed_iters=fixed,
            out_dir=out_dir,
            params_override=override,
            use_platt=True,
        )
        score = preds["calibrated"]
    else:
        override = {
            "learning_rate": cfg["learning_rate"],
            "max_depth": cfg["max_depth"],
            "subsample": cfg["subsample"],
            "colsample_bytree": cfg["colsample_bytree"],
            "min_child_weight": cfg["min_child_weight"],
            "nthread": 8,
            "seed": SEED,
        }
        if spw is not None:
            override["scale_pos_weight"] = spw
        preds = fit_xgb_deterministic(
            X_train=X_tr,
            y_train=y_tr,
            elig_train=elig_tr,
            times_train=times_tr,
            X_val=X_va,
            fixed_iters=fixed,
            out_dir=out_dir,
            params_override=override,
            use_platt=True,
        )
        score = preds["calibrated"]

    yy = y_va[elig_va]
    ss = score[elig_va]
    met = metrics_block(yy, ss)

    # Frozen V2 baseline comparison from phase4 preds / CORE metrics
    core = json.loads(CORE_METRICS.read_text(encoding="utf-8"))
    fam_key = "lightgbm" if family == "lightgbm" else "xgboost"
    head_core = core[fam_key]["heads"][head_id]
    # CORE calibrated stores PR/Brier/ECE; ROC is under raw (rank-invariant to Platt).
    frozen = {
        "roc_auc": float(head_core["raw"]["roc_auc"]),
        "pr_auc": float(head_core["calibrated"]["pr_auc"]),
        "raw_pr_auc": float(head_core["raw"]["pr_auc"]),
    }
    # Bootstrap vs frozen scores from phase4 parquet if available
    import pyarrow.parquet as pq
    from scripts.review.phase5.common import PHASE4_PREDS

    p4 = pq.read_table(PHASE4_PREDS)
    # Align by observation_id
    fam_col = "lgb" if family == "lightgbm" else "xgb"
    p4_oids = np.array(p4["observation_id"].to_pylist(), dtype=object)
    p4_score = p4[f"{fam_col}_{head_id}"].to_numpy(zero_copy_only=False).astype(np.float64)
    p4_y = np.nan_to_num(
        p4[f"y_{head_id}"].to_numpy(zero_copy_only=False), nan=0.0
    ).astype(int)
    p4_elig = p4[f"elig_{head_id}"].to_numpy(zero_copy_only=False).astype(bool)
    p4_mints = np.array(p4["mint"].to_pylist(), dtype=object)

    oid_to_i = {str(o): i for i, o in enumerate(val["observation_id"])}
    # Build aligned arrays on p4 eligible rows that exist in val
    aligned_y = []
    aligned_tuned = []
    aligned_frozen = []
    aligned_mints = []
    for i, oid in enumerate(p4_oids):
        if not p4_elig[i]:
            continue
        j = oid_to_i.get(str(oid))
        if j is None or not elig_va[j]:
            continue
        aligned_y.append(int(p4_y[i]))
        aligned_frozen.append(float(p4_score[i]))
        aligned_tuned.append(float(score[j]))
        aligned_mints.append(p4_mints[i])
    ay = np.asarray(aligned_y, dtype=int)
    at = np.asarray(aligned_tuned, dtype=np.float64)
    af = np.asarray(aligned_frozen, dtype=np.float64)
    am = np.asarray(aligned_mints, dtype=object)

    print(
        f"  bootstrap CIs n={ay.size} n_mints={len({str(x) for x in am})}...",
        flush=True,
    )
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_roc = pool.submit(
            mint_bootstrap_delta, ay, af, at, am, metric="roc_auc"
        )
        fut_pr = pool.submit(
            mint_bootstrap_delta, ay, af, at, am, metric="pr_auc"
        )
        delta = {"roc_auc": fut_roc.result(), "pr_auc": fut_pr.result()}
    return {
        "family": family,
        "head_id": head_id,
        "config": cfg,
        "fixed_iterations": fixed,
        "permanent_val_metrics": met,
        "frozen_v2_calibrated_from_core": {
            "roc_auc": frozen["roc_auc"],
            "pr_auc": frozen["pr_auc"],
        },
        "delta_tuned_minus_frozen_phase4": delta,
        "permanent_val_scored": True,
        "note": (
            "Selected on wf_fold_1/2 mean PR-AUC only; permanent val scored once."
        ),
        **Q85,
    }


def section_tune() -> dict[str, Any]:
    folds = _selection_folds()
    out: dict[str, Any] = {
        **Q85,
        "permanent_val_configs_scored": [],
        "by_family_head": {},
        "selection_protocol": {
            "folds": ["wf_fold_1", "wf_fold_2"],
            "metric": "mean_pr_auc",
            "permanent_val": "score_best_once_per_family_head",
            "max_configs_per_family": 24,
        },
    }
    for head_id in ("explosion_60s", "explosion_300s"):
        train = load_head_bundle(head_id, "train")
        val = load_head_bundle(head_id, "val")
        fold_masks = walkforward_masks_from_detected(train["detected_at"], folds)
        for family in ("lightgbm", "xgboost"):
            key = f"{family}/{head_id}"
            tune_path = SCRATCH / "cache" / f"tune_{family}_{head_id}.json"
            score_path = SCRATCH / "cache" / f"score_{family}_{head_id}.json"
            if tune_path.exists():
                print(f"=== reuse cached selection {key} ===", flush=True)
                tune_res = json.loads(tune_path.read_text(encoding="utf-8"))
            else:
                print(f"=== tuning {key} ===", flush=True)
                tune_res = tune_family_head(
                    family=family,
                    head_id=head_id,
                    train=train,
                    fold_masks=fold_masks,
                )
                write_json(tune_path, tune_res)
            if score_path.exists():
                print(f"=== reuse cached permanent-val score {key} ===", flush=True)
                scored = json.loads(score_path.read_text(encoding="utf-8"))
            else:
                print(f"=== scoring best {key} on permanent val ===", flush=True)
                scored = score_best_on_permanent_val(
                    family=family,
                    head_id=head_id,
                    best=tune_res["best"],
                    train=train,
                    val=val,
                )
                write_json(score_path, scored)
            out["permanent_val_configs_scored"].append(
                {
                    "family": family,
                    "head_id": head_id,
                    "config_id": tune_res["best"]["config_id"],
                    "config": tune_res["best"]["config"],
                    "fixed_iterations": scored["fixed_iterations"],
                }
            )
            out["by_family_head"][key] = {
                "selection": {
                    "best_config_id": tune_res["best"]["config_id"],
                    "best_config": tune_res["best"]["config"],
                    "mean_pr_auc_wf": tune_res["best"]["mean_pr_auc"],
                    "median_best_iterations": tune_res["best"]["median_best_iterations"],
                    "n_configs": tune_res["n_configs"],
                },
                "permanent_val": scored,
            }
    write_json(SCRATCH / "cache" / "section_tune.json", out)
    return out
