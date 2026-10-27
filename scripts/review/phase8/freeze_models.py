#!/usr/bin/env python3
"""Freeze the five V2 locked models under scratch/test_models_v2/ (gitignored)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb
from scripts.review.phase5.common import (
    fit_lgbm_deterministic,
    fit_xgb_deterministic,
    lgbm_deterministic_params,
    metrics_block,
)
from scripts.review.phase5.matrices import load_head_bundle
from scripts.review.phase8.common import (
    AGE_CURVE_FEATURES,
    AGE_CURVE_FIXED,
    HEADS,
    LOGISTIC_C,
    MANIFEST_PATH,
    MODELS_DIR,
    PHASE4_PREDS,
    Q85,
    REPO,
    SEED,
    STAGE_B_CKPT_SRC,
    STAGE_B_PREDS,
    STAGE_B_SHA256,
    TWOX_FIXED_ITERS,
    core_fixed,
    ensure_dirs,
    sha256_file,
    sha256_ids,
    write_json,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trencher.baselines.calibration import PlattCalibrator


def _align_p4(oids: np.ndarray, col: str) -> np.ndarray:
    trees = pq.read_table(PHASE4_PREDS)
    oid_to_i = {str(o): i for i, o in enumerate(trees["observation_id"].to_pylist())}
    idx = np.array([oid_to_i[str(o)] for o in oids], dtype=np.int64)
    return trees[col].to_numpy(zero_copy_only=False).astype(np.float64)[idx]


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _save_platt(path: Path, cal: PlattCalibrator) -> None:
    path.write_text(json.dumps(cal.to_dict(), indent=2) + "\n", encoding="utf-8")


def _feature_list(names: list[str]) -> list[str]:
    return list(names)


def freeze_stage_b() -> dict[str, Any]:
    dst = MODELS_DIR / "stage_b_broad" / "latest.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not STAGE_B_CKPT_SRC.exists():
        raise FileNotFoundError(f"missing Stage-B ckpt: {STAGE_B_CKPT_SRC}")
    shutil.copy2(STAGE_B_CKPT_SRC, dst)
    digest = sha256_file(dst)
    if digest != STAGE_B_SHA256:
        raise RuntimeError(f"Stage-B SHA mismatch: got {digest} want {STAGE_B_SHA256}")
    # Also pin the phase4 permanent-val parquet (CPU infer artifact) for val proof reuse
    preds_dst = MODELS_DIR / "stage_b_broad" / "permanent_val_preds.parquet"
    shutil.copy2(STAGE_B_PREDS, preds_dst)
    return {
        "model_id": "stage_b_broad",
        "path": str(dst.relative_to(REPO)),
        "sha256": digest,
        "config": {
            "checkpoint": "stage_b_broad/latest.pt",
            "mode": "permanent_val_eval",
            "population": "broad",
            "phase": "broad",
        },
        "fixed_iterations": None,
        "feature_list": None,
        "training_rows_hash": None,
        "val_reproduction_max_abs_diff": 0.0,
        "val_preds_parquet": str(preds_dst.relative_to(REPO)),
        "val_preds_sha256": sha256_file(preds_dst),
        "note": "Weights byte-identical to ccf27af7…; val parquet from phase4 step17 infer.",
        **Q85,
    }


def freeze_xgb_champion() -> list[dict[str, Any]]:
    """Copy phase4 XGB + Platt; verify raw preds vs trees parquet (max abs ≤ 1e-6)."""
    out: list[dict[str, Any]] = []
    for head in HEADS:
        src = REPO / "scratch/phase4_clean_v1/trees/xgboost" / head / "final_fit"
        dst = MODELS_DIR / "xgb_champion" / head
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "model.json", dst / "model.json")
        shutil.copy2(src / "calibration.json", dst / "calibration.json")
        fixed = core_fixed("xgboost", head)
        (dst / "config.json").write_text(
            json.dumps(
                {
                    "family": "xgboost",
                    "role": "champion_v2",
                    "head_id": head,
                    "fixed_iterations": fixed,
                    "score_for_metrics": "raw",
                    "params": "baseline_v2_frozen",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        val = load_head_bundle(head, "val")
        train = load_head_bundle(head, "train")
        booster = xgb.Booster()
        booster.load_model(str(dst / "model.json"))
        end = min(fixed, int(booster.num_boosted_rounds()))
        raw = np.asarray(
            booster.predict(xgb.DMatrix(np.asarray(val["X"])), iteration_range=(0, end)),
            dtype=np.float64,
        )
        ref = _align_p4(val["observation_id"], f"xgb_{head}")
        diff = _max_abs(raw, ref)
        if diff > 1e-6:
            raise RuntimeError(f"xgb champion {head} val max_abs={diff} > 1e-6")
        entry = {
            "model_id": f"xgb_champion_{head}",
            "path": str((dst / "model.json").relative_to(REPO)),
            "calibrator_path": str((dst / "calibration.json").relative_to(REPO)),
            "sha256": sha256_file(dst / "model.json"),
            "calibrator_sha256": sha256_file(dst / "calibration.json"),
            "config": json.loads((dst / "config.json").read_text()),
            "fixed_iterations": fixed,
            "feature_list": _feature_list(val["feature_names"]),
            "training_rows_hash": sha256_ids(train["observation_id"]),
            "val_reproduction_max_abs_diff": diff,
            "score_for_metrics": "raw",
            **Q85,
        }
        out.append(entry)
        print(f"  frozen xgb_champion {head} max_abs={diff:.3e}", flush=True)
    return out


def freeze_xgb_twox() -> dict[str, Any]:
    """Deterministic retrain of 2× head; match phase7 twox_val_scores.npz."""
    from scripts.review.phase7.twox import _build_twox_labels, _load_reach_and_upside_maps

    head = "explosion_300s"
    maps = _load_reach_and_upside_maps()
    train = load_head_bundle(head, "train")
    val = load_head_bundle(head, "val")
    y_tr, elig_tr = _build_twox_labels(
        train["observation_id"], train["y"], train["elig"], maps["reach"]
    )
    y_va, elig_va = _build_twox_labels(
        val["observation_id"], val["y"], val["elig"], maps["reach"]
    )
    dst = MODELS_DIR / "xgb_twox" / "explosion_300s_reaches_plus_100pct"
    dst.mkdir(parents=True, exist_ok=True)
    print("  retraining xgb 2×...", flush=True)
    fitted = fit_xgb_deterministic(
        X_train=np.asarray(train["X"]),
        y_train=y_tr.astype(np.float64),
        elig_train=elig_tr,
        times_train=train["times"],
        X_val=np.asarray(val["X"]),
        fixed_iters=TWOX_FIXED_ITERS,
        seed=SEED,
        params_override={"nthread": 8, "seed": SEED},
        use_platt=True,
        out_dir=dst,
    )
    # Persist calibrator explicitly (fit_xgb writes model under final_fit/)
    # Re-fit Platt to save coeffs (same chrono split)
    from trencher.baselines.calibration import chronological_calibration_split

    tr = elig_tr
    fit_idx, cal_idx = chronological_calibration_split(train["times"][tr])
    booster = xgb.Booster()
    booster.load_model(str(dst / "final_fit" / "model.json"))
    end = min(TWOX_FIXED_ITERS, int(booster.num_boosted_rounds()))
    X_tr = np.asarray(train["X"])[tr]
    cal_raw = np.asarray(
        booster.predict(xgb.DMatrix(X_tr[cal_idx]), iteration_range=(0, end)),
        dtype=np.float64,
    )
    cal = PlattCalibrator().fit(cal_raw, y_tr[tr][cal_idx])
    _save_platt(dst / "final_fit" / "calibration.json", cal)
    pred = fitted["calibrated"]
    ref_path = REPO / "scratch/phase7_pretest_v1/cache/twox_val_scores.npz"
    ref = np.load(ref_path, allow_pickle=True)
    # Align by observation_id
    ref_oid = ref["observation_id"]
    oid_to_i = {str(o): i for i, o in enumerate(ref_oid.tolist())}
    idx = np.array([oid_to_i[str(o)] for o in val["observation_id"]], dtype=np.int64)
    ref_s = np.asarray(ref["scores"][idx], dtype=np.float64)
    diff = _max_abs(pred, ref_s)
    if diff > 1e-6:
        raise RuntimeError(f"xgb twox val max_abs={diff} > 1e-6")
    (dst / "config.json").write_text(
        json.dumps(
            {
                "family": "xgboost",
                "role": "twox_reaches_plus_100pct",
                "definition": "within_300s AND reaches_plus_100pct",
                "fixed_iterations": TWOX_FIXED_ITERS,
                "score_for_metrics": "calibrated",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  frozen xgb_twox max_abs={diff:.3e}", flush=True)
    return {
        "model_id": "xgb_champion_twox_reaches_plus_100pct",
        "path": str((dst / "final_fit" / "model.json").relative_to(REPO)),
        "calibrator_path": str((dst / "final_fit" / "calibration.json").relative_to(REPO)),
        "sha256": sha256_file(dst / "final_fit" / "model.json"),
        "calibrator_sha256": sha256_file(dst / "final_fit" / "calibration.json"),
        "config": json.loads((dst / "config.json").read_text()),
        "fixed_iterations": TWOX_FIXED_ITERS,
        "feature_list": _feature_list(val["feature_names"]),
        "training_rows_hash": sha256_ids(train["observation_id"]),
        "val_reproduction_max_abs_diff": diff,
        "score_for_metrics": "calibrated",
        **Q85,
    }


def freeze_lgbm_tuned() -> list[dict[str, Any]]:
    """Retrain phase5 tuned LightGBM (no SPW); match phase5 permanent_val metrics via scores."""
    out: list[dict[str, Any]] = []
    for head in HEADS:
        train = load_head_bundle(head, "train")
        val = load_head_bundle(head, "val")
        # Phase5 selected median_best_iterations=50 for both heads
        fixed = 50
        cfg = {
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "feature_fraction": 0.8,
            "learning_rate": 0.05,
            "min_data_in_leaf": 100,
            "num_leaves": 63,
            "use_scale_pos_weight": False,
        }
        params = lgbm_deterministic_params(
            seed=SEED,
            num_threads=1,
            learning_rate=cfg["learning_rate"],
            num_leaves=cfg["num_leaves"],
            min_data_in_leaf=cfg["min_data_in_leaf"],
            feature_fraction=cfg["feature_fraction"],
            bagging_fraction=cfg["bagging_fraction"],
            bagging_freq=cfg["bagging_freq"],
            scale_pos_weight=None,
        )
        dst = MODELS_DIR / "lgbm_tuned" / head
        dst.mkdir(parents=True, exist_ok=True)
        print(f"  retraining lgbm_tuned {head}...", flush=True)
        fitted = fit_lgbm_deterministic(
            X_train=np.asarray(train["X"]),
            y_train=train["y"],
            elig_train=train["elig"],
            times_train=train["times"],
            X_val=np.asarray(val["X"]),
            fixed_iters=fixed,
            seed=SEED,
            params_override=params,
            use_platt=True,
            out_dir=dst,
        )
        # Save calibrator
        import lightgbm as lgb

        from trencher.baselines.calibration import chronological_calibration_split

        tr = train["elig"].astype(bool)
        fit_idx, cal_idx = chronological_calibration_split(train["times"][tr])
        booster = lgb.Booster(model_file=str(dst / "final_fit" / "model.txt"))
        X_tr = np.asarray(train["X"])[tr]
        cal_raw = np.asarray(
            booster.predict(X_tr[cal_idx], num_iteration=fixed), dtype=np.float64
        )
        cal = PlattCalibrator().fit(cal_raw, train["y"][tr][cal_idx])
        _save_platt(dst / "final_fit" / "calibration.json", cal)
        pred = fitted["calibrated"]
        # Reference: re-score against phase5 metrics (no score parquet); also
        # compare to a second deterministic predict from saved model.
        pred2 = cal.transform(
            np.asarray(booster.predict(np.asarray(val["X"]), num_iteration=fixed), dtype=np.float64)
        )
        diff_self = _max_abs(pred, pred2)
        elig = val["elig"].astype(bool)
        met = metrics_block(val["y"][elig].astype(int), pred[elig])
        ref = json.loads(
            (
                REPO / f"scratch/phase5_trees_v1/cache/score_lightgbm_{head}.json"
            ).read_text(encoding="utf-8")
        )["permanent_val_metrics"]
        roc_diff = abs(met["roc_auc"] - ref["roc_auc"])
        pr_diff = abs(met["pr_auc"] - ref["pr_auc"])
        if roc_diff > 1e-6 or pr_diff > 1e-6 or diff_self > 1e-6:
            raise RuntimeError(
                f"lgbm {head} mismatch roc_diff={roc_diff} pr_diff={pr_diff} "
                f"self={diff_self}"
            )
        (dst / "config.json").write_text(
            json.dumps(
                {
                    "family": "lightgbm",
                    "role": "phase5_tuned",
                    "head_id": head,
                    "fixed_iterations": fixed,
                    "config": cfg,
                    "score_for_metrics": "calibrated",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        # Store val scores for scorer reproduction
        np.savez_compressed(
            dst / "val_scores.npz",
            scores=pred,
            observation_id=val["observation_id"],
        )
        print(
            f"  frozen lgbm_tuned {head} roc_diff={roc_diff:.3e} pr_diff={pr_diff:.3e}",
            flush=True,
        )
        out.append(
            {
                "model_id": f"lgbm_tuned_{head}",
                "path": str((dst / "final_fit" / "model.txt").relative_to(REPO)),
                "calibrator_path": str(
                    (dst / "final_fit" / "calibration.json").relative_to(REPO)
                ),
                "sha256": sha256_file(dst / "final_fit" / "model.txt"),
                "calibrator_sha256": sha256_file(dst / "final_fit" / "calibration.json"),
                "config": json.loads((dst / "config.json").read_text()),
                "fixed_iterations": fixed,
                "feature_list": _feature_list(val["feature_names"]),
                "training_rows_hash": sha256_ids(train["observation_id"]),
                "val_reproduction_max_abs_diff": max(roc_diff, pr_diff, diff_self),
                "val_metric_abs_diff": {"roc_auc": roc_diff, "pr_auc": pr_diff},
                "score_for_metrics": "calibrated",
                **Q85,
            }
        )
    return out


def freeze_logistic_l2() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for head in HEADS:
        train = load_head_bundle(head, "train")
        val = load_head_bundle(head, "val")
        tr = train["elig"].astype(bool)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(np.asarray(train["X"][tr], dtype=np.float64))
        clf = LogisticRegression(
            C=LOGISTIC_C,
            penalty="l2",
            solver="lbfgs",
            max_iter=500,
            random_state=SEED,
        )
        clf.fit(Xtr, train["y"][tr].astype(int))
        Xva = scaler.transform(np.asarray(val["X"], dtype=np.float64))
        scores = clf.predict_proba(Xva)[:, 1]
        dst = MODELS_DIR / "logistic_l2" / head
        dst.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": scaler, "clf": clf}, dst / "model.joblib")
        (dst / "config.json").write_text(
            json.dumps(
                {
                    "family": "logistic_l2",
                    "C": LOGISTIC_C,
                    "head_id": head,
                    "selection": "phase6_wf_fold_1_2_mean_pr",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            dst / "val_scores.npz", scores=scores, observation_id=val["observation_id"]
        )
        elig = val["elig"].astype(bool)
        met = metrics_block(val["y"][elig].astype(int), scores[elig])
        # Compare to phase7 strata overall logistic
        st = json.loads(
            (REPO / "scratch/phase7_pretest_v1/cache/section_strata.json").read_text()
        )
        ref = st["by_head"][head]["overall"]["models"]["logistic_l2"]
        roc_diff = abs(met["roc_auc"] - ref["roc_auc"])
        pr_diff = abs(met["pr_auc"] - ref["pr_auc"])
        if roc_diff > 1e-6 or pr_diff > 1e-6:
            raise RuntimeError(
                f"logistic {head} mismatch roc_diff={roc_diff} pr_diff={pr_diff}"
            )
        print(
            f"  frozen logistic_l2 {head} roc_diff={roc_diff:.3e} pr_diff={pr_diff:.3e}",
            flush=True,
        )
        out.append(
            {
                "model_id": f"logistic_l2_{head}",
                "path": str((dst / "model.joblib").relative_to(REPO)),
                "sha256": sha256_file(dst / "model.joblib"),
                "config": json.loads((dst / "config.json").read_text()),
                "fixed_iterations": None,
                "feature_list": _feature_list(val["feature_names"]),
                "training_rows_hash": sha256_ids(train["observation_id"]),
                "val_reproduction_max_abs_diff": max(roc_diff, pr_diff),
                "val_metric_abs_diff": {"roc_auc": roc_diff, "pr_auc": pr_diff},
                **Q85,
            }
        )
    return out


def freeze_age_curve_xgb() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for head in HEADS:
        train = load_head_bundle(head, "train")
        val = load_head_bundle(head, "val")
        names = list(train["feature_names"])
        idx = [names.index(n) for n in AGE_CURVE_FEATURES]
        fixed = AGE_CURVE_FIXED[head]
        dst = MODELS_DIR / "age_curve_xgb" / head
        dst.mkdir(parents=True, exist_ok=True)
        print(f"  retraining age_curve_xgb {head}...", flush=True)
        fitted = fit_xgb_deterministic(
            X_train=np.asarray(train["X"][:, idx]),
            y_train=train["y"],
            elig_train=train["elig"],
            times_train=train["times"],
            X_val=np.asarray(val["X"][:, idx]),
            fixed_iters=fixed,
            seed=SEED,
            params_override={"nthread": 8, "seed": SEED},
            use_platt=True,
            out_dir=dst,
        )
        from trencher.baselines.calibration import chronological_calibration_split

        tr = train["elig"].astype(bool)
        fit_idx, cal_idx = chronological_calibration_split(train["times"][tr])
        booster = xgb.Booster()
        booster.load_model(str(dst / "final_fit" / "model.json"))
        end = min(fixed, int(booster.num_boosted_rounds()))
        X_tr = np.asarray(train["X"][:, idx])[tr]
        cal_raw = np.asarray(
            booster.predict(xgb.DMatrix(X_tr[cal_idx]), iteration_range=(0, end)),
            dtype=np.float64,
        )
        cal = PlattCalibrator().fit(cal_raw, train["y"][tr][cal_idx])
        _save_platt(dst / "final_fit" / "calibration.json", cal)
        pred = fitted["calibrated"]
        np.savez_compressed(
            dst / "val_scores.npz", scores=pred, observation_id=val["observation_id"]
        )
        elig = val["elig"].astype(bool)
        met = metrics_block(val["y"][elig].astype(int), pred[elig])
        st = json.loads(
            (REPO / "scratch/phase7_pretest_v1/cache/section_strata.json").read_text()
        )
        ref = st["by_head"][head]["overall"]["models"]["age_curve_xgb"]
        roc_diff = abs(met["roc_auc"] - ref["roc_auc"])
        pr_diff = abs(met["pr_auc"] - ref["pr_auc"])
        if roc_diff > 1e-6 or pr_diff > 1e-6:
            raise RuntimeError(
                f"age_curve {head} mismatch roc_diff={roc_diff} pr_diff={pr_diff}"
            )
        (dst / "config.json").write_text(
            json.dumps(
                {
                    "family": "xgboost",
                    "role": "age_curve_baseline",
                    "head_id": head,
                    "fixed_iterations": fixed,
                    "features": AGE_CURVE_FEATURES,
                    "score_for_metrics": "calibrated",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"  frozen age_curve_xgb {head} roc_diff={roc_diff:.3e} pr_diff={pr_diff:.3e}",
            flush=True,
        )
        out.append(
            {
                "model_id": f"age_curve_xgb_{head}",
                "path": str((dst / "final_fit" / "model.json").relative_to(REPO)),
                "calibrator_path": str(
                    (dst / "final_fit" / "calibration.json").relative_to(REPO)
                ),
                "sha256": sha256_file(dst / "final_fit" / "model.json"),
                "calibrator_sha256": sha256_file(dst / "final_fit" / "calibration.json"),
                "config": json.loads((dst / "config.json").read_text()),
                "fixed_iterations": fixed,
                "feature_list": list(AGE_CURVE_FEATURES),
                "training_rows_hash": sha256_ids(train["observation_id"]),
                "val_reproduction_max_abs_diff": max(roc_diff, pr_diff),
                "val_metric_abs_diff": {"roc_auc": roc_diff, "pr_auc": pr_diff},
                "score_for_metrics": "calibrated",
                **Q85,
            }
        )
    return out


def build_manifest() -> dict[str, Any]:
    ensure_dirs()
    print("=== freezing Stage B ===", flush=True)
    models: list[dict[str, Any]] = [freeze_stage_b()]
    print("=== freezing XGB champion ===", flush=True)
    models.extend(freeze_xgb_champion())
    print("=== freezing XGB 2× ===", flush=True)
    models.append(freeze_xgb_twox())
    print("=== freezing tuned LightGBM ===", flush=True)
    models.extend(freeze_lgbm_tuned())
    print("=== freezing logistic_l2 ===", flush=True)
    models.extend(freeze_logistic_l2())
    print("=== freezing age+curve XGB ===", flush=True)
    models.extend(freeze_age_curve_xgb())

    bad = [m for m in models if (m.get("val_reproduction_max_abs_diff") or 0) > 1e-6]
    if bad:
        raise RuntimeError(f"freeze verification failed: {[m['model_id'] for m in bad]}")

    manifest = {
        **Q85,
        "protocol": "TEST_PROTOCOL_V2.md",
        "models_root": str(MODELS_DIR.relative_to(REPO)),
        "models": {m["model_id"]: m for m in models},
        "n_models": len(models),
        "all_val_reproduction_pass": True,
    }
    write_json(MANIFEST_PATH, manifest)
    write_json(MODELS_DIR / "MANIFEST.json", manifest)
    print(f"wrote {MANIFEST_PATH}", flush=True)
    return manifest


def main() -> int:
    build_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
