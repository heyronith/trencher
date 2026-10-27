"""Shared paths and helpers for Phase-5."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.review.lead_intake_lib import (  # noqa: E402
    BOOT_SEED,
    N_BOOT,
    binary_auc_ap,
    mint_bootstrap_metric_delta,
    topk_precision_recall,
    write_json,
)

WORLD = REPO / "data" / "external" / "world_model"
SPLIT = WORLD / "splits" / "split_v1"
CLEAN_ROOT = REPO / "data" / "external" / "world_model_clean_v1"
SCRATCH = REPO / "scratch" / "phase5_trees_v1"
OUT = REPO / "manifests" / "world_model_data" / "evaluation" / "phase5_trees_v1"
PHASE4_PREDS = REPO / "scratch" / "phase4_clean_v1" / "preds" / "trees_broad_permanent_val.parquet"
CORE_METRICS = (
    REPO / "manifests" / "world_model_data" / "baseline_v2" / "BASELINE_CORE_METRICS_V2.json"
)
RICH_ABLATION = (
    REPO / "manifests" / "world_model_data" / "baseline_v2" / "BASELINE_RICH_ABLATION_V2.json"
)
Q85 = {"q85_test_accessed": False}
TOP_FRACS = [0.001, 0.005, 0.01, 0.02]
SEED = 0


def ensure_dirs() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "cache").mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    CLEAN_ROOT.mkdir(parents=True, exist_ok=True)


def cached_json(name: str, fn):  # noqa: ANN001
    path = SCRATCH / "cache" / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    val = fn()
    write_json(path, val)
    return val


def metrics_block(y: np.ndarray, s: np.ndarray) -> dict[str, Any]:
    met = binary_auc_ap(y, s)
    top = topk_precision_recall(y, s, TOP_FRACS)
    return {
        "n": int(met["n"]),
        "n_pos": int(y.sum()),
        "prevalence": float(met["prevalence"]),
        "roc_auc": float(met["roc_auc"]),
        "pr_auc": float(met["pr_auc"]),
        "precision_at": {
            f"top_{f:g}": float(top[f"top_{f:g}"]["precision"]) for f in TOP_FRACS
        },
        "recall_at": {f"top_{f:g}": float(top[f"top_{f:g}"]["recall"]) for f in TOP_FRACS},
        **Q85,
    }


def lgbm_deterministic_params(
    *,
    seed: int = SEED,
    num_threads: int = 1,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    min_data_in_leaf: int = 100,
    feature_fraction: float = 0.8,
    bagging_fraction: float = 0.8,
    bagging_freq: int = 1,
    scale_pos_weight: float | None = None,
) -> dict[str, Any]:
    """Fully pinned LightGBM params (deterministic hist + single thread)."""
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": bagging_freq,
        "verbosity": -1,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "num_threads": num_threads,
        "deterministic": True,
        "force_row_wise": True,
    }
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)
    return params


def xgb_deterministic_params(
    *,
    seed: int = SEED,
    nthread: int = 1,
    learning_rate: float = 0.05,
    max_depth: int = 8,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: float | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "tree_method": "hist",
        "seed": seed,
        "nthread": nthread,
        "verbosity": 0,
    }
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)
    return params


def fit_lgbm_deterministic(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    elig_train: np.ndarray,
    times_train: np.ndarray,
    X_val: np.ndarray,
    fixed_iters: int,
    out_dir: Path | None = None,
    seed: int = SEED,
    params_override: dict[str, Any] | None = None,
    use_platt: bool = False,
) -> dict[str, np.ndarray]:
    """Chrono fit/cal split; train on fit_idx with fixed iters; optional Platt."""
    import lightgbm as lgb

    from trencher.baselines.calibration import (
        PlattCalibrator,
        chronological_calibration_split,
    )
    from trencher.data.world_model.io import ensure_dir

    tr = np.asarray(elig_train, dtype=bool)
    X_tr, y_tr = X_train[tr], y_train[tr]
    times = times_train[tr]
    fit_idx, cal_idx = chronological_calibration_split(times)
    params = lgbm_deterministic_params(seed=seed)
    if params_override:
        params.update(params_override)
    dtrain = lgb.Dataset(X_tr[fit_idx], label=y_tr[fit_idx])
    booster = lgb.train(params, dtrain, num_boost_round=int(fixed_iters))
    n_it = int(fixed_iters)
    val_raw = np.asarray(booster.predict(X_val, num_iteration=n_it), dtype=np.float64)
    out: dict[str, np.ndarray] = {"raw": val_raw}
    if use_platt:
        cal_raw = np.asarray(
            booster.predict(X_tr[cal_idx], num_iteration=n_it), dtype=np.float64
        )
        cal = PlattCalibrator().fit(cal_raw, y_tr[cal_idx])
        out["calibrated"] = cal.transform(val_raw)
    if out_dir is not None:
        out_dir = ensure_dir(out_dir / "final_fit")
        booster.save_model(str(out_dir / "model.txt"))
    return out


def fit_xgb_deterministic(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    elig_train: np.ndarray,
    times_train: np.ndarray,
    X_val: np.ndarray,
    fixed_iters: int,
    out_dir: Path | None = None,
    seed: int = SEED,
    params_override: dict[str, Any] | None = None,
    use_platt: bool = True,
) -> dict[str, np.ndarray]:
    import xgboost as xgb

    from trencher.baselines.calibration import (
        PlattCalibrator,
        chronological_calibration_split,
    )
    from trencher.data.world_model.io import ensure_dir

    tr = np.asarray(elig_train, dtype=bool)
    X_tr, y_tr = X_train[tr], y_train[tr]
    times = times_train[tr]
    fit_idx, cal_idx = chronological_calibration_split(times)
    params = xgb_deterministic_params(seed=seed)
    if params_override:
        params.update(params_override)
    dtrain = xgb.DMatrix(X_tr[fit_idx], label=y_tr[fit_idx])
    booster = xgb.train(params, dtrain, num_boost_round=int(fixed_iters))
    end = min(int(fixed_iters), int(booster.num_boosted_rounds()))
    val_raw = np.asarray(
        booster.predict(xgb.DMatrix(X_val), iteration_range=(0, end)),
        dtype=np.float64,
    )
    out: dict[str, np.ndarray] = {"raw": val_raw}
    if use_platt:
        cal_raw = np.asarray(
            booster.predict(xgb.DMatrix(X_tr[cal_idx]), iteration_range=(0, end)),
            dtype=np.float64,
        )
        cal = PlattCalibrator().fit(cal_raw, y_tr[cal_idx])
        out["calibrated"] = cal.transform(val_raw)
    if out_dir is not None:
        out_dir = ensure_dir(out_dir / "final_fit")
        booster.save_model(str(out_dir / "model.json"))
    return out


__all__ = [
    "BOOT_SEED",
    "CLEAN_ROOT",
    "CORE_METRICS",
    "N_BOOT",
    "OUT",
    "PHASE4_PREDS",
    "Q85",
    "REPO",
    "RICH_ABLATION",
    "SCRATCH",
    "SEED",
    "SPLIT",
    "TOP_FRACS",
    "WORLD",
    "binary_auc_ap",
    "cached_json",
    "ensure_dirs",
    "fit_lgbm_deterministic",
    "fit_xgb_deterministic",
    "lgbm_deterministic_params",
    "metrics_block",
    "mint_bootstrap_metric_delta",
    "topk_precision_recall",
    "write_json",
    "xgb_deterministic_params",
]
