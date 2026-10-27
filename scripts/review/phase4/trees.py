"""Frozen-protocol LightGBM / XGBoost final-fit helpers (fixed iters + Platt)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def fit_lgbm_binary_platt(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    elig_train: np.ndarray,
    times_train: np.ndarray,
    X_val: np.ndarray,
    fixed_iters: int,
    out_dir: Path,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Train LightGBM with fixed iterations; Platt on chronological train holdout."""
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
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": seed,
        "num_threads": 1,
    }
    dtrain = lgb.Dataset(X_tr[fit_idx], label=y_tr[fit_idx])
    booster = lgb.train(params, dtrain, num_boost_round=int(fixed_iters))
    out_dir = ensure_dir(out_dir / "final_fit")
    booster.save_model(str(out_dir / "model.txt"))
    n_it = int(fixed_iters)
    cal_raw = np.asarray(booster.predict(X_tr[cal_idx], num_iteration=n_it), dtype=np.float64)
    val_raw = np.asarray(booster.predict(X_val, num_iteration=n_it), dtype=np.float64)
    cal = PlattCalibrator().fit(cal_raw, y_tr[cal_idx])
    (out_dir / "calibration.json").write_text(
        __import__("json").dumps(cal.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return {
        "raw": val_raw,
        "calibrated": cal.transform(val_raw),
    }


def fit_xgb_binary_platt(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    elig_train: np.ndarray,
    times_train: np.ndarray,
    X_val: np.ndarray,
    fixed_iters: int,
    out_dir: Path,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Train XGBoost with fixed iterations; Platt on chronological train holdout."""
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
    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "seed": seed,
        "nthread": 1,
        "verbosity": 0,
    }
    dtrain = xgb.DMatrix(X_tr[fit_idx], label=y_tr[fit_idx])
    booster = xgb.train(params, dtrain, num_boost_round=int(fixed_iters))
    out_dir = ensure_dir(out_dir / "final_fit")
    model_path = out_dir / "model.json"
    booster.save_model(str(model_path))
    n_it = int(fixed_iters)
    # iteration_range is [begin, end); after num_boost_round=n_it, BoostedRounds()==n_it
    end = min(n_it, int(booster.num_boosted_rounds()))
    cal_raw = np.asarray(
        booster.predict(xgb.DMatrix(X_tr[cal_idx]), iteration_range=(0, end)),
        dtype=np.float64,
    )
    val_raw = np.asarray(
        booster.predict(xgb.DMatrix(X_val), iteration_range=(0, end)),
        dtype=np.float64,
    )
    cal = PlattCalibrator().fit(cal_raw, y_tr[cal_idx])
    (out_dir / "calibration.json").write_text(
        __import__("json").dumps(cal.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return {
        "raw": val_raw,
        "calibrated": cal.transform(val_raw),
    }
