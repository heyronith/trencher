"""Load frozen V2 models from the manifest and predict (no fitting)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from scripts.review.phase8.common import (
    AGE_CURVE_FEATURES,
    MANIFEST_PATH,
    REPO,
    sha256_file,
)

from trencher.baselines.calibration import PlattCalibrator


class ModelHashMismatch(RuntimeError):
    pass


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    path = Path(path or MANIFEST_PATH)
    return json.loads(path.read_text(encoding="utf-8"))


def _verify(path: Path, expected_sha: str) -> None:
    got = sha256_file(path)
    if got != expected_sha:
        raise ModelHashMismatch(f"{path}: sha256 {got} != manifest {expected_sha}")


def _load_platt(path: Path) -> PlattCalibrator:
    return PlattCalibrator.from_dict(json.loads(path.read_text(encoding="utf-8")))


def predict_xgb(
    *,
    model_path: Path,
    X: np.ndarray,
    fixed_iterations: int,
    calibrator_path: Path | None = None,
    use_calibrated: bool = False,
) -> np.ndarray:
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    end = min(int(fixed_iterations), int(booster.num_boosted_rounds()))
    raw = np.asarray(
        booster.predict(xgb.DMatrix(np.asarray(X)), iteration_range=(0, end)),
        dtype=np.float64,
    )
    if use_calibrated:
        if calibrator_path is None:
            raise ValueError("calibrator_path required when use_calibrated=True")
        return _load_platt(calibrator_path).transform(raw)
    return raw


def predict_lgbm(
    *,
    model_path: Path,
    X: np.ndarray,
    fixed_iterations: int,
    calibrator_path: Path,
) -> np.ndarray:
    booster = lgb.Booster(model_file=str(model_path))
    raw = np.asarray(
        booster.predict(np.asarray(X), num_iteration=int(fixed_iterations)),
        dtype=np.float64,
    )
    return _load_platt(calibrator_path).transform(raw)


def predict_logistic(*, model_path: Path, X: np.ndarray) -> np.ndarray:
    bundle = joblib.load(model_path)
    Xv = bundle["scaler"].transform(np.asarray(X, dtype=np.float64))
    return bundle["clf"].predict_proba(Xv)[:, 1]


def predict_all_tree_models(
    *,
    manifest: dict[str, Any],
    head_id: str,
    X: np.ndarray,
    feature_names: list[str],
) -> dict[str, np.ndarray]:
    """Predict xgb champion, lgbm, logistic, age_curve for one head."""
    models = manifest["models"]
    out: dict[str, np.ndarray] = {}

    # XGB champion — raw scores (phase4/6/7 protocol)
    m = models[f"xgb_champion_{head_id}"]
    path = REPO / m["path"]
    _verify(path, m["sha256"])
    out["xgb_champion"] = predict_xgb(
        model_path=path,
        X=X,
        fixed_iterations=int(m["fixed_iterations"]),
        calibrator_path=REPO / m["calibrator_path"],
        use_calibrated=False,
    )

    # Tuned LGBM — calibrated
    m = models[f"lgbm_tuned_{head_id}"]
    path = REPO / m["path"]
    _verify(path, m["sha256"])
    out["lgbm_tuned"] = predict_lgbm(
        model_path=path,
        X=X,
        fixed_iterations=int(m["fixed_iterations"]),
        calibrator_path=REPO / m["calibrator_path"],
    )

    # logistic_l2
    m = models[f"logistic_l2_{head_id}"]
    path = REPO / m["path"]
    _verify(path, m["sha256"])
    out["logistic_l2"] = predict_logistic(model_path=path, X=X)

    # age + curve
    m = models[f"age_curve_xgb_{head_id}"]
    path = REPO / m["path"]
    _verify(path, m["sha256"])
    name_to_i = {n: i for i, n in enumerate(feature_names)}
    ac_idx = [name_to_i[n] for n in AGE_CURVE_FEATURES]
    out["age_curve_xgb"] = predict_xgb(
        model_path=path,
        X=np.asarray(X)[:, ac_idx],
        fixed_iterations=int(m["fixed_iterations"]),
        calibrator_path=REPO / m["calibrator_path"],
        use_calibrated=True,
    )
    return out


def predict_twox(
    *,
    manifest: dict[str, Any],
    X: np.ndarray,
) -> np.ndarray:
    m = manifest["models"]["xgb_champion_twox_reaches_plus_100pct"]
    path = REPO / m["path"]
    _verify(path, m["sha256"])
    return predict_xgb(
        model_path=path,
        X=X,
        fixed_iterations=int(m["fixed_iterations"]),
        calibrator_path=REPO / m["calibrator_path"],
        use_calibrated=True,
    )


def load_stage_b_preds(
    manifest: dict[str, Any],
    *,
    split: str = "val",
    preds_parquet: Path | None = None,
) -> dict[str, Any]:
    """Load Stage-B per-row parquet for ``split``.

    val: prefer pinned manifest artifact (sha verified).
    test: require ``preds_parquet`` produced after TEST_ACCESSED.lock.
    """
    import pyarrow.parquet as pq

    m = manifest["models"]["stage_b_broad"]
    ckpt = REPO / m["path"]
    _verify(ckpt, m["sha256"])
    if split == "val":
        preds_path = Path(preds_parquet) if preds_parquet else REPO / m["val_preds_parquet"]
        if preds_path == REPO / m["val_preds_parquet"]:
            _verify(preds_path, m["val_preds_sha256"])
    elif split == "test":
        if preds_parquet is None:
            raise ValueError("TEST Stage-B preds require preds_parquet after unlock")
        preds_path = Path(preds_parquet)
    else:
        raise ValueError(f"unknown split {split!r}")
    table = pq.read_table(preds_path)
    return {
        "table": table,
        "observation_id": np.array(table["observation_id"].to_pylist(), dtype=object),
        "p_explosion_60s": table["p_explosion_60s"]
        .to_numpy(zero_copy_only=False)
        .astype(np.float64),
        "p_explosion_300s": table["p_explosion_300s"]
        .to_numpy(zero_copy_only=False)
        .astype(np.float64),
        "checkpoint_sha256": m["sha256"],
        "preds_path": str(preds_path),
        "split": split,
    }


def load_stage_b_val_preds(manifest: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible alias for permanent-val Stage-B preds."""
    return load_stage_b_preds(manifest, split="val")
