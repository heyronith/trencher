"""Tree baseline training: LightGBM / XGBoost / CatBoost with real 1–100 progress."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from trencher.baselines.calibration import PlattCalibrator, chronological_calibration_split
from trencher.baselines.dataset import (
    load_joined_matrices,
    merge_eligibility_into_targets,
    walkforward_mask,
)
from trencher.baselines.exec_overlay import merge_impact_into_targets
from trencher.baselines.metrics import (
    binary_metrics,
    decile_trading_table,
    regression_metrics,
)
from trencher.baselines.progress import BoostingProgressCallback, ProgressEmitter
from trencher.baselines.slices import (
    load_creator_flags,
    slice_binary_metrics,
    slice_regression_metrics,
)
from trencher.baselines.split_guard import refuse_test_split
from trencher.baselines.targets import (
    CORE_TARGET_SUITE,
    TargetSpec,
    extract_barrier_labels,
    extract_binary_labels,
    extract_regression_labels,
)
from trencher.baselines.trading import trading_arrays_from_targets
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.splits.walkforward import build_walkforward_folds


def _checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _library_versions() -> dict[str, str]:
    out = {}
    for name in ("lightgbm", "xgboost", "catboost", "sklearn", "numpy"):
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            out[name] = "missing"
    return out


def _prepare_xy(bundle: dict[str, Any], spec: TargetSpec, split_root: Path):
    merge_eligibility_into_targets(bundle, split_root, spec)
    rows = bundle["targets"]
    if spec.kind == "binary":
        y, elig, counts = extract_binary_labels(rows, spec)
        return y, elig, counts, "binary"
    if spec.kind == "regression":
        y, elig, counts = extract_regression_labels(rows, spec)
        return y, elig, counts, "regression"
    if spec.kind == "multiclass":
        y, elig, counts = extract_barrier_labels(rows, spec)
        return y, elig, counts, "multiclass"
    raise ValueError(f"unsupported kind for trees: {spec.kind}")


class _XGBProgress(ProgressEmitter):
    pass


def train_lightgbm_head(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    kind: str,
    out_dir: Path,
    tag: str,
    num_boost_round: int = 400,
    early_stopping_rounds: int | None = 40,
    seed: int = 0,
    scale_pos_weight: float | None = None,
    use_gpu: bool = False,
) -> dict[str, Any]:
    import lightgbm as lgb

    refuse_test_split("train")
    emitter = ProgressEmitter(tag, status_path=out_dir / "progress.json")
    emitter.emit(1, message="start", force=True)
    t0 = time.time()

    params: dict[str, Any] = {
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": seed,
    }
    if use_gpu:
        # Prefer CUDA/OpenCL when Modal image supports it; fall back silently.
        params["device"] = "gpu"
        params["gpu_platform_id"] = 0
        params["gpu_device_id"] = 0
    if kind == "binary":
        params["objective"] = "binary"
        params["metric"] = "average_precision"
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = float(scale_pos_weight)
    elif kind == "multiclass":
        params["objective"] = "multiclass"
        params["num_class"] = int(len(np.unique(y_train)))
        params["metric"] = "multi_logloss"
    else:
        params["objective"] = "regression"
        params["metric"] = "mae"
        params["alpha"] = 0.5  # unused unless quantile

    dtrain = lgb.Dataset(X_train, label=y_train)
    valid_sets = []
    valid_names = []
    if X_val is not None and len(X_val) > 0 and y_val is not None and len(y_val) > 0:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        valid_sets = [dval]
        valid_names = ["val"]
    cb = BoostingProgressCallback(emitter, num_boost_round)
    callbacks = [lgb.log_evaluation(period=0), cb]
    if early_stopping_rounds and valid_sets:
        callbacks.insert(0, lgb.early_stopping(early_stopping_rounds, verbose=False))
    train_kwargs: dict[str, Any] = {
        "params": params,
        "train_set": dtrain,
        "num_boost_round": num_boost_round,
        "callbacks": callbacks,
    }
    if valid_sets:
        train_kwargs["valid_sets"] = valid_sets
        train_kwargs["valid_names"] = valid_names
    try:
        booster = lgb.train(**train_kwargs)
    except Exception:
        if use_gpu:
            params.pop("device", None)
            params.pop("gpu_platform_id", None)
            params.pop("gpu_device_id", None)
            train_kwargs["params"] = params
            booster = lgb.train(**train_kwargs)
        else:
            raise
    best_it = int(booster.best_iteration or booster.current_iteration() or num_boost_round)
    emitter.complete(
        message="COMPLETE",
        best_iteration=best_it,
        actual_iterations=booster.current_iteration(),
    )
    model_path = out_dir / "model.txt"
    booster.save_model(str(model_path))
    if X_val is not None and len(X_val) > 0:
        raw = booster.predict(X_val, num_iteration=best_it)
    else:
        raw = np.array([], dtype=np.float64)
    return {
        "family": "lightgbm",
        "model_path": str(model_path),
        "checksum": _checksum(model_path),
        "best_iteration": best_it,
        "actual_iterations": int(booster.current_iteration()),
        "planned_iterations": num_boost_round,
        "raw_val_pred": np.asarray(raw, dtype=np.float64),
        "runtime_sec": time.time() - t0,
        "params": params,
        "early_stopping_rounds": early_stopping_rounds,
    }


def train_xgboost_head(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    kind: str,
    out_dir: Path,
    tag: str,
    num_boost_round: int = 400,
    early_stopping_rounds: int | None = 40,
    seed: int = 0,
    scale_pos_weight: float | None = None,
    use_gpu: bool = True,
) -> dict[str, Any]:
    import xgboost as xgb

    emitter = ProgressEmitter(tag, status_path=out_dir / "progress.json")
    emitter.emit(1, message="start", force=True)
    t0 = time.time()
    params: dict[str, Any] = {
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,
        "verbosity": 0,
    }
    if use_gpu:
        params["device"] = "cuda"
        params["tree_method"] = "hist"
    else:
        params["tree_method"] = "hist"
    if kind == "binary":
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "aucpr"
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = float(scale_pos_weight)
    elif kind == "multiclass":
        params["objective"] = "multi:softprob"
        params["num_class"] = int(len(np.unique(y_train)))
        params["eval_metric"] = "mlogloss"
    else:
        params["objective"] = "reg:squarederror"
        params["eval_metric"] = "mae"

    dtrain = xgb.DMatrix(X_train, label=y_train)
    has_val = X_val is not None and len(X_val) > 0 and y_val is not None and len(y_val) > 0
    dval = xgb.DMatrix(X_val, label=y_val) if has_val else None

    class _CB(xgb.callback.TrainingCallback):
        def __init__(self) -> None:
            self.best = None
            self.best_it = None

        def after_iteration(self, model, epoch, evals_log):  # noqa: ANN001
            it = epoch + 1
            msg = f"iter={it}/{num_boost_round}"
            if evals_log:
                for _dn, metrics in evals_log.items():
                    for mn, vals in metrics.items():
                        msg += f" {mn}={vals[-1]:.6g}"
                        self.best = vals[-1]
                        self.best_it = it
            emitter.update(
                it,
                num_boost_round,
                message=msg,
                current_iteration=it,
                best_metric=self.best,
                best_iteration=self.best_it,
            )
            return False

    train_kwargs: dict[str, Any] = {
        "params": params,
        "dtrain": dtrain,
        "num_boost_round": num_boost_round,
        "callbacks": [_CB()],
        "verbose_eval": False,
    }
    if has_val and dval is not None:
        train_kwargs["evals"] = [(dval, "val")]
        if early_stopping_rounds:
            train_kwargs["early_stopping_rounds"] = early_stopping_rounds
    booster = xgb.train(**train_kwargs)
    best_it = int(getattr(booster, "best_iteration", None) or (booster.num_boosted_rounds() - 1))
    emitter.complete(
        message="COMPLETE",
        best_iteration=best_it,
        actual_iterations=booster.num_boosted_rounds(),
    )
    model_path = out_dir / "model.json"
    booster.save_model(str(model_path))
    if has_val and dval is not None:
        raw = booster.predict(dval, iteration_range=(0, best_it + 1))
    else:
        raw = np.array([], dtype=np.float64)
    return {
        "family": "xgboost",
        "model_path": str(model_path),
        "checksum": _checksum(model_path),
        "best_iteration": best_it,
        "actual_iterations": int(booster.num_boosted_rounds()),
        "planned_iterations": num_boost_round,
        "raw_val_pred": np.asarray(raw, dtype=np.float64),
        "runtime_sec": time.time() - t0,
        "params": params,
        "early_stopping_rounds": early_stopping_rounds,
    }


def train_catboost_head(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    kind: str,
    out_dir: Path,
    tag: str,
    num_boost_round: int = 400,
    early_stopping_rounds: int | None = 40,
    seed: int = 0,
    class_weights: list[float] | None = None,
    use_gpu: bool = True,
) -> dict[str, Any]:
    import re
    import sys
    import threading

    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    emitter = ProgressEmitter(tag, status_path=out_dir / "progress.json")
    emitter.emit(1, message="start", force=True)
    t0 = time.time()
    common: dict[str, Any] = {
        "iterations": num_boost_round,
        "learning_rate": 0.05,
        "depth": 8,
        "random_seed": seed,
        "allow_writing_files": False,
        "task_type": "GPU" if use_gpu else "CPU",
        "metric_period": 1,
        "logging_level": "Verbose",
        "verbose": True,
    }
    if early_stopping_rounds:
        common["od_type"] = "Iter"
        common["od_wait"] = int(early_stopping_rounds)
    train_pool = Pool(X_train, y_train)
    has_val = X_val is not None and len(X_val) > 0 and y_val is not None and len(y_val) > 0
    val_pool = Pool(X_val, y_val) if has_val else None

    iter_re = re.compile(r"^(\d+)\s*:")

    class _CB:
        def after_iteration(self, info):  # noqa: ANN001
            it = int(info.iteration)
            msg = f"iter={it}/{num_boost_round}"
            metrics = getattr(info, "metrics", None) or {}
            best = None
            if metrics:
                for _split, m in metrics.items():
                    for name, vals in m.items():
                        if vals:
                            best = vals[-1]
                            msg += f" {name}={best:.6g}"
            emitter.update(
                it,
                num_boost_round,
                message=msg,
                current_iteration=it,
                best_metric=best,
            )
            return True

    class _StdoutIterTee:
        """Parse CatBoost Verbose stdout lines into ProgressEmitter updates (GPU path)."""

        def __init__(self, wrapped):  # noqa: ANN001
            self._wrapped = wrapped
            self._buf = ""
            self._lock = threading.Lock()

        def write(self, data: str) -> int:
            with self._lock:
                self._wrapped.write(data)
                self._buf += data
                while "\n" in self._buf:
                    line, self._buf = self._buf.split("\n", 1)
                    m = iter_re.match(line.strip())
                    if m:
                        it = int(m.group(1)) + 1  # CatBoost prints 0-based iter index
                        emitter.update(
                            it,
                            num_boost_round,
                            message=f"gpu iter={it}/{num_boost_round} {line.strip()[:120]}",
                            current_iteration=it,
                        )
            return len(data)

        def flush(self) -> None:
            self._wrapped.flush()

        def fileno(self):  # noqa: ANN001
            return self._wrapped.fileno()

    if kind in ("binary", "multiclass"):
        params = dict(common)
        if kind == "binary":
            params["loss_function"] = "Logloss"
            if has_val:
                n_pos = float(np.sum(np.asarray(y_val) >= 0.5))
                n_neg = float(len(y_val) - n_pos)
                params["eval_metric"] = "PRAUC" if n_pos > 0 and n_neg > 0 else "Logloss"
            else:
                params["eval_metric"] = "Logloss"
            if class_weights is not None:
                params["class_weights"] = class_weights
        else:
            params["loss_function"] = "MultiClass"
            params["eval_metric"] = "MultiClass"
        model = CatBoostClassifier(**params)
    else:
        params = dict(common)
        model = CatBoostRegressor(loss_function="MAE", eval_metric="MAE", **common)

    fit_kwargs: dict[str, Any] = {}
    if has_val and val_pool is not None:
        fit_kwargs["eval_set"] = val_pool

    if use_gpu:
        emitter.emit(5, message="gpu fit with stdout iteration progress", force=True)
        old_out, old_err = sys.stdout, sys.stderr
        tee = _StdoutIterTee(old_out)
        try:
            sys.stdout = tee  # type: ignore[assignment]
            sys.stderr = tee  # type: ignore[assignment]
            try:
                model.fit(train_pool, **fit_kwargs)
            except Exception as exc:  # noqa: BLE001
                emitter.emit(
                    15,
                    message=f"gpu failed ({type(exc).__name__}); CPU+callbacks",
                    force=True,
                )
                cpu_common = {
                    **common,
                    "task_type": "CPU",
                    "logging_level": "Silent",
                    "verbose": False,
                }
                if kind in ("binary", "multiclass"):
                    model = CatBoostClassifier(
                        **{**params, **cpu_common, "task_type": "CPU"}
                    )
                else:
                    model = CatBoostRegressor(
                        loss_function="MAE", eval_metric="MAE", **cpu_common
                    )
                cb_kwargs = dict(fit_kwargs)
                cb_kwargs["callbacks"] = [_CB()]
                model.fit(train_pool, **cb_kwargs)
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        trees = int(model.tree_count_ or num_boost_round)
        emitter.update(trees, num_boost_round, message=f"gpu trees={trees}")
    else:
        cb_kwargs = dict(fit_kwargs)
        cb_kwargs["callbacks"] = [_CB()]
        # Quiet native logging when callbacks drive progress.
        if hasattr(model, "set_params"):
            model.set_params(logging_level="Silent", verbose=False)
        model.fit(train_pool, **cb_kwargs)

    if kind in ("binary", "multiclass"):
        raw = model.predict_proba(X_val) if has_val else np.array([])
        if kind == "binary" and has_val:
            raw = raw[:, 1]
    else:
        raw = model.predict(X_val) if has_val else np.array([])

    emitter.complete(
        message="COMPLETE",
        best_iteration=int(model.get_best_iteration() or 0),
        actual_iterations=int(model.tree_count_),
    )
    model_path = out_dir / "model.cbm"
    model.save_model(str(model_path))
    return {
        "family": "catboost",
        "model_path": str(model_path),
        "checksum": _checksum(model_path),
        "best_iteration": int(model.get_best_iteration() or model.tree_count_ or 0),
        "actual_iterations": int(model.tree_count_),
        "planned_iterations": num_boost_round,
        "raw_val_pred": np.asarray(raw, dtype=np.float64),
        "runtime_sec": time.time() - t0,
        "params": common,
        "early_stopping_rounds": early_stopping_rounds,
    }



def run_family_core_suite(
    *,
    family: str,
    world_root: Path,
    split_root: Path | None = None,
    artifacts_root: Path,
    seed: int = 0,
    max_partitions: int | None = None,
    use_gpu: bool = False,
    smoke: bool = False,
    protocol_version: str = "v2",
) -> dict[str, Any]:
    """V2: wf1+wf2 selection only; permanent val is evaluation-only."""
    refuse_test_split("val")
    world_root = Path(world_root)
    split_root = Path(split_root or world_root / "splits" / "split_v1")
    artifacts_root = ensure_dir(Path(artifacts_root) / family)
    cutoffs = json.loads((split_root / "cutoffs.json").read_text(encoding="utf-8"))
    all_folds = build_walkforward_folds(cutoffs)["folds"]
    # Permanent val [q70,q85) must never enter selection / early stopping.
    selection_folds = [f for f in all_folds if f["fold_id"] in ("wf_fold_1", "wf_fold_2")]

    overall = ProgressEmitter(
        f"{family} overall", status_path=artifacts_root / "overall_progress.json"
    )
    heads = list(CORE_TARGET_SUITE)
    if smoke:
        heads = [h for h in heads if h.head_id in ("explosion_60s", "return_5m")]
        max_partitions = max_partitions or 2
    heads = [h for h in heads if h.kind in ("binary", "regression", "multiclass")]

    train_bundle = load_joined_matrices(
        world_root=world_root,
        split_root=split_root,
        split="train",
        max_partitions=max_partitions,
        population="broad",
    )
    val_bundle = load_joined_matrices(
        world_root=world_root,
        split_root=split_root,
        split="val",
        max_partitions=max_partitions,
        population="broad",
    )
    # Side-load impact-adjusted exec fields for trading tables (eval only).
    merge_impact_into_targets(val_bundle["targets"], val_bundle["meta"], world_root)

    total_runs = len(heads) * (len(selection_folds) + 1)
    done_runs = 0
    overall.emit(1, message="start", force=True)
    eval_only_proof = {
        "protocol_version": protocol_version,
        "label": "final_full_population_validation_control"
        if protocol_version == "v2"
        else "legacy",
        "selection_folds": [f["fold_id"] for f in selection_folds],
        "excluded_from_selection": ["wf_fold_3", "permanent_val_q70_q85"],
        "permanent_val_used_for": [
            "final_score_once",
            "trading_tables",
            "slice_reports",
        ],
        "permanent_val_forbidden_for": [
            "early_stopping",
            "best_iteration_selection",
            "hyperparameter_selection",
            "threshold_selection",
            "calibration_fitting",
            "model_family_selection",
        ],
        "calibration": "platt_on_chronological_train_holdout_only",
        "final_fit": "fixed_iterations_from_wf1_wf2_median_no_es_on_permanent_val",
        "permanent_test_q85_plus": "untouched",
    }
    results: dict[str, Any] = {
        "family": family,
        "generated_at": utc_now(),
        "seed": seed,
        "library_versions": _library_versions(),
        "heads": {},
        "train_n": train_bundle["n"],
        "val_n": val_bundle["n"],
        "feature_n": len(train_bundle["feature_names"]),
        "smoke": smoke,
        "use_gpu": use_gpu,
        "protocol_version": protocol_version,
        "eval_only_proof": eval_only_proof,
    }

    planned_rounds = 50 if smoke else 400

    for spec in heads:
        head_dir = ensure_dir(artifacts_root / spec.head_id)
        y_tr, elig_tr, counts_tr, kind = _prepare_xy(train_bundle, spec, split_root)
        y_va, elig_va, counts_va, _ = _prepare_xy(val_bundle, spec, split_root)
        Xtr_all = train_bundle["X"]
        Xva_all = val_bundle["X"]

        fold_summaries = []
        best_iters: list[int] = []
        for fold in selection_folds:
            tr_m = walkforward_mask(train_bundle["meta"], fold, "train") & elig_tr
            va_m = walkforward_mask(train_bundle["meta"], fold, "val") & elig_tr
            X_va_f, y_va_f = Xtr_all[va_m], y_tr[va_m]
            X_tr_f, y_tr_f = Xtr_all[tr_m], y_tr[tr_m]
            if X_tr_f.shape[0] < 50 or X_va_f.shape[0] < 20:
                done_runs += 1
                overall.update(
                    done_runs, total_runs, message=f"skip {spec.head_id} {fold['fold_id']}"
                )
                continue

            spw = None
            cw = None
            if kind == "binary":
                pos = float(y_tr_f.sum())
                neg = float(len(y_tr_f) - pos)
                spw = neg / max(pos, 1.0)
                cw = [1.0, spw]

            tag = f"{family} {spec.head_id} {fold['fold_id']}"
            fdir = ensure_dir(head_dir / fold["fold_id"])
            common_kw = dict(
                X_train=X_tr_f,
                y_train=y_tr_f,
                X_val=X_va_f,
                y_val=y_va_f,
                kind=kind,
                out_dir=fdir,
                tag=tag,
                num_boost_round=planned_rounds,
                seed=seed,
                use_gpu=use_gpu,
            )
            if family == "lightgbm":
                art = train_lightgbm_head(**common_kw, scale_pos_weight=spw)
            elif family == "xgboost":
                art = train_xgboost_head(**common_kw, scale_pos_weight=spw)
            else:
                art = train_catboost_head(
                    **common_kw, class_weights=cw if kind == "binary" else None
                )
            bi = int(art["best_iteration"] or art["actual_iterations"] or planned_rounds)
            best_iters.append(max(bi, 1))
            fold_summaries.append(
                {
                    "fold_id": fold["fold_id"],
                    "train_n": int(X_tr_f.shape[0]),
                    "val_n": int(X_va_f.shape[0]),
                    "best_iteration": bi,
                    "runtime_sec": art["runtime_sec"],
                    "dev_val_window": fold["rule"],
                }
            )
            done_runs += 1
            overall.update(done_runs, total_runs, message=f"done {tag}")

        if best_iters:
            fixed_iters = int(np.median(np.asarray(best_iters, dtype=np.float64)))
        else:
            fixed_iters = planned_rounds
        fixed_iters = max(fixed_iters, 10 if smoke else 50)

        # Final fit: train-only chrono calib split; NO early stopping on permanent val.
        tr_m = elig_tr
        va_m = elig_va
        X_tr_f, y_tr_f = Xtr_all[tr_m], y_tr[tr_m]
        X_va_f, y_va_f = Xva_all[va_m], y_va[va_m]
        times = np.array(
            [
                float(m["observation_time_t"] or 0.0)
                for m, e in zip(train_bundle["meta"], elig_tr, strict=True)
                if e
            ],
            dtype=np.float64,
        )
        fit_idx, cal_idx = chronological_calibration_split(times)
        X_fit, y_fit = X_tr_f[fit_idx], y_tr_f[fit_idx]
        X_cal, y_cal = X_tr_f[cal_idx], y_tr_f[cal_idx]

        spw = None
        cw = None
        if kind == "binary":
            pos = float(y_fit.sum())
            neg = float(len(y_fit) - pos)
            spw = neg / max(pos, 1.0)
            cw = [1.0, spw]

        tag = f"{family} {spec.head_id} final_fit"
        fdir = ensure_dir(head_dir / "final_fit")
        # Use cal slice only as optional monitor set; no ES / no permanent val in fit.
        common_kw = dict(
            X_train=X_fit,
            y_train=y_fit,
            X_val=X_cal,
            y_val=y_cal,
            kind=kind,
            out_dir=fdir,
            tag=tag,
            num_boost_round=fixed_iters,
            early_stopping_rounds=None,
            seed=seed,
            use_gpu=use_gpu,
        )
        if family == "lightgbm":
            art = train_lightgbm_head(**common_kw, scale_pos_weight=spw)
        elif family == "xgboost":
            art = train_xgboost_head(**common_kw, scale_pos_weight=spw)
        else:
            art = train_catboost_head(
                **common_kw, class_weights=cw if kind == "binary" else None
            )

        metrics: dict[str, Any] = {}
        calib_info = None
        # Score permanent val exactly once (evaluation-only).
        if family == "lightgbm":
            import lightgbm as lgb

            booster = lgb.Booster(model_file=art["model_path"])
            n_it = int(art["best_iteration"] or fixed_iters)
            cal_scores = booster.predict(X_cal, num_iteration=n_it)
            val_scores = booster.predict(X_va_f, num_iteration=n_it)
            pred = val_scores
        elif family == "xgboost":
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(art["model_path"])
            n_it = int(art["best_iteration"] or fixed_iters)
            cal_scores = booster.predict(xgb.DMatrix(X_cal), iteration_range=(0, n_it + 1))
            val_scores = booster.predict(xgb.DMatrix(X_va_f), iteration_range=(0, n_it + 1))
            pred = val_scores
        else:
            from catboost import CatBoostClassifier, CatBoostRegressor

            if kind in ("binary", "multiclass"):
                model = CatBoostClassifier()
                model.load_model(art["model_path"])
                cal_scores = model.predict_proba(X_cal)
                val_scores = model.predict_proba(X_va_f)
                if kind == "binary":
                    cal_scores = cal_scores[:, 1]
                    val_scores = val_scores[:, 1]
                pred = val_scores
            else:
                model = CatBoostRegressor()
                model.load_model(art["model_path"])
                cal_scores = model.predict(X_cal)
                pred = model.predict(X_va_f)
                val_scores = pred

        if kind == "binary":
            cal = PlattCalibrator().fit(cal_scores, y_cal)
            write_json(fdir / "calibration.json", cal.to_dict())
            calib_info = cal.to_dict()
            metrics["raw"] = binary_metrics(y_va_f, val_scores)
            metrics["calibrated"] = binary_metrics(y_va_f, cal.transform(val_scores))
            tgt_va = [t for t, e in zip(val_bundle["targets"], elig_va, strict=True) if e]
            trade = trading_arrays_from_targets(tgt_va)
            metrics["trading_deciles"] = decile_trading_table(
                cal.transform(val_scores),
                gross_return=trade["gross_return"],
                entry_impact_adjusted_return=trade["entry_impact_adjusted_return"],
                curve_impact_adjusted_return=trade["curve_impact_adjusted_return"],
                mfe=trade["mfe"],
                mae=trade["mae"],
                explosion=y_va_f.astype(np.float64),
            )
            meta_va = [m for m, e in zip(val_bundle["meta"], elig_va, strict=True) if e]
            creator_flags = load_creator_flags(split_root / "eval_creator_flags.parquet")
            metrics["slices"] = slice_binary_metrics(
                y_va_f,
                cal.transform(val_scores),
                meta_va,
                creator_flags=creator_flags,
            )
        elif kind == "regression":
            metrics["regression"] = regression_metrics(y_va_f, pred)
            meta_va = [m for m, e in zip(val_bundle["meta"], elig_va, strict=True) if e]
            metrics["slices"] = slice_regression_metrics(y_va_f, pred, meta_va)
        else:
            if getattr(pred, "ndim", 1) == 1:
                metrics["note"] = "multiclass_pred_1d"
            else:
                p = np.clip(pred, 1e-7, 1 - 1e-7)
                p = p / p.sum(axis=1, keepdims=True)
                ll = -np.mean(np.log(p[np.arange(len(y_va_f)), y_va_f.astype(int)]))
                metrics["multiclass_logloss"] = float(ll)
                upper = (y_va_f == 0).astype(int)
                metrics["upper_first"] = binary_metrics(upper, p[:, 0])

        write_json(
            fdir / "metrics.json",
            {
                "counts_train": counts_tr,
                "counts_val": counts_va,
                "metrics": metrics,
                "calibration": calib_info,
                "runtime_sec": art["runtime_sec"],
                "checksum": art["checksum"],
                "best_iteration": art["best_iteration"],
                "fixed_iterations_from_selection": fixed_iters,
                "selection_best_iters": best_iters,
                "eval_only_proof": eval_only_proof,
            },
        )
        results["heads"][spec.head_id] = {
            "kind": kind,
            "counts_train": counts_tr,
            "counts_val": counts_va,
            "folds": fold_summaries,
            "selection_best_iters": best_iters,
            "fixed_iterations": fixed_iters,
            "final": {
                "runtime_sec": art["runtime_sec"],
                "checksum": art["checksum"],
                "best_iteration": art["best_iteration"],
                "fixed_iterations": fixed_iters,
                "early_stopping_rounds": None,
                "metrics": metrics,
                "calibration": calib_info,
            },
        }
        done_runs += 1
        overall.update(done_runs, total_runs, message=f"final {spec.head_id}")

    overall.complete(message="family complete")
    write_json(artifacts_root / "results.json", results)
    return results
