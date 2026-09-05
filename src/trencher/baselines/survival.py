"""Survival / TTE baselines: Cox PH + discrete-time logistic hazard."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from trencher.baselines.dataset import load_joined_matrices, merge_eligibility_into_targets
from trencher.baselines.metrics import concordance_index
from trencher.baselines.progress import ProgressEmitter
from trencher.baselines.split_guard import refuse_test_split
from trencher.baselines.targets import CORE_TARGET_SUITE, extract_tte_labels
from trencher.data.world_model.io import ensure_dir, utc_now, write_json


def _cox_fit_predict(
    X_train: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    X_val: np.ndarray,
    emitter: ProgressEmitter,
) -> tuple[np.ndarray, dict[str, Any]]:
    import pandas as pd
    from lifelines import CoxPHFitter

    emitter.emit(1, message="cox prepare", force=True)
    df = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
    # drop constant cols
    nunique = df.nunique()
    keep = [c for c in df.columns if nunique[c] > 1]
    df = df[keep]
    df["duration"] = duration
    df["event"] = event.astype(int)
    emitter.emit(20, message="cox fit start")
    cph = CoxPHFitter(penalizer=0.1)
    # lifelines doesn't give iteration callbacks easily — map fit stages
    emitter.emit(40, message="cox fitting")
    cph.fit(df, duration_col="duration", event_col="event", show_progress=False)
    emitter.emit(80, message="cox predict")
    dfv = pd.DataFrame(X_val, columns=[f"f{i}" for i in range(X_val.shape[1])])
    for c in keep:
        if c not in dfv.columns:
            dfv[c] = 0.0
    dfv = dfv[keep]
    risk = np.asarray(cph.predict_partial_hazard(dfv), dtype=np.float64).reshape(-1)
    emitter.complete(message="COMPLETE")
    return risk, {"model": "cox_ph", "n_features_used": len(keep), "penalizer": 0.1}


def _discrete_hazard_fit_predict(
    X_train: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    X_val: np.ndarray,
    emitter: ProgressEmitter,
    *,
    n_bins: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Boosted discrete-time hazard via LightGBM on person-period data (subsampled)."""
    import lightgbm as lgb

    emitter.emit(1, message="discrete hazard prepare", force=True)
    # Quantile time bins on train
    edges = np.quantile(duration[np.isfinite(duration)], np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        edges = np.linspace(float(np.nanmin(duration)), float(np.nanmax(duration)) + 1e-6, 5)

    rng = np.random.default_rng(seed)
    # Subsample individuals for person-period expansion
    n = X_train.shape[0]
    take = min(n, 50_000)
    idx = rng.choice(n, size=take, replace=False) if n > take else np.arange(n)

    rows_x = []
    rows_y = []
    for i in idx:
        d = float(duration[i])
        e = int(event[i])
        # bins fully elapsed
        for b in range(len(edges) - 1):
            t0, t1 = edges[b], edges[b + 1]
            if d < t0:
                break
            # at risk in bin
            feat = np.concatenate([X_train[i], np.array([b], dtype=np.float64)])
            rows_x.append(feat)
            # event in this bin?
            y = 1 if (e == 1 and t0 <= d < t1) else 0
            rows_y.append(y)
            if y == 1:
                break
    Xpp = np.asarray(rows_x, dtype=np.float32)
    ypp = np.asarray(rows_y, dtype=np.float32)
    emitter.emit(30, message=f"person-period n={len(ypp)}")

    planned = 200
    dtrain = lgb.Dataset(Xpp, label=ypp)
    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbosity": -1,
        "seed": seed,
        "metric": "binary_logloss",
    }

    def _cb(env):  # noqa: ANN001
        it = env.iteration + 1
        emitter.update(30 + int(60 * it / planned), 100, message=f"hazard iter={it}/{planned}")

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=planned,
        callbacks=[lgb.callback.LambdaCallback(on_iteration=_cb)],
    )
    # Risk score = predicted hazard in first bins aggregate
    risks = []
    for i in range(X_val.shape[0]):
        acc = 0.0
        surv = 1.0
        for b in range(min(5, len(edges) - 1)):
            feat = np.concatenate([X_val[i], np.array([b], dtype=np.float64)]).reshape(1, -1)
            h = float(booster.predict(feat)[0])
            acc += surv * h
            surv *= 1.0 - h
        risks.append(acc)
    emitter.complete(message="COMPLETE")
    return np.asarray(risks, dtype=np.float64), {
        "model": "discrete_time_lgbm_hazard",
        "n_bins": int(len(edges) - 1),
        "person_period_n": int(len(ypp)),
    }


def run_survival_suite(
    *,
    world_root: Path,
    split_root: Path | None = None,
    artifacts_root: Path,
    seed: int = 0,
    max_partitions: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    refuse_test_split("val")
    world_root = Path(world_root)
    split_root = Path(split_root or world_root / "splits" / "split_v1")
    artifacts_root = ensure_dir(Path(artifacts_root) / "survival")
    overall = ProgressEmitter("survival overall", status_path=artifacts_root / "overall_progress.json")
    overall.emit(1, message="start", force=True)

    if smoke:
        max_partitions = max_partitions or 2

    train_bundle = load_joined_matrices(
        world_root=world_root, split_root=split_root, split="train", max_partitions=max_partitions
    )
    val_bundle = load_joined_matrices(
        world_root=world_root, split_root=split_root, split="val", max_partitions=max_partitions
    )

    heads = [h for h in CORE_TARGET_SUITE if h.kind == "tte"]
    results: dict[str, Any] = {
        "family": "survival",
        "generated_at": utc_now(),
        "seed": seed,
        "heads": {},
        "train_n": train_bundle["n"],
        "val_n": val_bundle["n"],
        "smoke": smoke,
    }
    for hi, spec in enumerate(heads):
        merge_eligibility_into_targets(train_bundle, split_root, spec)
        merge_eligibility_into_targets(val_bundle, split_root, spec)
        d_tr, e_tr, ar_tr, c_tr = extract_tte_labels(train_bundle["targets"], spec)
        d_va, e_va, ar_va, c_va = extract_tte_labels(val_bundle["targets"], spec)
        head_dir = ensure_dir(artifacts_root / spec.head_id)
        honesty = {
            "note": (
                "Graduation TTE is sparse/event-only under V4 source-aware censoring; "
                "report at-risk counts honestly."
                if spec.head_id == "graduation_tte"
                else "Explosion TTE uses Pump continuous-coverage censoring."
            ),
            "counts_train": c_tr,
            "counts_val": c_va,
        }
        if int(ar_tr.sum()) < 30 or int(ar_va.sum()) < 10:
            write_json(head_dir / "metrics.json", {**honesty, "skipped": "insufficient_at_risk"})
            results["heads"][spec.head_id] = {**honesty, "skipped": True}
            overall.update(hi + 1, len(heads), message=f"skip {spec.head_id}")
            continue

        Xtr = train_bundle["X"][ar_tr]
        Xva = val_bundle["X"][ar_va]
        # Subsample features for Cox stability
        Xtr_s = Xtr[:, : min(64, Xtr.shape[1])]
        Xva_s = Xva[:, : min(64, Xva.shape[1])]

        tag = f"survival {spec.head_id} final_fit"
        emitter = ProgressEmitter(tag, status_path=head_dir / "progress.json")
        t0 = time.time()
        try:
            risk, info = _cox_fit_predict(Xtr_s, d_tr[ar_tr], e_tr[ar_tr], Xva_s, emitter)
            method = "cox"
        except Exception as exc:  # noqa: BLE001
            emitter.emit(10, message=f"cox failed: {exc}; discrete hazard")
            risk, info = _discrete_hazard_fit_predict(
                Xtr_s, d_tr[ar_tr], e_tr[ar_tr], Xva_s, emitter, seed=seed
            )
            method = "discrete_hazard"
            info["cox_error"] = str(exc)

        cidx = concordance_index(d_va[ar_va], e_va[ar_va], risk)
        metrics = {
            **honesty,
            "method": method,
            "model_info": info,
            "c_index": cidx,
            "runtime_sec": time.time() - t0,
            "events_val": int(e_va[ar_va].sum()),
            "at_risk_val": int(ar_va.sum()),
        }
        write_json(head_dir / "metrics.json", metrics)
        results["heads"][spec.head_id] = metrics
        overall.update(hi + 1, len(heads), message=f"done {spec.head_id}")

    overall.complete(message="survival complete")
    write_json(artifacts_root / "results.json", results)
    return results
