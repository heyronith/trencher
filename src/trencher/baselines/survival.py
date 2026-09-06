"""Survival / TTE baselines: discrete-time hazard (primary) + optional subsampled Cox."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import numpy as np

from trencher.baselines.dataset import load_joined_matrices, merge_eligibility_into_targets
from trencher.baselines.metrics import concordance_index
from trencher.baselines.progress import ProgressEmitter
from trencher.baselines.split_guard import refuse_test_split
from trencher.baselines.targets import CORE_TARGET_SUITE, extract_tte_labels
from trencher.data.world_model.io import ensure_dir, utc_now, write_json

SurvivalMethod = Literal["discrete_hazard", "cox_subsample"]


def _cox_fit_predict(
    X_train: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    X_val: np.ndarray,
    emitter: ProgressEmitter,
    *,
    max_train_rows: int = 80_000,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Documented subsampled Cox control — not a full-population lifelines fit."""
    import pandas as pd
    from lifelines import CoxPHFitter

    emitter.emit(1, message="cox prepare (subsample)", force=True)
    n = X_train.shape[0]
    rng = np.random.default_rng(seed)
    if n > max_train_rows:
        idx = rng.choice(n, size=max_train_rows, replace=False)
        X_fit, d_fit, e_fit = X_train[idx], duration[idx], event[idx]
    else:
        X_fit, d_fit, e_fit = X_train, duration, event
        idx = np.arange(n)

    df = pd.DataFrame(X_fit, columns=[f"f{i}" for i in range(X_fit.shape[1])])
    nunique = df.nunique()
    keep = [c for c in df.columns if nunique[c] > 1]
    df = df[keep]
    df["duration"] = d_fit
    df["event"] = e_fit.astype(int)
    emitter.emit(20, message=f"cox fit start n={len(df)}")
    cph = CoxPHFitter(penalizer=0.1)
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
    return risk, {
        "model": "cox_ph_subsample",
        "n_features_used": len(keep),
        "penalizer": 0.1,
        "train_rows_fit": int(len(df)),
        "train_rows_available": int(n),
        "subsample": bool(n > max_train_rows),
        "population_validation": "full_val",
        "note": "Cox fit uses deterministic-ish RNG subsample; val scored full population.",
    }


def _discrete_hazard_fit_predict(
    X_train: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    X_val: np.ndarray,
    emitter: ProgressEmitter,
    *,
    n_bins: int = 20,
    seed: int = 0,
    max_individuals: int = 80_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Boosted discrete-time hazard via LightGBM on person-period data.

    Train person-period expansion is capped at ``max_individuals`` (documented).
    Validation risk is vectorized over the full val population.
    """
    import lightgbm as lgb

    emitter.emit(1, message="discrete hazard prepare", force=True)
    edges = np.quantile(duration[np.isfinite(duration)], np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        edges = np.linspace(float(np.nanmin(duration)), float(np.nanmax(duration)) + 1e-6, 5)
    n_bin = int(len(edges) - 1)

    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    take = min(n, max_individuals)
    idx = rng.choice(n, size=take, replace=False) if n > take else np.arange(n)

    rows_x: list[np.ndarray] = []
    rows_y: list[int] = []
    for j, i in enumerate(idx):
        d = float(duration[i])
        e = int(event[i])
        for b in range(n_bin):
            t0, t1 = float(edges[b]), float(edges[b + 1])
            if d < t0:
                break
            feat = np.concatenate([X_train[i], np.array([b], dtype=np.float64)])
            rows_x.append(feat)
            y = 1 if (e == 1 and t0 <= d < t1) else 0
            rows_y.append(y)
            if y == 1:
                break
        if (j + 1) % 10_000 == 0:
            emitter.emit(
                min(25, 1 + int(24 * (j + 1) / max(take, 1))),
                message=f"person-period expand {j + 1}/{take}",
                force=True,
            )
    Xpp = np.asarray(rows_x, dtype=np.float32)
    ypp = np.asarray(rows_y, dtype=np.float32)
    emitter.emit(30, message=f"person-period n={len(ypp)} individuals={take}", force=True)

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
        emitter.update(
            30 + int(50 * it / planned),
            100,
            message=f"hazard iter={it}/{planned}",
        )

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=planned,
        callbacks=[lgb.callback.LambdaCallback(on_iteration=_cb)],
    )

    # Vectorized val risk over first K bins (no per-row booster.predict loop).
    emitter.emit(85, message="vectorized val hazard", force=True)
    k_bins = min(5, n_bin)
    n_val = X_val.shape[0]
    feat_dim = X_val.shape[1]
    # [n_val * k_bins, feat_dim+1]
    flat = np.empty((n_val * k_bins, feat_dim + 1), dtype=np.float32)
    for b in range(k_bins):
        sl = slice(b * n_val, (b + 1) * n_val)
        flat[sl, :feat_dim] = X_val.astype(np.float32, copy=False)
        flat[sl, feat_dim] = float(b)
    haz = booster.predict(flat).astype(np.float64).reshape(k_bins, n_val)
    surv = np.ones(n_val, dtype=np.float64)
    risk = np.zeros(n_val, dtype=np.float64)
    for b in range(k_bins):
        h = np.clip(haz[b], 0.0, 1.0)
        risk += surv * h
        surv *= 1.0 - h

    emitter.complete(message="COMPLETE")
    return risk, {
        "model": "discrete_time_lgbm_hazard",
        "n_bins": n_bin,
        "person_period_n": int(len(ypp)),
        "train_individuals_expanded": int(take),
        "train_individuals_available": int(n),
        "val_rows_scored": int(n_val),
        "vectorized_val_predict": True,
        "note": (
            "Full-population discrete-time hazard control. "
            "Person-period train expansion may be subsampled; validation is full population."
        ),
    }


def run_survival_suite(
    *,
    world_root: Path,
    split_root: Path | None = None,
    artifacts_root: Path,
    seed: int = 0,
    max_partitions: int | None = None,
    smoke: bool = False,
    method: SurvivalMethod = "discrete_hazard",
) -> dict[str, Any]:
    refuse_test_split("val")
    world_root = Path(world_root)
    split_root = Path(split_root or world_root / "splits" / "split_v1")
    artifacts_root = ensure_dir(Path(artifacts_root) / "survival")
    overall = ProgressEmitter(
        "survival overall", status_path=artifacts_root / "overall_progress.json"
    )
    overall.emit(1, message=f"start method={method}", force=True)

    if smoke:
        max_partitions = max_partitions or 2

    def _load_progress(done: int, total: int, msg: str) -> None:
        # Train load occupies 5..22; val load occupies 22..30.
        frac = float(done) / float(max(total, 1))
        if "val" in msg:
            overall.emit(
                22 + int(8 * frac),
                message=msg,
                force=True,
                load_done=done,
                load_total=total,
            )
        else:
            overall.emit(
                5 + int(17 * frac),
                message=msg,
                force=True,
                load_done=done,
                load_total=total,
            )

    overall.emit(5, message="load train matrices", force=True)
    train_bundle = load_joined_matrices(
        world_root=world_root,
        split_root=split_root,
        split="train",
        max_partitions=max_partitions,
        population="broad",
        progress_cb=_load_progress,
    )
    overall.emit(25, message="load val matrices", force=True)
    val_bundle = load_joined_matrices(
        world_root=world_root,
        split_root=split_root,
        split="val",
        max_partitions=max_partitions,
        population="broad",
        progress_cb=_load_progress,
    )
    overall.emit(30, message=f"matrices ready train={train_bundle['n']} val={val_bundle['n']}", force=True)

    heads = [h for h in CORE_TARGET_SUITE if h.kind == "tte"]
    results: dict[str, Any] = {
        "family": "survival",
        "generated_at": utc_now(),
        "seed": seed,
        "heads": {},
        "train_n": train_bundle["n"],
        "val_n": val_bundle["n"],
        "smoke": smoke,
        "protocol_version": "v2",
        "method_default": method,
        "family_status": "complete",
        "eval_only_proof": {
            "permanent_val": "evaluation_only",
            "permanent_test": "untouched",
            "selection": "none_for_survival_final_fit",
        },
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
            results["heads"][spec.head_id] = {
                **honesty,
                "skipped": True,
                "status": "skipped_insufficient_supervision",
            }
            overall.update(hi + 1, len(heads), message=f"skip {spec.head_id}")
            continue

        Xtr = train_bundle["X"][ar_tr]
        Xva = val_bundle["X"][ar_va]
        Xtr_s = Xtr[:, : min(64, Xtr.shape[1])]
        Xva_s = Xva[:, : min(64, Xva.shape[1])]

        tag = f"survival {spec.head_id} final_fit"
        emitter = ProgressEmitter(tag, status_path=head_dir / "progress.json")
        t0 = time.time()
        used = method
        try:
            if method == "cox_subsample":
                risk, info = _cox_fit_predict(
                    Xtr_s, d_tr[ar_tr], e_tr[ar_tr], Xva_s, emitter, seed=seed
                )
            else:
                risk, info = _discrete_hazard_fit_predict(
                    Xtr_s,
                    d_tr[ar_tr],
                    e_tr[ar_tr],
                    Xva_s,
                    emitter,
                    seed=seed,
                    max_individuals=20_000 if smoke else 80_000,
                )
        except Exception as exc:  # noqa: BLE001
            emitter.emit(10, message=f"{method} failed: {exc}; fallback discrete")
            risk, info = _discrete_hazard_fit_predict(
                Xtr_s, d_tr[ar_tr], e_tr[ar_tr], Xva_s, emitter, seed=seed
            )
            used = "discrete_hazard"
            info["primary_error"] = str(exc)

        cidx = concordance_index(d_va[ar_va], e_va[ar_va], risk)
        metrics = {
            **honesty,
            "method": used,
            "model_info": info,
            "c_index": cidx,
            "runtime_sec": time.time() - t0,
            "events_val": int(e_va[ar_va].sum()),
            "at_risk_val": int(ar_va.sum()),
            "status": "complete",
        }
        write_json(head_dir / "metrics.json", metrics)
        results["heads"][spec.head_id] = metrics
        overall.update(hi + 1, len(heads), message=f"done {spec.head_id} method={used}")

    overall.complete(message="survival complete")
    write_json(artifacts_root / "results.json", results)
    write_json(
        artifacts_root / "FAMILY_STATUS.json",
        {
            "family": "survival",
            "status": "complete",
            "method": method,
            "generated_at": utc_now(),
            "prior_lifelines_cox_fullpop": {
                "stopped": True,
                "reason": "pathological_fullpop_lifelines_cox_worker_restarts",
                "app_id": "ap-LAAgokVVzmuQUN6T6G2ZdN",
            },
        },
    )
    return results
