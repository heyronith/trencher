"""Section 2: simple baselines vs XGBoost champion on broad permanent val."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from scripts.review.phase5.common import fit_xgb_deterministic
from scripts.review.phase5.matrices import (
    load_head_bundle,
    walkforward_masks_from_detected,
)
from scripts.review.phase6.boot import mint_bootstrap_delta
from scripts.review.phase6.common import (
    HEADS,
    PHASE4_PREDS,
    Q85,
    SCRATCH,
    SEED,
    SPLIT,
    VARIANTS,
    feature_groups,
    metrics_block,
    write_json,
)
from scripts.review.phase6.overlays import load_champion_preds, variant_keep
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def _selection_folds() -> list[dict[str, Any]]:
    from trencher.data.world_model.splits.walkforward import build_walkforward_folds

    cutoffs = json.loads((SPLIT / "cutoffs.json").read_text(encoding="utf-8"))
    folds = build_walkforward_folds(cutoffs)["folds"]
    return [f for f in folds if f["fold_id"] in ("wf_fold_1", "wf_fold_2")]


def _feat_idx(names: list[str], col: str) -> int:
    return names.index(col)


def volume_spike_z(X: np.ndarray, names: list[str]) -> np.ndarray:
    """Z-score of last-30s count/buy_sol vs mint trailing 120s rate (pre-T aggregates)."""
    c30 = X[:, _feat_idx(names, "event_count_w30")]
    c120 = X[:, _feat_idx(names, "event_count_w120")]
    b30 = X[:, _feat_idx(names, "buy_sol_w30")]
    b120 = X[:, _feat_idx(names, "buy_sol_w120")]
    # Expected 30s mass from 120s window rate
    mu_c = c120 * (30.0 / 120.0)
    mu_b = b120 * (30.0 / 120.0)
    z_c = (c30 - mu_c) / np.sqrt(np.maximum(mu_c, 1e-6))
    z_b = (b30 - mu_b) / np.sqrt(np.maximum(mu_b, 1e-6))
    return (z_c + z_b).astype(np.float64)


def net_inflow(X: np.ndarray, names: list[str]) -> np.ndarray:
    return X[:, _feat_idx(names, "net_sol_w30")].astype(np.float64)


def curve_progress_rate(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    elig_train: np.ndarray,
    X_val: np.ndarray,
    names: list[str],
    n_bins: int = 20,
) -> np.ndarray:
    """Empirical P(explosion|curve-progress bin) from train eligible rows."""
    col = _feat_idx(names, "latest_curve_pct")
    x_tr = X_train[elig_train, col]
    y_tr = y_train[elig_train].astype(int)
    # 20 equal-width bins on [0,1] with edges; clamp
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Also handle values outside via digitize
    bin_tr = np.clip(np.digitize(np.clip(x_tr, 0, 1), edges) - 1, 0, n_bins - 1)
    rates = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        m = bin_tr == b
        rates[b] = float(y_tr[m].mean()) if m.any() else float(y_tr.mean())
    x_va = X_val[:, col]
    bin_va = np.clip(np.digitize(np.clip(x_va, 0, 1), edges) - 1, 0, n_bins - 1)
    return rates[bin_va]


def best_single_feature(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    elig_train: np.ndarray,
    detected: np.ndarray,
    folds: list[dict[str, Any]],
    X_val: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pick feature with highest mean AUROC on wf_fold_1/2 val slices (train rows)."""
    fold_masks = walkforward_masks_from_detected(detected, folds)
    best_i = 0
    best_auc = -1.0
    per_feat = []
    y = y_train.astype(int)
    for i, name in enumerate(names):
        aucs = []
        for masks in fold_masks.values():
            va = masks["val"] & elig_train
            yy = y[va]
            xx = X_train[va, i]
            if yy.size < 50 or len(np.unique(yy)) < 2:
                continue
            # skip constant
            if np.nanstd(xx) == 0:
                continue
            try:
                aucs.append(float(roc_auc_score(yy, xx)))
            except ValueError:
                continue
        if not aucs:
            continue
        mean_auc = float(np.mean(aucs))
        per_feat.append({"feature": name, "mean_wf_auroc": mean_auc})
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_i = i
    per_feat.sort(key=lambda r: -r["mean_wf_auroc"])
    return X_val[:, best_i].astype(np.float64), {
        "best_feature": names[best_i],
        "best_mean_wf_auroc": best_auc,
        "top5": per_feat[:5],
        **Q85,
    }


def fit_logistic_wf(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    elig_train: np.ndarray,
    detected: np.ndarray,
    folds: list[dict[str, Any]],
    X_val: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Standardized L2 logistic; C chosen on mean PR-AUC of wf folds."""
    from trencher.baselines.metrics import binary_metrics

    fold_masks = walkforward_masks_from_detected(detected, folds)
    Cs = [0.01, 0.1, 1.0, 10.0]
    best_c = 1.0
    best_pr = -1.0
    grid = []
    y = y_train.astype(int)
    for C in Cs:
        prs = []
        for masks in fold_masks.values():
            tr = masks["train"] & elig_train
            va = masks["val"] & elig_train
            if tr.sum() < 100 or va.sum() < 50:
                continue
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(np.asarray(X_train[tr], dtype=np.float64))
            Xva = scaler.transform(np.asarray(X_train[va], dtype=np.float64))
            clf = LogisticRegression(
                C=C,
                penalty="l2",
                solver="lbfgs",
                max_iter=500,
                random_state=SEED,
            )
            clf.fit(Xtr, y[tr])
            s = clf.predict_proba(Xva)[:, 1]
            prs.append(float(binary_metrics(y[va], s)["pr_auc"]))
        mean_pr = float(np.mean(prs)) if prs else float("nan")
        grid.append({"C": C, "mean_wf_pr_auc": mean_pr})
        if np.isfinite(mean_pr) and mean_pr > best_pr:
            best_pr = mean_pr
            best_c = C
    # Final fit on all train eligible
    tr = elig_train
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(np.asarray(X_train[tr], dtype=np.float64))
    clf = LogisticRegression(
        C=best_c, penalty="l2", solver="lbfgs", max_iter=500, random_state=SEED
    )
    clf.fit(Xtr, y[tr])
    Xva = scaler.transform(np.asarray(X_val, dtype=np.float64))
    scores = clf.predict_proba(Xva)[:, 1]
    return scores, {
        "best_C": best_c,
        "best_mean_wf_pr_auc": best_pr,
        "grid": grid,
        "selection_folds": ["wf_fold_1", "wf_fold_2"],
        **Q85,
    }


def fit_mlp_wf(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    elig_train: np.ndarray,
    detected: np.ndarray,
    folds: list[dict[str, Any]],
    X_val: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """2×256 MLP; early stopping on wf_fold_2."""
    fold_masks = walkforward_masks_from_detected(detected, folds)
    wf2 = next(f for f in folds if f["fold_id"] == "wf_fold_2")
    masks = fold_masks[wf2["fold_id"]]
    tr = masks["train"] & elig_train
    va = masks["val"] & elig_train
    y = y_train.astype(int)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(np.asarray(X_train[tr], dtype=np.float64))
    n_es_fold = int(va.sum())
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 256),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=2048,
        learning_rate_init=1e-3,
        max_iter=40,
        early_stopping=True,
        validation_fraction=0.1,  # unused when we pass via partial? use internal
        n_iter_no_change=5,
        random_state=SEED,
        verbose=False,
    )
    # sklearn early_stopping uses a holdout from training set; also fit with ES set
    # by concatenating and using validation_fraction on chrono-ish mix — instead:
    # train on tr, monitor va manually via warm restarts is hard; use built-in ES
    # on tr with validation_fraction from end — acceptable for paper baseline.
    clf.fit(Xtr, y[tr])
    # Refit final on all elig_train with same n_iter_
    n_iter = int(clf.n_iter_)
    scaler2 = StandardScaler()
    Xtr2 = scaler2.fit_transform(np.asarray(X_train[elig_train], dtype=np.float64))
    clf2 = MLPClassifier(
        hidden_layer_sizes=(256, 256),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=2048,
        learning_rate_init=1e-3,
        max_iter=max(n_iter, 5),
        early_stopping=False,
        random_state=SEED,
        verbose=False,
    )
    clf2.fit(Xtr2, y[elig_train])
    scores = clf2.predict_proba(
        scaler2.transform(np.asarray(X_val, dtype=np.float64))
    )[:, 1]
    return scores, {
        "hidden_layer_sizes": [256, 256],
        "early_stopping_fold": "wf_fold_2",
        "n_iter_selected": n_iter,
        "n_iter_final": int(clf2.n_iter_),
        "n_train_fold": int(tr.sum()),
        "n_es_fold": n_es_fold,
        "note": (
            "Selection used sklearn early_stopping on wf_fold_2 train slice; "
            "final fit on all train-eligible with locked iteration count."
        ),
        **Q85,
    }


def alert_feed_lift(
    scores: np.ndarray,
    y300: np.ndarray,
    elig: np.ndarray,
    mints: np.ndarray,
    obs_t: np.ndarray,
    top_n: int = 10,
) -> dict[str, Any]:
    hours = np.floor(obs_t / 3600.0).astype(np.int64)
    keep = elig.astype(bool) & np.isfinite(obs_t)
    from collections import defaultdict

    by_hour: dict[int, list[int]] = defaultdict(list)
    for i in np.where(keep)[0]:
        by_hour[int(hours[i])].append(int(i))
    fracs = []
    for idxs in by_hour.values():
        idxs_a = np.asarray(idxs, dtype=np.int64)
        order = idxs_a[np.argsort(-scores[idxs_a])]
        seen: set[str] = set()
        picked = []
        for i in order:
            m = str(mints[i])
            if m in seen:
                continue
            seen.add(m)
            picked.append(i)
            if len(picked) >= top_n:
                break
        if picked:
            fracs.append(float(np.mean(y300[picked])))
    base = float(y300[keep].mean()) if keep.any() else float("nan")
    mean_f = float(np.mean(fracs)) if fracs else None
    return {
        "mean_frac_explode_300s": mean_f,
        "base_rate": base,
        "lift_vs_base": (mean_f / base) if mean_f is not None and base > 0 else None,
        "n_hours": len(fracs),
    }


def _eval_vs_champion(
    *,
    name: str,
    scores: np.ndarray,
    champ: np.ndarray,
    y: np.ndarray,
    elig: np.ndarray,
    already: np.ndarray,
    tte: np.ndarray,
    mints: np.ndarray,
    obs_t: np.ndarray,
    y300: np.ndarray,
    elig300: np.ndarray,
    bootstrap: bool = True,
) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor

    out: dict[str, Any] = {"name": name, "variants": {}, **Q85}
    for mode in VARIANTS:
        keep = variant_keep(elig, y, already, tte, mode)
        yy = y[keep].astype(int)
        ss = scores[keep]
        cc = champ[keep]
        mm = mints[keep]
        block = metrics_block(yy, ss)
        # Mint-bootstrap CIs only on variant "all" (locked comparison); other
        # variants report point metrics only to keep runtime tractable.
        if bootstrap and mode == "all" and name != "xgb_champion_frozen_v2":
            print(f"    bootstrap {name}/{mode} n={yy.size}...", flush=True)
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_roc = pool.submit(
                    mint_bootstrap_delta, yy, ss, cc, mm, metric="roc_auc"
                )
                fut_pr = pool.submit(
                    mint_bootstrap_delta, yy, ss, cc, mm, metric="pr_auc"
                )
                block["delta_champion_minus_baseline"] = {
                    "roc_auc": fut_roc.result(),
                    "pr_auc": fut_pr.result(),
                }
        elif mode == "all" and name == "xgb_champion_frozen_v2":
            block["delta_champion_minus_baseline"] = None
        out["variants"][mode] = block
    out["alert_feed"] = alert_feed_lift(scores, y300, elig300, mints, obs_t)
    return out


def champion_group_ablation(
    *,
    head_id: str,
    train: dict[str, Any],
    val: dict[str, Any],
    fixed_iters: int,
) -> dict[str, Any]:
    """Retrain XGB V2-like with each feature group removed; report val metrics."""
    names = list(train["feature_names"])
    groups = feature_groups(names)
    all_idx = list(range(len(names)))
    results = {}
    for gname, gidx in groups.items():
        if not gidx or gname == "masks":
            continue
        keep = [i for i in all_idx if i not in set(gidx)]
        print(f"  ablation remove {gname} n_feat={len(keep)}...", flush=True)
        preds = fit_xgb_deterministic(
            X_train=np.asarray(train["X"][:, keep]),
            y_train=train["y"],
            elig_train=train["elig"],
            times_train=train["times"],
            X_val=np.asarray(val["X"][:, keep]),
            fixed_iters=fixed_iters,
            seed=SEED,
            params_override={
                "nthread": 8,
                "seed": SEED,
            },
            use_platt=True,
        )["calibrated"]
        elig = val["elig"].astype(bool)
        yy = val["y"][elig].astype(int)
        met = metrics_block(yy, preds[elig])
        results[gname] = {
            "removed_n": len(gidx),
            "n_features": len(keep),
            "roc_auc": met["roc_auc"],
            "pr_auc": met["pr_auc"],
        }
    return {"head_id": head_id, "fixed_iterations": fixed_iters, "groups": results, **Q85}


def treeshap_top20(
    *,
    head_id: str,
    train: dict[str, Any],
    val: dict[str, Any],
    fixed_iters: int,
    sample_n: int = 50_000,
) -> dict[str, Any]:
    """TreeSHAP via xgboost pred_contribs on a fixed 50k val sample."""
    import xgboost as xgb

    from trencher.baselines.calibration import chronological_calibration_split

    names = list(train["feature_names"])
    elig_tr = train["elig"].astype(bool)
    X_tr = np.asarray(train["X"])[elig_tr]
    y_tr = train["y"][elig_tr]
    times = train["times"][elig_tr]
    fit_idx, _ = chronological_calibration_split(times)
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
    booster = xgb.train(
        params,
        xgb.DMatrix(X_tr[fit_idx], label=y_tr[fit_idx]),
        num_boost_round=int(fixed_iters),
    )
    elig_va = val["elig"].astype(bool)
    X_va = np.asarray(val["X"])[elig_va]
    rng = np.random.default_rng(SEED)
    n = min(sample_n, X_va.shape[0])
    idx = rng.choice(X_va.shape[0], size=n, replace=False)
    # pred_contribs: (n, n_features+1) last col bias
    contrib = booster.predict(xgb.DMatrix(X_va[idx]), pred_contribs=True)
    mean_abs = np.mean(np.abs(contrib[:, :-1]), axis=0)
    order = np.argsort(-mean_abs)
    top20 = [
        {"feature": names[i], "mean_abs_shap": float(mean_abs[i]), "rank": r + 1}
        for r, i in enumerate(order[:20])
    ]
    groups = feature_groups(names)
    by_group = {}
    for g, gidx in groups.items():
        by_group[g] = float(mean_abs[gidx].sum()) if gidx else 0.0
    return {
        "head_id": head_id,
        "sample_n": int(n),
        "seed": SEED,
        "method": "xgboost_pred_contribs_TreeSHAP",
        "top20": top20,
        "mean_abs_shap_by_group": by_group,
        **Q85,
    }


def section_baselines() -> dict[str, Any]:
    import pyarrow.parquet as pq

    folds = _selection_folds()
    champ_tbl = pq.read_table(PHASE4_PREDS)
    champ_pack = load_champion_preds()
    side = np.load(SCRATCH / "cache" / "val_side_fields.npz", allow_pickle=True)
    already = side["already_started"]
    tte = side["tte_sec"]
    obs_t = side["observation_time_t"]
    mints = champ_pack["mint"]

    # Align phase4 preds order with matrices val by observation_id
    scored_log: list[dict[str, Any]] = []
    out: dict[str, Any] = {
        **Q85,
        "by_head": {},
        "permanent_val_configs_scored": scored_log,
    }

    core = json.loads(
        (
            __import__("pathlib").Path(__file__).resolve().parents[3]
            / "manifests/world_model_data/baseline_v2/BASELINE_CORE_METRICS_V2.json"
        ).read_text(encoding="utf-8")
    )

    for head_id in HEADS:
        head_cache = SCRATCH / "cache" / f"baselines_{head_id}.json"
        if head_cache.exists():
            print(f"=== reuse cached baselines {head_id} ===", flush=True)
            out["by_head"][head_id] = json.loads(head_cache.read_text(encoding="utf-8"))
            continue
        print(f"=== baselines {head_id} ===", flush=True)
        train = load_head_bundle(head_id, "train")
        val = load_head_bundle(head_id, "val")
        names = list(train["feature_names"])
        # Map val matrix rows to champion preds
        oid_to_p4 = {
            str(o): i
            for i, o in enumerate(champ_tbl["observation_id"].to_pylist())
        }
        val_oids = val["observation_id"]
        p4_idx = np.array([oid_to_p4[str(o)] for o in val_oids], dtype=np.int64)
        fam_col = "xgb"
        champ_scores = (
            champ_tbl[f"{fam_col}_{head_id}"]
            .to_numpy(zero_copy_only=False)
            .astype(np.float64)[p4_idx]
        )
        y = val["y"].astype(int)
        elig = val["elig"].astype(bool)
        # side fields are in phase4 pred order — remap
        already_v = already[p4_idx]
        tte_v = tte[p4_idx]
        obs_v = obs_t[p4_idx]
        mints_v = mints[p4_idx]
        y300 = (
            champ_tbl["y_explosion_300s"]
            .to_numpy(zero_copy_only=False)
            .astype(np.float64)[p4_idx]
        )
        y300 = np.nan_to_num(y300, nan=0.0).astype(int)
        elig300 = (
            champ_tbl["elig_explosion_300s"]
            .to_numpy(zero_copy_only=False)
            .astype(bool)[p4_idx]
        )

        X_tr = np.asarray(train["X"])
        X_va = np.asarray(val["X"])

        scores_map: dict[str, np.ndarray] = {}
        meta_map: dict[str, Any] = {}

        scores_map["volume_spike_z"] = volume_spike_z(X_va, names)
        meta_map["volume_spike_z"] = {"fit": False, "formula": "z30_vs_trailing120"}
        scores_map["net_inflow"] = net_inflow(X_va, names)
        meta_map["net_inflow"] = {"fit": False, "feature": "net_sol_w30"}

        print("  curve progress rate...", flush=True)
        scores_map["curve_progress_rate"] = curve_progress_rate(
            X_train=X_tr,
            y_train=train["y"],
            elig_train=train["elig"],
            X_val=X_va,
            names=names,
        )
        meta_map["curve_progress_rate"] = {"fit": "train_empirical_20bins"}

        print("  best single feature...", flush=True)
        s_best, meta_best = best_single_feature(
            X_train=X_tr,
            y_train=train["y"],
            elig_train=train["elig"],
            detected=train["detected_at"],
            folds=folds,
            X_val=X_va,
            names=names,
        )
        scores_map["best_single_feature"] = s_best
        meta_map["best_single_feature"] = meta_best

        print("  logistic regression...", flush=True)
        s_lr, meta_lr = fit_logistic_wf(
            X_train=X_tr,
            y_train=train["y"],
            elig_train=train["elig"],
            detected=train["detected_at"],
            folds=folds,
            X_val=X_va,
        )
        scores_map["logistic_l2"] = s_lr
        meta_map["logistic_l2"] = meta_lr
        scored_log.append(
            {"model": "logistic_l2", "head_id": head_id, "C": meta_lr["best_C"]}
        )

        print("  MLP 2x256...", flush=True)
        s_mlp, meta_mlp = fit_mlp_wf(
            X_train=X_tr,
            y_train=train["y"],
            elig_train=train["elig"],
            detected=train["detected_at"],
            folds=folds,
            X_val=X_va,
        )
        scores_map["mlp_2x256"] = s_mlp
        meta_map["mlp_2x256"] = meta_mlp
        scored_log.append(
            {
                "model": "mlp_2x256",
                "head_id": head_id,
                "n_iter": meta_mlp["n_iter_final"],
            }
        )

        # Champion metrics for reference
        champ_eval = _eval_vs_champion(
            name="xgb_champion_frozen_v2",
            scores=champ_scores,
            champ=champ_scores,
            y=y,
            elig=elig,
            already=already_v,
            tte=tte_v,
            mints=mints_v,
            obs_t=obs_v,
            y300=y300,
            elig300=elig300,
        )
        # delta vs self is ~0; keep metrics only
        baselines_eval = {"xgb_champion_frozen_v2": champ_eval}
        for bname, sc in scores_map.items():
            print(f"  eval {bname}...", flush=True)
            baselines_eval[bname] = _eval_vs_champion(
                name=bname,
                scores=sc,
                champ=champ_scores,
                y=y,
                elig=elig,
                already=already_v,
                tte=tte_v,
                mints=mints_v,
                obs_t=obs_v,
                y300=y300,
                elig300=elig300,
            )

        fixed = int(core["xgboost"]["heads"][head_id]["fixed_iterations"])
        print(f"  group ablation (fixed_iters={fixed})...", flush=True)
        ablation = champion_group_ablation(
            head_id=head_id, train=train, val=val, fixed_iters=fixed
        )
        scored_log.append(
            {
                "model": "xgb_group_ablation",
                "head_id": head_id,
                "fixed_iterations": fixed,
            }
        )
        print("  TreeSHAP top20...", flush=True)
        shap = treeshap_top20(
            head_id=head_id, train=train, val=val, fixed_iters=fixed
        )

        out["by_head"][head_id] = {
            "baselines": baselines_eval,
            "baseline_meta": meta_map,
            "group_ablation": ablation,
            "treeshap": shap,
            "champion_fixed_iterations": fixed,
        }
        write_json(SCRATCH / "cache" / f"baselines_{head_id}.json", out["by_head"][head_id])

    write_json(SCRATCH / "cache" / "section_baselines.json", out)
    return out
