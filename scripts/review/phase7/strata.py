"""Section 2: stratified evaluation + age×curve baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scripts.review.phase5.common import fit_xgb_deterministic
from scripts.review.phase5.matrices import (
    load_head_bundle,
    walkforward_masks_from_detected,
)
from scripts.review.phase6.baselines import fit_logistic_wf
from scripts.review.phase7.common import (
    AGE_CURVE_FEATURES,
    PHASE4_PREDS,
    Q85,
    SCRATCH,
    SEED,
    SPLIT,
    STAGE_B_PREDS,
    WORLD,
    core_xgb_fixed,
    metrics_block,
    mint_bootstrap_delta,
    write_json,
)


def _load_triggers(oids: np.ndarray) -> np.ndarray:
    want = set(map(str, oids.tolist()))
    out = {str(o): None for o in oids}
    root = WORLD / "broad" / "broad_targets_v1" / "observation_targets"
    for part in sorted(root.glob("part-*.parquet")):
        schema = set(pq.read_schema(part).names)
        if "observation_trigger" not in schema:
            continue
        t = pq.read_table(part, columns=["observation_id", "observation_trigger"])
        for oid, tr in zip(
            t["observation_id"].to_pylist(),
            t["observation_trigger"].to_pylist(),
            strict=True,
        ):
            if str(oid) in want:
                out[str(oid)] = tr
    return np.array([out[str(o)] for o in oids], dtype=object)


def _age_bucket(age: np.ndarray) -> np.ndarray:
    """Label age buckets."""
    labels = np.empty(len(age), dtype=object)
    labels[:] = "unknown"
    m = np.isfinite(age)
    labels[m & (age < 30)] = "[0,30s)"
    labels[m & (age >= 30) & (age < 120)] = "[30s,2m)"
    labels[m & (age >= 120) & (age < 600)] = "[2m,10m)"
    labels[m & (age >= 600) & (age < 3600)] = "[10m,1h)"
    labels[m & (age >= 3600)] = ">=1h"
    return labels


def _approx_trades_last_60s(event_w30: np.ndarray, event_w120: np.ndarray) -> np.ndarray:
    """Approximate last-60s trade count from w30/w120 (no w60 feature).

    Assumes uniform intensity on (30,120]: count_60 ≈ c30 + (c120-c30)*(30/90).
    """
    c30 = np.asarray(event_w30, dtype=np.float64)
    c120 = np.asarray(event_w120, dtype=np.float64)
    return c30 + np.maximum(c120 - c30, 0.0) * (30.0 / 90.0)


def _stratum_metrics(
    y: np.ndarray, scores: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, Any]:
    m = mask.astype(bool)
    yy = y[m].astype(int)
    out: dict[str, Any] = {
        "n": int(m.sum()),
        "n_pos": int(yy.sum()) if yy.size else 0,
        "prevalence": float(yy.mean()) if yy.size else None,
        "models": {},
    }
    for name, s in scores.items():
        ss = s[m]
        if yy.size == 0 or len(np.unique(yy)) < 2:
            out["models"][name] = {
                "roc_auc": None,
                "pr_auc": None,
                "precision_at_1pct": None,
            }
            continue
        met = metrics_block(yy, ss)
        out["models"][name] = {
            "roc_auc": met["roc_auc"],
            "pr_auc": met["pr_auc"],
            "precision_at_1pct": met["precision_at"]["top_0.01"],
        }
    return out


def section_strata() -> dict[str, Any]:
    import json

    from trencher.data.world_model.splits.walkforward import build_walkforward_folds

    print("loading matrices + champion / stage-b preds...", flush=True)
    results: dict[str, Any] = {**Q85, "by_head": {}}
    trees = pq.read_table(PHASE4_PREDS)
    stage_b = pq.read_table(STAGE_B_PREDS)
    sb_oid = {
        str(o): i for i, o in enumerate(stage_b["observation_id"].to_pylist())
    }
    p4_oid = {
        str(o): i for i, o in enumerate(trees["observation_id"].to_pylist())
    }

    for head_id in ("explosion_60s", "explosion_300s"):
        print(f"=== strata {head_id} ===", flush=True)
        train = load_head_bundle(head_id, "train")
        val = load_head_bundle(head_id, "val")
        names = list(train["feature_names"])
        name_to_i = {n: i for i, n in enumerate(names)}

        p4_idx = np.array(
            [p4_oid[str(o)] for o in val["observation_id"]], dtype=np.int64
        )
        xgb = (
            trees[f"xgb_{head_id}"]
            .to_numpy(zero_copy_only=False)
            .astype(np.float64)[p4_idx]
        )
        sb_col = f"p_{head_id}"
        sb_np = stage_b[sb_col].to_numpy(zero_copy_only=False).astype(np.float64)
        sb_scores = np.array(
            [
                sb_np[sb_oid[str(o)]] if str(o) in sb_oid else np.nan
                for o in val["observation_id"]
            ],
            dtype=np.float64,
        )

        print("  fitting logistic_l2 on train...", flush=True)
        folds = [
            f
            for f in build_walkforward_folds(
                json.loads((SPLIT / "cutoffs.json").read_text(encoding="utf-8"))
            )["folds"]
            if f["fold_id"] in ("wf_fold_1", "wf_fold_2")
        ]
        log_scores, log_meta = fit_logistic_wf(
            X_train=np.asarray(train["X"]),
            y_train=train["y"],
            elig_train=train["elig"],
            detected=train["detected_at"],
            folds=folds,
            X_val=np.asarray(val["X"]),
        )

        scores = {
            "xgb_champion": xgb,
            "logistic_l2": log_scores,
            "stage_b": sb_scores,
        }

        elig = val["elig"].astype(bool) & np.isfinite(xgb) & np.isfinite(sb_scores)
        y = val["y"].astype(int)
        age = np.asarray(val["X"][:, name_to_i["token_age_seconds"]], dtype=np.float64)
        triggers = _load_triggers(val["observation_id"])
        c30 = np.asarray(val["X"][:, name_to_i["event_count_w30"]], dtype=np.float64)
        c120 = np.asarray(val["X"][:, name_to_i["event_count_w120"]], dtype=np.float64)
        trades_60 = _approx_trades_last_60s(c30, c120)
        buckets = _age_bucket(age)

        strata: dict[str, Any] = {
            "by_trigger": {},
            "by_age_bucket": {},
            "active_only": {},
            "trade_count_proxy_note": (
                "No event_count_w60 in feature_spec (windows 10/30/120). "
                "Last-60s count ≈ event_count_w30 + (event_count_w120-event_count_w30)*(30/90)."
            ),
        }
        for trig in ("fixed_age", "event_count"):
            m = elig & (triggers == trig)
            strata["by_trigger"][trig] = _stratum_metrics(y, scores, m)
        for b in ("[0,30s)", "[30s,2m)", "[2m,10m)", "[10m,1h)", ">=1h"):
            m = elig & (buckets == b)
            strata["by_age_bucket"][b] = _stratum_metrics(y, scores, m)
        strata["active_only"]["ge1_trade_last_60s_approx"] = _stratum_metrics(
            y, scores, elig & (trades_60 >= 1)
        )
        strata["active_only"]["ge5_trades_last_60s_approx"] = _stratum_metrics(
            y, scores, elig & (trades_60 >= 5)
        )

        # Age + curve XGB
        print("  training age+curve XGB...", flush=True)
        ac_idx = [name_to_i[n] for n in AGE_CURVE_FEATURES if n in name_to_i]
        fixed = core_xgb_fixed(head_id)
        # Select iters on WF for this subset
        fold_masks = walkforward_masks_from_detected(train["detected_at"], folds)
        import xgboost as xgb_lib

        best_iters = []
        Xtr_full = np.asarray(train["X"][:, ac_idx])
        ytr = train["y"]
        eltr = train["elig"].astype(bool)
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
            tr = fold_masks[fold["fold_id"]]["train"] & eltr
            va = fold_masks[fold["fold_id"]]["val"] & eltr
            if tr.sum() < 100 or va.sum() < 50:
                continue
            booster = xgb_lib.train(
                params,
                xgb_lib.DMatrix(Xtr_full[tr], label=ytr[tr]),
                num_boost_round=200,
                evals=[(xgb_lib.DMatrix(Xtr_full[va], label=ytr[va]), "val")],
                early_stopping_rounds=40,
                verbose_eval=False,
            )
            best_iters.append(max(int(booster.best_iteration) + 1, 50))
        fixed_ac = int(np.median(best_iters)) if best_iters else fixed
        ac_scores = fit_xgb_deterministic(
            X_train=Xtr_full,
            y_train=ytr,
            elig_train=eltr,
            times_train=train["times"],
            X_val=np.asarray(val["X"][:, ac_idx]),
            fixed_iters=fixed_ac,
            seed=SEED,
            params_override={"nthread": 8, "seed": SEED},
            use_platt=True,
        )["calibrated"]
        scores["age_curve_xgb"] = ac_scores

        # Overall + gain CI champion - age_curve
        overall = _stratum_metrics(y, scores, elig)
        yy = y[elig]
        delta = mint_bootstrap_delta(
            yy, ac_scores[elig], xgb[elig], val["mint"][elig], metric="roc_auc"
        )
        delta_pr = mint_bootstrap_delta(
            yy, ac_scores[elig], xgb[elig], val["mint"][elig], metric="pr_auc"
        )

        results["by_head"][head_id] = {
            "strata": strata,
            "overall": overall,
            "age_curve": {
                "features": AGE_CURVE_FEATURES,
                "fixed_iterations": fixed_ac,
                "metrics": overall["models"]["age_curve_xgb"],
                "delta_champion_minus_age_curve": {
                    "roc_auc": delta,
                    "pr_auc": delta_pr,
                },
            },
            "logistic_meta": log_meta,
            "permanent_val_configs_scored": [
                {"model": "logistic_l2", "head": head_id, "C": log_meta["best_C"]},
                {
                    "model": "age_curve_xgb",
                    "head": head_id,
                    "fixed_iterations": fixed_ac,
                },
            ],
        }
        write_json(
            SCRATCH / "cache" / f"strata_{head_id}.json", results["by_head"][head_id]
        )

    write_json(SCRATCH / "cache" / "section_strata.json", results)
    return results
