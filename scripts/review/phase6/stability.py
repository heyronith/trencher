"""Section 4: champion stability over calendar days and WF folds."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scripts.review.phase5.common import fit_xgb_deterministic
from scripts.review.phase5.matrices import (
    load_head_bundle,
    walkforward_masks_from_detected,
)
from scripts.review.phase6.common import (
    HEADS,
    PHASE4_PREDS,
    Q85,
    SCRATCH,
    SEED,
    SPLIT,
    metrics_block,
    write_json,
)


def _daily_metrics(
    scores: np.ndarray,
    y: np.ndarray,
    elig: np.ndarray,
    obs_t: np.ndarray,
) -> list[dict[str, Any]]:
    keep = elig.astype(bool) & np.isfinite(obs_t)
    days = np.floor(obs_t / 86400.0).astype(np.int64)
    out = []
    for d in sorted(set(days[keep].tolist())):
        m = keep & (days == d)
        yy = y[m].astype(int)
        if yy.size == 0 or len(np.unique(yy)) < 2:
            out.append(
                {
                    "day_unix": int(d),
                    "n": int(m.sum()),
                    "n_pos": int(yy.sum()) if yy.size else 0,
                    "roc_auc": None,
                    "pr_auc": None,
                }
            )
            continue
        met = metrics_block(yy, scores[m])
        out.append(
            {
                "day_unix": int(d),
                "n": int(m.sum()),
                "n_pos": int(yy.sum()),
                "roc_auc": met["roc_auc"],
                "pr_auc": met["pr_auc"],
            }
        )
    return out


def section_stability() -> dict[str, Any]:
    preds = pq.read_table(PHASE4_PREDS)
    side = np.load(SCRATCH / "cache" / "val_side_fields.npz", allow_pickle=True)
    obs_t = side["observation_time_t"]
    out: dict[str, Any] = {**Q85, "permanent_val_by_day": {}, "wf_folds": {}}

    for head in HEADS:
        y = np.nan_to_num(
            preds[f"y_{head}"].to_numpy(zero_copy_only=False), nan=0.0
        ).astype(int)
        elig = preds[f"elig_{head}"].to_numpy(zero_copy_only=False).astype(bool)
        for fam in ("xgb", "lgb"):
            scores = preds[f"{fam}_{head}"].to_numpy(zero_copy_only=False).astype(
                np.float64
            )
            key = f"{fam}_{head}"
            out["permanent_val_by_day"][key] = _daily_metrics(scores, y, elig, obs_t)

    # WF fold slices: retrain champion XGB on fold train, score fold val (3 time slices
    # with permanent val as the third).
    from pathlib import Path

    core = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "manifests/world_model_data/baseline_v2/BASELINE_CORE_METRICS_V2.json"
        ).read_text(encoding="utf-8")
    )
    from trencher.data.world_model.splits.walkforward import build_walkforward_folds

    cutoffs = json.loads((SPLIT / "cutoffs.json").read_text(encoding="utf-8"))
    folds = [
        f
        for f in build_walkforward_folds(cutoffs)["folds"]
        if f["fold_id"] in ("wf_fold_1", "wf_fold_2")
    ]
    scored = []
    for head in HEADS:
        train = load_head_bundle(head, "train")
        fixed = int(core["xgboost"]["heads"][head]["fixed_iterations"])
        fold_masks = walkforward_masks_from_detected(train["detected_at"], folds)
        X = np.asarray(train["X"])
        y = train["y"]
        elig = train["elig"].astype(bool)
        times = train["times"]
        head_out = {}
        for fold in folds:
            fid = fold["fold_id"]
            tr = fold_masks[fid]["train"] & elig
            va = fold_masks[fid]["val"] & elig
            # Fit on fold-train eligible only: pass elig mask = tr
            elig_fit = tr
            print(f"  stability {head} {fid}...", flush=True)
            preds_va = fit_xgb_deterministic(
                X_train=X,
                y_train=y,
                elig_train=elig_fit,
                times_train=times,
                X_val=X,
                fixed_iters=fixed,
                seed=SEED,
                params_override={"nthread": 8, "seed": SEED},
                use_platt=True,
            )["calibrated"]
            yy = y[va].astype(int)
            met = metrics_block(yy, preds_va[va])
            head_out[fid] = {
                "train_cutoff": fold["train"],
                "val_window": fold["val"],
                "n": int(va.sum()),
                "n_pos": int(yy.sum()),
                "roc_auc": met["roc_auc"],
                "pr_auc": met["pr_auc"],
                "fixed_iterations": fixed,
            }
            scored.append({"head": head, "fold": fid, "fixed_iterations": fixed})
        # permanent val from phase4 champion
        yv = np.nan_to_num(
            preds[f"y_{head}"].to_numpy(zero_copy_only=False), nan=0.0
        ).astype(int)
        elv = preds[f"elig_{head}"].to_numpy(zero_copy_only=False).astype(bool)
        sv = preds[f"xgb_{head}"].to_numpy(zero_copy_only=False).astype(np.float64)
        met = metrics_block(yv[elv], sv[elv])
        head_out["permanent_val"] = {
            "n": int(elv.sum()),
            "n_pos": int(yv[elv].sum()),
            "roc_auc": met["roc_auc"],
            "pr_auc": met["pr_auc"],
            "source": "phase4_frozen_v2_xgb_preds",
        }
        out["wf_folds"][head] = head_out

    out["permanent_val_configs_scored"] = scored
    write_json(SCRATCH / "cache" / "section_stability.json", out)
    return out
