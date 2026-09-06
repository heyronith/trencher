"""RICH-additive ablation: broad-only vs broad+rich on identical frozen rich observations.

Permanent q70–q85 validation is evaluation-only. Selection / early stopping only on
wf1+wf2. Arm selection must never use permanent validation during fitting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from trencher.baselines.calibration import chronological_calibration_split
from trencher.baselines.dataset import load_joined_matrices, walkforward_mask
from trencher.baselines.feature_spec import (
    all_feature_names,
    broad_feature_names,
    rich_additive_feature_names,
)
from trencher.baselines.metrics import binary_metrics, regression_metrics
from trencher.baselines.split_guard import refuse_test_split
from trencher.baselines.targets import CORE_TARGET_SUITE
from trencher.baselines.train_trees import _prepare_xy, train_lightgbm_head
from trencher.data.world_model.io import ensure_dir, utc_now, write_json
from trencher.data.world_model.splits.walkforward import build_walkforward_folds

ABLATION_HEADS = (
    "explosion_60s",
    "explosion_300s",
    "barrier_u25_l15_5m",
    "barrier_u100_l30_1h",
    "return_5m",
    "return_1h",
)


def _score_metrics(kind: str, y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    if kind == "binary":
        return binary_metrics(y, pred)
    if kind == "regression":
        return regression_metrics(y, pred)
    # multiclass: pred may be class ids or probabilities
    out: dict[str, Any] = {}
    if getattr(pred, "ndim", 1) == 2:
        p = np.clip(pred, 1e-7, 1 - 1e-7)
        p = p / p.sum(axis=1, keepdims=True)
        ll = -np.mean(np.log(p[np.arange(len(y)), y.astype(int)]))
        out["multiclass_logloss"] = float(ll)
        out["upper_first"] = binary_metrics((y == 0).astype(int), p[:, 0])
    else:
        out["note"] = "multiclass_pred_1d"
        out["accuracy"] = float(np.mean(pred.astype(int) == y.astype(int)))
    return out


def _fit_arm_v2(
    *,
    X: np.ndarray,
    y: np.ndarray,
    elig: np.ndarray,
    meta: list[dict[str, Any]],
    X_val: np.ndarray,
    y_val: np.ndarray,
    elig_val: np.ndarray,
    kind: str,
    selection_folds: list[dict[str, Any]],
    out_dir: Path,
    tag: str,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    """wf1/wf2 ES → fixed iters → final fit without permanent-val ES → score val once."""
    planned = 50 if smoke else 200
    best_iters: list[int] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold in selection_folds:
        tr_m = walkforward_mask(meta, fold, "train") & elig
        va_m = walkforward_mask(meta, fold, "val") & elig
        if tr_m.sum() < 30 or va_m.sum() < 10:
            continue
        art = train_lightgbm_head(
            X_train=X[tr_m],
            y_train=y[tr_m],
            X_val=X[va_m],
            y_val=y[va_m],
            kind=kind,
            out_dir=ensure_dir(out_dir / fold["fold_id"]),
            tag=f"{tag} {fold['fold_id']}",
            num_boost_round=planned,
            early_stopping_rounds=30,
            seed=seed,
            use_gpu=False,
        )
        bi = int(art["best_iteration"] or art["actual_iterations"] or planned)
        best_iters.append(max(bi, 1))
        fold_summaries.append(
            {
                "fold_id": fold["fold_id"],
                "train_n": int(tr_m.sum()),
                "val_n": int(va_m.sum()),
                "best_iteration": bi,
            }
        )

    fixed_iters = int(np.median(np.asarray(best_iters, dtype=np.float64))) if best_iters else planned
    fixed_iters = max(fixed_iters, 10 if smoke else 50)

    tr = elig
    times = np.array(
        [
            float(m["observation_time_t"] or 0.0)
            for m, e in zip(meta, elig, strict=True)
            if e
        ],
        dtype=np.float64,
    )
    X_tr, y_tr = X[tr], y[tr]
    fit_idx, cal_idx = chronological_calibration_split(times)
    art = train_lightgbm_head(
        X_train=X_tr[fit_idx],
        y_train=y_tr[fit_idx],
        X_val=X_tr[cal_idx],
        y_val=y_tr[cal_idx],
        kind=kind,
        out_dir=ensure_dir(out_dir / "final_fit"),
        tag=f"{tag} final_fit",
        num_boost_round=fixed_iters,
        early_stopping_rounds=None,
        seed=seed,
        use_gpu=False,
    )
    # Score permanent validation once (eval-only).
    va = elig_val
    import lightgbm as lgb

    booster = lgb.Booster(model_file=art["model_path"])
    n_it = int(art["best_iteration"] or fixed_iters)
    if kind == "multiclass" or kind == "binary":
        pred = booster.predict(X_val[va], num_iteration=n_it + 1)
    else:
        pred = booster.predict(X_val[va], num_iteration=n_it + 1)

    return {
        "fixed_iterations": fixed_iters,
        "selection_best_iters": best_iters,
        "fold_summaries": fold_summaries,
        "train_n": int(tr.sum()),
        "val_n": int(va.sum()),
        "metrics": _score_metrics(kind, y_val[va], np.asarray(pred)),
        "runtime_sec": art["runtime_sec"],
        "eval_only": True,
    }


def run_rich_ablation_v2(
    *,
    world_root: Path,
    split_root: Path | None = None,
    artifacts_root: Path,
    seed: int = 0,
    max_partitions: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Compare A=124 broad vs B=146 broad+rich on identical frozen rich rows."""
    refuse_test_split("val")
    world_root = Path(world_root)
    split_root = Path(split_root or world_root / "splits" / "split_v1")
    out_dir = ensure_dir(Path(artifacts_root) / "rich_ablation_v2")

    broad_names = broad_feature_names()
    rich_names = all_feature_names(include_rich=True)
    assert len(broad_names) == 124
    assert len(rich_names) == 124 + len(rich_additive_feature_names())

    # Single rich-population cache supplies both arms (identical observation IDs).
    train = load_joined_matrices(
        world_root=world_root,
        split_root=split_root,
        split="train",
        feature_names=rich_names,
        max_partitions=max_partitions or (2 if smoke else None),
        population="rich",
        include_rich_features=True,
    )
    val = load_joined_matrices(
        world_root=world_root,
        split_root=split_root,
        split="val",
        feature_names=rich_names,
        max_partitions=max_partitions or (2 if smoke else None),
        population="rich",
        include_rich_features=True,
    )
    if train["n"] == 0 or val["n"] == 0:
        payload = {
            "ablation_version": "BASELINE_RICH_ABLATION_V2",
            "generated_at": utc_now(),
            "skipped": True,
            "reason": "rich feature cache empty; build population=rich first",
            "train_n": train["n"],
            "val_n": val["n"],
        }
        write_json(out_dir / "BASELINE_RICH_ABLATION_V2.json", payload)
        return payload

    # Coverage invariants before any fitting.
    if train["X"].shape[1] != len(rich_names) or val["X"].shape[1] != len(rich_names):
        raise RuntimeError(
            f"rich feature dims mismatch: train={train['X'].shape[1]} "
            f"val={val['X'].shape[1]} expected={len(rich_names)}"
        )
    tr_oids = [str(m["observation_id"]) for m in train["meta"]]
    va_oids = [str(m["observation_id"]) for m in val["meta"]]
    if len(tr_oids) != len(set(tr_oids)) or len(va_oids) != len(set(va_oids)):
        raise RuntimeError("rich feature cache has duplicate observation_ids")
    # Target-join coverage must be 100% (every feature row has a target dict).
    if any(not t for t in train["targets"]) or any(not t for t in val["targets"]):
        n_miss_tr = sum(1 for t in train["targets"] if not t)
        n_miss_va = sum(1 for t in val["targets"] if not t)
        raise RuntimeError(
            f"rich target-join incomplete: train_missing={n_miss_tr} val_missing={n_miss_va}"
        )
    # Derive expected train/val sizes from frozen rich assignments (not hardcoded).
    from trencher.baselines.feature_cache import _index_assignments
    from trencher.baselines.progress import ProgressEmitter

    cutoffs = json.loads((split_root / "cutoffs.json").read_text(encoding="utf-8"))
    q85 = float(cutoffs["quantiles_unix"]["q85"])
    by_mp, _ = _index_assignments(
        assign_dir=split_root / "observation_assignments_rich",
        q85=q85,
        max_partitions=max_partitions or (2 if smoke else None),
        progress=ProgressEmitter("rich ablation assign check"),
    )
    expect_tr = sum(1 for rows in by_mp.values() for r in rows if r["base_split"] == "train")
    expect_va = sum(1 for rows in by_mp.values() for r in rows if r["base_split"] == "val")
    if train["n"] != expect_tr or val["n"] != expect_va:
        raise RuntimeError(
            f"rich feature rows != assignment population: "
            f"train {train['n']}!={expect_tr} val {val['n']}!={expect_va}"
        )

    # Align arms on identical rows / targets (bit-identical oid order).
    X_b_tr = train["X"][:, : len(broad_names)]
    X_r_tr = train["X"]
    X_b_va = val["X"][:, : len(broad_names)]
    X_r_va = val["X"]
    assert X_b_tr.shape[0] == X_r_tr.shape[0]
    assert X_b_va.shape[0] == X_r_va.shape[0]
    assert tr_oids == [str(m["observation_id"]) for m in train["meta"]]
    assert va_oids == [str(m["observation_id"]) for m in val["meta"]]

    selection_folds = [
        f
        for f in build_walkforward_folds(cutoffs)["folds"]
        if f["fold_id"] in ("wf_fold_1", "wf_fold_2")
    ]
    eval_only_proof = {
        "selection_folds": [f["fold_id"] for f in selection_folds],
        "excluded_from_selection": ["wf_fold_3", "permanent_val_q70_q85"],
        "permanent_val_used_for": ["final_score_once"],
        "permanent_val_forbidden_for": [
            "early_stopping",
            "best_iteration_selection",
            "arm_selection",
            "hyperparameter_selection",
        ],
        "arm_selection": "not_based_on_permanent_val",
        "population": "frozen_rich_observation_assignments",
        "permanent_test_q85_plus": "untouched",
    }

    payload: dict[str, Any] = {
        "ablation_version": "BASELINE_RICH_ABLATION_V2",
        "generated_at": utc_now(),
        "feature_dims": {
            "broad": len(broad_names),
            "broad_plus_rich": len(rich_names),
            "rich_additive": len(rich_additive_feature_names()),
        },
        "rich_train_n": int(train["n"]),
        "rich_val_n": int(val["n"]),
        "observation_ids_aligned": True,
        "eval_only_proof": eval_only_proof,
        "heads": {},
        "note": (
            "Same frozen rich observation ids for arms A (broad-only columns) "
            "and B (broad+rich). Rich features from sequence_dataset_v2."
        ),
    }

    specs = [s for s in CORE_TARGET_SUITE if s.head_id in ABLATION_HEADS]
    for spec in specs:
        y_tr, elig_tr, counts_tr, kind = _prepare_xy(train, spec, split_root)
        y_va, elig_va, counts_va, _ = _prepare_xy(val, spec, split_root)
        tr = elig_tr
        va = elig_va
        if tr.sum() < 30 or va.sum() < 10:
            payload["heads"][spec.head_id] = {
                "skipped": True,
                "kind": kind,
                "train_n": int(tr.sum()),
                "val_n": int(va.sum()),
                "counts_train": counts_tr,
                "counts_val": counts_va,
            }
            continue

        arms: dict[str, Any] = {}
        for arm, Xtr, Xva in (
            ("A_broad_only", X_b_tr, X_b_va),
            ("B_broad_plus_rich", X_r_tr, X_r_va),
        ):
            arms[arm] = _fit_arm_v2(
                X=Xtr,
                y=y_tr,
                elig=elig_tr,
                meta=train["meta"],
                X_val=Xva,
                y_val=y_va,
                elig_val=elig_va,
                kind=kind,
                selection_folds=selection_folds,
                out_dir=ensure_dir(out_dir / spec.head_id / arm),
                tag=f"rich_ablation {arm} {spec.head_id}",
                seed=seed,
                smoke=smoke,
            )

        uplift: dict[str, Any] = {}
        a_m = arms["A_broad_only"].get("metrics") or {}
        b_m = arms["B_broad_plus_rich"].get("metrics") or {}
        for k in ("pr_auc", "roc_auc", "mae", "rmse", "spearman", "multiclass_logloss"):
            if a_m.get(k) is not None and b_m.get(k) is not None:
                uplift[k] = float(b_m[k] - a_m[k])
        if "upper_first" in a_m and "upper_first" in b_m:
            uplift["upper_first_pr_auc"] = float(
                (b_m["upper_first"] or {}).get("pr_auc", 0)
                - (a_m["upper_first"] or {}).get("pr_auc", 0)
            )

        payload["heads"][spec.head_id] = {
            "kind": kind,
            "train_n": int(tr.sum()),
            "val_n": int(va.sum()),
            "counts_train": counts_tr,
            "counts_val": counts_va,
            "arms": arms,
            "uplift_B_minus_A": uplift,
            "eval_only_proof": eval_only_proof,
        }

    write_json(out_dir / "BASELINE_RICH_ABLATION_V2.json", payload)
    return payload
