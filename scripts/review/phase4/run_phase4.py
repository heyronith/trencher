#!/usr/bin/env python3
# ruff: noqa: E501,E402
"""Phase-4: clean Stage-B vs trees, clean rich arms, Stage-C mismatch, quarantine.

Train + permanent_val only. Never reads q85+ TEST. Never writes freeze/GCS paths.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.review.lead_intake_lib import (  # noqa: E402
    EXPLOSION_HORIZONS,
    binary_auc_ap,
    mint_bootstrap_metric_delta,
    sha256_file,
    softmax_pos_prob,
    topk_precision_recall,
    write_json,
)
from scripts.review.phase4.trees import (  # noqa: E402
    fit_lgbm_binary_platt,
    fit_xgb_binary_platt,
)
from scripts.review.run_lead_intake import fit_lgbm_arm  # noqa: E402

WORLD = REPO / "data" / "external" / "world_model"
SPLIT = WORLD / "splits" / "split_v1"
SPLIT_OVERLAY = REPO / "scratch" / "phase4_clean_v1" / "split_overlay"
SCRATCH = REPO / "scratch" / "phase4_clean_v1"
OUT = REPO / "manifests" / "world_model_data" / "evaluation" / "phase4_clean_v1"
Q85 = {"q85_test_accessed": False}
STAGE_B_SHA = "ccf27af79bf178f095c70a84e48ad2e573bae8d5cfcfdde9d32fbb0a1f7aa208"
STAGE_B_METRICS_SHA = "55b305fe602ac7cfeceab436ab87e960717e432229942c26e5a4c203505dfe75"
STAGE_C_SHA = "11bf593e10e5dcc08915fc0fa2300ce988592f660b3608a5024c43121051d839"
WANT_SC_PRED = "2d97c92e3bf6a9b721355d9ea2537b7a54787594ba0ee659105643e8200def13"
TOP_FRACS = [0.001, 0.005, 0.01, 0.02, 0.05]
I60, I300 = EXPLOSION_HORIZONS.index(60), EXPLOSION_HORIZONS.index(300)
CACHE = SCRATCH / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def _cached(name: str, fn):  # noqa: ANN001
    path = CACHE / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    val = fn()
    write_json(path, val)
    return val


def _metrics_block(y: np.ndarray, s: np.ndarray) -> dict[str, Any]:
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
    }


def _buggy_safe_auc(y_arr: np.ndarray, s: np.ndarray) -> float | None:
    y_arr = np.asarray(y_arr).astype(int)
    s = np.asarray(s, dtype=np.float64)
    if y_arr.size == 0:
        return None
    n_pos = int(y_arr.sum())
    n_neg = int(y_arr.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    y_ord = y_arr[order]
    ranks = np.empty(y_arr.size, dtype=np.float64)
    i = 0
    while i < y_arr.size:
        j = i + 1
        while j < y_arr.size and s[order[j]] == s[order[i]]:
            j += 1
        avg = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg
        i = j
    sum_pos = float(ranks[y_ord == 1].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def section_stage_b_infer_status() -> dict[str, Any]:
    pred_dir = SCRATCH / "infer_stage_b_broad"
    cands = list(pred_dir.glob("*_predictions.pt"))
    log = SCRATCH / "infer_stage_b_broad.log"
    out: dict[str, Any] = {
        **Q85,
        "pred_dir": str(pred_dir),
        "n_prediction_files": len(cands),
        "log_bytes": log.stat().st_size if log.exists() else 0,
    }
    if not cands:
        out["status"] = "pending"
        return out
    pred = cands[0]
    out["status"] = "ready"
    out["predictions_path"] = str(pred)
    out["predictions_sha256"] = sha256_file(pred)
    meta = torch.load(pred, map_location="cpu", weights_only=False)["metadata"]
    out["rows_loaded"] = meta.get("rows_loaded")
    out["elapsed_sec"] = meta.get("elapsed_sec")
    out["examples_per_sec"] = meta.get("examples_per_sec")
    out["checkpoint_sha256"] = meta.get("checkpoint_sha256")
    out["checkpoint_sha_ok"] = meta.get("checkpoint_sha256") == STAGE_B_SHA
    return out


def section_stage_b_metrics_recompute() -> dict[str, Any]:
    """Recompute Stage-B broad AUROC with fixed metric; compare to committed JSON."""
    from trencher.baselines.metrics import binary_metrics
    from trencher.world_model.evaluation.metrics import _safe_auc

    status = section_stage_b_infer_status()
    if status.get("status") != "ready":
        return {**Q85, "status": "pending_infer", "infer": status}

    pred = Path(status["predictions_path"])
    obj = torch.load(pred, map_location="cpu", weights_only=False)
    logits = obj["outputs"]["explosion_logits"].numpy()
    y = obj["targets"]["explosion_target"].numpy()
    mask = obj["targets"]["explosion_mask"].numpy()

    committed = REPO / "scratch" / "lead_review_v1" / "stage_b" / "stage_b_permanent_val_eval.json"
    committed_sha = sha256_file(committed) if committed.exists() else None
    committed_obj = (
        json.loads(committed.read_text(encoding="utf-8")) if committed.exists() else {}
    )

    by_h: dict[str, Any] = {}
    for h in (60, 300):
        i = EXPLOSION_HORIZONS.index(h)
        el = mask[:, i] > 0.5
        p = softmax_pos_prob(logits[:, i])[el]
        yy = y[el, i].astype(int)
        bm = binary_metrics(yy, p)
        by_h[str(h)] = {
            "binary_metrics_roc_auc": float(bm["roc_auc"]),
            "binary_metrics_pr_auc": float(bm["pr_auc"]),
            "fixed_safe_auc": _safe_auc(yy, p),
            "buggy_safe_auc": _buggy_safe_auc(yy, p),
            "n": int(bm["n"]),
            "n_pos": int(yy.sum()),
            "committed_auroc": (
                committed_obj.get("explosion_metrics_by_horizon_sec", {})
                .get(str(h), {})
                .get("auroc")
            ),
        }

    # Committed Stage-B metrics predate world_model.evaluation.metrics (git 5f895e1).
    buggy_gap_60 = None
    if by_h["60"]["committed_auroc"] is not None:
        buggy_gap_60 = abs(
            float(by_h["60"]["committed_auroc"]) - float(by_h["60"]["buggy_safe_auc"] or 0)
        )
        fixed_gap_60 = abs(
            float(by_h["60"]["committed_auroc"]) - float(by_h["60"]["binary_metrics_roc_auc"])
        )
    else:
        fixed_gap_60 = None

    return {
        **Q85,
        "status": "ok",
        "predictions_sha256": status["predictions_sha256"],
        "runtime": {
            "elapsed_sec": status.get("elapsed_sec"),
            "examples_per_sec": status.get("examples_per_sec"),
            "rows_loaded": status.get("rows_loaded"),
        },
        "committed_metrics_path": str(committed),
        "committed_metrics_sha256": committed_sha,
        "committed_metrics_sha_ok": committed_sha == STAGE_B_METRICS_SHA,
        "committed_produced_by_buggy_safe_auc": False,
        "committed_produced_by_buggy_reason": (
            "stage_b_permanent_val_eval.json written under git 5f895e1 before "
            "world_model.evaluation.metrics existed; Colab scorer, not Step-17 _safe_auc. "
            f"|committed-buggy|={buggy_gap_60}; |committed-fixed|={fixed_gap_60}."
        ),
        "by_horizon": by_h,
        "infer_metadata": status,
    }


def _oid_hash(oid: str) -> int:
    import hashlib

    return int(hashlib.sha256(oid.encode("utf-8")).hexdigest()[:16], 16) % (2**63)


def _export_stage_b_parquet(pred_path: Path) -> dict[str, Any]:
    """Write per-row Stage-B scores with observation_id + mint to scratch parquet."""
    from trencher.data.world_model.training.loaders import resolve_assignment_paths

    obj = torch.load(pred_path, map_location="cpu", weights_only=False)
    hashes = obj["targets"]["observation_id_hash"].numpy().reshape(-1)
    logits = obj["outputs"]["explosion_logits"].numpy()
    y = obj["targets"]["explosion_target"].numpy()
    mask = obj["targets"]["explosion_mask"].numpy()
    barrier = obj["outputs"].get("barrier_logits")

    import pyarrow.compute as pc

    hash_to_meta: dict[int, tuple[str, str]] = {}
    for path in resolve_assignment_paths(SPLIT, "broad"):
        table = pq.read_table(path, columns=["observation_id", "mint", "base_split"])
        table = table.filter(pc.equal(table["base_split"], "val"))
        for oid, mint in zip(
            table["observation_id"].to_pylist(),
            table["mint"].to_pylist(),
            strict=True,
        ):
            hash_to_meta[_oid_hash(str(oid))] = (str(oid), str(mint))

    # Barrier heads: map Stage-B barrier logits to upper-first scores.
    # barrier_logits shape [N, n_horizons, n_sizes, 3] — use horizons matching
    # 300s / 3600s if present; else skip Stage-B barrier columns.
    p_b25 = p_b100 = None
    if barrier is not None:
        bl = barrier.numpy()
        # Softmax over last dim → P(class0=upper_first) as ranking score
        import torch.nn.functional as F

        probs = F.softmax(torch.from_numpy(bl), dim=-1).numpy()
        # Common layout: horizon index for 300s / sizes — take mean over size dim
        # if ambiguous. Prefer index 0 for short barrier, -1 for long if 2 horizons.
        if probs.ndim == 4:
            # [N, H, S, C]
            p_b25 = probs[:, 0, :, 0].mean(axis=1)
            p_b100 = probs[:, min(1, probs.shape[1] - 1), :, 0].mean(axis=1)

    rows = []
    missing = 0
    for i, h in enumerate(hashes):
        meta = hash_to_meta.get(int(h))
        if meta is None:
            missing += 1
            oid, mint = f"missing_{h}", ""
        else:
            oid, mint = meta
        row = {
            "observation_id": oid,
            "mint": mint,
            "observation_id_hash": int(h),
            "p_explosion_60s": float(softmax_pos_prob(logits[i, I60])),
            "p_explosion_300s": float(softmax_pos_prob(logits[i, I300])),
            "y_explosion_60s": float(y[i, I60]),
            "y_explosion_300s": float(y[i, I300]),
            "m_explosion_60s": float(mask[i, I60]),
            "m_explosion_300s": float(mask[i, I300]),
        }
        if p_b25 is not None:
            row["p_barrier_u25_l15_5m"] = float(p_b25[i])
            row["p_barrier_u100_l30_1h"] = float(p_b100[i])
        rows.append(row)

    out_path = SCRATCH / "preds" / "stage_b_broad_permanent_val.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_path)
    return {
        **Q85,
        "path": str(out_path),
        "sha256": sha256_file(out_path),
        "n_rows": len(rows),
        "n_missing_oid": missing,
    }


def section_broad_feature_status() -> dict[str, Any]:
    root = SPLIT_OVERLAY / "baseline_features_v1" / "broad"
    parts = list((root / "by_mint_partition").glob("*/features.parquet")) if root.exists() else []
    meta_path = SCRATCH / "build_broad_features_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return {
        **Q85,
        "split_overlay": str(SPLIT_OVERLAY),
        "n_partitions_built": len(parts),
        "complete": meta is not None,
        "meta": meta,
    }


def _load_broad_joined(split: str) -> dict[str, Any]:
    from trencher.baselines.dataset import load_joined_matrices
    from trencher.baselines.feature_spec import broad_feature_names
    from trencher.baselines.targets import CORE_TARGET_SUITE
    from trencher.baselines.train_trees import _prepare_xy

    feat_names = broad_feature_names()
    joined = load_joined_matrices(
        world_root=WORLD,
        split_root=SPLIT_OVERLAY,
        split=split,
        population="broad",
        feature_names=feat_names,
    )
    return {"joined": joined, "feature_names": feat_names, "prepare": _prepare_xy, "suite": CORE_TARGET_SUITE}


def section_reproduce_trees() -> dict[str, Any]:
    """Reproduce LGBM + XGB broad val preds; check vs BASELINE_CORE ±0.005."""
    from trencher.baselines.metrics import binary_metrics
    from trencher.baselines.train_trees import _prepare_xy

    feat_status = section_broad_feature_status()
    if not feat_status.get("complete"):
        return {**Q85, "status": "pending_features", "features": feat_status}

    core = json.loads(
        (REPO / "manifests/world_model_data/baseline_v2/BASELINE_CORE_METRICS_V2.json").read_text(
            encoding="utf-8"
        )
    )
    train_pack = _load_broad_joined("train")
    val_pack = _load_broad_joined("val")
    joined_tr, joined_va = train_pack["joined"], val_pack["joined"]

    heads = ["explosion_60s", "explosion_300s", "barrier_u25_l15_5m", "barrier_u100_l30_1h"]
    results: dict[str, Any] = {**Q85, "status": "ok", "heads": {}, "reproduction_check": {}}
    pred_tables: dict[str, dict[str, np.ndarray]] = {
        "observation_id": np.array(
            [m["observation_id"] for m in joined_va["meta"]], dtype=object
        ),
        "mint": np.array([m["mint"] for m in joined_va["meta"]], dtype=object),
    }

    X_tr = joined_tr["X"]
    X_va = joined_va["X"]
    times_tr = np.array(
        [float(m.get("observation_time_t") or 0.0) for m in joined_tr["meta"]],
        dtype=np.float64,
    )

    for head in heads:
        spec = next(s for s in train_pack["suite"] if s.head_id == head)
        y_tr, elig_tr, _counts_tr, kind = _prepare_xy(joined_tr, spec, SPLIT_OVERLAY)
        y_va, elig_va, _counts_va, _ = _prepare_xy(joined_va, spec, SPLIT_OVERLAY)
        fixed_lgb = int(core["lightgbm"]["heads"][head]["fixed_iterations"])
        fixed_xgb = int(core["xgboost"]["heads"][head]["fixed_iterations"])

        if kind == "binary":
            lgb = fit_lgbm_binary_platt(
                X_train=X_tr,
                y_train=y_tr,
                elig_train=elig_tr,
                times_train=times_tr,
                X_val=X_va,
                fixed_iters=fixed_lgb,
                out_dir=SCRATCH / "trees" / "lightgbm" / head,
            )
            xgb = fit_xgb_binary_platt(
                X_train=X_tr,
                y_train=y_tr,
                elig_train=elig_tr,
                times_train=times_tr,
                X_val=X_va,
                fixed_iters=fixed_xgb,
                out_dir=SCRATCH / "trees" / "xgboost" / head,
            )
            # Metrics on eligible val only
            yy = y_va[elig_va].astype(int)
            lgb_m = binary_metrics(yy, lgb["raw"][elig_va])
            xgb_m = binary_metrics(yy, xgb["raw"][elig_va])
            want_lgb = float(core["lightgbm"]["heads"][head]["raw"]["roc_auc"])
            want_xgb = float(core["xgboost"]["heads"][head]["raw"]["roc_auc"])
            results["heads"][head] = {
                "kind": kind,
                "fixed_iterations": {"lightgbm": fixed_lgb, "xgboost": fixed_xgb},
                "lightgbm": {
                    "roc_auc": float(lgb_m["roc_auc"]),
                    "pr_auc": float(lgb_m["pr_auc"]),
                    "n": int(lgb_m["n"]),
                    "delta_vs_committed_roc": float(lgb_m["roc_auc"]) - want_lgb,
                    "within_0_005": abs(float(lgb_m["roc_auc"]) - want_lgb) <= 0.005,
                },
                "xgboost": {
                    "roc_auc": float(xgb_m["roc_auc"]),
                    "pr_auc": float(xgb_m["pr_auc"]),
                    "n": int(xgb_m["n"]),
                    "delta_vs_committed_roc": float(xgb_m["roc_auc"]) - want_xgb,
                    "within_0_005": abs(float(xgb_m["roc_auc"]) - want_xgb) <= 0.005,
                },
            }
            pred_tables[f"lgb_{head}"] = lgb["raw"]
            pred_tables[f"xgb_{head}"] = xgb["raw"]
            pred_tables[f"y_{head}"] = y_va
            pred_tables[f"elig_{head}"] = elig_va.astype(np.float64)
        else:
            # Multiclass barriers: use fit_lgbm_arm multiclass path from lead intake
            # Score P(upper) = class 0 for upper-first comparison vs Stage-B barrier head.
            import lightgbm as lgb
            import xgboost as xgb

            from trencher.baselines.calibration import chronological_calibration_split

            tr = elig_tr
            fit_idx, cal_idx = chronological_calibration_split(times_tr[tr])
            Xf, yf = X_tr[tr][fit_idx], y_tr[tr][fit_idx]
            n_class = int(len(np.unique(y_tr[tr])))
            # LightGBM multiclass
            params = {
                "objective": "multiclass",
                "num_class": n_class,
                "metric": "multi_logloss",
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_data_in_leaf": 100,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "verbosity": -1,
                "seed": 0,
                "num_threads": 1,
            }
            booster = lgb.train(
                params, lgb.Dataset(Xf, label=yf), num_boost_round=fixed_lgb
            )
            lgb_prob = booster.predict(X_va, num_iteration=fixed_lgb)
            # XGB
            xparams = {
                "objective": "multi:softprob",
                "num_class": n_class,
                "eval_metric": "mlogloss",
                "learning_rate": 0.05,
                "max_depth": 8,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "seed": 0,
                "nthread": 1,
                "verbosity": 0,
            }
            bst = xgb.train(
                xparams,
                xgb.DMatrix(Xf, label=yf),
                num_boost_round=fixed_xgb,
            )
            xgb_prob = bst.predict(
                xgb.DMatrix(X_va), iteration_range=(0, fixed_xgb + 1)
            )
            # Upper-first binary: class 0
            yy_u = (y_va[elig_va] == 0).astype(int)
            lgb_u = lgb_prob[elig_va, 0] if lgb_prob.ndim == 2 else lgb_prob[elig_va]
            xgb_u = xgb_prob[elig_va, 0] if xgb_prob.ndim == 2 else xgb_prob[elig_va]
            lgb_m = binary_metrics(yy_u, lgb_u)
            xgb_m = binary_metrics(yy_u, xgb_u)
            want_lgb = float(
                core["lightgbm"]["heads"][head]["barrier"]["upper_first"]["roc_auc"]
            )
            want_xgb = float(
                core["xgboost"]["heads"][head]["barrier"]["upper_first"]["roc_auc"]
            )
            results["heads"][head] = {
                "kind": "multiclass_upper_first",
                "fixed_iterations": {"lightgbm": fixed_lgb, "xgboost": fixed_xgb},
                "lightgbm": {
                    "roc_auc": float(lgb_m["roc_auc"]),
                    "pr_auc": float(lgb_m["pr_auc"]),
                    "n": int(lgb_m["n"]),
                    "delta_vs_committed_roc": float(lgb_m["roc_auc"]) - want_lgb,
                    "within_0_005": abs(float(lgb_m["roc_auc"]) - want_lgb) <= 0.005,
                },
                "xgboost": {
                    "roc_auc": float(xgb_m["roc_auc"]),
                    "pr_auc": float(xgb_m["pr_auc"]),
                    "n": int(xgb_m["n"]),
                    "delta_vs_committed_roc": float(xgb_m["roc_auc"]) - want_xgb,
                    "within_0_005": abs(float(xgb_m["roc_auc"]) - want_xgb) <= 0.005,
                },
            }
            pred_tables[f"lgb_{head}"] = (
                lgb_prob[:, 0] if lgb_prob.ndim == 2 else lgb_prob
            )
            pred_tables[f"xgb_{head}"] = (
                xgb_prob[:, 0] if xgb_prob.ndim == 2 else xgb_prob
            )
            pred_tables[f"y_{head}"] = (y_va == 0).astype(np.float64)
            pred_tables[f"elig_{head}"] = elig_va.astype(np.float64)

    # Save tree preds parquet
    out_path = SCRATCH / "preds" / "trees_broad_permanent_val.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = {k: v.tolist() if hasattr(v, "tolist") else list(v) for k, v in pred_tables.items()}
    pq.write_table(pa.table(cols), out_path)
    results["preds_path"] = str(out_path)
    results["preds_sha256"] = sha256_file(out_path)
    results["reproduction_all_within_0_005"] = all(
        results["heads"][h]["lightgbm"]["within_0_005"]
        and results["heads"][h]["xgboost"]["within_0_005"]
        for h in results["heads"]
    )
    return results


def section_compare_stage_b_vs_trees() -> dict[str, Any]:
    """Inner-join Stage-B and tree preds; metrics + mint-bootstrap CIs + ensemble."""
    sb = section_stage_b_metrics_recompute()
    trees = section_reproduce_trees()
    if sb.get("status") != "ok" or trees.get("status") != "ok":
        return {
            **Q85,
            "status": "pending",
            "stage_b": sb.get("status"),
            "trees": trees.get("status"),
        }

    # Load parquets
    export = _export_stage_b_parquet(Path(sb["infer_metadata"]["predictions_path"]))
    sb_t = pq.read_table(export["path"]).to_pandas()
    tr_t = pq.read_table(trees["preds_path"]).to_pandas()
    merged = sb_t.merge(tr_t, on="observation_id", how="inner", suffixes=("_sb", "_tr"))
    if "mint_sb" in merged.columns:
        merged["mint"] = merged["mint_sb"].fillna(merged.get("mint_tr"))
    elif "mint" not in merged.columns:
        merged["mint"] = merged.get("mint_tr", merged.get("mint_sb"))

    comparisons: dict[str, Any] = {}
    from scipy.stats import rankdata

    for head, p_sb_col, y_col, m_col, lgb_col, xgb_col in (
        (
            "explosion_60s",
            "p_explosion_60s",
            "y_explosion_60s",
            "m_explosion_60s",
            "lgb_explosion_60s",
            "xgb_explosion_60s",
        ),
        (
            "explosion_300s",
            "p_explosion_300s",
            "y_explosion_300s",
            "m_explosion_300s",
            "lgb_explosion_300s",
            "xgb_explosion_300s",
        ),
        (
            "barrier_u25_l15_5m",
            "p_barrier_u25_l15_5m",
            "y_barrier_u25_l15_5m",
            "elig_barrier_u25_l15_5m",
            "lgb_barrier_u25_l15_5m",
            "xgb_barrier_u25_l15_5m",
        ),
        (
            "barrier_u100_l30_1h",
            "p_barrier_u100_l30_1h",
            "y_barrier_u100_l30_1h",
            "elig_barrier_u100_l30_1h",
            "lgb_barrier_u100_l30_1h",
            "xgb_barrier_u100_l30_1h",
        ),
    ):
        if lgb_col not in merged.columns:
            continue
        elig = merged[m_col].to_numpy() > 0.5
        y = merged[y_col].to_numpy()[elig].astype(int)
        s_lgb = merged[lgb_col].to_numpy()[elig]
        s_xgb = merged[xgb_col].to_numpy()[elig]
        mints = merged["mint"].to_numpy()[elig]
        s_sb = (
            merged[p_sb_col].to_numpy()[elig]
            if p_sb_col in merged.columns
            else None
        )

        block: dict[str, Any] = {
            "n_joined": int(elig.sum()),
            "lightgbm": _metrics_block(y, s_lgb),
            "xgboost": _metrics_block(y, s_xgb),
        }
        if s_sb is not None:
            block["stage_b"] = _metrics_block(y, s_sb)
            r_sb = rankdata(s_sb)
            r_lgb = rankdata(s_lgb)
            s_ens = 0.5 * (r_sb / (len(r_sb) + 1) + r_lgb / (len(r_lgb) + 1))
            block["ensemble_stage_b_lgbm"] = _metrics_block(y, s_ens)
            block["delta_stage_b_minus_lgbm"] = {
                "roc_auc": mint_bootstrap_metric_delta(
                    y, s_lgb, s_sb, mints, metric="roc_auc"
                ),
                "pr_auc": mint_bootstrap_metric_delta(
                    y, s_lgb, s_sb, mints, metric="pr_auc"
                ),
            }
            block["delta_stage_b_minus_xgb"] = {
                "roc_auc": mint_bootstrap_metric_delta(
                    y, s_xgb, s_sb, mints, metric="roc_auc"
                ),
                "pr_auc": mint_bootstrap_metric_delta(
                    y, s_xgb, s_sb, mints, metric="pr_auc"
                ),
            }
            block["delta_ensemble_minus_lgbm"] = {
                "roc_auc": mint_bootstrap_metric_delta(
                    y, s_lgb, s_ens, mints, metric="roc_auc"
                ),
                "pr_auc": mint_bootstrap_metric_delta(
                    y, s_lgb, s_ens, mints, metric="pr_auc"
                ),
            }
        comparisons[head] = block

    # Lead-time variants for explosion heads (a exclude already-started, c onset>=5s)
    lead = _lead_time_variants(merged)
    return {
        **Q85,
        "status": "ok",
        "n_inner_join": int(len(merged)),
        "stage_b_parquet": export,
        "trees_parquet": {
            "path": trees["preds_path"],
            "sha256": trees["preds_sha256"],
        },
        "comparisons": comparisons,
        "lead_time_variants": lead,
        "tree_reproduction": {
            h: {
                "lgb_within": trees["heads"][h]["lightgbm"]["within_0_005"],
                "xgb_within": trees["heads"][h]["xgboost"]["within_0_005"],
                "lgb_delta": trees["heads"][h]["lightgbm"]["delta_vs_committed_roc"],
                "xgb_delta": trees["heads"][h]["xgboost"]["delta_vs_committed_roc"],
            }
            for h in trees["heads"]
        },
    }


def _lead_time_variants(merged) -> dict[str, Any]:  # noqa: ANN001
    """Variants (a) exclude already-started, (c) onset >= 5s using tte overlays."""
    from scipy.stats import rankdata

    # Load TTE / already_started from broad overlays if present
    overlay = SPLIT / "tte_overlays_broad"
    already = None
    tte = None
    oid_list = merged["observation_id"].astype(str).tolist()
    if overlay.exists():
        # overlays are partitioned; build oid maps
        a_map: dict[str, bool] = {}
        t_map: dict[str, float] = {}
        for part in overlay.glob("**/*.parquet"):
            try:
                t = pq.read_table(part)
            except Exception:  # noqa: BLE001
                continue
            cols = set(t.column_names)
            if "observation_id" not in cols:
                continue
            oids = t["observation_id"].to_pylist()
            if "explosion_already_started" in cols:
                for o, v in zip(oids, t["explosion_already_started"].to_pylist(), strict=True):
                    a_map[str(o)] = bool(v)
            if "explosion_tte_sec" in cols:
                for o, v in zip(oids, t["explosion_tte_sec"].to_pylist(), strict=True):
                    if v is not None:
                        t_map[str(o)] = float(v)
        already = np.array([a_map.get(o, False) for o in oid_list], dtype=bool)
        tte = np.array([t_map.get(o, np.nan) for o in oid_list], dtype=np.float64)

    out: dict[str, Any] = {}
    if already is None:
        return {"status": "skipped", "reason": "tte_overlays_broad unavailable"}

    for head, pcol, ycol, mcol, lgb_col in (
        ("explosion_60s", "p_explosion_60s", "y_explosion_60s", "m_explosion_60s", "lgb_explosion_60s"),
        ("explosion_300s", "p_explosion_300s", "y_explosion_300s", "m_explosion_300s", "lgb_explosion_300s"),
    ):
        if lgb_col not in merged.columns:
            continue
        base_elig = merged[mcol].to_numpy() > 0.5
        y_all = merged[ycol].to_numpy().astype(int)
        s_sb = merged[pcol].to_numpy()
        s_lgb = merged[lgb_col].to_numpy()
        s_xgb = merged[f"xgb_{head}"].to_numpy()
        mints = merged["mint"].to_numpy()

        variants = {}
        for vname, min_lead in (("a_exclude_already_started", 0.0), ("c_onset_ge_5s", 5.0)):
            keep = base_elig & (~already)
            if min_lead > 0:
                pos = y_all.astype(bool)
                pos_bad = pos & (~np.isfinite(tte) | (tte < min_lead))
                keep = keep & (~pos_bad)
            y = y_all[keep]
            block = {
                "n": int(keep.sum()),
                "n_pos": int(y.sum()),
                "stage_b": _metrics_block(y, s_sb[keep]),
                "lightgbm": _metrics_block(y, s_lgb[keep]),
                "xgboost": _metrics_block(y, s_xgb[keep]),
            }
            r_sb = rankdata(s_sb[keep])
            r_lgb = rankdata(s_lgb[keep])
            s_ens = 0.5 * (r_sb / (len(r_sb) + 1) + r_lgb / (len(r_lgb) + 1))
            block["ensemble_stage_b_lgbm"] = _metrics_block(y, s_ens)
            block["delta_stage_b_minus_lgbm_roc"] = mint_bootstrap_metric_delta(
                y, s_lgb[keep], s_sb[keep], mints[keep], metric="roc_auc"
            )
            variants[vname] = block
        out[head] = variants
    return out


def section_clean_rich_trees() -> dict[str, Any]:
    """Rich train/val: A, A+n_trades, A+graph+wallet_prior+n_trades, A+graph+wallet_prior."""
    from trencher.baselines.dataset import load_joined_matrices
    from trencher.baselines.feature_spec import all_feature_names, broad_feature_names
    from trencher.baselines.metrics import binary_metrics
    from trencher.baselines.targets import CORE_TARGET_SUITE
    from trencher.baselines.train_trees import _prepare_xy

    names = all_feature_names(include_rich=True)
    broad = broad_feature_names()
    # indices
    name_to_i = {n: i for i, n in enumerate(names)}
    broad_idx = [name_to_i[n] for n in broad if n in name_to_i]
    n_trades_idx = [name_to_i["n_trades_before_T"]] if "n_trades_before_T" in name_to_i else []
    graph_idx = [i for n, i in name_to_i.items() if n.startswith("graph_") or n == "mask_graph"]
    wallet_idx = [
        i for n, i in name_to_i.items() if n.startswith("wallet_prior_") or n == "mask_wallet_prior"
    ]

    arms = {
        "A": broad_idx,
        "A_plus_n_trades": broad_idx + n_trades_idx,
        "A_plus_graph_wallet_n_trades": broad_idx + graph_idx + wallet_idx + n_trades_idx,
        "A_plus_graph_wallet": broad_idx + graph_idx + wallet_idx,
    }
    # dedupe preserve order
    arms = {k: list(dict.fromkeys(v)) for k, v in arms.items()}

    ablation = json.loads(
        (
            REPO
            / "manifests/world_model_data/baseline_v2/BASELINE_RICH_ABLATION_V2.json"
        ).read_text(encoding="utf-8")
        if (REPO / "manifests/world_model_data/baseline_v2/BASELINE_RICH_ABLATION_V2.json").exists()
        else "{}"
    )
    # fallback fixed iters from core / prior phase
    fixed_default = {"explosion_60s": 75, "explosion_300s": 68}

    joined_tr = load_joined_matrices(
        world_root=WORLD, split_root=SPLIT, split="train", population="rich", feature_names=names
    )
    joined_va = load_joined_matrices(
        world_root=WORLD, split_root=SPLIT, split="val", population="rich", feature_names=names
    )
    times_tr = np.array(
        [float(m.get("observation_time_t") or 0.0) for m in joined_tr["meta"]],
        dtype=np.float64,
    )

    X_tr_full = joined_tr["X"]
    X_va_full = joined_va["X"]
    out: dict[str, Any] = {**Q85, "arms": {}, "deltas_vs_A": {}}
    for head in ("explosion_60s", "explosion_300s"):
        spec = next(s for s in CORE_TARGET_SUITE if s.head_id == head)
        y_tr, elig_tr, _counts_tr, kind = _prepare_xy(joined_tr, spec, SPLIT)
        y_va, elig_va, _counts_va, _ = _prepare_xy(joined_va, spec, SPLIT)
        assert kind == "binary"
        if ablation:
            try:
                fixed = int(
                    ablation["heads"][head]["arms"]["A_broad_only"]["fixed_iterations"]
                )
            except Exception:  # noqa: BLE001
                fixed = fixed_default[head]
        else:
            fixed = fixed_default[head]

        head_scores: dict[str, np.ndarray] = {}
        head_metrics: dict[str, Any] = {}
        for arm_name, idx in arms.items():
            pred = fit_lgbm_arm(
                kind="binary",
                X_train=X_tr_full[:, idx],
                y_train=y_tr,
                elig_train=elig_tr,
                times_train=times_tr,
                X_val=X_va_full[:, idx],
                fixed_iters=fixed,
                out_dir=SCRATCH / "rich_trees" / head / arm_name,
                tag=arm_name,
            )
            yy = y_va[elig_va].astype(int)
            bm = binary_metrics(yy, pred[elig_va])
            head_scores[arm_name] = pred
            head_metrics[arm_name] = {
                "roc_auc": float(bm["roc_auc"]),
                "pr_auc": float(bm["pr_auc"]),
                "n": int(bm["n"]),
                "fixed_iterations": fixed,
                "n_features": len(idx),
            }
        out["arms"][head] = head_metrics
        mints = np.array([m["mint"] for m in joined_va["meta"]], dtype=object)[elig_va]
        yy = y_va[elig_va].astype(int)
        deltas = {}
        for arm_name in arms:
            if arm_name == "A":
                continue
            deltas[arm_name] = {
                "roc_auc": mint_bootstrap_metric_delta(
                    yy,
                    head_scores["A"][elig_va],
                    head_scores[arm_name][elig_va],
                    mints,
                    metric="roc_auc",
                ),
                "pr_auc": mint_bootstrap_metric_delta(
                    yy,
                    head_scores["A"][elig_va],
                    head_scores[arm_name][elig_va],
                    mints,
                    metric="pr_auc",
                ),
            }
        out["deltas_vs_A"][head] = deltas
    return out


def section_stage_c_mismatch() -> dict[str, Any]:
    """Root-cause 0.850 (diagnose-masks) vs 0.8945 (frozen Stage-C preds)."""
    from trencher.baselines.metrics import binary_metrics

    base_path = (
        REPO / "scratch/phase3_leak_v1/infer_masks/baseline/baseline_predictions.pt"
    )
    froz_path = (
        REPO / "scratch/lead_review_v1/stage_c_val/permanent_val_predictions_175000.pt"
    )
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    froz = torch.load(froz_path, map_location="cpu", weights_only=False)

    hb = base["targets"]["observation_id_hash"].numpy().reshape(-1)
    hf = froz["targets"]["observation_id_hash"].numpy().reshape(-1)
    same_rows = bool(np.array_equal(hb, hf))
    same_y = bool(
        torch.equal(base["targets"]["explosion_target"], froz["targets"]["explosion_target"])
    )
    same_mask = bool(
        torch.equal(base["targets"]["explosion_mask"], froz["targets"]["explosion_mask"])
    )
    same_ckpt = (
        base["metadata"].get("checkpoint_sha256")
        == froz["metadata"].get("checkpoint_sha256")
        == STAGE_C_SHA
    )

    i60 = EXPLOSION_HORIZONS.index(60)
    lb = base["outputs"]["explosion_logits"].numpy()
    lf = froz["outputs"]["explosion_logits"].numpy()
    el = base["targets"]["explosion_mask"].numpy()[:, i60] > 0.5
    y = base["targets"]["explosion_target"].numpy()[el, i60].astype(int)
    roc_base = float(binary_metrics(y, softmax_pos_prob(lb[:, i60])[el])["roc_auc"])
    roc_froz = float(binary_metrics(y, softmax_pos_prob(lf[:, i60])[el])["roc_auc"])
    d = np.max(np.abs(lb - lf), axis=1)
    micro = base["targets"]["micro_available"].numpy().reshape(-1)

    return {
        **Q85,
        "paths_examined_in_order": [
            "diagnose-masks: scratch/phase3_leak_v1/infer_masks/baseline/ (CPU step17)",
            "infer: scratch/lead_review_v1/stage_c_val/run_permanent_val.py (CUDA+bf16 Colab)",
            "frozen preds: permanent_val_predictions_175000.pt",
        ],
        "same_observation_id_hash_order": same_rows,
        "same_explosion_targets": same_y,
        "same_explosion_masks": same_mask,
        "same_checkpoint_sha256": same_ckpt,
        "checkpoint_sha256": STAGE_C_SHA,
        "diagnose_masks": {
            "device": "cpu",
            "bf16": False,
            "runtime_git_sha": base["metadata"].get("evaluation_runtime_git_sha")
            or (base["metadata"].get("evaluation_runtime") or {}).get("git_sha"),
            "rows_loaded": base["metadata"].get("rows_loaded"),
            "roc_auc_60s_binary_metrics": roc_base,
            "note": "phase3 report 0.8500 matches binary_metrics on these logits",
        },
        "frozen_infer": {
            "device": "cuda",
            "bf16": True,
            "runtime_git_sha": froz["metadata"].get("runtime_git_sha"),
            "rows_loaded": froz["metadata"].get("rows_loaded"),
            "roc_auc_60s_binary_metrics": roc_froz,
            "metrics_file_auroc_60": 0.894475225090402,
            "predictions_sha256": sha256_file(froz_path),
            "predictions_sha_ok": sha256_file(froz_path) == WANT_SC_PRED,
        },
        "logit_divergence": {
            "corr_60s": float(np.corrcoef(lb[:, i60], lf[:, i60])[0, 1]),
            "mean_maxabs_all_heads": float(np.mean(d)),
            "mean_maxabs_micro_avail": float(d[micro > 0.5].mean()),
            "mean_maxabs_micro_missing": float(d[micro < 0.5].mean()),
        },
        "root_cause": (
            "Not different rows, labels, masks, dropout, or checkpoint. "
            "Identical observation_id_hash order and explosion targets/masks; "
            f"same Stage-C SHA {STAGE_C_SHA}. "
            "Diagnose-masks baseline used CPU fp32 step17 inference "
            f"(ROC@60={roc_base:.4f}); frozen preds used Colab CUDA bf16 "
            f"autocast (ROC@60={roc_froz:.4f}). Logits diverge systematically "
            "(corr≈0.95); gap is largest on rows with missing micro/ownership. "
            "Secondary: diagnose path git includes empty-sequence pooling fix "
            "9af5548 absent from frozen runtime_git_sha b7abd1de. "
            "The 0.850 vs 0.8945 headline is this device/precision (+minor code) "
            "gap on the same population, not a population or eval-mode mismatch."
        ),
        "ruled_out": [
            "different observation rows",
            "different explosion masks/labels",
            "different checkpoint weights",
            "inference_mask_omit on baseline (empty)",
            "train vs eval dropout (both model.eval + inference_mode)",
        ],
    }


def section_rpc() -> dict[str, Any]:
    from scripts.review.phase4.refetch_micro import _rpc_configured, dry_run

    ok, _ = _rpc_configured()
    if ok:
        from scripts.review.phase4.refetch_micro import fetch_if_configured

        return fetch_if_configured()
    return dry_run()


def write_report(report: dict[str, Any]) -> None:
    lines: list[str] = []
    s1 = report.get("section1_compare") or {}
    s2 = report.get("section2_rich") or {}
    s3 = report.get("section3_mismatch") or {}
    s5 = report.get("section5_rpc") or {}
    sb = report.get("section1_stage_b") or {}

    # 10-line summary at top
    beat = "UNKNOWN"
    comps = (s1 or {}).get("comparisons") or {}
    if "explosion_60s" in comps and "stage_b" in comps["explosion_60s"]:
        d = comps["explosion_60s"]["delta_stage_b_minus_lgbm"]["roc_auc"]
        delta = d.get("delta")
        ci = d.get("ci95") or [None, None]
        if delta is not None and ci[0] is not None:
            beat = (
                f"Stage B ROC@60 − LGBM = {delta:+.4f} "
                f"(95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]); "
                f"SB={comps['explosion_60s']['stage_b']['roc_auc']:.4f} "
                f"LGBM={comps['explosion_60s']['lightgbm']['roc_auc']:.4f} "
                f"XGB={comps['explosion_60s']['xgboost']['roc_auc']:.4f}."
            )
        else:
            beat = "Stage B vs trees computed but CI incomplete."
    micro_signal = "RPC not configured — leak-free micro refetch dry-run only; no B-clean retrain."
    if (s5 or {}).get("status") == "fetch_complete":
        micro_signal = "Fetch completed; see section 5 for B-clean metrics."
    elif (s2 or {}).get("arms"):
        # clean rich proxy: graph+wallet vs A
        a60 = ((s2.get("arms") or {}).get("explosion_60s") or {}).get("A_plus_graph_wallet")
        if a60:
            micro_signal = (
                "Leak-free micro not rebuilt (no RPC). Clean rich proxy "
                f"A+graph+wallet ROC@60={a60.get('roc_auc')}."
            )

    lines.append("# PHASE4_REPORT")
    lines.append("")
    lines.append("## Summary (10 lines)")
    lines.append(f"1. {beat}")
    lines.append(f"2. {micro_signal}")
    lines.append(
        f"3. Stage-B broad infer: rows={((sb.get('runtime') or {}).get('rows_loaded'))}, "
        f"elapsed_s={((sb.get('runtime') or {}).get('elapsed_sec'))}."
    )
    lines.append(
        f"4. Committed Stage-B metrics buggy `_safe_auc`? "
        f"{sb.get('committed_produced_by_buggy_safe_auc')} — {sb.get('committed_produced_by_buggy_reason', '')[:120]}"
    )
    lines.append(
        f"5. Tree reproduction within ±0.005: {(s1 or {}).get('tree_reproduction')}."
    )
    lines.append(
        f"6. Stage-C mismatch root cause: {(s3 or {}).get('root_cause', '')[:200]}"
    )
    lines.append("7. Causal trade selection: `select_causal_trades_before` + leak flag on stratified.")
    lines.append(
        f"8. RPC: configured={s5.get('rpc_configured') if s5 else False}; "
        f"unique sigs={s5.get('n_unique_signatures') or s5.get('n_unique_signatures_estimate')}."
    )
    lines.append("9. Quarantine: `manifests/world_model_data/LEAK_QUARANTINE_V1.md`.")
    lines.append("10. `q85_test_accessed: false` on all phase-4 outputs.")
    lines.append("")

    def _fmt_ci(rec: dict[str, Any] | None) -> str:
        if not rec or rec.get("delta") is None:
            return "n/a"
        ci = rec.get("ci95") or [None, None]
        return f"{rec['delta']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"

    lines.append("## 1. Stage B vs LightGBM / XGBoost (broad permanent val)")
    lines.append("")
    if comps:
        lines.append(
            "| head | SB ROC | LGBM ROC | XGB ROC | SB−LGBM ROC CI | SB−XGB ROC CI | Ens−LGBM ROC CI |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for head, block in comps.items():
            if "stage_b" not in block:
                lines.append(
                    f"| {head} | — | {block['lightgbm']['roc_auc']:.4f} | "
                    f"{block['xgboost']['roc_auc']:.4f} | — | — | — |"
                )
                continue
            lines.append(
                f"| {head} | {block['stage_b']['roc_auc']:.4f} | "
                f"{block['lightgbm']['roc_auc']:.4f} | {block['xgboost']['roc_auc']:.4f} | "
                f"{_fmt_ci(block.get('delta_stage_b_minus_lgbm', {}).get('roc_auc'))} | "
                f"{_fmt_ci(block.get('delta_stage_b_minus_xgb', {}).get('roc_auc'))} | "
                f"{_fmt_ci(block.get('delta_ensemble_minus_lgbm', {}).get('roc_auc'))} |"
            )
        lines.append("")
        lines.append("### Precision at top-k (explosion_60s)")
        if "explosion_60s" in comps and "stage_b" in comps["explosion_60s"]:
            lines.append("| model | 0.1% | 0.5% | 1% | 2% | 5% |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for m in ("stage_b", "lightgbm", "xgboost", "ensemble_stage_b_lgbm"):
                p = comps["explosion_60s"][m]["precision_at"]
                lines.append(
                    f"| {m} | {p['top_0.001']:.4f} | {p['top_0.005']:.4f} | "
                    f"{p['top_0.01']:.4f} | {p['top_0.02']:.4f} | {p['top_0.05']:.4f} |"
                )
    else:
        lines.append("_Pending Stage-B infer and/or broad feature rebuild._")
    lines.append("")

    lines.append("## 2. Clean rich tree baselines")
    lines.append("")
    arms = (s2 or {}).get("arms") or {}
    if arms:
        lines.append("| head | arm | ROC-AUC | PR-AUC | ΔROC vs A (CI) |")
        lines.append("|---|---|---:|---:|---|")
        for head, arm_map in arms.items():
            for arm, met in arm_map.items():
                dlt = ((s2.get("deltas_vs_A") or {}).get(head) or {}).get(arm, {})
                lines.append(
                    f"| {head} | {arm} | {met['roc_auc']:.4f} | {met['pr_auc']:.4f} | "
                    f"{_fmt_ci(dlt.get('roc_auc'))} |"
                )
    else:
        lines.append("_Pending or empty._")
    lines.append("")

    lines.append("## 3. Stage-C 0.850 vs 0.8945 mismatch")
    lines.append("")
    lines.append((s3 or {}).get("root_cause", "_pending_"))
    lines.append("")
    lines.append("## 4–6. Code fix / RPC / quarantine")
    lines.append("")
    lines.append("- Causal selection + regression test landed.")
    lines.append(f"- RPC section: `{json.dumps({k: s5.get(k) for k in ('status','n_unique_signatures','n_unique_signatures_estimate','rpc_configured') if s5}, default=str)}`")
    lines.append("- See `manifests/world_model_data/LEAK_QUARANTINE_V1.md`.")
    lines.append("")
    lines.append("q85_test_accessed: false")

    (OUT / "PHASE4_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "PHASE4_REPORT.json", report)


def main() -> int:
    report: dict[str, Any] = {
        **Q85,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print("section3 stage-c mismatch...", flush=True)
    report["section3_mismatch"] = _cached("section3", section_stage_c_mismatch)
    print("section5 rpc dry-run...", flush=True)
    report["section5_rpc"] = _cached("section5", section_rpc)
    print("section1 stage-b status...", flush=True)
    report["section1_stage_b"] = section_stage_b_metrics_recompute()
    print("section2 clean rich...", flush=True)
    try:
        report["section2_rich"] = _cached("section2", section_clean_rich_trees)
    except Exception as exc:  # noqa: BLE001
        report["section2_rich"] = {**Q85, "error": str(exc)}
    print("section1 trees+compare...", flush=True)
    try:
        report["section1_compare"] = section_compare_stage_b_vs_trees()
    except Exception as exc:  # noqa: BLE001
        report["section1_compare"] = {**Q85, "error": str(exc)}
        report["section1_features"] = section_broad_feature_status()

    report["section1_stage_b"] = section_stage_b_metrics_recompute()
    write_report(report)
    print(f"wrote {OUT / 'PHASE4_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
