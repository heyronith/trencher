#!/usr/bin/env python3
"""Lead intake A–G analysis. Train + permanent_val only. Never reads TEST rows."""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.review.lead_intake_lib import (  # noqa: E402
    BOOT_SEED,
    EXPLOSION_HORIZONS,
    N_BOOT,
    OUT,
    SCRATCH,
    SPLIT,
    WANT_METRICS,
    WANT_PRED,
    WANT_STAGE_B,
    WORLD,
    binary_auc_ap,
    mint_bootstrap_metric_delta,
    oid_hash,
    sha256_file,
    softmax_pos_prob,
    topk_precision_recall,
    write_json,
)

Q85_FLAG = {"q85_test_accessed": False}
TOP_FRACS = [0.005, 0.01, 0.02, 0.05, 0.10]


def section_hashes() -> dict[str, Any]:
    pred = SCRATCH / "stage_c_val" / "permanent_val_predictions_175000.pt"
    met = SCRATCH / "stage_c_val" / "permanent_val_metrics_175000.json"
    sb = SCRATCH / "stage_b" / "stage_b_permanent_val_eval.json"
    got_pred, got_met, got_sb = sha256_file(pred), sha256_file(met), sha256_file(sb)
    return {
        **Q85_FLAG,
        "stage_c_predictions": {
            "sha256": got_pred,
            "matches_expected": got_pred == WANT_PRED,
            "expected": WANT_PRED,
        },
        "stage_c_metrics": {
            "sha256": got_met,
            "matches_expected": got_met == WANT_METRICS,
            "expected": WANT_METRICS,
        },
        "stage_b_metrics": {
            "sha256": got_sb,
            "matches_expected": got_sb == WANT_STAGE_B,
            "expected": WANT_STAGE_B,
        },
    }


def fit_lgbm_arm(
    *,
    kind: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    elig_train: np.ndarray,
    times_train: np.ndarray,
    X_val: np.ndarray,
    fixed_iters: int,
    out_dir: Path,
    tag: str,
    seed: int = 0,
) -> np.ndarray:
    """Final-fit with exact fixed_iterations (no ES on permanent_val).

    Uses a minimal LightGBM train path (no progress callbacks) to avoid a
    macOS OpenMP segfault observed with BoostingProgressCallback.
    """
    import lightgbm as lgb

    from trencher.baselines.calibration import chronological_calibration_split
    from trencher.data.world_model.io import ensure_dir

    del tag  # retained for call-site clarity
    tr = elig_train
    X_tr, y_tr = X_train[tr], y_train[tr]
    times = times_train[tr]
    fit_idx, _cal_idx = chronological_calibration_split(times)
    # Match frozen protocol: chronological fit/cal split exists, but final fit
    # uses fixed_iters with no early stopping. Train on fit slice only (same as
    # rich_ablation final_fit which trains on fit_idx with cal as unused ES set
    # when early_stopping_rounds is None — train_lightgbm still passes val but
    # without ES). For exact fixed trees, training on fit_idx is sufficient.
    params: dict[str, Any] = {
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
    if kind == "binary":
        params["objective"] = "binary"
        params["metric"] = "average_precision"
    elif kind == "multiclass":
        params["objective"] = "multiclass"
        params["num_class"] = int(len(np.unique(y_tr[fit_idx])))
        params["metric"] = "multi_logloss"
    else:
        params["objective"] = "regression"
        params["metric"] = "mae"

    dtrain = lgb.Dataset(X_tr[fit_idx], label=y_tr[fit_idx])
    booster = lgb.train(params, dtrain, num_boost_round=int(fixed_iters))
    out_dir = ensure_dir(out_dir / "final_fit")
    model_path = out_dir / "model.txt"
    booster.save_model(str(model_path))
    return np.asarray(booster.predict(X_val, num_iteration=int(fixed_iters)))


def section_a() -> dict[str, Any]:
    from trencher.baselines.dataset import load_joined_matrices
    from trencher.baselines.feature_spec import all_feature_names, broad_feature_names
    from trencher.baselines.targets import CORE_TARGET_SUITE
    from trencher.baselines.train_trees import _prepare_xy

    hashes = section_hashes()
    for key in ("stage_c_predictions", "stage_c_metrics", "stage_b_metrics"):
        if not hashes[key]["matches_expected"]:
            return {"error": f"{key} hash mismatch", "hashes": hashes, **Q85_FLAG}

    pred_obj = torch.load(
        SCRATCH / "stage_c_val" / "permanent_val_predictions_175000.pt",
        map_location="cpu",
        weights_only=False,
    )
    sc_meta = pred_obj["metadata"]
    sc_out = pred_obj["outputs"]
    sc_tgt = pred_obj["targets"]
    sc_oid_hash = sc_tgt["observation_id_hash"].numpy()
    n_sc = int(sc_oid_hash.shape[0])

    rich_names = all_feature_names(include_rich=True)
    broad_names = broad_feature_names()
    train = load_joined_matrices(
        world_root=WORLD,
        split_root=SPLIT,
        split="train",
        feature_names=rich_names,
        population="rich",
        include_rich_features=True,
    )
    val = load_joined_matrices(
        world_root=WORLD,
        split_root=SPLIT,
        split="val",
        feature_names=rich_names,
        population="rich",
        include_rich_features=True,
    )

    val_oids = [str(m["observation_id"]) for m in val["meta"]]
    val_mints = [str(m["mint"]) for m in val["meta"]]
    hash_to_i = {oid_hash(o): i for i, o in enumerate(val_oids)}
    sc_to_val = np.array([hash_to_i.get(int(h), -1) for h in sc_oid_hash], dtype=np.int64)
    matched = sc_to_val >= 0

    ablation = json.loads(
        (REPO / "manifests/world_model_data/baseline_v2/BASELINE_RICH_ABLATION_V2.json").read_text(
            encoding="utf-8"
        )
    )
    X_tr_rich, X_va_rich = train["X"], val["X"]
    X_tr_broad, X_va_broad = X_tr_rich[:, : len(broad_names)], X_va_rich[:, : len(broad_names)]
    train_times = np.array(
        [float(m["observation_time_t"] or 0.0) for m in train["meta"]], dtype=np.float64
    )

    specs = {t.head_id: t for t in CORE_TARGET_SUITE}
    heads_out: dict[str, Any] = {}
    paired_by_oid: dict[str, dict[str, Any]] = {}
    exp_logits = sc_out["explosion_logits"].numpy()
    exp_target = sc_tgt["explosion_target"].numpy()
    exp_mask = sc_tgt["explosion_mask"].numpy()
    h_idx = {h: i for i, h in enumerate(EXPLOSION_HORIZONS)}

    for head_id in ("explosion_60s", "explosion_300s"):
        print(f"  fitting LGBM arms for {head_id}...", flush=True)
        spec = specs[head_id]
        y_tr, elig_tr, _counts_tr, kind = _prepare_xy(train, spec, SPLIT)
        y_va, elig_va, _counts_va, _ = _prepare_xy(val, spec, SPLIT)
        fixed_a = int(ablation["heads"][head_id]["arms"]["A_broad_only"]["fixed_iterations"])
        fixed_b = int(ablation["heads"][head_id]["arms"]["B_broad_plus_rich"]["fixed_iterations"])
        arm_a = ablation["heads"][head_id]["arms"]["A_broad_only"]
        arm_b = ablation["heads"][head_id]["arms"]["B_broad_plus_rich"]
        committed_a = float(arm_a["metrics"]["roc_auc"])
        committed_b = float(arm_b["metrics"]["roc_auc"])

        pred_a_all = fit_lgbm_arm(
            kind=kind,
            X_train=X_tr_broad,
            y_train=y_tr,
            elig_train=elig_tr,
            times_train=train_times,
            X_val=X_va_broad,
            fixed_iters=fixed_a,
            out_dir=SCRATCH / "lgbm" / head_id / "A_broad_only",
            tag=f"A {head_id}",
        )
        pred_b_all = fit_lgbm_arm(
            kind=kind,
            X_train=X_tr_rich,
            y_train=y_tr,
            elig_train=elig_tr,
            times_train=train_times,
            X_val=X_va_rich,
            fixed_iters=fixed_b,
            out_dir=SCRATCH / "lgbm" / head_id / "B_broad_plus_rich",
            tag=f"B {head_id}",
        )
        met_a = binary_auc_ap(y_va[elig_va], pred_a_all[elig_va])
        met_b = binary_auc_ap(y_va[elig_va], pred_b_all[elig_va])

        hi = 60 if head_id.endswith("60s") else 300
        i_h = h_idx[hi]
        sc_score = np.full(val["n"], np.nan)
        sc_elig = np.zeros(val["n"], dtype=bool)
        for j in np.flatnonzero(matched):
            vi = int(sc_to_val[j])
            sc_score[vi] = float(softmax_pos_prob(exp_logits[j, i_h]))
            sc_elig[vi] = bool(exp_mask[j, i_h] > 0.5)
            # optionally check label
            _ = exp_target[j, i_h]

        pair_mask = sc_elig & elig_va & np.isfinite(sc_score)
        y_p = y_va[pair_mask].astype(int)
        s_c = sc_score[pair_mask]
        s_a = pred_a_all[pair_mask]
        s_b = pred_b_all[pair_mask]
        mints_p = np.array(val_mints, dtype=object)[pair_mask]

        heads_out[head_id] = {
            "fixed_iterations": {"A_broad_only": fixed_a, "B_broad_plus_rich": fixed_b},
            "reproduction_eligible_val": {
                "lgbm_a_roc_auc": met_a["roc_auc"],
                "lgbm_b_roc_auc": met_b["roc_auc"],
                "lgbm_a_pr_auc": met_a["pr_auc"],
                "lgbm_b_pr_auc": met_b["pr_auc"],
                "committed_a_roc_auc": committed_a,
                "committed_b_roc_auc": committed_b,
                "a_within_0_005": abs(float(met_a["roc_auc"]) - committed_a) <= 0.005,
                "b_within_0_005": abs(float(met_b["roc_auc"]) - committed_b) <= 0.005,
                "n_elig": int(elig_va.sum()),
            },
            "paired_n": int(pair_mask.sum()),
            "models": {
                "stage_c": {
                    **binary_auc_ap(y_p, s_c),
                    "topk": topk_precision_recall(y_p, s_c, TOP_FRACS),
                },
                "lgbm_a": {
                    **binary_auc_ap(y_p, s_a),
                    "topk": topk_precision_recall(y_p, s_a, TOP_FRACS),
                },
                "lgbm_b": {
                    **binary_auc_ap(y_p, s_b),
                    "topk": topk_precision_recall(y_p, s_b, TOP_FRACS),
                },
            },
            "bootstrap": {
                "stage_c_minus_lgbm_b": {
                    "roc_auc": mint_bootstrap_metric_delta(
                        y_p, s_b, s_c, mints_p, metric="roc_auc"
                    ),
                    "pr_auc": mint_bootstrap_metric_delta(
                        y_p, s_b, s_c, mints_p, metric="pr_auc"
                    ),
                },
                "stage_c_minus_lgbm_a": {
                    "roc_auc": mint_bootstrap_metric_delta(
                        y_p, s_a, s_c, mints_p, metric="roc_auc"
                    ),
                    "pr_auc": mint_bootstrap_metric_delta(
                        y_p, s_a, s_c, mints_p, metric="pr_auc"
                    ),
                },
                "lgbm_b_minus_lgbm_a": {
                    "roc_auc": mint_bootstrap_metric_delta(
                        y_p, s_a, s_b, mints_p, metric="roc_auc"
                    ),
                    "pr_auc": mint_bootstrap_metric_delta(
                        y_p, s_a, s_b, mints_p, metric="pr_auc"
                    ),
                },
            },
        }
        suffix = "60s" if hi == 60 else "300s"
        for vi in np.flatnonzero(pair_mask):
            oid = val_oids[int(vi)]
            row = paired_by_oid.setdefault(
                oid, {"observation_id": oid, "mint": val_mints[int(vi)]}
            )
            row[f"y_explosion_{suffix}"] = int(y_va[int(vi)])
            row[f"stage_c_{suffix}"] = float(sc_score[int(vi)])
            row[f"lgbm_a_{suffix}"] = float(pred_a_all[int(vi)])
            row[f"lgbm_b_{suffix}"] = float(pred_b_all[int(vi)])

    # Barriers upper-first
    barrier_out: dict[str, Any] = {}
    barrier_logits = sc_out["barrier_logits"].numpy()
    barrier_mask = sc_tgt["barrier_mask"].numpy()
    from trencher.world_model.config import load_world_model_config

    cfg = load_world_model_config(REPO / "configs/world_model/world_model_candidate_v2.yaml")
    pair_ids = [f"u{int(u)}_l{int(lo)}" for u, lo in cfg.heads.barrier_pairs]
    h_list = list(cfg.heads.return_horizons)

    for head_id, pair_name, horizon in (
        ("barrier_u25_l15_5m", "u25_l15", 300),
        ("barrier_u100_l30_1h", "u100_l30", 3600),
    ):
        print(f"  fitting LGBM arms for {head_id}...", flush=True)
        if pair_name not in pair_ids or horizon not in h_list:
            barrier_out[head_id] = {
                "error": "index mapping failed",
                "pair_ids": pair_ids,
                "horizons": h_list,
            }
            continue
        pi, hi = pair_ids.index(pair_name), h_list.index(horizon)
        spec = specs[head_id]
        y_tr, elig_tr, _, kind = _prepare_xy(train, spec, SPLIT)
        y_va, elig_va, _, _ = _prepare_xy(val, spec, SPLIT)
        fixed_a = int(ablation["heads"][head_id]["arms"]["A_broad_only"]["fixed_iterations"])
        fixed_b = int(ablation["heads"][head_id]["arms"]["B_broad_plus_rich"]["fixed_iterations"])
        pred_a = fit_lgbm_arm(
            kind=kind,
            X_train=X_tr_broad,
            y_train=y_tr,
            elig_train=elig_tr,
            times_train=train_times,
            X_val=X_va_broad,
            fixed_iters=fixed_a,
            out_dir=SCRATCH / "lgbm" / head_id / "A",
            tag=f"A {head_id}",
        )
        pred_b = fit_lgbm_arm(
            kind=kind,
            X_train=X_tr_rich,
            y_train=y_tr,
            elig_train=elig_tr,
            times_train=train_times,
            X_val=X_va_rich,
            fixed_iters=fixed_b,
            out_dir=SCRATCH / "lgbm" / head_id / "B",
            tag=f"B {head_id}",
        )
        s_a = pred_a[:, 0] if pred_a.ndim == 2 else (pred_a == 0).astype(float)
        s_b = pred_b[:, 0] if pred_b.ndim == 2 else (pred_b == 0).astype(float)
        y_bin = (y_va == 0).astype(int)
        sc_score = np.full(val["n"], np.nan)
        sc_elig = np.zeros(val["n"], dtype=bool)
        for j in np.flatnonzero(matched):
            vi = int(sc_to_val[j])
            logits = barrier_logits[j, pi, hi]
            e = np.exp(logits - logits.max())
            sc_score[vi] = float(e[0] / e.sum())
            sc_elig[vi] = bool(barrier_mask[j, pi, hi] > 0.5)
        pair_mask = sc_elig & elig_va & np.isfinite(sc_score)
        y_p = y_bin[pair_mask]
        mints_p = np.array(val_mints, dtype=object)[pair_mask]
        s_c = sc_score[pair_mask]
        barrier_out[head_id] = {
            "paired_n": int(pair_mask.sum()),
            "models_upper_first": {
                "stage_c": binary_auc_ap(y_p, s_c),
                "lgbm_a": binary_auc_ap(y_p, s_a[pair_mask]),
                "lgbm_b": binary_auc_ap(y_p, s_b[pair_mask]),
            },
            "bootstrap_stage_c_minus_lgbm_b_roc": mint_bootstrap_metric_delta(
                y_p, s_b[pair_mask], s_c, mints_p, metric="roc_auc"
            ),
        }

    sb = json.loads((SCRATCH / "stage_b" / "stage_b_permanent_val_eval.json").read_text())
    core = json.loads(
        (REPO / "manifests/world_model_data/baseline_v2/BASELINE_CORE_METRICS_V2.json").read_text()
    )
    lgbm_heads = core["lightgbm"]["heads"]

    def _core_exp(hid: str) -> dict[str, Any]:
        h = lgbm_heads.get(hid) or {}
        m = h.get("metrics") or h.get("raw") or h
        if isinstance(m, dict) and "raw" in m:
            m = m["raw"]
        return {
            "roc_auc": m.get("roc_auc"),
            "pr_auc": m.get("pr_auc"),
            "n": m.get("n"),
            "prevalence": m.get("prevalence"),
            "fixed_iterations": h.get("fixed_iterations"),
        }

    csv_path = OUT / "rich_val_paired_preds.csv"
    fields = [
        "observation_id",
        "mint",
        "y_explosion_60s",
        "y_explosion_300s",
        "stage_c_60s",
        "stage_c_300s",
        "lgbm_b_60s",
        "lgbm_b_300s",
        "lgbm_a_60s",
        "lgbm_a_300s",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in paired_by_oid.values():
            if "y_explosion_60s" in r and "y_explosion_300s" in r:
                w.writerow({k: r.get(k) for k in fields})

    return {
        **Q85_FLAG,
        "hashes": hashes,
        "row_counts": {
            "assignment_val_rows": sc_meta.get("assignment_val_rows"),
            "stage_c_rows_loaded": n_sc,
            "rich_val_feature_rows": val["n"],
            "rich_train_feature_rows": train["n"],
            "stage_c_hash_matched_to_val": int(matched.sum()),
            "stage_c_unmatched_hashes": int((~matched).sum()),
            "dropped_vs_assignment": int(sc_meta.get("assignment_val_rows", 10159)) - n_sc,
            "drop_explanation": (
                "permanent_val eval script skips empty DataLoader batches. "
                f"Predictions contain {n_sc} rows vs {sc_meta.get('assignment_val_rows')} "
                "assignment val rows; per-row skip reasons are not logged."
            ),
        },
        "explosion": heads_out,
        "barriers": barrier_out,
        "stage_b_vs_broad_lgbm_descriptive": {
            "note": (
                "Descriptive only. Stage-B permanent-val is broad; rich LGBM arms are rich. "
                "Row sets are not identical."
            ),
            "row_sets_identical": False,
            "stage_b_sha256_ok": True,
            "stage_b_explosion_60s": (sb.get("explosion") or {}).get("60"),
            "stage_b_explosion_300s": (sb.get("explosion") or {}).get("300"),
            "broad_lgbm_core_explosion_60s": _core_exp("explosion_60s"),
            "broad_lgbm_core_explosion_300s": _core_exp("explosion_300s"),
        },
        "paired_csv": str(csv_path.relative_to(REPO)),
        "bootstrap_protocol": {"cluster": "mint", "n_boot": N_BOOT, "seed": BOOT_SEED},
    }


def section_b() -> dict[str, Any]:
    events = pq.read_table(WORLD / "targets/target_dataset_v3/explosion_events_v1.parquet")
    tokens = pq.read_table(SPLIT / "token_assignments.parquet")
    mint_to_split = {
        m: s
        for m, s in zip(tokens["mint"].to_pylist(), tokens["base_split"].to_pylist(), strict=True)
        if s in ("train", "val")
    }
    rich = pq.read_table(SPLIT / "observation_assignments_rich.parquet")
    rich_tv = rich.filter(pc.field("base_split").isin(["train", "val"]))
    rich_mints = set(rich_tv["mint"].to_pylist())
    cols = events.column_names
    out: dict[str, Any] = {"event_columns": cols, **Q85_FLAG}

    def _analyze(definition: str) -> dict[str, Any]:
        sub = events.filter(pc.field("definition_id") == definition)
        if "explosion_exists" in cols:
            sub = sub.filter(pc.field("explosion_exists") == True)  # noqa: E712
        mints = sub["mint"].to_pylist()
        keep_i = [i for i, m in enumerate(mints) if mint_to_split.get(m) in ("train", "val")]
        if not keep_i:
            return {"n_events": 0}
        mints_k = [mints[i] for i in keep_i]
        splits_k = [mint_to_split[m] for m in mints_k]
        peak_col = next(
            (
                c
                for c in (
                    "peak_mfe_from_onset",
                    "explosion_magnitude",
                    "explosion_peak_future_return",
                )
                if c in cols
            ),
            None,
        )
        ttp_col = next(
            (c for c in ("onset_to_peak_sec", "time_to_peak_sec") if c in cols),
            None,
        )
        peak = [sub[peak_col].to_pylist()[i] for i in keep_i] if peak_col else []
        ttp = [sub[ttp_col].to_pylist()[i] for i in keep_i] if ttp_col else []

        def _pct(vals: list[Any]) -> dict[str, float] | None:
            a = np.array(
                [float(x) for x in vals if x is not None and np.isfinite(float(x))],
                dtype=np.float64,
            )
            if a.size == 0:
                return None
            return {f"p{q}": float(np.percentile(a, q)) for q in (10, 25, 50, 75, 90, 99)}

        by_split = {}
        for sp in ("train", "val"):
            ii = [j for j, s in enumerate(splits_k) if s == sp]
            by_split[sp] = {
                "n_events": len(ii),
                "n_mints_with_ge1": len({mints_k[j] for j in ii}),
            }
        peak_f = [float(x) for x in peak if x is not None and np.isfinite(float(x))]
        fracs = None
        if peak_f:
            a = np.asarray(peak_f)
            fracs = {
                "ge_0p5x_gain": float(np.mean(a >= 0.5)),
                "ge_1x_gain_2x": float(np.mean(a >= 1.0)),
                "ge_4x_gain_5x": float(np.mean(a >= 4.0)),
                "ge_9x_gain_10x": float(np.mean(a >= 9.0)),
            }
        # rich-only
        rich_i = [j for j, m in enumerate(mints_k) if m in rich_mints]
        rich_by_split = {}
        for sp in ("train", "val"):
            ii = [j for j in rich_i if splits_k[j] == sp]
            rich_by_split[sp] = {
                "n_events": len(ii),
                "n_mints_with_ge1": len({mints_k[j] for j in ii}),
            }
        return {
            "by_split_train_val": by_split,
            "peak_column": peak_col,
            "ttp_column": ttp_col,
            "peak_return_percentiles": _pct(peak_f),
            "time_to_peak_sec_percentiles": _pct(ttp) if ttp_col else None,
            "fraction_peak_gain_thresholds": fracs,
            "rich_population_only": {"by_split": rich_by_split},
        }

    for defn in (
        "explosion_primary_v1",
        "explosion_conservative_v1",
        "explosion_aggressive_v1",
    ):
        out[defn] = _analyze(defn)

    # Per-mint max multiple vs first observation — use joined first price if present
    joined = pq.read_table(SPLIT / "training_cache_v1/rich/joined.parquet")
    joined = joined.filter(pc.field("base_split").isin(["train", "val"]))
    jcols = joined.column_names
    out["mint_max_multiple_vs_first_obs"] = {
        "status": "could_not_complete",
        "reason": (
            "No frozen per-mint max-future-multiple column found in rich joined cache; "
            "available price-like cols sample: "
            f"{[c for c in jcols if 'price' in c or 'mfe' in c or 'return' in c][:30]}"
        ),
    }
    return out


def section_c() -> dict[str, Any]:
    from trencher.data.world_model.splits.tte_overlay import STATUS_LEFT_TRUNCATED

    joined = pq.read_table(SPLIT / "training_cache_v1/rich/joined.parquet")
    joined = joined.filter(pc.field("base_split").isin(["train", "val"]))
    already = joined["explosion_already_started_at_t"].to_pylist()
    n = len(already)
    n_already = sum(1 for x in already if x is True)
    out: dict[str, Any] = {
        **Q85_FLAG,
        "observation_sampling": {
            "source": "manifests/world_model_data/OBSERVATION_DATASET_V1_AUDIT.md",
            "triggers": {"fixed_age": 40000, "event_count": 11353},
            "obs_per_token": {
                "min": 10,
                "p50": 13.0,
                "p90": 16.0,
                "p99": 20.0,
                "max": 20,
                "mean": 12.83825,
            },
            "tokens": 4000,
            "observations": 51353,
        },
        "left_truncation": {
            "file": "src/trencher/data/world_model/splits/tte_overlay.py",
            "function": "apply_explosion_tte_overlay",
            "status": STATUS_LEFT_TRUNCATED,
            "rule": (
                "If explosion_already_started_at_t, TTE is left_truncated with at_risk=False."
            ),
            "rich_train_val_fraction_already_started": float(n_already / n) if n else None,
            "n": n,
            "n_already_started": n_already,
        },
    }
    for h, col in ((60, "explosion_within_60s"), (300, "explosion_within_300s")):
        y = joined[col].to_pylist()
        tte = joined["time_to_explosion_sec"].to_pylist()
        secs = [
            float(tte[i])
            for i, v in enumerate(y)
            if v is True and tte[i] is not None and np.isfinite(float(tte[i]))
        ]
        a = np.asarray(secs, dtype=np.float64)
        pct = (
            {f"p{q}": float(np.percentile(a, q)) for q in (10, 25, 50, 75, 90, 99)}
            if a.size
            else {}
        )
        out[f"seconds_T_to_explosion_start_positive_{h}s"] = {
            "n": int(a.size),
            **pct,
            "mean": float(a.mean()) if a.size else None,
        }
    out["already_pumping_top1pct"] = {
        "status": "could_not_complete",
        "reason": (
            "Requires causal pre-T price path (±60s) joined to Stage-C/LGBM top-1% "
            "observation_ids; not available as a frozen extract in this pass."
        ),
    }
    return out


def section_d() -> dict[str, Any]:
    contract_path = REPO / "manifests/world_model_data/TARGET_CONTRACT_V3.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    lines = contract_path.read_text(encoding="utf-8").splitlines()

    def _line(substr: str) -> int | None:
        for i, ln in enumerate(lines, 1):
            if substr in ln:
                return i
        return None

    joined = pq.read_table(SPLIT / "training_cache_v1/rich/joined.parquet")
    cols = joined.column_names
    surv_cols = [c for c in cols if "surviv" in c.lower() or "active_after" in c.lower()]
    sc_met = json.loads(
        (SCRATCH / "stage_c_val" / "permanent_val_metrics_175000.json").read_text(encoding="utf-8")
    )
    cutoffs = json.loads((SPLIT / "cutoffs.json").read_text(encoding="utf-8"))

    def _counts(col: str, split: str) -> dict[str, Any]:
        t = joined.filter(pc.field("base_split") == split)
        if col not in t.column_names:
            return {"error": f"missing {col}"}
        labs = t[col].to_pylist()
        st_col = col + "_status"
        statuses = t[st_col].to_pylist() if st_col in t.column_names else [None] * len(labs)
        pos = neg = other = 0
        for y, st in zip(labs, statuses, strict=True):
            if y is True or y == 1:
                pos += 1
            elif y is False or y == 0:
                neg += 1
            else:
                other += 1
                _ = st
        return {"positives": pos, "negatives": neg, "null_or_other": other, "n": len(labs)}

    out: dict[str, Any] = {
        **Q85_FLAG,
        "definitions": {
            "survival_definition": {
                "file": "manifests/world_model_data/TARGET_CONTRACT_V3.json",
                "line": _line('"survival_definition"'),
                "text": contract.get("survival_definition"),
            },
            "survival_positive_definition": {
                "file": "manifests/world_model_data/TARGET_CONTRACT_V3.json",
                "line": _line("survival_positive_definition"),
                "text": contract["status_contract"].get("survival_positive_definition"),
            },
            "survival_negative_definition": {
                "file": "manifests/world_model_data/TARGET_CONTRACT_V3.json",
                "line": _line("survival_negative_definition"),
                "text": contract["status_contract"].get("survival_negative_definition"),
            },
            "survival_30m": {
                "meaning": "Active@H survival with H=1800s (30m) per survival_horizons",
                "horizon_sec": 1800,
            },
            "survival_24h": {
                "meaning": "Active@H survival with H=86400s (24h) per survival_horizons",
                "horizon_sec": 86400,
            },
            "explosion_tte": {
                "file": "src/trencher/data/world_model/splits/tte_overlay.py",
                "function": "apply_explosion_tte_overlay",
                "line_approx": 315,
            },
            "graduation_tte": {
                "file": "src/trencher/data/world_model/splits/tte_overlay.py",
                "note": "graduation TTE overlay with administrative censoring",
            },
        },
        "survival_like_columns_rich_joined": surv_cols,
        "loss_weights_v2": {
            "file": "configs/world_model/world_model_candidate_v2.yaml",
            "survival_weight": 1.0,
            "survival_tte_weight": 1.0,
            "explosion_tte_weight": 1.0,
            "graduation_tte_weight": 1.0,
        },
        "stage_c_survival_metrics_rich_permanent_val": sc_met.get("survival"),
        "survival_24h_zero_positives_evidence": {
            "stage_c_horizon_86400_positives": (sc_met.get("survival") or {})
            .get("86400", {})
            .get("positives"),
            "stage_c_horizon_86400_support": (sc_met.get("survival") or {})
            .get("86400", {})
            .get("support"),
            "cutoffs_iso_q70_q85": {
                "q70": "2026-07-01T01:34:01.017306+00:00",
                "q85": "2026-07-07T21:19:00.308182+00:00",
            },
            "cutoffs_unix": cutoffs.get("quantiles_unix"),
            "explanation": (
                "Stage-C rich permanent_val metrics report positives=0 for survival "
                "horizon 86400 with support=4382. Survival positives require genuine "
                "activity at/after T+24h with source coverage for negatives. Val tokens "
                "launch in a ~7-day [q70,q85) window; for many T, T+24h exceeds corpus "
                "coverage / confirmation window, so confirmed 24h-survival positives are "
                "absent while shorter horizons still have positives."
            ),
        },
    }
    for name, candidates in (
        ("survival_30m", ["survival_h1800", "active_after_h1800", "survived_h1800"]),
        ("survival_24h", ["survival_h86400", "active_after_h86400", "survived_h86400"]),
    ):
        found = next((c for c in candidates if c in cols), None)
        if found:
            out[name] = {
                "column": found,
                "train": _counts(found, "train"),
                "permanent_val": _counts(found, "val"),
            }
        else:
            # fall back to Stage-C metric horizons as evidence
            out[name] = {
                "column_in_joined": None,
                "candidates_tried": candidates,
                "stage_c_proxy_horizons": {
                    "1800": (sc_met.get("survival") or {}).get("1800"),
                    "86400": (sc_met.get("survival") or {}).get("86400"),
                },
            }
    return out


def section_e() -> dict[str, Any]:
    progress = json.loads((SCRATCH / "stage_c_run" / "progress.json").read_text())
    phase = json.loads((SCRATCH / "stage_c_run" / "phase_summary.json").read_text())
    metas = {
        name: json.loads((SCRATCH / "stage_c_run" / f"{name}.pt.meta.json").read_text())
        for name in ("latest", "periodic", "stage_c_init_150000")
    }
    csv_path = OUT / "stage_c_training_curve.csv"
    csv_path.write_text(
        "step,split,head,loss\n# no per-step loss history available under stage_c_rich/\n",
        encoding="utf-8",
    )
    return {
        **Q85_FLAG,
        "gcs_prefix": (
            "gs://trencher-world-model-v1-ronithworks-20260907/world_model_freeze_v1/"
            "runs/stage_c_rich"
        ),
        "artifacts": [
            "latest.pt",
            "latest.pt.meta.json",
            "periodic.pt",
            "periodic.pt.meta.json",
            "stage_c_init_150000.pt (+ meta + sha256)",
            "progress.json",
            "phase_summary.json",
            "derived_artifacts/sequence_graph_wallet_prior_v2_entity_index.parquet",
            "derived_artifacts/wallet_prior_entity_index_provenance.json",
            "validation/stage_c_175000_permanent_val/*",
        ],
        "checkpoints": metas,
        "progress_final": progress,
        "phase_summary": phase,
        "training_curve_available": False,
        "training_curve_csv": str(csv_path.relative_to(REPO)),
        "could_not_complete": (
            "No per-step train/val loss history logged for steps 150000→175000; "
            f"only final_loss={phase.get('final_loss')} in phase_summary.json."
        ),
        "validation_bottomed_before_175000": None,
    }


def _synthetic_batch(cfg: Any, *, batch: int, device: torch.device) -> dict[str, torch.Tensor]:
    b = batch
    t = 32

    def seq(n_ch: int, tt: int = t) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.randn(b, tt, n_ch, device=device),
            torch.ones(b, tt, dtype=torch.bool, device=device),
            torch.linspace(0, 1, tt, device=device).unsqueeze(0).expand(b, -1),
        )

    ex, em, et = seq(cfg.event.n_channels)
    cx, cm, ct = seq(cfg.curve.n_channels, 16)
    gx, gm, gt = seq(cfg.global_temporal.n_bucket_channels, 8)
    crx, crm, crt = seq(4, 8)
    mx, mm, mt = seq(cfg.microstructure.n_channels, 8)
    ox, om, ot = seq(cfg.ownership.n_channels, 8)
    wx, wm, wt = seq(cfg.wallet_prior.n_channels, 16)
    return {
        "event_x": ex,
        "event_mask": em,
        "event_times": et,
        "event_available": torch.ones(b, device=device),
        "curve_x": cx,
        "curve_mask": cm,
        "curve_times": ct,
        "curve_available": torch.ones(b, device=device),
        "global_buckets": gx,
        "global_scalars": torch.randn(b, cfg.global_temporal.n_scalars, device=device),
        "global_mask": gm,
        "global_times": gt,
        "global_available": torch.ones(b, device=device),
        "creator_x": crx,
        "creator_mask": crm,
        "creator_times": crt,
        "creator_available": torch.ones(b, device=device),
        "micro_x": mx,
        "micro_mask": mm,
        "micro_times": mt,
        "micro_available": torch.ones(b, device=device),
        "own_x": ox,
        "own_mask": om,
        "own_times": ot,
        "own_available": torch.ones(b, device=device),
        "wallet_x": wx,
        "wallet_mask": wm,
        "wallet_times": wt,
        "wallet_available": torch.ones(b, device=device),
        "graph_edge_feat": torch.randn(b, 16, cfg.graph.n_edge_channels, device=device),
        "graph_times": torch.linspace(0, 1, 16, device=device).unsqueeze(0).expand(b, -1),
        "graph_mask": torch.ones(b, 16, dtype=torch.bool, device=device),
        "graph_available": torch.ones(b, device=device),
    }


def section_f() -> dict[str, Any]:
    from trencher.world_model.config import load_world_model_config
    from trencher.world_model.model import WorldModel

    cfg = load_world_model_config(REPO / "configs/world_model/world_model_candidate_v2.yaml")
    model = WorldModel(cfg)
    total = int(sum(p.numel() for p in model.parameters()))
    by_mod = {
        name: int(sum(p.numel() for p in module.parameters()))
        for name, module in model.named_children()
    }
    device = torch.device("cpu")
    model = model.to(device).eval()
    b1 = _synthetic_batch(cfg, batch=1, device=device)
    b64 = _synthetic_batch(cfg, batch=64, device=device)

    def _bench(batch: dict[str, torch.Tensor]) -> dict[str, float]:
        with torch.inference_mode():
            for _ in range(20):
                model(batch, phase="rich")
            times = []
            for _ in range(200):
                t0 = time.perf_counter()
                model(batch, phase="rich")
                times.append(time.perf_counter() - t0)
        a = np.asarray(times) * 1000.0
        return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95))}

    cpu = {"batch_1": _bench(b1), "batch_64": _bench(b64)}
    gpu = None
    if torch.cuda.is_available():
        md = torch.device("cuda")
        mg = WorldModel(cfg).to(md).eval()
        g1 = _synthetic_batch(cfg, batch=1, device=md)
        g64 = _synthetic_batch(cfg, batch=64, device=md)

        def _bench_g(batch: dict[str, torch.Tensor]) -> dict[str, float]:
            with torch.inference_mode():
                for _ in range(20):
                    mg(batch, phase="rich")
                torch.cuda.synchronize()
                times = []
                for _ in range(200):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    mg(batch, phase="rich")
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
            a = np.asarray(times) * 1000.0
            return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95))}

        gpu = {"batch_1": _bench_g(g1), "batch_64": _bench_g(g64)}

    return {
        **Q85_FLAG,
        "parameters_total": total,
        "parameters_by_module": by_mod,
        "stage_c_val_log_parameters": 15_679_133,
        "latency_cpu_ms": cpu,
        "latency_gpu_ms": gpu,
        "gpu_available": bool(torch.cuda.is_available()),
        "modality_table": [
            {
                "modality": "events",
                "raw_sources": "Pump trades → broad_sequence_events_v1 / Dune pumpdotfun",
                "lookback": "causal pre-T event window (config max_len=512)",
                "realtime_rpc_geyser": "yes",
                "why": "Trades streamable via Geyser/RPC",
            },
            {
                "modality": "curve",
                "raw_sources": "Bonding-curve reserves from trades / execution_curve_states",
                "lookback": "causal curve path to T",
                "realtime_rpc_geyser": "yes",
                "why": "Reserves update with trades",
            },
            {
                "modality": "global",
                "raw_sources": "daily_pump_network_context, GeckoTerminal SOL/USDC, BQ blocks",
                "lookback": "completed-second global buckets",
                "realtime_rpc_geyser": "partial",
                "why": "Needs global indexer, not single-account RPC",
            },
            {
                "modality": "creator",
                "raw_sources": "broad_sequence_creator_v1 / Pump creator metadata",
                "lookback": "creator history before T",
                "realtime_rpc_geyser": "partial",
                "why": "Requires creator↔token linkage store",
            },
            {
                "modality": "microstructure",
                "raw_sources": "Solana RPC getTransaction archival subset",
                "lookback": "recent txs before T",
                "realtime_rpc_geyser": "yes",
                "why": "Tx metadata available from Geyser/RPC",
            },
            {
                "modality": "ownership",
                "raw_sources": "BigQuery Token Transfers + RPC balance samples",
                "lookback": "transfers/balances before T",
                "realtime_rpc_geyser": "partial",
                "why": "Full holder ledger needs indexer",
            },
            {
                "modality": "wallet_prior",
                "raw_sources": "wallet graph (pumpfun+cohort+BQ SPL+native SOL RPC)",
                "lookback": "wallet history before T (Stage-C length-capped)",
                "realtime_rpc_geyser": "partial",
                "why": "Maintained multi-wallet graph",
            },
            {
                "modality": "graph",
                "raw_sources": "freeze graph edges_wallet_token / shared-funder graph",
                "lookback": "neighborhood before T",
                "realtime_rpc_geyser": "partial",
                "why": "Offline-derived multi-entity graph",
            },
        ],
    }


def section_g() -> dict[str, Any]:
    cutoffs = json.loads((SPLIT / "cutoffs.json").read_text(encoding="utf-8"))
    contract = json.loads(
        (REPO / "manifests/world_model_data/SPLIT_CONTRACT_V1.json").read_text(encoding="utf-8")
    )
    tokens = pq.read_table(SPLIT / "token_assignments.parquet")
    rich = pq.read_table(SPLIT / "observation_assignments_rich.parquet")
    launch_col = "detected_at_unix"

    def _iso(u: float) -> str:
        return datetime.fromtimestamp(u, tz=UTC).isoformat()

    def _median_mints_per_day(table) -> float | None:
        launches = np.asarray(table[launch_col].to_pylist(), dtype=np.float64)
        days = np.floor(launches / 86400.0).astype(np.int64)
        counts = Counter(days.tolist())
        return float(np.median(list(counts.values()))) if counts else None

    train_tok = tokens.filter(pc.field("base_split") == "train")
    val_tok = tokens.filter(pc.field("base_split") == "val")
    rich_tv = rich.filter(pc.field("base_split").isin(["train", "val"]))
    rich_mints = set(rich_tv["mint"].to_pylist())
    # token rows for rich mints in train+val
    tok_tv = tokens.filter(pc.field("base_split").isin(["train", "val"]))
    rich_launch = [
        float(u)
        for m, u, s in zip(
            tok_tv["mint"].to_pylist(),
            tok_tv[launch_col].to_pylist(),
            tok_tv["base_split"].to_pylist(),
            strict=True,
        )
        if m in rich_mints and s in ("train", "val") and u is not None
    ]
    # assignment value_counts include test totals only as counts — do not analyze test rows
    vc_rich = {r["values"]: r["counts"] for r in pc.value_counts(rich["base_split"]).to_pylist()}
    vc_tok = {r["values"]: r["counts"] for r in pc.value_counts(tokens["base_split"]).to_pylist()}

    return {
        **Q85_FLAG,
        "cutoffs_unix": cutoffs.get("quantiles_unix") or contract.get("cutoffs_unix"),
        "cutoffs_iso": contract.get("cutoffs_iso"),
        "min_train_launch_unix": float(pc.min(train_tok[launch_col]).as_py()),
        "min_train_launch_iso": _iso(float(pc.min(train_tok[launch_col]).as_py())),
        "counts": {
            "tokens_by_split_including_test_count_only": vc_tok,
            "rich_observations_by_split_including_test_count_only": vc_rich,
            "broad_tokens_train": train_tok.num_rows,
            "broad_tokens_val": val_tok.num_rows,
            "rich_obs_train": int(vc_rich.get("train", 0)),
            "rich_obs_val": int(vc_rich.get("val", 0)),
            "rich_mints_train_val": len(rich_mints),
            "note": (
                "TEST counts shown only as frozen assignment totals; "
                "TEST rows not read for metrics"
            ),
        },
        "median_new_mints_per_day": {
            "train": _median_mints_per_day(train_tok),
            "permanent_val": _median_mints_per_day(val_tok),
        },
        "rich_population_date_range": {
            "min_unix": min(rich_launch) if rich_launch else None,
            "max_unix": max(rich_launch) if rich_launch else None,
            "min_iso": _iso(min(rich_launch)) if rich_launch else None,
            "max_iso": _iso(max(rich_launch)) if rich_launch else None,
            "n_launch_timestamps": len(rich_launch),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    parts = [
        "# Lead Intake Report v1",
        "",
        "`q85_test_accessed: false`",
        "",
    ]
    for key, title in (
        ("A", "Stage C vs LightGBM vs Stage B (explosion)"),
        ("B", "Pump size distribution"),
        ("C", "Timing and already pumping"),
        ("D", "Survival heads"),
        ("E", "Stage C overfitting check"),
        ("F", "Model size and inference latency"),
        ("G", "Data timeline and volume"),
    ):
        parts.append(f"## {key}. {title}")
        parts.append("")
        parts.append("```json")
        parts.append(json.dumps(report.get(key), indent=2, default=str)[:20000])
        parts.append("```")
        parts.append("")
    parts.append("## Could not complete")
    parts.append("")
    for item in report.get("could_not_complete") or []:
        parts.append(f"- {item}")
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-a",
        action="store_true",
        help="Load section A from scratch/section_a.json if present",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    could_not: list[str] = []
    report: dict[str, Any] = {**Q85_FLAG, "contract": "LEAD_INTAKE_V1"}
    a_cache = SCRATCH / "section_a.json"

    print("A...", flush=True)
    if args.resume_a and a_cache.is_file():
        report["A"] = json.loads(a_cache.read_text(encoding="utf-8"))
        print("  resumed from", a_cache, flush=True)
    else:
        report["A"] = section_a()
        write_json(a_cache, report["A"])

    print("B...", flush=True)
    report["B"] = section_b()
    print("C...", flush=True)
    report["C"] = section_c()
    print("D...", flush=True)
    report["D"] = section_d()
    print("E...", flush=True)
    report["E"] = section_e()
    if not report["E"].get("training_curve_available"):
        could_not.append(report["E"]["could_not_complete"])
    print("F...", flush=True)
    try:
        report["F"] = section_f()
    except Exception as exc:  # noqa: BLE001
        report["F"] = {"error": str(exc), **Q85_FLAG}
        could_not.append(f"F failed: {exc}")
    print("G...", flush=True)
    report["G"] = section_g()

    if report["C"].get("already_pumping_top1pct", {}).get("status") == "could_not_complete":
        could_not.append(report["C"]["already_pumping_top1pct"]["reason"])
    if report["B"].get("mint_max_multiple_vs_first_obs", {}).get("status") == "could_not_complete":
        could_not.append(report["B"]["mint_max_multiple_vs_first_obs"]["reason"])
    report["could_not_complete"] = could_not
    write_json(OUT / "LEAD_INTAKE_REPORT.json", report)
    (OUT / "LEAD_INTAKE_REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
