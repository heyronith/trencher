#!/usr/bin/env python3
# ruff: noqa: E501,E402
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.review.lead_intake_lib import sha256_file, write_json
from scripts.review.phase4.broad_matrices import head_spec, load_broad_head_arrays
from scripts.review.phase4.trees import fit_lgbm_binary_platt, fit_xgb_binary_platt

from trencher.baselines.metrics import binary_metrics

SPLIT_OVERLAY = REPO / "scratch/phase4_clean_v1/split_overlay"
SCRATCH = REPO / "scratch/phase4_clean_v1"
CORE = json.loads((REPO / "manifests/world_model_data/baseline_v2/BASELINE_CORE_METRICS_V2.json").read_text())
Q85 = {"q85_test_accessed": False}
HEADS = ["explosion_60s", "explosion_300s", "barrier_u25_l15_5m", "barrier_u100_l30_1h"]

def main():
    results = {**Q85, "status": "ok", "heads": {}}
    # Load val once for explosion heads after train-per-head to limit peak RAM:
    # For each head: load train -> fit -> free train -> load val -> predict -> free val
    pred_store = {}
    for head in HEADS:
        print(f"=== {head} load train ===", flush=True)
        spec = head_spec(head)
        tr = load_broad_head_arrays(split_root=SPLIT_OVERLAY, split="train", spec=spec)
        print(f"  train X={tr['X'].shape} elig={tr['elig'].sum()}", flush=True)
        print(f"=== {head} load val ===", flush=True)
        va = load_broad_head_arrays(split_root=SPLIT_OVERLAY, split="val", spec=spec)
        print(f"  val X={va['X'].shape} elig={va['elig'].sum()}", flush=True)
        fixed_lgb = int(CORE["lightgbm"]["heads"][head]["fixed_iterations"])
        fixed_xgb = int(CORE["xgboost"]["heads"][head]["fixed_iterations"])
        if spec.kind == "binary":
            print(f"  fit lgbm iters={fixed_lgb}", flush=True)
            lgb = fit_lgbm_binary_platt(
                X_train=tr["X"], y_train=tr["y"], elig_train=tr["elig"], times_train=tr["times"],
                X_val=va["X"], fixed_iters=fixed_lgb,
                out_dir=SCRATCH / "trees" / "lightgbm" / head,
            )
            print(f"  fit xgb iters={fixed_xgb}", flush=True)
            xgb = fit_xgb_binary_platt(
                X_train=tr["X"], y_train=tr["y"], elig_train=tr["elig"], times_train=tr["times"],
                X_val=va["X"], fixed_iters=fixed_xgb,
                out_dir=SCRATCH / "trees" / "xgboost" / head,
            )
            yy = va["y"][va["elig"]].astype(int)
            lgb_m = binary_metrics(yy, lgb["raw"][va["elig"]])
            xgb_m = binary_metrics(yy, xgb["raw"][va["elig"]])
            want_lgb = float(CORE["lightgbm"]["heads"][head]["raw"]["roc_auc"])
            want_xgb = float(CORE["xgboost"]["heads"][head]["raw"]["roc_auc"])
            results["heads"][head] = {
                "kind": "binary",
                "fixed_iterations": {"lightgbm": fixed_lgb, "xgboost": fixed_xgb},
                "lightgbm": {
                    "roc_auc": float(lgb_m["roc_auc"]), "pr_auc": float(lgb_m["pr_auc"]),
                    "n": int(lgb_m["n"]), "delta_vs_committed_roc": float(lgb_m["roc_auc"]) - want_lgb,
                    "within_0_005": abs(float(lgb_m["roc_auc"]) - want_lgb) <= 0.005,
                },
                "xgboost": {
                    "roc_auc": float(xgb_m["roc_auc"]), "pr_auc": float(xgb_m["pr_auc"]),
                    "n": int(xgb_m["n"]), "delta_vs_committed_roc": float(xgb_m["roc_auc"]) - want_xgb,
                    "within_0_005": abs(float(xgb_m["roc_auc"]) - want_xgb) <= 0.005,
                },
            }
            if "observation_id" not in pred_store:
                pred_store["observation_id"] = va["observation_id"]
                pred_store["mint"] = va["mint"]
            pred_store[f"lgb_{head}"] = lgb["raw"]
            pred_store[f"xgb_{head}"] = xgb["raw"]
            pred_store[f"y_{head}"] = va["y"]
            pred_store[f"elig_{head}"] = va["elig"].astype(np.float64)
        else:
            import lightgbm as lgb
            import xgboost as xgb

            from trencher.baselines.calibration import chronological_calibration_split
            tr_mask = tr["elig"]
            fit_idx, _ = chronological_calibration_split(tr["times"][tr_mask])
            Xf, yf = tr["X"][tr_mask][fit_idx], tr["y"][tr_mask][fit_idx]
            n_class = int(len(np.unique(tr["y"][tr_mask])))
            params = {"objective":"multiclass","num_class":n_class,"metric":"multi_logloss",
                      "learning_rate":0.05,"num_leaves":63,"min_data_in_leaf":100,
                      "feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":1,
                      "verbosity":-1,"seed":0,"num_threads":1}
            print(f"  fit lgbm multiclass iters={fixed_lgb} n_class={n_class}", flush=True)
            booster = lgb.train(params, lgb.Dataset(Xf, label=yf), num_boost_round=fixed_lgb)
            lgb_prob = booster.predict(va["X"], num_iteration=fixed_lgb)
            xparams = {"objective":"multi:softprob","num_class":n_class,"eval_metric":"mlogloss",
                       "learning_rate":0.05,"max_depth":8,"subsample":0.8,"colsample_bytree":0.8,
                       "tree_method":"hist","seed":0,"nthread":1,"verbosity":0}
            print(f"  fit xgb multiclass iters={fixed_xgb}", flush=True)
            bst = xgb.train(xparams, xgb.DMatrix(Xf, label=yf), num_boost_round=fixed_xgb)
            xgb_prob = bst.predict(xgb.DMatrix(va["X"]), iteration_range=(0, fixed_xgb+1))
            yy_u = (va["y"][va["elig"]] == 0).astype(int)
            lgb_u = lgb_prob[va["elig"], 0]
            xgb_u = xgb_prob[va["elig"], 0]
            lgb_m = binary_metrics(yy_u, lgb_u)
            xgb_m = binary_metrics(yy_u, xgb_u)
            want_lgb = float(CORE["lightgbm"]["heads"][head]["barrier"]["upper_first"]["roc_auc"])
            want_xgb = float(CORE["xgboost"]["heads"][head]["barrier"]["upper_first"]["roc_auc"])
            results["heads"][head] = {
                "kind": "multiclass_upper_first",
                "fixed_iterations": {"lightgbm": fixed_lgb, "xgboost": fixed_xgb},
                "lightgbm": {
                    "roc_auc": float(lgb_m["roc_auc"]), "pr_auc": float(lgb_m["pr_auc"]),
                    "n": int(lgb_m["n"]), "delta_vs_committed_roc": float(lgb_m["roc_auc"]) - want_lgb,
                    "within_0_005": abs(float(lgb_m["roc_auc"]) - want_lgb) <= 0.005,
                },
                "xgboost": {
                    "roc_auc": float(xgb_m["roc_auc"]), "pr_auc": float(xgb_m["pr_auc"]),
                    "n": int(xgb_m["n"]), "delta_vs_committed_roc": float(xgb_m["roc_auc"]) - want_xgb,
                    "within_0_005": abs(float(xgb_m["roc_auc"]) - want_xgb) <= 0.005,
                },
            }
            if "observation_id" not in pred_store:
                pred_store["observation_id"] = va["observation_id"]
                pred_store["mint"] = va["mint"]
            pred_store[f"lgb_{head}"] = lgb_prob[:, 0] if lgb_prob.ndim==2 else lgb_prob
            pred_store[f"xgb_{head}"] = xgb_prob[:, 0] if xgb_prob.ndim==2 else xgb_prob
            pred_store[f"y_{head}"] = (va["y"] == 0).astype(np.float64)
            pred_store[f"elig_{head}"] = va["elig"].astype(np.float64)
        del tr, va
        gc.collect()
        print(json.dumps(results["heads"][head], indent=2), flush=True)

    out_path = SCRATCH / "preds" / "trees_broad_permanent_val.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = {k: (v.tolist() if hasattr(v, "tolist") else list(v)) for k, v in pred_store.items()}
    pq.write_table(pa.table(cols), out_path)
    results["preds_path"] = str(out_path)
    results["preds_sha256"] = sha256_file(out_path)
    results["reproduction_all_within_0_005"] = all(
        results["heads"][h]["lightgbm"]["within_0_005"] and results["heads"][h]["xgboost"]["within_0_005"]
        for h in results["heads"]
    )
    write_json(SCRATCH / "cache" / "section1_trees.json", results)
    print("DONE", results["reproduction_all_within_0_005"], flush=True)

if __name__ == "__main__":
    main()
