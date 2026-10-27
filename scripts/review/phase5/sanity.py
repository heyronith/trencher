"""Section 1: n_trades_before_T diagnosis + deterministic rich trees."""

from __future__ import annotations

from typing import Any

import numpy as np
from scripts.review.phase3.leak_lib import load_event_times_by_mint, n_before
from scripts.review.phase5.common import (
    Q85,
    RICH_ABLATION,
    SCRATCH,
    SEED,
    SPLIT,
    WORLD,
    fit_lgbm_deterministic,
    metrics_block,
    mint_bootstrap_metric_delta,
    write_json,
)


def _n_trades_vector(meta: list[dict], evt_by: dict[str, np.ndarray]) -> np.ndarray:
    out = np.zeros(len(meta), dtype=np.float32)
    for i, m in enumerate(meta):
        mint = str(m["mint"])
        T = float(m.get("observation_time_t") or 0.0)
        times = evt_by.get(mint)
        out[i] = float(n_before(times, T)) if times is not None else 0.0
    return out


def _dist(x: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {"n": 0, "all_nan": True}
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    return {
        "n": int(finite.size),
        "n_nan": int(np.isnan(x).sum()),
        "n_zero": int((finite == 0).sum()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "quantiles": {f"p{q}": float(np.percentile(finite, q)) for q in qs},
        "n_unique": int(np.unique(finite).size),
        "is_constant": bool(np.unique(finite).size <= 1),
    }


def diagnose_n_trades() -> dict[str, Any]:
    """Why Phase-4 A_plus_n_trades matched A: column never in feature_spec."""
    from trencher.baselines.feature_spec import all_feature_names, broad_feature_names

    names = all_feature_names(include_rich=True)
    in_spec = "n_trades_before_T" in names
    in_broad = "n_trades_before_T" in broad_feature_names()
    return {
        **Q85,
        "root_cause": (
            "n_trades_before_T is not part of all_feature_names()/feature_spec; "
            "Phase-4 looked it up in name_to_i and got [], so A_plus_n_trades == A "
            "(identical n_features and predictions). Not constant/NaN in data — never injected."
        ),
        "in_all_feature_names": in_spec,
        "in_broad_feature_names": in_broad,
        "phase4_n_trades_idx_was_empty": True,
        "code_location": "scripts/review/phase4/run_phase4.py section_clean_rich_trees",
    }


def section_sanity() -> dict[str, Any]:
    """Fix n_trades injection; re-run rich arms; prove deterministic preds."""
    import json

    from trencher.baselines.dataset import load_joined_matrices
    from trencher.baselines.feature_spec import all_feature_names, broad_feature_names
    from trencher.baselines.targets import CORE_TARGET_SUITE
    from trencher.baselines.train_trees import _prepare_xy

    diag = diagnose_n_trades()
    names = all_feature_names(include_rich=True)
    broad = broad_feature_names()
    name_to_i = {n: i for i, n in enumerate(names)}
    broad_idx = [name_to_i[n] for n in broad if n in name_to_i]
    graph_idx = [i for n, i in name_to_i.items() if n.startswith("graph_") or n == "mask_graph"]
    wallet_idx = [
        i for n, i in name_to_i.items() if n.startswith("wallet_prior_") or n == "mask_wallet_prior"
    ]

    print("loading rich joined matrices...", flush=True)
    joined_tr = load_joined_matrices(
        world_root=WORLD,
        split_root=SPLIT,
        split="train",
        population="rich",
        feature_names=names,
        include_rich_features=True,
    )
    joined_va = load_joined_matrices(
        world_root=WORLD,
        split_root=SPLIT,
        split="val",
        population="rich",
        feature_names=names,
        include_rich_features=True,
    )
    print("computing n_trades_before_T from sequence_events_v2...", flush=True)
    evt_by = load_event_times_by_mint()
    n_tr = _n_trades_vector(joined_tr["meta"], evt_by)
    n_va = _n_trades_vector(joined_va["meta"], evt_by)
    dist = {
        "train": _dist(n_tr),
        "val": _dist(n_va),
        "source": "sequence_events_v2 via load_event_times_by_mint + n_before (<=T)",
    }

    # Append as extra column (index = len(names))
    X_tr = np.concatenate([joined_tr["X"], n_tr.reshape(-1, 1)], axis=1)
    X_va = np.concatenate([joined_va["X"], n_va.reshape(-1, 1)], axis=1)
    n_trades_col = X_tr.shape[1] - 1
    arms = {
        "A": broad_idx,
        "A_plus_n_trades": broad_idx + [n_trades_col],
        "A_plus_graph_wallet": broad_idx + graph_idx + wallet_idx,
        "A_plus_graph_wallet_n_trades": broad_idx + graph_idx + wallet_idx + [n_trades_col],
    }
    arms = {k: list(dict.fromkeys(v)) for k, v in arms.items()}

    ablation = json.loads(RICH_ABLATION.read_text(encoding="utf-8"))
    times_tr = np.array(
        [float(m.get("observation_time_t") or 0.0) for m in joined_tr["meta"]],
        dtype=np.float64,
    )

    # Compare phase3 vs phase4 fixed-iter choice for graph/wallet
    phase3_vs_phase4_iters = {}
    for head in ("explosion_60s", "explosion_300s"):
        fixed_a = int(ablation["heads"][head]["arms"]["A_broad_only"]["fixed_iterations"])
        fixed_b = int(ablation["heads"][head]["arms"]["B_broad_plus_rich"]["fixed_iterations"])
        phase3_vs_phase4_iters[head] = {
            "phase3_clean_used_B_fixed_iters": fixed_b,
            "phase4_used_A_fixed_iters_for_all_arms": fixed_a,
            "same": fixed_a == fixed_b,
        }

    out_arms: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    det_check: dict[str, Any] = {}

    for head in ("explosion_60s", "explosion_300s"):
        spec = next(s for s in CORE_TARGET_SUITE if s.head_id == head)
        y_tr, elig_tr, _, kind = _prepare_xy(joined_tr, spec, SPLIT)
        y_va, elig_va, _, _ = _prepare_xy(joined_va, spec, SPLIT)
        assert kind == "binary"
        # Match phase3 clean protocol: B iters for graph/wallet arms, A for A / A+n_trades
        fixed_a = int(ablation["heads"][head]["arms"]["A_broad_only"]["fixed_iterations"])
        fixed_b = int(ablation["heads"][head]["arms"]["B_broad_plus_rich"]["fixed_iterations"])

        head_scores: dict[str, np.ndarray] = {}
        head_metrics: dict[str, Any] = {}
        for arm_name, idx in arms.items():
            fixed = fixed_b if "graph_wallet" in arm_name else fixed_a
            pred = fit_lgbm_deterministic(
                X_train=X_tr[:, idx],
                y_train=y_tr,
                elig_train=elig_tr,
                times_train=times_tr,
                X_val=X_va[:, idx],
                fixed_iters=fixed,
                out_dir=SCRATCH / "rich_trees" / head / arm_name,
                seed=SEED,
            )["raw"]
            yy = y_va[elig_va].astype(int)
            bm = metrics_block(yy, pred[elig_va])
            head_scores[arm_name] = pred
            head_metrics[arm_name] = {
                "roc_auc": bm["roc_auc"],
                "pr_auc": bm["pr_auc"],
                "n": bm["n"],
                "fixed_iterations": fixed,
                "n_features": len(idx),
                "feature_extra": (
                    ["n_trades_before_T"] if n_trades_col in idx else []
                ),
            }
            print(
                f"  {head}/{arm_name}: roc={bm['roc_auc']:.4f} pr={bm['pr_auc']:.4f} "
                f"n_feat={len(idx)} iters={fixed}",
                flush=True,
            )

        # Determinism: two runs of A_plus_graph_wallet must match exactly
        idx_gw = arms["A_plus_graph_wallet"]
        fixed_gw = fixed_b
        p1 = fit_lgbm_deterministic(
            X_train=X_tr[:, idx_gw],
            y_train=y_tr,
            elig_train=elig_tr,
            times_train=times_tr,
            X_val=X_va[:, idx_gw],
            fixed_iters=fixed_gw,
            seed=SEED,
        )["raw"]
        p2 = fit_lgbm_deterministic(
            X_train=X_tr[:, idx_gw],
            y_train=y_tr,
            elig_train=elig_tr,
            times_train=times_tr,
            X_val=X_va[:, idx_gw],
            fixed_iters=fixed_gw,
            seed=SEED,
        )["raw"]
        max_abs = float(np.max(np.abs(p1 - p2)))
        det_check[head] = {
            "arm": "A_plus_graph_wallet",
            "identical_predictions": bool(np.array_equal(p1, p2)),
            "max_abs_diff": max_abs,
            "n_compared": int(p1.size),
            "params": {
                "deterministic": True,
                "force_row_wise": True,
                "num_threads": 1,
                "seed": SEED,
                "bagging_seed": SEED,
                "feature_fraction_seed": SEED,
            },
        }

        out_arms[head] = head_metrics
        mints = np.array([m["mint"] for m in joined_va["meta"]], dtype=object)[elig_va]
        yy = y_va[elig_va].astype(int)
        deltas[head] = {}
        for arm_name in arms:
            if arm_name == "A":
                continue
            deltas[head][arm_name] = {
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

    phase4_mismatch = {
        "phase3_lgbm_clean_only_explosion_300s_roc": 0.8173310701014166,
        "phase4_A_plus_graph_wallet_explosion_300s_roc": 0.7962,
        "investigation": {
            "columns": (
                "Same CLEAN set: mask_graph + graph_* + mask_wallet_prior + wallet_prior_* "
                "(134 features). Phase-4 did not accidentally include micro/ownership."
            ),
            "fixed_iterations": phase3_vs_phase4_iters,
            "rows": "Same rich train/val via load_joined_matrices on freeze split_v1.",
            "seed": "Both used seed=0; Phase-4 lacked deterministic/force_row_wise/"
            "bagging_seed/feature_fraction_seed, so bagging+feature subsample was not bit-stable.",
            "protocol_fix": (
                "Phase-5 re-runs with deterministic LightGBM and Phase-3 B fixed_iters "
                "for graph/wallet arms."
            ),
        },
    }

    result = {
        **Q85,
        "n_trades_diagnosis": diag,
        "n_trades_distribution": dist,
        "arms": out_arms,
        "deltas_vs_A": deltas,
        "determinism_check": det_check,
        "phase3_vs_phase4_graph_wallet": phase4_mismatch,
        "feature_columns_graph_wallet": [
            names[i] for i in (graph_idx + wallet_idx)
        ],
    }
    write_json(SCRATCH / "cache" / "section_sanity.json", result)
    return result
