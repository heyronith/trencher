#!/usr/bin/env python3
# ruff: noqa: E501,E402
"""Join Stage-B + tree preds, mint-bootstrap CIs, write PHASE4 report."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.review.lead_intake_lib import (  # noqa: E402
    BOOT_SEED,
    N_BOOT,
    sha256_file,
    topk_precision_recall,
    write_json,
)
from scripts.review.phase4.run_phase4 import (  # noqa: E402
    OUT,
    Q85,
    SCRATCH,
    section_rpc,
    section_stage_b_metrics_recompute,
    section_stage_c_mismatch,
    write_report,
)

TOP_FRACS = [0.001, 0.005, 0.01, 0.02, 0.05]


def _metrics_block(y: np.ndarray, s: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    roc = float(roc_auc_score(y, s)) if y.min() != y.max() else float("nan")
    pr = float(average_precision_score(y, s)) if y.min() != y.max() else float("nan")
    top = topk_precision_recall(y, s, TOP_FRACS)
    return {
        "n": int(y.size),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": roc,
        "pr_auc": pr,
        "precision_at": {
            f"top_{f:g}": float(top[f"top_{f:g}"]["precision"]) for f in TOP_FRACS
        },
        "recall_at": {f"top_{f:g}": float(top[f"top_{f:g}"]["recall"]) for f in TOP_FRACS},
    }


def _boot_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Process-pool worker: mint-clustered Δmetric bootstrap."""
    y = payload["y"]
    a = payload["a"]
    b = payload["b"]
    groups = payload["groups"]
    n_mints = payload["n_mints"]
    metric = payload["metric"]
    n_boot = payload["n_boot"]
    seed = payload["seed"]
    tag = payload["tag"]

    def _m(yy: np.ndarray, ss: np.ndarray) -> float | None:
        if yy.size == 0 or yy.min() == yy.max():
            return None
        if metric == "roc_auc":
            return float(roc_auc_score(yy, ss))
        return float(average_precision_score(yy, ss))

    point_a = _m(y, a)
    point_b = _m(y, b)
    out: dict[str, Any] = {
        "tag": tag,
        "metric": metric,
        "point_a": point_a,
        "point_b": point_b,
        "delta": None if point_a is None or point_b is None else float(point_b - point_a),
        "ci95": [None, None],
        "n": int(y.size),
        "n_mints": int(n_mints),
        "n_boot": n_boot,
        "seed": seed,
    }
    if point_a is None or point_b is None:
        return out
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    finite = 0
    for i in range(n_boot):
        chosen = rng.integers(0, n_mints, size=n_mints)
        idx = np.concatenate([groups[j] for j in chosen])
        pa = _m(y[idx], a[idx])
        pb = _m(y[idx], b[idx])
        if pa is None or pb is None:
            deltas[i] = np.nan
            continue
        deltas[i] = pb - pa
        finite += 1
    ok = deltas[np.isfinite(deltas)]
    if ok.size:
        out["ci95"] = [float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))]
    out["n_boot_finite"] = int(finite)
    return out


def _groups(mints: np.ndarray) -> tuple[list[np.ndarray], int]:
    uniq, inv = np.unique(mints, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(uniq.size)]
    return groups, int(uniq.size)


def compare() -> dict[str, Any]:
    from scipy.stats import rankdata

    sb_path = SCRATCH / "preds" / "stage_b_broad_permanent_val.parquet"
    tr_path = SCRATCH / "preds" / "trees_broad_permanent_val.parquet"
    print("loading parquets...", flush=True)
    sb_t = pq.read_table(sb_path).to_pandas()
    tr_t = pq.read_table(tr_path).to_pandas()
    merged = sb_t.merge(tr_t, on="observation_id", how="inner", suffixes=("_sb", "_tr"))
    if "mint_sb" in merged.columns:
        merged["mint"] = merged["mint_sb"].where(
            merged["mint_sb"].astype(str).str.len() > 0, merged.get("mint_tr")
        )
    elif "mint" not in merged.columns:
        merged["mint"] = merged.get("mint_tr", "")

    comparisons: dict[str, Any] = {}
    boot_jobs: list[dict[str, Any]] = []

    specs = (
        ("explosion_60s", "p_explosion_60s", "y_explosion_60s", "m_explosion_60s", "lgb_explosion_60s", "xgb_explosion_60s"),
        ("explosion_300s", "p_explosion_300s", "y_explosion_300s", "m_explosion_300s", "lgb_explosion_300s", "xgb_explosion_300s"),
        ("barrier_u25_l15_5m", "p_barrier_u25_l15_5m", "y_barrier_u25_l15_5m", "elig_barrier_u25_l15_5m", "lgb_barrier_u25_l15_5m", "xgb_barrier_u25_l15_5m"),
        ("barrier_u100_l30_1h", "p_barrier_u100_l30_1h", "y_barrier_u100_l30_1h", "elig_barrier_u100_l30_1h", "lgb_barrier_u100_l30_1h", "xgb_barrier_u100_l30_1h"),
    )

    for head, p_sb, y_col, m_col, lgb_col, xgb_col in specs:
        if lgb_col not in merged.columns:
            continue
        print(f"metrics {head}...", flush=True)
        elig = merged[m_col].to_numpy() > 0.5
        y_src = y_col if y_col in merged.columns else f"{y_col}_sb"
        if y_src not in merged.columns:
            y_src = f"{y_col}_tr"
        y = merged[y_src].to_numpy()[elig].astype(np.int8)
        if head.startswith("explosion"):
            s_sb = merged[p_sb].to_numpy()[elig].astype(np.float64)
        else:
            s_sb = (
                merged[p_sb].to_numpy()[elig].astype(np.float64)
                if p_sb in merged.columns
                else None
            )
        s_lgb = merged[lgb_col].to_numpy()[elig].astype(np.float64)
        s_xgb = merged[xgb_col].to_numpy()[elig].astype(np.float64)
        mints = merged["mint"].to_numpy()[elig]
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
            groups, n_mints = _groups(mints)
            # ROC + PR mint-bootstrap (protocol: 2000 reps, seed 0).
            for metric in ("roc_auc", "pr_auc"):
                for tag, aa, bb in (
                    (f"{head}|sb_minus_lgbm|{metric}", s_lgb, s_sb),
                    (f"{head}|sb_minus_xgb|{metric}", s_xgb, s_sb),
                    (f"{head}|ens_minus_lgbm|{metric}", s_lgb, s_ens),
                ):
                    boot_jobs.append(
                        {
                            "tag": tag,
                            "y": y,
                            "a": aa,
                            "b": bb,
                            "groups": groups,
                            "n_mints": n_mints,
                            "metric": metric,
                            "n_boot": N_BOOT,
                            "seed": BOOT_SEED,
                        }
                    )
        comparisons[head] = block

    print(f"running {len(boot_jobs)} bootstrap jobs (n_boot={N_BOOT})...", flush=True)
    # Threads: sklearn roc_auc releases the GIL; avoid macOS spawn RAM blowup.
    boot_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_boot_worker, job): job["tag"] for job in boot_jobs}
        for fut in as_completed(futs):
            tag = futs[fut]
            rec = fut.result()
            boot_results.append(rec)
            print(
                f"  done {tag} delta={rec.get('delta')} ci={rec.get('ci95')}",
                flush=True,
            )

    by_tag = {r["tag"]: r for r in boot_results}
    for head, block in comparisons.items():
        if "stage_b" not in block:
            continue
        block["delta_stage_b_minus_lgbm"] = {
            "roc_auc": by_tag[f"{head}|sb_minus_lgbm|roc_auc"],
            "pr_auc": by_tag[f"{head}|sb_minus_lgbm|pr_auc"],
        }
        block["delta_stage_b_minus_xgb"] = {
            "roc_auc": by_tag[f"{head}|sb_minus_xgb|roc_auc"],
            "pr_auc": by_tag[f"{head}|sb_minus_xgb|pr_auc"],
        }
        block["delta_ensemble_minus_lgbm"] = {
            "roc_auc": by_tag[f"{head}|ens_minus_lgbm|roc_auc"],
            "pr_auc": by_tag[f"{head}|ens_minus_lgbm|pr_auc"],
        }

    trees = json.loads((SCRATCH / "cache" / "section1_trees.json").read_text(encoding="utf-8"))
    return {
        **Q85,
        "status": "ok",
        "n_inner_join": int(len(merged)),
        "stage_b_parquet": {"path": str(sb_path), "sha256": sha256_file(sb_path)},
        "trees_parquet": {"path": str(tr_path), "sha256": sha256_file(tr_path)},
        "comparisons": comparisons,
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


def main() -> int:
    t0 = time.time()
    report: dict[str, Any] = {
        **Q85,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "section1_stage_b": section_stage_b_metrics_recompute(),
        "section1_trees": json.loads(
            (SCRATCH / "cache" / "section1_trees.json").read_text(encoding="utf-8")
        ),
        "section2_rich": json.loads(
            (SCRATCH / "cache" / "section2.json").read_text(encoding="utf-8")
        ),
        "section3_mismatch": section_stage_c_mismatch(),
        "section5_rpc": section_rpc(),
    }
    sb_par = SCRATCH / "cache" / "section1_sb_parquet.json"
    if sb_par.exists():
        report["section1_stage_b"]["per_row_parquet"] = json.loads(
            sb_par.read_text(encoding="utf-8")
        )
    print("section1 compare...", flush=True)
    report["section1_compare"] = compare()
    report["compare_elapsed_sec"] = time.time() - t0
    write_report(report)
    write_json(OUT / "PHASE4_REPORT.json", report)
    print(f"wrote {OUT} in {report['compare_elapsed_sec']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
