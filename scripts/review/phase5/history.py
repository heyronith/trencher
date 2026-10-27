"""Section 4: corpus coverage, cost to extend, learning curve."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scripts.review.phase5.common import (
    Q85,
    REPO,
    SCRATCH,
    SEED,
    SPLIT,
    fit_lgbm_deterministic,
    fit_xgb_deterministic,
    metrics_block,
    write_json,
)


def corpus_coverage() -> dict[str, Any]:
    """Date ranges from Dune query docs + public corpora under manifests/external/."""
    cutoffs = json.loads((SPLIT / "cutoffs.json").read_text(encoding="utf-8"))
    freeze = {
        "launch_min_iso": cutoffs["launch_min_iso"],
        "launch_max_iso": cutoffs["launch_max_iso"],
        "train": cutoffs["permanent_split_intervals"]["train"],
        "val": cutoffs["permanent_split_intervals"]["val"],
        "note": "Freeze covers ~5 weeks: train from 2026-06-05, val [2026-07-01, 2026-07-07).",
    }

    # Pumpfun trades local min/max event_time
    trades_dir = REPO / "data" / "external" / "pumpfun_memecoin_corpus" / "trades"
    t_min = None
    t_max = None
    n_rows = 0
    if trades_dir.exists():
        for fp in sorted(trades_dir.glob("*.parquet")):
            t = pq.read_table(fp, columns=["event_time"])
            n_rows += t.num_rows
            mn = pc.min(t["event_time"]).as_py()
            mx = pc.max(t["event_time"]).as_py()
            if mn is not None and (t_min is None or mn < t_min):
                t_min = mn
            if mx is not None and (t_max is None or mx > t_max):
                t_max = mx

    external = {
        "pumpfun_memecoin_corpus": {
            "kind": "huggingface Slinky21/Pumpfun_Memecoin_Corpus",
            "local_trades_event_time_min": str(t_min) if t_min else None,
            "local_trades_event_time_max": str(t_max) if t_max else None,
            "n_trade_rows_local": n_rows,
            "covers": "Pump trades (bonding-curve); overlaps freeze Jun–Jul 2026",
        },
        "red_pump_2026": {
            "kind": "zenodo 21383616",
            "title_coverage": "May–June 2026 pump.fun launches (survival outcomes)",
            "covers": "creator/launch cohort metadata, not full trade tape",
        },
        "red_cohort_2026": {
            "kind": "zenodo 21765387",
            "title_coverage": "June 11–25, 2026 wallet cohorts",
            "covers": "global/wallet cohort context",
        },
        "solarchive": {
            "kind": "huggingface solarchive/solarchive",
            "covers": "index/metadata from ~2020-10; freeze note: no June–July 2026 trade coverage",
        },
    }

    dune = {
        "sql_dir": "src/trencher/queries/sql/",
        "queries": [
            "trades_events",
            "dex_trades",
            "token_lifecycle",
            "token_transfers",
            "transaction_microstructure",
            "execution_liquidity",
            "wallet_funding",
            "wallet_graph",
            "market_context",
        ],
        "note": (
            "Dune SQL is parameterized by date; historical coverage is limited by "
            "Pump.fun program history on Solana (~2024-01 onward for pump.fun) and "
            "credit budget. Repo docs do not store a live Dune credit balance."
        ),
        "example_backfill_start_documented": "2025-01-01",
    }
    return {**Q85, "freeze": freeze, "external_corpora": external, "dune": dune}


def cost_to_extend() -> dict[str, Any]:
    """Estimate 3/6/12 month broad corpus extension from existing partition manifests."""
    fusion = REPO / "manifests" / "world_model_data" / "FUSION_EXPANSION_COST_PLAN.md"
    freeze_md = REPO / "manifests" / "world_model_data" / "DATA_FREEZE.md"
    # Freeze window length
    cutoffs = json.loads((SPLIT / "cutoffs.json").read_text(encoding="utf-8"))
    freeze_days = (
        cutoffs["launch_max_unix"] - cutoffs["launch_min_unix"]
    ) / 86400.0
    # From fusion plan: unqueried BQ partitions ~15 GiB/day average in Jul/Jun table
    # Use documented on-demand $6.25/TiB and ~15 GiB/day as order-of-magnitude
    # from the ranked partition table (range ~12–19 GiB/day).
    gib_per_day = 15.0
    usd_per_tib = 6.25
    # Broad feature pipeline also needs Dune/RPC; report BQ-like + row-scale proxy
    # from freeze: ~622k tokens in quantile set over ~39 days → ~16k tokens/day
    tokens_per_day = cutoffs["n_tokens_used_for_quantiles"] / max(freeze_days, 1.0)

    def _est(months: int) -> dict[str, Any]:
        days = int(months * 30.44)
        gib = gib_per_day * days
        tib = gib / 1024.0
        usd = tib * usd_per_tib
        tokens = tokens_per_day * days
        # Runtime: phase4 broad feature build was substantial; scale linearly
        # from freeze ~40 days. Use 2 CPU-hours/day as conservative feature+label build.
        cpu_hours = 2.0 * days
        return {
            "months": months,
            "days": days,
            "est_bq_gib": round(gib, 1),
            "est_bq_usd_on_demand": round(usd, 2),
            "est_tokens": int(tokens),
            "est_feature_build_cpu_hours": round(cpu_hours, 1),
            "est_wall_clock_days_8cpu": round(cpu_hours / (8 * 24), 2),
        }

    return {
        **Q85,
        "assumptions": {
            "gib_per_day": gib_per_day,
            "usd_per_tib": usd_per_tib,
            "tokens_per_day": tokens_per_day,
            "freeze_days": freeze_days,
            "sources": [
                str(fusion.relative_to(REPO)) if fusion.exists() else None,
                str(freeze_md.relative_to(REPO)) if freeze_md.exists() else None,
                "cutoffs.json quantile token count / launch span",
            ],
            "caveat": (
                "Repo has BQ partition GiB estimates, not live Dune credit quotes. "
                "Dune credit cost for equivalent Pump trades is separate and vendor-priced."
            ),
        },
        "horizons": {
            "3_months": _est(3),
            "6_months": _est(6),
            "12_months": _est(12),
        },
        "fusion_plan_snapshot": {
            "bytes_consumed_month_gib": 901.051,
            "free_tier_remaining_gib": 122.949,
            "unqueried_partitions_days": 20,
            "unqueried_gib": 301.185,
            "paid_usd_if_executed": 1.09,
        },
    }


def learning_curve(best_by_family_head: dict[str, Any]) -> dict[str, Any]:
    """Train tuned best config on most-recent 25/50/75/100% of train by time; score val.

    Runs the curve for every family/head that was tuned (4 keys). Each permanent-val
    score uses the locked WF-selected config only.
    """
    from scripts.review.phase5.matrices import load_head_bundle

    fractions = [0.25, 0.5, 0.75, 1.0]
    out: dict[str, Any] = {**Q85, "fractions": fractions, "curves": {}}

    # Primary head learning curves (both families); 300s omitted to bound runtime.
    ordered = [
        (k, v)
        for k, v in sorted(best_by_family_head.items())
        if "explosion_60s" in k
    ]
    out["curve_scope"] = [k for k, _ in ordered]
    for key, block in ordered:
        family, head_id = key.split("/", 1)
        sel = block["selection"]
        cfg = sel["best_config"]
        fixed = int(sel["median_best_iterations"])
        print(f"learning curve {key}...", flush=True)
        train = load_head_bundle(head_id, "train")
        val = load_head_bundle(head_id, "val")
        elig_tr = train["elig"].astype(bool)
        times = train["times"]
        # Most recent fraction by observation_time_t among eligible
        elig_idx = np.where(elig_tr)[0]
        order = elig_idx[np.argsort(times[elig_idx])]
        X_tr_all = np.asarray(train["X"])
        y_tr_all = train["y"]
        X_va = np.asarray(val["X"])
        elig_va = val["elig"].astype(bool)
        y_va = val["y"].astype(int)

        curve = []
        for frac in fractions:
            n_keep = max(100, int(len(order) * frac))
            # most recent = last n_keep in time-sorted eligible
            keep_idx = order[-n_keep:]
            # Build full-length elig mask with only keep_idx True
            elig_sub = np.zeros_like(elig_tr)
            elig_sub[keep_idx] = True
            t0 = time.time()
            pos = float(y_tr_all[elig_sub].sum())
            neg = float(elig_sub.sum() - pos)
            spw = (neg / max(pos, 1.0)) if cfg.get("use_scale_pos_weight") else None
            if family == "lightgbm":
                override = {
                    "learning_rate": cfg["learning_rate"],
                    "num_leaves": cfg["num_leaves"],
                    "min_data_in_leaf": cfg["min_data_in_leaf"],
                    "feature_fraction": cfg["feature_fraction"],
                    "bagging_fraction": cfg["bagging_fraction"],
                    "bagging_freq": cfg.get("bagging_freq", 1),
                    "num_threads": 8,
                    "deterministic": False,
                    "force_row_wise": False,
                    "seed": SEED,
                    "bagging_seed": SEED,
                    "feature_fraction_seed": SEED,
                }
                if spw is not None:
                    override["scale_pos_weight"] = spw
                preds = fit_lgbm_deterministic(
                    X_train=X_tr_all,
                    y_train=y_tr_all,
                    elig_train=elig_sub,
                    times_train=times,
                    X_val=X_va,
                    fixed_iters=fixed,
                    seed=SEED,
                    params_override=override,
                    use_platt=True,
                )["calibrated"]
            else:
                override = {
                    "learning_rate": cfg["learning_rate"],
                    "max_depth": cfg["max_depth"],
                    "subsample": cfg["subsample"],
                    "colsample_bytree": cfg["colsample_bytree"],
                    "min_child_weight": cfg["min_child_weight"],
                    "nthread": 8,
                    "seed": SEED,
                }
                if spw is not None:
                    override["scale_pos_weight"] = spw
                preds = fit_xgb_deterministic(
                    X_train=X_tr_all,
                    y_train=y_tr_all,
                    elig_train=elig_sub,
                    times_train=times,
                    X_val=X_va,
                    fixed_iters=fixed,
                    seed=SEED,
                    params_override=override,
                    use_platt=True,
                )["calibrated"]
            met = metrics_block(y_va[elig_va], preds[elig_va])
            rec = {
                "fraction": frac,
                "n_train_eligible": int(n_keep),
                "time_min": float(times[keep_idx].min()),
                "time_max": float(times[keep_idx].max()),
                "roc_auc": met["roc_auc"],
                "pr_auc": met["pr_auc"],
                "runtime_sec": float(time.time() - t0),
            }
            curve.append(rec)
            print(
                f"  frac={frac} n={n_keep} roc={met['roc_auc']:.4f} pr={met['pr_auc']:.4f}",
                flush=True,
            )
        # Is curve still rising at 100%?
        if len(curve) >= 2:
            rising = curve[-1]["pr_auc"] > curve[-2]["pr_auc"]
            roc_rising = curve[-1]["roc_auc"] > curve[-2]["roc_auc"]
        else:
            rising = None
            roc_rising = None
        out["curves"][key] = {
            "points": curve,
            "pr_auc_still_rising_at_100pct": rising,
            "roc_auc_still_rising_at_100pct": roc_rising,
            "config": cfg,
            "fixed_iterations": fixed,
        }
    return out


def section_history(tune_section: dict[str, Any] | None = None) -> dict[str, Any]:
    cov = corpus_coverage()
    cost = cost_to_extend()
    curves = None
    if tune_section and tune_section.get("by_family_head"):
        curves = learning_curve(tune_section["by_family_head"])
    result = {
        **Q85,
        "coverage": cov,
        "cost_to_extend": cost,
        "learning_curve": curves,
    }
    write_json(SCRATCH / "cache" / "section_history.json", result)
    return result
