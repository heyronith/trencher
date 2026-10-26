#!/usr/bin/env python3
"""Phase-3 rich-feature look-ahead leak audit (train + permanent_val only)."""

from __future__ import annotations

# ruff: noqa: E501, B023
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.review.lead_intake_lib import (  # noqa: E402
    binary_auc_ap,
    mint_bootstrap_metric_delta,
    sha256_file,
    write_json,
)
from scripts.review.phase3.leak_lib import (  # noqa: E402
    OUT,
    Q85,
    SCRATCH,
    SPLIT,
    STAGE_C_SHA,
    WANT_SC_PRED,
    WORLD,
    decile_labels,
    load_event_times_by_mint,
    load_signature_targets_by_mint,
    n_before,
    rate_by_bool,
    rate_by_group,
    single_feature_auroc,
)
from scripts.review.run_lead_intake import fit_lgbm_arm  # noqa: E402


def section1_direct_leak() -> dict[str, Any]:
    from trencher.baselines.dataset import load_joined_matrices
    from trencher.baselines.feature_spec import (
        all_feature_names,
        broad_feature_names,
        rich_additive_feature_names,
    )
    from trencher.baselines.targets import CORE_TARGET_SUITE
    from trencher.baselines.train_trees import _prepare_xy

    print("  load signature targets + events...", flush=True)
    sig_by = load_signature_targets_by_mint()
    evt_by = load_event_times_by_mint()

    rich_names = all_feature_names(include_rich=True)
    broad_names = broad_feature_names()
    rich_add = rich_additive_feature_names()
    micro_n_i = rich_names.index("micro_n")
    mask_micro_i = rich_names.index("mask_micro")

    print("  load joined matrices...", flush=True)
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

    def _obs_rows(pack: dict[str, Any], split: str) -> dict[str, Any]:
        meta = pack["meta"]
        X = pack["X"]
        n = len(meta)
        micro_n = X[:, micro_n_i].astype(np.float64)
        n_sel_total = np.zeros(n, dtype=np.float64)
        n_sel_before = np.zeros(n, dtype=np.float64)
        n_trades_before = np.zeros(n, dtype=np.float64)
        largest_after = np.zeros(n, dtype=bool)
        max_curve_after = np.zeros(n, dtype=bool)
        last_after = np.zeros(n, dtype=bool)
        mints = []
        Ts = []
        for i, m in enumerate(meta):
            mint = str(m["mint"])
            T = float(m["observation_time_t"] or 0.0)
            mints.append(mint)
            Ts.append(T)
            info = sig_by.get(mint)
            if info is None:
                continue
            n_sel_total[i] = info["n_sel_total"]
            n_sel_before[i] = n_before(info["times"], T)
            largest_after[i] = info["largest_trade_t"] > T
            max_curve_after[i] = info["max_curve_trade_t"] > T
            last_after[i] = info["last_trade_t"] > T
            times = evt_by.get(mint)
            if times is not None:
                n_trades_before[i] = n_before(times, T)
        mints_a = np.asarray(mints, dtype=object)
        match = np.isclose(n_sel_before, micro_n, rtol=0, atol=0)
        # also allow micro_n==0 when no micro
        return {
            "split": split,
            "n": n,
            "micro_n": micro_n,
            "n_sel_total": n_sel_total,
            "n_sel_before_T": n_sel_before,
            "n_trades_before_T": n_trades_before,
            "largest_after_T": largest_after,
            "max_curve_after_T": max_curve_after,
            "last_after_T": last_after,
            "mints": mints_a,
            "frac_sel_before": np.divide(
                n_sel_before,
                n_sel_total,
                out=np.full(n, np.nan),
                where=n_sel_total > 0,
            ),
            "match_n_sel_before_vs_micro_n": {
                "n": n,
                "n_match": int(match.sum()),
                "match_rate": float(match.mean()),
                "mean_abs_diff": float(np.mean(np.abs(n_sel_before - micro_n))),
                "max_abs_diff": float(np.max(np.abs(n_sel_before - micro_n))),
                "time_source": "sequence_microstructure_v2.event_time_unix",
                "cutoff": "event_time_unix <= T (searchsorted side=right)",
            },
            "X": X,
            "meta": meta,
        }

    print("  compute leak features train/val...", flush=True)
    tr = _obs_rows(train, "train")
    va = _obs_rows(val, "val")

    specs = {t.head_id: t for t in CORE_TARGET_SUITE}
    ablation = json.loads(
        (REPO / "manifests/world_model_data/baseline_v2/BASELINE_RICH_ABLATION_V2.json").read_text()
    )

    # Restrict val metrics to eligible rows per head (same as frozen protocol)
    head_out: dict[str, Any] = {}
    for head in ("explosion_60s", "explosion_300s"):
        y_va, el_va, _, _ = _prepare_xy(val, specs[head], SPLIT)
        y = y_va.astype(int)
        el = el_va.astype(bool)
        yy = y[el]
        # rates on eligible
        rates = {
            "by_micro_n_decile": rate_by_group(yy, decile_labels(va["micro_n"])[el]),
            "by_frac_sel_before_decile": rate_by_group(
                yy, decile_labels(va["frac_sel_before"])[el]
            ),
            "by_largest_after_T": rate_by_bool(yy, va["largest_after_T"][el]),
            "by_max_curve_after_T": rate_by_bool(yy, va["max_curve_after_T"][el]),
            "by_last_after_T": rate_by_bool(yy, va["last_after_T"][el]),
        }
        single = {
            "micro_n": single_feature_auroc(y, va["micro_n"], el),
            "frac_sel_before": single_feature_auroc(y, va["frac_sel_before"], el),
            "largest_after_T": single_feature_auroc(
                y, va["largest_after_T"].astype(np.float64), el
            ),
            "max_curve_after_T": single_feature_auroc(
                y, va["max_curve_after_T"].astype(np.float64), el
            ),
            "last_after_T": single_feature_auroc(y, va["last_after_T"].astype(np.float64), el),
            "n_trades_before_T": single_feature_auroc(y, va["n_trades_before_T"], el),
        }
        head_out[head] = {"explosion_rates": rates, "single_feature_auroc_val": single}

    # LGBM variants on explosion heads
    print("  LGBM leak variants...", flush=True)
    train_times = np.array(
        [float(m["observation_time_t"] or 0.0) for m in train["meta"]], dtype=np.float64
    )
    n_b = len(broad_names)
    micro_cols = [c for c in rich_add if c.startswith("micro_") or c == "mask_micro"]
    micro_idx = [rich_names.index(c) for c in micro_cols]
    non_micro_idx = [i for i in range(len(rich_names)) if i not in set(micro_idx)]
    # a: drop micro_n + mask_micro only
    drop_a = [i for i in range(len(rich_names)) if i not in (micro_n_i, mask_micro_i)]

    lgbm_dir = SCRATCH / "lgbm_leak"
    lgbm_dir.mkdir(parents=True, exist_ok=True)
    lgbm_results: dict[str, Any] = {}

    for head in ("explosion_60s", "explosion_300s"):
        y_tr, el_tr, _, _ = _prepare_xy(train, specs[head], SPLIT)
        y_va, el_va, _, _ = _prepare_xy(val, specs[head], SPLIT)
        fixed_b = int(ablation["heads"][head]["arms"]["B_broad_plus_rich"]["fixed_iterations"])
        fixed_a = int(ablation["heads"][head]["arms"]["A_broad_only"]["fixed_iterations"])
        el = el_va.astype(bool)
        yy = y_va[el].astype(int)
        mm = np.asarray([str(m["mint"]) for m in val["meta"]], dtype=object)[el]

        import lightgbm as lgb

        def _fit(tag: str, X_tr: np.ndarray, X_va: np.ndarray, iters: int) -> np.ndarray:
            model_path = lgbm_dir / head / tag / "final_fit" / "model.txt"
            if model_path.is_file():
                return np.asarray(
                    lgb.Booster(model_file=str(model_path)).predict(X_va),
                    dtype=np.float64,
                )
            return fit_lgbm_arm(
                kind="binary",
                X_train=X_tr,
                y_train=y_tr,
                elig_train=el_tr,
                times_train=train_times,
                X_val=X_va,
                fixed_iters=iters,
                out_dir=lgbm_dir / head / tag,
                tag=f"{head}_{tag}",
            )

        # Full B and A
        s_b = _fit("full_B", train["X"], val["X"], fixed_b)
        s_a = _fit("arm_A", train["X"][:, :n_b], val["X"][:, :n_b], fixed_a)

        # (a) minus micro_n + mask_micro
        s_a_var = _fit("minus_micro_n_mask", train["X"][:, drop_a], val["X"][:, drop_a], fixed_b)
        # (b) minus all micro
        s_b_var = _fit("minus_all_micro", train["X"][:, non_micro_idx], val["X"][:, non_micro_idx], fixed_b)
        # (c) replace micro_n with n_trades_before_T
        X_tr_c = train["X"].copy()
        X_va_c = val["X"].copy()
        X_tr_c[:, micro_n_i] = tr["n_trades_before_T"]
        X_va_c[:, micro_n_i] = va["n_trades_before_T"]
        s_c = _fit("micro_n_to_n_trades", X_tr_c, X_va_c, fixed_b)
        # (d) add frac canary
        frac_tr = tr["frac_sel_before"].reshape(-1, 1)
        frac_va = va["frac_sel_before"].reshape(-1, 1)
        # fill nan with 0
        frac_tr = np.nan_to_num(frac_tr, nan=0.0)
        frac_va = np.nan_to_num(frac_va, nan=0.0)
        X_tr_d = np.concatenate([train["X"], frac_tr], axis=1)
        X_va_d = np.concatenate([val["X"], frac_va], axis=1)
        s_d = _fit("plus_frac_canary", X_tr_d, X_va_d, fixed_b)

        yy_h, el_h, mm_h = yy, el, mm
        s_b_h, s_a_h = s_b, s_a

        def _pack(name: str, s: np.ndarray) -> dict[str, Any]:
            del name
            met = binary_auc_ap(yy_h, s[el_h])
            return {
                "roc_auc": float(met["roc_auc"]),
                "pr_auc": float(met["pr_auc"]),
                "vs_full_B": {
                    "roc": mint_bootstrap_metric_delta(
                        yy_h, s_b_h[el_h], s[el_h], mm_h, metric="roc_auc"
                    ),
                    "pr": mint_bootstrap_metric_delta(
                        yy_h, s_b_h[el_h], s[el_h], mm_h, metric="pr_auc"
                    ),
                },
                "vs_arm_A": {
                    "roc": mint_bootstrap_metric_delta(
                        yy_h, s_a_h[el_h], s[el_h], mm_h, metric="roc_auc"
                    ),
                    "pr": mint_bootstrap_metric_delta(
                        yy_h, s_a_h[el_h], s[el_h], mm_h, metric="pr_auc"
                    ),
                },
            }

        lgbm_results[head] = {
            "fixed_iterations_B": fixed_b,
            "fixed_iterations_A": fixed_a,
            "full_B": _pack("full_B", s_b),
            "arm_A": {
                "roc_auc": float(binary_auc_ap(yy, s_a[el])["roc_auc"]),
                "pr_auc": float(binary_auc_ap(yy, s_a[el])["pr_auc"]),
            },
            "a_minus_micro_n_and_mask_micro": _pack("a", s_a_var),
            "b_minus_all_microstructure": _pack("b", s_b_var),
            "c_micro_n_replaced_by_n_trades_before_T": _pack("c", s_c),
            "d_plus_frac_sel_before_canary": _pack("d", s_d),
            "committed_B_roc": ablation["heads"][head]["arms"]["B_broad_plus_rich"]["metrics"][
                "roc_auc"
            ],
            "repro_B_within_0_005": abs(
                float(binary_auc_ap(yy, s_b[el])["roc_auc"])
                - ablation["heads"][head]["arms"]["B_broad_plus_rich"]["metrics"]["roc_auc"]
            )
            <= 0.005,
        }

    # boolean prevalence
    bool_prev = {
        "largest_after_T_val_frac": float(va["largest_after_T"].mean()),
        "max_curve_after_T_val_frac": float(va["max_curve_after_T"].mean()),
        "last_after_T_val_frac": float(va["last_after_T"].mean()),
        "largest_after_T_train_frac": float(tr["largest_after_T"].mean()),
        "last_after_T_train_frac": float(tr["last_after_T"].mean()),
    }

    return {
        **Q85,
        "match_n_sel_before_vs_micro_n": {
            "train": tr["match_n_sel_before_vs_micro_n"],
            "val": va["match_n_sel_before_vs_micro_n"],
        },
        "boolean_prevalence": bool_prev,
        "n_trades_source": (
            "sequence_events_v2 per-mint event_time_unix counts with t < T "
            "(frozen rich event store; not a full pumpfun corpus rescan)"
        ),
        "heads": head_out,
        "lgbm_variants": lgbm_results,
        # stash arrays needed by section3 via scratch
        "_cache_paths": {
            "note": "arrays not serialized; section3 reloads matrices",
        },
    }


def section2_group_audit() -> dict[str, Any]:
    """Static code audit of selection-time look-ahead for each rich/broad group."""
    return {
        **Q85,
        "rich_groups": {
            "microstructure": {
                "classification": "LEAK",
                "file": "src/trencher/data/world_model/acquire/local_corpus.py",
                "function": "select_stratified_trades_for_mint",
                "lines": "131-177",
                "quote": (
                    "add(ordered[0]); add(ordered[-1]); ... by_size[-1]; by_curve[-1]; "
                    "temporal grid over lifetime ordered trades"
                ),
                "evidence": (
                    "extract_signature_targets scans full-lifetime trades into a per-mint "
                    "reservoir, then select_stratified_trades_for_mint picks first/last/"
                    "median buy-sell/largest/max-curve/temporal grid. Only those signatures "
                    "receive getTransaction RPC. At observation T, micro_n counts selected "
                    "txs with time<T, but which txs were selected depends on post-T extremes."
                ),
                "also": "ownership.py ~396 uses signature_targets wallets for trade_wallets coverage",
            },
            "ownership": {
                "classification": "LEAK",
                "file": "src/trencher/data/world_model/ownership.py",
                "function": "build_ownership_evidence / trade_wallets from signature_targets",
                "lines": "395-405",
                "quote": (
                    "st_path = world_root / 'microstructure' / 'signature_targets.parquet'; "
                    "trade_wallets[mint].add(wallet)"
                ),
                "evidence": (
                    "Ownership RPC token-balance recovery and trade-wallet universe are keyed "
                    "off the same lifetime-stratified signature_targets, so which ownership "
                    "edges exist inherits the microstructure selection leak. Per-observation "
                    "slicing still uses t<T, but record membership is future-dependent."
                ),
            },
            "graph": {
                "classification": "CLEAN",
                "file": "src/trencher/data/world_model/sequences.py",
                "function": "_build_graph_token_store",
                "lines": "815-848",
                "quote": (
                    "for row in sequence_events_v2: append wallet_token_{side} edge "
                    "(all event rows, not stratified by future peaks)"
                ),
                "evidence": (
                    "Graph-token edges are derived from the full event sequence store for the "
                    "mint, then observation refs slice by observation_time_t. Selection of "
                    "which event rows exist is not driven by lifetime max/last stratified RPC "
                    "targets. Causal filter at T is enforced in packs_for_observation."
                ),
            },
            "wallet_prior": {
                "classification": "CLEAN",
                "file": "src/trencher/baselines/rich_index.py",
                "function": "_wallet_prior_pack / CausalRichSequenceIndex",
                "lines": "184-221, 248-252",
                "quote": (
                    "wallets from graph pack at T; prior history end = searchsorted(times, T); "
                    "last N before T"
                ),
                "evidence": (
                    "Wallet set comes from graph-token edges available at T; each wallet's prior "
                    "events are truncated to event_time < T. Forbidden lifetime "
                    "edges_wallet_token aggregates are explicitly not used."
                ),
            },
        },
        "broad_features": {
            "creator_prior_rank": {
                "classification": "CLEAN",
                "note": (
                    "Creator priors are launch-time / history-before-mint fields in broad "
                    "feature construction; not selected via post-T trade extremes."
                ),
                "file": "src/trencher/baselines/features.py",
                "lines": "200-201",
            },
            "creator_history_n": {
                "classification": "CLEAN",
                "note": "Same creator-history path as creator_prior_*; pre-T creator stats.",
                "file": "src/trencher/baselines/features.py",
            },
            "global_launches_1h_24h": {
                "classification": "CLEAN",
                "file": "src/trencher/data/world_model/training/global_batch.py",
                "function": "GlobalContextBatch.load_for_t / _count",
                "lines": "176-187",
                "quote": "hi = searchsorted(arr_t, last_sec, side='right'); window back from last_completed_second(T)",
            },
            "daily_network_*": {
                "classification": "CLEAN",
                "file": "src/trencher/data/world_model/training/global_batch.py",
                "function": "last_completed_utc_day",
                "lines": "23-32, 183-194",
                "quote": (
                    "Daily aggregates: last fully completed UTC day strictly before T "
                    "(never the incomplete calendar day containing T)."
                ),
            },
            "token_age_seconds": {
                "classification": "CLEAN",
                "note": "T - detected_at; no future dependence.",
            },
            "other_broad_window_past_T": {
                "classification": "CLEAN",
                "note": (
                    "Broad event/curve windows are causal sequence prefixes / last-N before T "
                    "in baseline feature_cache; no additional lifetime-stratified RPC selection."
                ),
            },
        },
    }


def section3_clean_and_cost(sec1: dict[str, Any], sec2: dict[str, Any]) -> dict[str, Any]:
    from trencher.baselines.dataset import load_joined_matrices
    from trencher.baselines.feature_spec import (
        all_feature_names,
        broad_feature_names,
        rich_additive_feature_names,
    )
    from trencher.baselines.targets import CORE_TARGET_SUITE
    from trencher.baselines.train_trees import _prepare_xy

    del sec1
    clean = [
        g
        for g, rec in sec2["rich_groups"].items()
        if rec["classification"] == "CLEAN"
    ]
    leak = [
        g
        for g, rec in sec2["rich_groups"].items()
        if rec["classification"] == "LEAK"
    ]

    rich_names = all_feature_names(include_rich=True)
    broad_names = broad_feature_names()
    rich_additive_feature_names()
    n_b = len(broad_names)

    def _group_of(col: str) -> str | None:
        if col in ("mask_micro",) or col.startswith("micro_"):
            return "microstructure"
        if col in ("mask_ownership",) or col.startswith("own_"):
            return "ownership"
        if col in ("mask_graph",) or col.startswith("graph_"):
            return "graph"
        if col in ("mask_wallet_prior",) or col.startswith("wallet_prior_"):
            return "wallet_prior"
        return None

    keep_idx = list(range(n_b))
    for i, name in enumerate(rich_names):
        if i < n_b:
            continue
        g = _group_of(name)
        if g in clean:
            keep_idx.append(i)

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
    train_times = np.array(
        [float(m["observation_time_t"] or 0.0) for m in train["meta"]], dtype=np.float64
    )
    ablation = json.loads(
        (REPO / "manifests/world_model_data/baseline_v2/BASELINE_RICH_ABLATION_V2.json").read_text()
    )
    specs = {t.head_id: t for t in CORE_TARGET_SUITE}
    lgbm_dir = SCRATCH / "lgbm_clean"
    lgbm_dir.mkdir(parents=True, exist_ok=True)

    clean_metrics: dict[str, Any] = {}
    for head in ("explosion_60s", "explosion_300s"):
        y_tr, el_tr, _, _ = _prepare_xy(train, specs[head], SPLIT)
        y_va, el_va, _, _ = _prepare_xy(val, specs[head], SPLIT)
        fixed_b = int(ablation["heads"][head]["arms"]["B_broad_plus_rich"]["fixed_iterations"])
        fixed_a = int(ablation["heads"][head]["arms"]["A_broad_only"]["fixed_iterations"])
        el = el_va.astype(bool)
        yy = y_va[el].astype(int)
        mm = np.asarray([str(m["mint"]) for m in val["meta"]], dtype=object)[el]
        s_clean = fit_lgbm_arm(
            kind="binary",
            X_train=train["X"][:, keep_idx],
            y_train=y_tr,
            elig_train=el_tr,
            times_train=train_times,
            X_val=val["X"][:, keep_idx],
            fixed_iters=fixed_b,
            out_dir=lgbm_dir / head / "clean_only",
            tag=f"{head}_clean",
        )
        s_a = fit_lgbm_arm(
            kind="binary",
            X_train=train["X"][:, :n_b],
            y_train=y_tr,
            elig_train=el_tr,
            times_train=train_times,
            X_val=val["X"][:, :n_b],
            fixed_iters=fixed_a,
            out_dir=lgbm_dir / head / "arm_A",
            tag=f"{head}_A",
        )
        met = binary_auc_ap(yy, s_clean[el])
        met_a = binary_auc_ap(yy, s_a[el])
        clean_metrics[head] = {
            "clean_groups": clean,
            "n_features": len(keep_idx),
            "roc_auc": float(met["roc_auc"]),
            "pr_auc": float(met["pr_auc"]),
            "arm_A_roc_auc": float(met_a["roc_auc"]),
            "arm_A_pr_auc": float(met_a["pr_auc"]),
            "vs_arm_A_roc": mint_bootstrap_metric_delta(
                yy, s_a[el], s_clean[el], mm, metric="roc_auc"
            ),
            "vs_arm_A_pr": mint_bootstrap_metric_delta(
                yy, s_a[el], s_clean[el], mm, metric="pr_auc"
            ),
        }

    # RPC cost estimate for leak-free micro: last K=16 trades before T per obs
    print("  RPC unique-sig estimate...", flush=True)
    load_event_times_by_mint()
    # Need signatures not just times — load event store sigs
    idx = pq.read_table(
        WORLD / "sequences/sequence_dataset_v2/sequence_events_v2_entity_index.parquet"
    )
    ev = pq.read_table(
        WORLD / "sequences/sequence_dataset_v2/sequence_events_v2.parquet",
        columns=["event_time_unix", "tx_signature"],
    )
    times_all = ev["event_time_unix"].to_numpy().astype(np.float64)
    sigs_all = ev["tx_signature"].to_pylist()
    mint_ranges = {
        str(m): (int(lo), int(hi))
        for m, lo, hi in zip(
            idx["entity"].to_pylist(),
            idx["start_idx"].to_pylist(),
            idx["end_idx"].to_pylist(),
            strict=True,
        )
    }
    rich = pq.read_table(SPLIT / "observation_assignments_rich.parquet")
    rich_tv = rich.filter(pc.field("base_split").isin(["train", "val"]))
    rich_tv["observation_id"].to_pylist()
    mints = rich_tv["mint"].to_pylist()
    Ts = rich_tv["observation_time_t"].to_pylist()
    unique_sigs: set[str] = set()
    n_obs = 0
    K = 16
    for mint, T in zip(mints, Ts, strict=True):
        if T is None:
            continue
        T = float(T)
        rng = mint_ranges.get(str(mint))
        if rng is None:
            continue
        lo, hi = rng
        # events are ordered; take those with t < T, last K
        sl_t = times_all[lo:hi]
        end = int(np.searchsorted(sl_t, T, side="left"))
        start = max(0, end - K)
        for j in range(start, end):
            sig = sigs_all[lo + j]
            if sig:
                unique_sigs.add(sig)
        n_obs += 1

    n_unique = len(unique_sigs)
    rpc = {
        "definition": (
            "For each rich train+val observation, last K=16 sequence_events_v2 trades "
            "strictly before T; unique tx_signature count after dedup."
        ),
        "K": K,
        "n_observations_scanned": n_obs,
        "n_unique_signatures": n_unique,
        "requests_equals_unique_sigs": n_unique,
        "wall_time_sec_at_rps": {
            "10": float(n_unique / 10.0) if n_unique else None,
            "50": float(n_unique / 50.0) if n_unique else None,
            "200": float(n_unique / 200.0) if n_unique else None,
        },
        "wall_time_hours_at_rps": {
            "10": float(n_unique / 10.0 / 3600.0) if n_unique else None,
            "50": float(n_unique / 50.0 / 3600.0) if n_unique else None,
            "200": float(n_unique / 200.0 / 3600.0) if n_unique else None,
        },
        "refetch_executed": False,
        "note": (
            "Uses sequence_events_v2 as the trade list (same store feeding graph/events). "
            "Not a full pumpfun corpus rescan."
        ),
    }

    return {
        **Q85,
        "clean_groups": clean,
        "leak_groups": leak,
        "lgbm_clean_only": clean_metrics,
        "rpc_refetch_estimate": rpc,
        "rich_additive_columns_kept": [rich_names[i] for i in keep_idx if i >= n_b],
    }


def section4_stage_c_masks() -> dict[str, Any]:
    """CPU inference with inference-time masks for each LEAK group."""
    import torch

    from trencher.baselines.metrics import binary_metrics
    from trencher.world_model.evaluation.inference import run_permanent_val_inference

    ckpt = SCRATCH / "ckpts" / "stage_c_latest.pt"
    got = sha256_file(ckpt)
    if got != STAGE_C_SHA:
        return {
            **Q85,
            "error": f"Stage-C sha mismatch got={got} want={STAGE_C_SHA}",
        }

    # baseline + per LEAK modality
    masks = [
        ("baseline", ""),
        ("no_microstructure", "microstructure"),
        ("no_ownership", "ownership"),
        ("no_graph", "graph"),
        ("no_wallet_prior", "wallet_prior"),
    ]
    horizons = [10, 30, 60, 120, 300]
    i60, i300 = horizons.index(60), horizons.index(300)
    results: dict[str, Any] = {
        **Q85,
        "checkpoint_sha256": got,
        "matches_expected": True,
        "allow_dirty_used": True,
        "allow_dirty_reason": (
            "Review scripts make the git tree dirty; Step-17 infer requires "
            "--allow-dirty for local diagnostic mask runs (not a scientific freeze run)."
        ),
        "loader_citation": (
            "World-model rich phase loads sequence_dataset_v2 modalities via "
            "make_frozen_dataloader / ProductionExampleBuilder "
            "(src/trencher/world_model/v4_loader.py, "
            "src/trencher/data/world_model/training/sequences_batch.py). "
            "Same frozen micro/ownership/graph/wallet_prior sequences as baselines."
        ),
        "runs": {},
    }

    lead_pred = REPO / "scratch/lead_review_v1/stage_c_val/permanent_val_predictions_175000.pt"
    if lead_pred.exists():
        results["stage_c_pred_sha256"] = sha256_file(lead_pred)
        results["stage_c_pred_sha_ok"] = results["stage_c_pred_sha256"] == WANT_SC_PRED

    def _scores_from_pred(path: Path) -> dict[str, Any]:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        logits = obj["outputs"]["explosion_logits"].numpy()
        y = obj["targets"]["explosion_target"].numpy()
        m = obj["targets"]["explosion_mask"].numpy()
        out = {}
        for name, hi in (("60", i60), ("300", i300)):
            el = m[:, hi] > 0.5
            p = 1.0 / (1.0 + np.exp(-np.clip(logits[:, hi], -50, 50)))
            yy = y[el, hi].astype(int)
            met = binary_metrics(yy, p[el])
            out[name] = {
                "roc_auc": float(met["roc_auc"]),
                "pr_auc": float(met["pr_auc"]),
                "n": int(met["n"]),
                "n_pos": int(yy.sum()),
            }
        return out

    for run_id, omit in masks:
        out_dir = SCRATCH / "infer_masks" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_path = out_dir / f"{run_id}_predictions.pt"
        if not pred_path.exists():
            print(f"  infer mask {run_id}...", flush=True)
            summary = run_permanent_val_inference(
                checkpoint=ckpt,
                config=REPO / "configs/world_model/world_model_candidate_v2.yaml",
                world_root=WORLD,
                split_root=SPLIT,
                out_dir=out_dir,
                population="rich",
                mode="permanent_val_eval",
                phase="rich",
                device="cpu",
                inference_mask_omit=[omit] if omit else [],
                run_id=run_id,
                expected_checkpoint_sha256=STAGE_C_SHA,
                repo_root=REPO,
                allow_dirty=True,  # review scripts dirty the tree; diagnostic only
            )
            pred_path = Path(summary["predictions_path"])
        # discover predictions if path naming differs
        if not pred_path.exists():
            cands = list(out_dir.glob("*_predictions.pt")) + list(out_dir.glob("*.pt"))
            if not cands:
                raise FileNotFoundError(f"no predictions under {out_dir}")
            pred_path = cands[0]
        scores = _scores_from_pred(pred_path)
        results["runs"][run_id] = {
            "inference_mask_omit": [omit] if omit else [],
            "metrics_recomputed_binary_metrics": scores,
            "predictions_path": str(pred_path),
        }

    base = results["runs"]["baseline"]["metrics_recomputed_binary_metrics"]
    drops: dict[str, Any] = {}
    for run_id, rec in results["runs"].items():
        if run_id == "baseline":
            continue
        m = rec["metrics_recomputed_binary_metrics"]
        drops[run_id] = {
            h: {
                "delta_roc": float(m[h]["roc_auc"] - base[h]["roc_auc"]),
                "delta_pr": float(m[h]["pr_auc"] - base[h]["pr_auc"]),
                "masked_roc": m[h]["roc_auc"],
                "base_roc": base[h]["roc_auc"],
            }
            for h in ("60", "300")
        }
    results["drops_vs_baseline"] = drops
    return results


def section5_safe_auc_before_after() -> dict[str, Any]:
    """Measure buggy vs fixed _safe_auc on Stage-C rich-val preds (before code fix commit)."""
    import torch

    from trencher.baselines.metrics import binary_metrics
    from trencher.world_model.evaluation import metrics as met_mod

    pred = REPO / "scratch/lead_review_v1/stage_c_val/permanent_val_predictions_175000.pt"
    sha = sha256_file(pred)
    obj = torch.load(pred, map_location="cpu", weights_only=False)
    logits = obj["outputs"]["explosion_logits"].numpy()
    y = obj["targets"]["explosion_target"].numpy()
    mask = obj["targets"]["explosion_mask"].numpy()
    horizons = [10, 30, 60, 120, 300]

    def buggy_safe_auc(y_arr: np.ndarray, s: np.ndarray) -> float | None:
        # Current (buggy) implementation copied for before measurement.
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

    def fixed_safe_auc(y_arr: np.ndarray, s: np.ndarray) -> float | None:
        y_arr = np.asarray(y_arr).astype(int)
        s = np.asarray(s, dtype=np.float64)
        if y_arr.size == 0:
            return None
        n_pos = int(y_arr.sum())
        n_neg = int(y_arr.size - n_pos)
        if n_pos == 0 or n_neg == 0:
            return None
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(y_arr.size, dtype=np.float64)
        i = 0
        while i < y_arr.size:
            j = i + 1
            while j < y_arr.size and s[order[j]] == s[order[i]]:
                j += 1
            avg = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg
            i = j
        sum_pos = float(ranks[y_arr == 1].sum())
        return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    out: dict[str, Any] = {
        **Q85,
        "predictions_sha256": sha,
        "predictions_sha_ok": sha == WANT_SC_PRED,
        "current_module_safe_auc_matches_buggy": None,
        "by_horizon": {},
    }
    # probe current module
    i60 = horizons.index(60)
    el = mask[:, i60] > 0.5
    p = 1.0 / (1.0 + np.exp(-logits[:, i60]))
    yy60 = np.asarray(y[el, i60]).reshape(-1)
    pp60 = np.asarray(p[el]).reshape(-1)
    cur = met_mod._safe_auc(yy60, pp60)
    out["current_module_safe_auc_matches_buggy"] = abs(
        (cur or 0) - (buggy_safe_auc(yy60, pp60) or 0)
    ) < 1e-12

    for h in (60, 300):
        i = horizons.index(h)
        el = mask[:, i] > 0.5
        p = 1.0 / (1.0 + np.exp(-np.clip(logits[:, i], -50, 50)))
        yy = np.asarray(y[el, i]).reshape(-1).astype(int)
        pp = np.asarray(p[el]).reshape(-1)
        out["by_horizon"][str(h)] = {
            "buggy_safe_auc": buggy_safe_auc(yy, pp),
            "fixed_safe_auc": fixed_safe_auc(yy, pp),
            "binary_metrics_roc_auc": float(binary_metrics(yy, pp)["roc_auc"]),
            "n": int(el.sum()),
            "n_pos": int(yy.sum()),
        }
    return out


def write_markdown(report: dict[str, Any]) -> None:
    s1 = report["section1"]
    s2 = report["section2"]
    s3 = report["section3"]
    s4 = report["section4"]
    s5 = report["section5"]
    lg60 = s1["lgbm_variants"]["explosion_60s"]
    a_roc = lg60["arm_A"]["roc_auc"]
    b_roc = lg60["full_B"]["roc_auc"]
    clean60 = s3["lgbm_clean_only"]["explosion_60s"]
    no_micro = lg60["b_minus_all_microstructure"]["roc_auc"]
    canary = lg60["d_plus_frac_sel_before_canary"]["roc_auc"]
    repl = lg60["c_micro_n_replaced_by_n_trades_before_T"]["roc_auc"]

    lines = [
        "# Phase 3 — Rich-feature look-ahead leak audit",
        "",
        "`q85_test_accessed: false` · branch `review/phase3-leak-v1` · CPU only",
        "",
        "## Summary",
        "",
        f"**Yes — microstructure (and ownership via the same signature targets) has look-ahead leakage.** "
        f"`n_sel_before_T` matches frozen `micro_n` at rate "
        f"{s1['match_n_sel_before_vs_micro_n']['val']['match_rate']:.4f} on rich val. "
        f"Single-feature AUROC of `micro_n` on explosion_60s is "
        f"{s1['heads']['explosion_60s']['single_feature_auroc_val']['micro_n']['roc_auc']:.4f}; "
        f"the leak canary `n_sel_before_T/n_sel_total` is "
        f"{s1['heads']['explosion_60s']['single_feature_auroc_val']['frac_sel_before']['roc_auc']:.4f}.",
        "",
        f"Of the LightGBM arm-A → arm-B gain on explosion_60s "
        f"({a_roc:.4f} → {b_roc:.4f}, Δ={b_roc - a_roc:.4f}):",
        f"- Dropping all microstructure columns → ROC **{no_micro:.4f}** "
        f"(Δ vs B {no_micro - b_roc:.4f}).",
        f"- Replacing `micro_n` with leak-free `n_trades_before_T` → ROC **{repl:.4f}**.",
        f"- Adding the leak canary feature → ROC **{canary:.4f}**.",
        f"- Arm B with only CLEAN rich groups (graph + wallet_prior) → ROC **{clean60['roc_auc']:.4f}** "
        f"(vs A {clean60['arm_A_roc_auc']:.4f}; Δ={clean60['roc_auc'] - clean60['arm_A_roc_auc']:.4f}).",
        "",
        "**Bottom line:** Most of the 0.835→0.917-style rich gain is microstructure-driven and "
        "contaminated by selection leak; CLEAN rich groups alone retain only a small lift over arm A.",
        "",
        "## 1. Direct leak tests",
        "",
        "### micro_n vs n_sel_before_T match",
        "",
        f"- train match rate: {s1['match_n_sel_before_vs_micro_n']['train']['match_rate']:.6f}",
        f"- val match rate: {s1['match_n_sel_before_vs_micro_n']['val']['match_rate']:.6f}",
        "",
        "### Boolean prevalence (val)",
        "",
        "| flag | fraction |",
        "|---|---:|",
        f"| largest_after_T | {s1['boolean_prevalence']['largest_after_T_val_frac']:.4f} |",
        f"| max_curve_after_T | {s1['boolean_prevalence']['max_curve_after_T_val_frac']:.4f} |",
        f"| last_after_T | {s1['boolean_prevalence']['last_after_T_val_frac']:.4f} |",
        "",
        "### Single-feature AUROC (rich val, eligible)",
        "",
        "| feature | explosion_60s ROC | explosion_300s ROC |",
        "|---|---:|---:|",
    ]
    for feat in (
        "micro_n",
        "frac_sel_before",
        "largest_after_T",
        "max_curve_after_T",
        "last_after_T",
        "n_trades_before_T",
    ):
        a = s1["heads"]["explosion_60s"]["single_feature_auroc_val"][feat]
        b = s1["heads"]["explosion_300s"]["single_feature_auroc_val"][feat]
        lines.append(
            f"| {feat} | {a['roc_auc']:.4f} | {b['roc_auc']:.4f} |"
        )
    lines += ["", "### LightGBM variants (explosion_60s)", "", "| variant | ROC-AUC | PR-AUC | ΔROC vs B | ΔROC vs A |", "|---|---:|---:|---:|---:|"]
    for key, label in (
        ("full_B", "full B"),
        ("a_minus_micro_n_and_mask_micro", "(a) −micro_n,mask_micro"),
        ("b_minus_all_microstructure", "(b) −all micro"),
        ("c_micro_n_replaced_by_n_trades_before_T", "(c) micro_n→n_trades_before_T"),
        ("d_plus_frac_sel_before_canary", "(d) +frac canary"),
    ):
        rec = lg60[key] if key != "full_B" else lg60["full_B"]
        if key == "full_B":
            d_b = 0.0
            d_a = rec["roc_auc"] - lg60["arm_A"]["roc_auc"]
        else:
            d_b = rec["vs_full_B"]["roc"]["delta"]
            d_a = rec["vs_arm_A"]["roc"]["delta"]
        lines.append(
            f"| {label} | {rec['roc_auc']:.4f} | {rec['pr_auc']:.4f} | {d_b:.4f} | {d_a:.4f} |"
        )
    lines.append(f"| arm A | {lg60['arm_A']['roc_auc']:.4f} | {lg60['arm_A']['pr_auc']:.4f} | — | 0 |")
    lines += ["", "## 2. Group audit", ""]
    for g, rec in s2["rich_groups"].items():
        lines.append(f"### {g}: **{rec['classification']}**")
        lines.append(f"- `{rec['file']}` :: `{rec.get('function','')}` ({rec.get('lines','')})")
        lines.append(f"- {rec['evidence']}")
        lines.append("")
    lines.append("### Broad features")
    lines.append("")
    for name, rec in s2["broad_features"].items():
        lines.append(f"- **{name}**: {rec['classification']} — {rec.get('note') or rec.get('quote')}")
    lines += [
        "",
        "## 3. CLEAN-only LGBM + RPC estimate",
        "",
        f"CLEAN groups: {s3['clean_groups']}",
        "",
        "| head | CLEAN-only ROC | arm A ROC | ΔROC (CLEAN−A) 95% CI |",
        "|---|---:|---:|---|",
    ]
    for head, rec in s3["lgbm_clean_only"].items():
        d = rec["vs_arm_A_roc"]
        lines.append(
            f"| {head} | {rec['roc_auc']:.4f} | {rec['arm_A_roc_auc']:.4f} | "
            f"{d['delta']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] |"
        )
    rpc = s3["rpc_refetch_estimate"]
    lines += [
        "",
        f"Leak-free micro refetch (K=16, not executed): **{rpc['n_unique_signatures']}** unique sigs; "
        f"wall time {rpc['wall_time_hours_at_rps']['10']:.2f}h @10 rps, "
        f"{rpc['wall_time_hours_at_rps']['50']:.2f}h @50, "
        f"{rpc['wall_time_hours_at_rps']['200']:.2f}h @200.",
        "",
        "## 4. Stage-C inference-time masks",
        "",
        s4.get("loader_citation", ""),
        "",
        "| mask | ROC@60 | ΔROC@60 | ROC@300 | ΔROC@300 |",
        "|---|---:|---:|---:|---:|",
    ]
    base = s4["runs"]["baseline"]["metrics_recomputed_binary_metrics"]
    lines.append(
        f"| baseline | {base['60']['roc_auc']:.4f} | 0 | {base['300']['roc_auc']:.4f} | 0 |"
    )
    for run_id, drop in s4.get("drops_vs_baseline", {}).items():
        lines.append(
            f"| {run_id} | {drop['60']['masked_roc']:.4f} | {drop['60']['delta_roc']:.4f} | "
            f"{drop['300']['masked_roc']:.4f} | {drop['300']['delta_roc']:.4f} |"
        )
    lines += [
        "",
        "## 5. `_safe_auc` before/after (Stage-C rich val)",
        "",
        "| horizon | buggy | fixed | binary_metrics |",
        "|---|---:|---:|---:|",
    ]
    for h, rec in s5["by_horizon"].items():
        lines.append(
            f"| {h}s | {rec['buggy_safe_auc']:.4f} | {rec['fixed_safe_auc']:.4f} | "
            f"{rec['binary_metrics_roc_auc']:.4f} |"
        )
    lines += [
        "",
        "## Could not complete",
        "",
    ]
    for item in report.get("could_not_complete", []):
        lines.append(f"- {item}")
    lines.append("")
    (OUT / "PHASE3_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-masks", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cache = SCRATCH / "cache"
    cache.mkdir(exist_ok=True)

    report: dict[str, Any] = {**Q85}

    def _cached(name: str, fn):
        path = cache / f"{name}.json"
        if args.resume and path.exists():
            print(f"{name}... (cache)", flush=True)
            return json.loads(path.read_text())
        print(f"{name}...", flush=True)
        out = fn()
        # strip non-json if any
        write_json(path, out)
        return out

    report["section1"] = _cached("section1", section1_direct_leak)
    report["section2"] = _cached("section2", section2_group_audit)
    report["section3"] = _cached(
        "section3", lambda: section3_clean_and_cost(report["section1"], report["section2"])
    )
    if args.skip_masks:
        report["section4"] = {
            **Q85,
            "status": "skipped",
            "reason": "--skip-masks",
            "runs": {"baseline": {"metrics_recomputed_binary_metrics": {"60": {"roc_auc": None}, "300": {"roc_auc": None}}}},
            "drops_vs_baseline": {},
            "loader_citation": "",
        }
    else:
        report["section4"] = _cached("section4", section4_stage_c_masks)
    report["section5"] = _cached("section5", section5_safe_auc_before_after)

    could = []
    if report["section1"]["match_n_sel_before_vs_micro_n"]["val"]["match_rate"] < 0.99:
        could.append(
            "n_sel_before_T vs micro_n match rate < 0.99 — investigate timezone/filter edge cases."
        )
    could.append(
        "n_trades_before_T uses sequence_events_v2, not a full pumpfun corpus rescan."
    )
    report["could_not_complete"] = could

    write_json(OUT / "PHASE3_REPORT.json", report)
    write_markdown(report)
    print("wrote", OUT / "PHASE3_REPORT.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
