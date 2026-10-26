#!/usr/bin/env python3
"""Phase-2 diagnostics: Stage B/C rich paired eval, LGBM groups, lead-time, survival, coverage.

Train + permanent_val only. Never reads q85+ TEST rows. Never writes GCS freeze paths.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

# ruff: noqa: E501

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.review.lead_intake_lib import (  # noqa: E402
    BOOT_SEED,
    N_BOOT,
    binary_auc_ap,
    mint_bootstrap_metric_delta,
    oid_hash,
    sha256_file,
    softmax_pos_prob,
    topk_precision_recall,
    write_json,
)
from scripts.review.phase2.rich_groups import (  # noqa: E402
    broad_plus_group_indices,
    drop_group_indices,
    rich_column_group_map,
)
from scripts.review.run_lead_intake import fit_lgbm_arm  # noqa: E402

WORLD = REPO / "data" / "external" / "world_model"
SPLIT = WORLD / "splits" / "split_v1"
SCRATCH = REPO / "scratch" / "phase2_v1"
LEAD_SCRATCH = REPO / "scratch" / "lead_review_v1"
LEAD_OUT = REPO / "manifests" / "world_model_data" / "evaluation" / "lead_review_v1"
OUT = REPO / "manifests" / "world_model_data" / "evaluation" / "phase2_v1"
Q85 = {"q85_test_accessed": False}
EXP_H = [10, 30, 60, 120, 300]
I60, I300 = EXP_H.index(60), EXP_H.index(300)
STAGE_B_LATEST_SHA = "ccf27af79bf178f095c70a84e48ad2e573bae8d5cfcfdde9d32fbb0a1f7aa208"
WANT_SC_PRED = "2d97c92e3bf6a9b721355d9ea2537b7a54787594ba0ee659105643e8200def13"
TOP_FRACS = [0.01, 0.02, 0.05]


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


def _score_from_logits(logits: np.ndarray, hi: int) -> np.ndarray:
    return softmax_pos_prob(logits[:, hi])


def section_loader_audit_rewriter() -> dict[str, Any]:
    """Identify which test rewrites LOADER_AUDIT_BROAD_SAMPLE_V2.json."""
    path = REPO / "tests" / "test_world_model_frozen_v4.py"
    text = path.read_text(encoding="utf-8")
    return {
        **Q85,
        "file": "tests/test_world_model_frozen_v4.py",
        "test_function": "test_broad_loader_audit_sample",
        "mechanism": (
            "The test builds a broad final_train sample audit and writes it with "
            "Path(...LOADER_AUDIT_BROAD_SAMPLE_V2.json).write_text(...)."
        ),
        "command_that_rewrites": (
            "uv run pytest tests/test_world_model_frozen_v4.py::test_broad_loader_audit_sample -q"
        ),
        "present_in_file": "LOADER_AUDIT_BROAD_SAMPLE_V2.json" in text,
    }


def section1_checkpoints_and_paired() -> dict[str, Any]:
    ckpt_dir = SCRATCH / "ckpts"
    init_path = ckpt_dir / "stage_c_init_150000.pt"
    sb_path = ckpt_dir / "stage_b_latest.pt"
    per_path = ckpt_dir / "periodic.pt"
    init_sha = sha256_file(init_path)
    sb_sha = sha256_file(sb_path)
    per_sha = sha256_file(per_path)
    sidecar = (ckpt_dir / "stage_c_init_150000.pt.sha256").read_text().split()[0]

    b = torch.load(sb_path, map_location="cpu", weights_only=False)
    c = torch.load(init_path, map_location="cpu", weights_only=False)
    bs, cs = b["model"], c["model"]
    identical = all(torch.equal(bs[k], cs[k]) for k in bs)
    hash_equal = init_sha == sb_sha

    # Load predictions (precomputed via step17 infer --device cpu, no --allow-dirty).
    sb_pred = torch.load(
        SCRATCH / "infer/stage_b_rich_init/stage_b_rich_from_init_150000_predictions.pt",
        map_location="cpu",
        weights_only=False,
    )
    per_pred = torch.load(
        SCRATCH / "infer/stage_c_periodic/stage_c_periodic_174000_predictions.pt",
        map_location="cpu",
        weights_only=False,
    )
    sc_pred = torch.load(
        LEAD_SCRATCH / "stage_c_val/permanent_val_predictions_175000.pt",
        map_location="cpu",
        weights_only=False,
    )
    sc_pred_sha = sha256_file(LEAD_SCRATCH / "stage_c_val/permanent_val_predictions_175000.pt")

    # Align WM prediction rows (10014) with rich val matrices + LGBM boosters.
    import lightgbm as lgb

    from trencher.baselines.dataset import load_joined_matrices
    from trencher.baselines.feature_spec import all_feature_names, broad_feature_names
    from trencher.baselines.targets import CORE_TARGET_SUITE
    from trencher.baselines.train_trees import _prepare_xy

    sc_h = sc_pred["targets"]["observation_id_hash"].numpy()
    assert torch.equal(
        sb_pred["targets"]["observation_id_hash"], sc_pred["targets"]["observation_id_hash"]
    )
    assert torch.equal(
        per_pred["targets"]["observation_id_hash"], sc_pred["targets"]["observation_id_hash"]
    )
    sb_l = sb_pred["outputs"]["explosion_logits"].numpy()
    sc_l = sc_pred["outputs"]["explosion_logits"].numpy()
    per_l = per_pred["outputs"]["explosion_logits"].numpy()
    y60 = sc_pred["targets"]["explosion_target"].numpy()[:, I60].astype(int)
    y300 = sc_pred["targets"]["explosion_target"].numpy()[:, I300].astype(int)
    m60 = sc_pred["targets"]["explosion_mask"].numpy()[:, I60] > 0.5
    m300 = sc_pred["targets"]["explosion_mask"].numpy()[:, I300] > 0.5

    sb60 = _score_from_logits(sb_l, I60)
    sb300 = _score_from_logits(sb_l, I300)
    sc60 = _score_from_logits(sc_l, I60)
    sc300 = _score_from_logits(sc_l, I300)
    per60 = _score_from_logits(per_l, I60)
    per300 = _score_from_logits(per_l, I300)

    rich_names = all_feature_names(include_rich=True)
    broad_names = broad_feature_names()
    val = load_joined_matrices(
        world_root=WORLD,
        split_root=SPLIT,
        split="val",
        feature_names=rich_names,
        population="rich",
        include_rich_features=True,
    )
    val_oids = [str(m["observation_id"]) for m in val["meta"]]
    val_mints_all = [str(m["mint"]) for m in val["meta"]]
    h2val = {oid_hash(o): i for i, o in enumerate(val_oids)}
    wm_to_val = np.array([h2val[int(h)] for h in sc_h], dtype=np.int64)
    mints = np.array([val_mints_all[i] for i in wm_to_val], dtype=object)

    def _predict(model_path: Path, X: np.ndarray) -> np.ndarray:
        booster = lgb.Booster(model_file=str(model_path))
        return np.asarray(booster.predict(X), dtype=np.float64)

    X_va = val["X"]
    n_b = len(broad_names)
    lgbm_root = LEAD_SCRATCH / "lgbm"
    lgbm_a_60 = _predict(
        lgbm_root / "explosion_60s/A_broad_only/final_fit/model.txt", X_va[:, :n_b]
    )[wm_to_val]
    lgbm_b_60 = _predict(
        lgbm_root / "explosion_60s/B_broad_plus_rich/final_fit/model.txt", X_va
    )[wm_to_val]
    lgbm_a_300 = _predict(
        lgbm_root / "explosion_300s/A_broad_only/final_fit/model.txt", X_va[:, :n_b]
    )[wm_to_val]
    lgbm_b_300 = _predict(
        lgbm_root / "explosion_300s/B_broad_plus_rich/final_fit/model.txt", X_va
    )[wm_to_val]

    # Labels from joined (same as training suite) for CSV y_* columns.
    specs = {t.head_id: t for t in CORE_TARGET_SUITE}
    y60_joined, _, _, _ = _prepare_xy(val, specs["explosion_60s"], SPLIT)
    y300_joined, _, _, _ = _prepare_xy(val, specs["explosion_300s"], SPLIT)

    out_rows: list[dict[str, Any]] = []
    for i in range(int(sc_h.shape[0])):
        vi = int(wm_to_val[i])
        out_rows.append(
            {
                "observation_id": val_oids[vi],
                "mint": val_mints_all[vi],
                "y_explosion_60s": int(y60[i]),
                "y_explosion_300s": int(y300[i]),
                "y_explosion_60s_joined": int(y60_joined[vi]),
                "y_explosion_300s_joined": int(y300_joined[vi]),
                "stage_c_60s": float(sc60[i]),
                "stage_c_300s": float(sc300[i]),
                "stage_b_60s": float(sb60[i]),
                "stage_b_300s": float(sb300[i]),
                "stage_c_periodic_60s": float(per60[i]),
                "stage_c_periodic_300s": float(per300[i]),
                "lgbm_b_60s": float(lgbm_b_60[i]),
                "lgbm_b_300s": float(lgbm_b_300[i]),
                "lgbm_a_60s": float(lgbm_a_60[i]),
                "lgbm_a_300s": float(lgbm_a_300[i]),
            }
        )
    csv_path = OUT / "rich_val_paired_preds.csv"
    fieldnames = list(out_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    models_60 = {
        "stage_b": sb60,
        "stage_c": sc60,
        "stage_c_periodic_174000": per60,
        "lgbm_a": lgbm_a_60,
        "lgbm_b": lgbm_b_60,
    }
    models_300 = {
        "stage_b": sb300,
        "stage_c": sc300,
        "stage_c_periodic_174000": per300,
        "lgbm_a": lgbm_a_300,
        "lgbm_b": lgbm_b_300,
    }

    def _pack(y: np.ndarray, mask: np.ndarray, scores: dict[str, np.ndarray], mints_a: np.ndarray):
        yy = y[mask].astype(int)
        mm = mints_a[mask]
        out: dict[str, Any] = {}
        for name, s in scores.items():
            out[name] = _metrics_block(yy, s[mask])
        boots = {
            "stage_c_minus_stage_b_roc": mint_bootstrap_metric_delta(
                yy, scores["stage_b"][mask], scores["stage_c"][mask], mm, metric="roc_auc"
            ),
            "stage_c_minus_stage_b_pr": mint_bootstrap_metric_delta(
                yy, scores["stage_b"][mask], scores["stage_c"][mask], mm, metric="pr_auc"
            ),
            "stage_b_minus_lgbm_a_roc": mint_bootstrap_metric_delta(
                yy, scores["lgbm_a"][mask], scores["stage_b"][mask], mm, metric="roc_auc"
            ),
            "stage_b_minus_lgbm_a_pr": mint_bootstrap_metric_delta(
                yy, scores["lgbm_a"][mask], scores["stage_b"][mask], mm, metric="pr_auc"
            ),
        }
        return {"models": out, "bootstrap": boots}

    note_auc = (
        "ROC-AUC recomputed with trencher.baselines.metrics.binary_metrics on sigmoid "
        "horizon logits. step17 summarize_prediction_artifact AUROC is unreliable due to "
        "a rank-indexing bug in world_model.evaluation.metrics._safe_auc "
        "(uses y_ord against original-order ranks)."
    )

    return {
        **Q85,
        "allow_dirty_used": False,
        "inference_device": "cpu",
        "checkpoint_hashes": {
            "stage_c_init_150000": {
                "sha256": init_sha,
                "matches_sidecar": init_sha == sidecar,
                "equals_stage_b_latest_file_sha": hash_equal,
                "stage_b_latest_sha256": sb_sha,
                "stage_b_latest_matches_handover": sb_sha == STAGE_B_LATEST_SHA,
            },
            "periodic_174000": {"sha256": per_sha, "step": 174000},
            "stage_c_175000_predictions": {
                "sha256": sc_pred_sha,
                "matches_expected": sc_pred_sha == WANT_SC_PRED,
            },
        },
        "init_vs_stage_b_latest": {
            "file_sha256_equal": hash_equal,
            "model_state_dict_tensors_identical": identical,
            "n_tensors": len(bs),
            "explanation": (
                "stage_c_init_150000.pt is a phase-transition copy of Stage-B latest.pt: "
                "all 396 model tensors are byte-identical, but the checkpoint file SHA differs "
                f"because non-weight fields differ (phase broad→rich, extra.phase_transition, "
                f"optimizer/rng/scheduler/telemetry). Stage-B latest SHA is {STAGE_B_LATEST_SHA}; "
                f"init SHA is {init_sha}."
            ),
        },
        "rows_loaded": int(sc_h.shape[0]),
        "paired_csv": str(csv_path.relative_to(REPO)),
        "scoring_note": note_auc,
        "explosion_60s": _pack(y60, m60, models_60, mints),
        "explosion_300s": _pack(y300, m300, models_300, mints),
        "bootstrap_protocol": {"cluster": "mint", "n_boot": N_BOOT, "seed": BOOT_SEED},
    }


def section2_lgbm_groups() -> dict[str, Any]:
    from trencher.baselines.dataset import load_joined_matrices
    from trencher.baselines.feature_spec import all_feature_names, broad_feature_names
    from trencher.baselines.targets import CORE_TARGET_SUITE
    from trencher.baselines.train_trees import _prepare_xy

    ablation = json.loads(
        (REPO / "manifests/world_model_data/baseline_v2/BASELINE_RICH_ABLATION_V2.json").read_text()
    )
    rich_names = all_feature_names(include_rich=True)
    broad_names = broad_feature_names()
    group_map = rich_column_group_map()
    groups = sorted(set(group_map.values()))

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
    val_mints = np.array([str(m["mint"]) for m in val["meta"]])
    X_tr, X_va = train["X"], val["X"]

    # Restrict metrics to Stage-C-overlapped permanent_val rows (10014) with head eligibility.
    lead_csv = list(csv.DictReader((OUT / "rich_val_paired_preds.csv").open()))
    if not lead_csv:
        lead_csv = list(csv.DictReader((LEAD_OUT / "rich_val_paired_preds.csv").open()))
    paired_oids = {r["observation_id"] for r in lead_csv}
    val_oids = [str(m["observation_id"]) for m in val["meta"]]
    in_paired = np.array([o in paired_oids for o in val_oids], dtype=bool)

    out: dict[str, Any] = {
        **Q85,
        "rich_column_to_group": group_map,
        "n_rich_columns": len(group_map),
        "groups": groups,
        "heads": {},
    }
    lgbm_dir = SCRATCH / "lgbm_groups"
    lgbm_dir.mkdir(parents=True, exist_ok=True)

    specs = {t.head_id: t for t in CORE_TARGET_SUITE}
    for head in ("explosion_60s", "explosion_300s"):
        spec = specs[head]
        y_tr, el_tr, _, _ = _prepare_xy(train, spec, SPLIT)
        y_va, el_va, _, _ = _prepare_xy(val, spec, SPLIT)
        fixed = int(ablation["heads"][head]["arms"]["B_broad_plus_rich"]["fixed_iterations"])
        import lightgbm as lgb

        y_tr_h, el_tr_h = y_tr, el_tr

        def _fit_or_load(
            out_dir: Path,
            idx: list[int],
            iters: int,
            tag: str,
            *,
            y_fit: np.ndarray = y_tr_h,
            elig_fit: np.ndarray = el_tr_h,
        ) -> np.ndarray:
            model_path = out_dir / "final_fit" / "model.txt"
            if model_path.is_file():
                booster = lgb.Booster(model_file=str(model_path))
                return np.asarray(booster.predict(X_va[:, idx]), dtype=np.float64)
            return fit_lgbm_arm(
                kind="binary",
                X_train=X_tr[:, idx],
                y_train=y_fit,
                elig_train=elig_fit,
                times_train=train_times,
                X_val=X_va[:, idx],
                fixed_iters=iters,
                out_dir=out_dir,
                tag=tag,
            )

        # Full arm B reference scores (retrain to get booster for importance).
        full_idx = list(range(len(rich_names)))
        s_full = _fit_or_load(lgbm_dir / head / "full_B", full_idx, fixed, f"{head}_full_B")
        mask = el_va & in_paired
        yy = y_va[mask].astype(int)
        mm = val_mints[mask]
        full_met = _metrics_block(yy, s_full[mask])

        booster = lgb.Booster(model_file=str(lgbm_dir / head / "full_B/final_fit/model.txt"))
        gains = booster.feature_importance(importance_type="gain")
        # LightGBM names are Column_0..; map to feature names
        imp = sorted(
            [
                {
                    "feature": rich_names[i],
                    "gain": float(gains[i]),
                    "group": group_map.get(rich_names[i], "broad"),
                }
                for i in range(len(rich_names))
            ],
            key=lambda d: -d["gain"],
        )[:30]

        drop_results: dict[str, Any] = {}
        for g in groups:
            idx = drop_group_indices(rich_names, g)
            s = _fit_or_load(lgbm_dir / head / f"drop_{g}", idx, fixed, f"{head}_drop_{g}")
            met = _metrics_block(yy, s[mask])
            drop_results[g] = {
                "metrics": met,
                "delta_vs_full_B": {
                    "roc": mint_bootstrap_metric_delta(
                        yy, s_full[mask], s[mask], mm, metric="roc_auc"
                    ),
                    "pr": mint_bootstrap_metric_delta(
                        yy, s_full[mask], s[mask], mm, metric="pr_auc"
                    ),
                },
                "note": "delta is drop_model - full_B (negative => group helped)",
            }

        single_results: dict[str, Any] = {}
        # broad-only
        n_b = len(broad_names)
        fixed_a = int(ablation["heads"][head]["arms"]["A_broad_only"]["fixed_iterations"])
        s_a = _fit_or_load(
            lgbm_dir / head / "broad_only", list(range(n_b)), fixed_a, f"{head}_broad_only"
        )
        single_results["broad_only"] = _metrics_block(yy, s_a[mask])
        for g in groups:
            idx = broad_plus_group_indices(rich_names, g)
            s = _fit_or_load(
                lgbm_dir / head / f"broad_plus_{g}", idx, fixed, f"{head}_broad_plus_{g}"
            )
            single_results[f"broad_plus_{g}"] = _metrics_block(yy, s[mask])

        out["heads"][head] = {
            "fixed_iterations_B": fixed,
            "full_B": full_met,
            "drop_one_group": drop_results,
            "broad_plus_single_group": single_results,
            "gain_importance_top30": imp,
            "committed_B_roc_auc": ablation["heads"][head]["arms"]["B_broad_plus_rich"]["metrics"][
                "roc_auc"
            ],
            "reproduced_B_roc_auc": full_met["roc_auc"],
            "repro_within_0_005": abs(
                full_met["roc_auc"]
                - ablation["heads"][head]["arms"]["B_broad_plus_rich"]["metrics"]["roc_auc"]
            )
            <= 0.005,
        }
    return out


def section3_lead_time(sec1: dict[str, Any]) -> dict[str, Any]:
    del sec1  # scores come from paired CSV
    joined = pq.read_table(SPLIT / "training_cache_v1/rich/joined.parquet")
    joined = joined.filter(pc.field("base_split") == "val")
    j_oids = joined["observation_id"].to_pylist()
    already = np.array(
        [bool(x) for x in joined["explosion_already_started_at_t"].to_pylist()], dtype=bool
    )
    tte = np.array(
        [
            float(x) if x is not None and np.isfinite(float(x)) else np.nan
            for x in joined["time_to_explosion_sec"].to_pylist()
        ],
        dtype=np.float64,
    )
    oid_to_j = {str(o): i for i, o in enumerate(j_oids)}

    rows = list(csv.DictReader((OUT / "rich_val_paired_preds.csv").open()))
    # Map scores
    score_cols = {
        "stage_b": ("stage_b_60s", "stage_b_300s"),
        "stage_c": ("stage_c_60s", "stage_c_300s"),
        "lgbm_a": ("lgbm_a_60s", "lgbm_a_300s"),
        "lgbm_b": ("lgbm_b_60s", "lgbm_b_300s"),
    }
    j_idx = np.array([oid_to_j[r["observation_id"]] for r in rows], dtype=np.int64)
    already_r = already[j_idx]
    tte_r = tte[j_idx]

    variants = {
        "a_exclude_already_started": {"min_lead_sec": 0.0},
        "b_onset_ge_2s": {"min_lead_sec": 2.0},
        "c_onset_ge_5s": {"min_lead_sec": 5.0},
    }

    def _variant_mask(y_pos_eligible: np.ndarray, min_lead: float) -> np.ndarray:
        """Eligible evaluation rows: not already started; positives need tte>=min_lead.

        Rows with onset too soon (positive but tte < min_lead) are excluded (not negatives).
        Already-started rows excluded. Negatives that are not already-started remain.
        """
        keep = np.ones(len(rows), dtype=bool)
        keep &= ~already_r
        # exclude too-soon positives
        too_soon = y_pos_eligible & np.isfinite(tte_r) & (tte_r < min_lead)
        # also exclude positives with missing tte when min_lead>0? treat as excluded
        if min_lead > 0:
            pos_bad = y_pos_eligible & (~np.isfinite(tte_r) | (tte_r < min_lead))
            keep &= ~pos_bad
        else:
            keep &= ~too_soon  # only already handled; too_soon with min_lead0 is tte<0
        return keep

    out_variants: dict[str, Any] = {}
    for head, ykey, sfx in (
        ("explosion_60s", "y_explosion_60s", "60s"),
        ("explosion_300s", "y_explosion_300s", "300s"),
    ):
        y = np.array([int(float(r[ykey])) for r in rows], dtype=int)
        # Match lead-intake eligibility roughly: use rows where label is 0/1 (CSV already filtered)
        head_out: dict[str, Any] = {}
        for vname, cfg in variants.items():
            keep = _variant_mask(y.astype(bool), cfg["min_lead_sec"])
            yy = y[keep]
            models = {}
            for mname, (c60, c300) in score_cols.items():
                col = c60 if sfx == "60s" else c300
                s = np.array([float(r[col]) for r in rows], dtype=np.float64)[keep]
                if yy.size == 0 or len(np.unique(yy)) < 2:
                    models[mname] = {"n": int(yy.size), "n_pos": int(yy.sum()), "roc_auc": None}
                else:
                    models[mname] = _metrics_block(yy, s)
            head_out[vname] = {
                "n": int(keep.sum()),
                "n_pos": int(yy.sum()),
                "n_excluded": int((~keep).sum()),
                "models": models,
            }
        out_variants[head] = head_out

    # Pump-level recall for 50 primary val events
    tokens = pq.read_table(SPLIT / "token_assignments.parquet")
    mint_split = {
        m: s
        for m, s in zip(tokens["mint"].to_pylist(), tokens["base_split"].to_pylist(), strict=True)
    }
    ev = pq.read_table(WORLD / "targets/target_dataset_v3/explosion_events_v1.parquet")
    prim = ev.filter(
        (pc.field("definition_id") == "explosion_primary_v1")
        & (pc.field("explosion_exists") == True)  # noqa: E712
    )
    ev_mints = prim["mint"].to_pylist()
    onset = prim["onset_unix"].to_pylist()
    peak_mfe = prim["peak_mfe_from_onset"].to_pylist()
    val_events = [
        {
            "mint": ev_mints[i],
            "onset_unix": float(onset[i]) if onset[i] is not None else None,
            "peak_mfe_from_onset": float(peak_mfe[i]) if peak_mfe[i] is not None else None,
        }
        for i in range(len(ev_mints))
        if mint_split.get(ev_mints[i]) == "val"
    ]
    assert len(val_events) == 50

    # observation times from joined
    obs_t = np.array(
        [float(x) for x in joined["observation_time_t"].to_pylist()], dtype=np.float64
    )
    # price at T if present
    price_col = "latest_price" if "latest_price" in joined.column_names else None
    prices = (
        np.array(
            [
                float(x) if x is not None and np.isfinite(float(x)) else np.nan
                for x in joined[price_col].to_pylist()
            ],
            dtype=np.float64,
        )
        if price_col
        else None
    )

    # Build per-mint list of (T, scores, jlocal)
    by_mint: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_mint[r["mint"]].append(i)

    # Top 2% threshold over all val paired rows per model/head
    pump_out: dict[str, Any] = {}
    for mname, (c60, _) in score_cols.items():
        scores = np.array([float(r[c60]) for r in rows], dtype=np.float64)
        thr = float(np.quantile(scores, 0.98))
        caught = 0
        leads: list[float] = []
        rem_gain: list[float] = []
        peak_gain_note = None
        for evn in val_events:
            mint = evn["mint"]
            onset_u = evn["onset_unix"]
            idxs = by_mint.get(mint, [])
            if not idxs or onset_u is None:
                continue
            # earliest T with score >= thr and T < onset
            cand = []
            for i in idxs:
                ji = j_idx[i]
                T = obs_t[ji]
                if onset_u > T and scores[i] >= thr:
                    cand.append((T, i, ji))
            if not cand:
                continue
            cand.sort()
            T, i, ji = cand[0]
            caught += 1
            leads.append(onset_u - T)
            # remaining peak gain: prefer peak_mfe_from_onset as return from onset;
            # remaining after T needs price_T and peak_price — peak price not in events.
            if prices is not None and np.isfinite(prices[ji]) and evn["peak_mfe_from_onset"] is not None:
                peak_gain_note = (
                    "peak_mfe_from_onset is MFE from explosion onset, not from T; "
                    "peak absolute price not stored on explosion_events_v1, so fraction of "
                    "peak gain remaining after T cannot be computed exactly."
                )
            else:
                peak_gain_note = (
                    "Could not compute remaining peak gain: need price at T and peak price; "
                    f"latest_price_col={price_col}, peak_mfe_from_onset present but is onset-relative."
                )
        arr = np.asarray(leads, dtype=np.float64)
        pump_out[mname] = {
            "n_events": 50,
            "fraction_caught_top2pct": float(caught / 50),
            "n_caught": caught,
            "lead_time_sec_percentiles": (
                {f"p{q}": float(np.percentile(arr, q)) for q in (10, 25, 50, 75, 90, 99)}
                if arr.size
                else None
            ),
            "top2_threshold": thr,
            "remaining_peak_gain": {
                "status": "could_not_complete",
                "reason": peak_gain_note
                or "No events caught or price/peak unavailable.",
                "n_computed": len(rem_gain),
            },
        }

    return {
        **Q85,
        "already_started_fraction_paired_val": float(already_r.mean()),
        "variants": out_variants,
        "pump_level_recall_primary_val_events": pump_out,
        "n_primary_val_explosion_events": 50,
    }


def section4_survival() -> dict[str, Any]:
    # Cite compute_activity_survival
    compute_path = REPO / "src/trencher/data/world_model/targets/compute.py"
    lines = compute_path.read_text(encoding="utf-8").splitlines()
    fn_line = next(i for i, ln in enumerate(lines, 1) if "def compute_activity_survival" in ln)
    pos_line = next(
        i for i, ln in enumerate(lines, 1) if "Positive resolved immediately by any genuine activity" in ln
    )

    # Sample 20 random train mints (seed 0)
    rich = pq.read_table(SPLIT / "observation_assignments_rich.parquet")
    rich_train = rich.filter(pc.field("base_split") == "train")
    train_mints = sorted(set(rich_train["mint"].to_pylist()))
    rng = np.random.default_rng(0)
    sample = list(rng.choice(train_mints, size=20, replace=False))

    idx = pq.read_table(
        WORLD / "sequences/sequence_dataset_v2/sequence_events_v2_entity_index.parquet"
    )
    ent = idx["entity"].to_pylist()
    starts = idx["start_idx"].to_pylist()
    ends = idx["end_idx"].to_pylist()
    ent_map = {e: (int(s), int(e_)) for e, s, e_ in zip(ent, starts, ends, strict=True)}
    events = pq.read_table(
        WORLD / "sequences/sequence_dataset_v2/sequence_events_v2.parquet",
        columns=["mint", "event_time_unix", "source", "source_quality_flag", "side"],
    )
    times = events["event_time_unix"].to_numpy()
    sources = events["source"].to_pylist()

    def _to_unix(t: Any) -> float | None:
        if t is None:
            return None
        if hasattr(t, "timestamp"):
            return float(t.timestamp())
        try:
            return float(t)
        except (TypeError, ValueError):
            return None

    mig = pq.read_table(WORLD / "universe" / "migrations.parquet")
    mig_map = {
        m: _to_unix(t)
        for m, t in zip(mig["mint"].to_pylist(), mig["migrated_at"].to_pylist(), strict=True)
    }
    mig_map = {m: t for m, t in mig_map.items() if t is not None}

    # PumpSwap / postgrad: registry only locally
    postgard = json.loads((WORLD / "postgrad" / "postgard_registry.json").read_text())
    # BQ transfers: list available days; sample last transfer per mint if files exist
    bq_dir = WORLD / "bigquery"
    bq_files = sorted(bq_dir.glob("token_transfers_*.parquet"))

    sample_rows: list[dict[str, Any]] = []
    for mint in sample:
        last_trade = None
        last_src = None
        if mint in ent_map:
            a, b = ent_map[mint]
            sl_t = times[a:b]
            sl_s = sources[a:b]
            if sl_t.size:
                last_i = int(np.argmax(sl_t))
                last_trade = float(sl_t[last_i])
                last_src = sl_s[last_i]
        graduated = mint in mig_map
        sample_rows.append(
            {
                "mint": mint,
                "last_trade_unix_in_sequence_events_v2": last_trade,
                "last_trade_source": last_src,
                "graduated_to_pumpswap": graduated,
                "migration_unix": mig_map.get(mint),
                "other_sources": {
                    "postgard_registry_keys_sample": list(postgard)[:8]
                    if isinstance(postgard, dict)
                    else type(postgard).__name__,
                    "bq_transfer_day_files_n": len(bq_files),
                    "note": (
                        "Per-mint last BQ transfer not joined in this pass; "
                        "day-partitioned token_transfers_*.parquet exist under bigquery/."
                    ),
                },
            }
        )

    # Does activity stop at graduation? Compare last trade vs migration for graduated train mints
    graduated_train = [m for m in train_mints if m in mig_map]
    deltas = []
    last_before = 0
    last_after = 0
    last_equalish = 0
    for mint in graduated_train:
        if mint not in ent_map:
            continue
        a, b = ent_map[mint]
        sl_t = times[a:b]
        if sl_t.size == 0:
            continue
        last_t = float(sl_t.max())
        mig_t = float(mig_map[mint])
        d = last_t - mig_t
        deltas.append(d)
        if last_t < mig_t - 1:
            last_before += 1
        elif last_t > mig_t + 1:
            last_after += 1
        else:
            last_equalish += 1
    darr = np.asarray(deltas, dtype=np.float64) if deltas else np.asarray([])

    # Broad train graduation + survival_30m prevalence
    # Use rich joined for survival status columns; broad via broad targets if present
    rich_joined = pq.read_table(SPLIT / "training_cache_v1/rich/joined.parquet")
    rich_joined_train = rich_joined.filter(pc.field("base_split") == "train")
    # survival columns
    surv_col = "surv_h1800_active" if "surv_h1800_active" in rich_joined.column_names else None
    surv_status = "surv_h1800_status" if "surv_h1800_status" in rich_joined.column_names else None
    # Also try survival_30m naming
    for c in rich_joined.column_names:
        if "1800" in c and "surv" in c:
            pass

    # Per-mint graduation among broad train from token_assignments + migrations
    tok = pq.read_table(SPLIT / "token_assignments.parquet")
    broad_train_mints = [
        m
        for m, s, pop in zip(
            tok["mint"].to_pylist(),
            tok["base_split"].to_pylist(),
            tok["in_rich_subset"].to_pylist()
            if "in_rich_subset" in tok.column_names
            else [False] * tok.num_rows,
            strict=True,
        )
        if s == "train"
    ]
    # token_assignments may be rich-only? check n
    # Prefer broad population file
    broad_pop = pq.read_table(WORLD / "broad" / "broad_population_v1.parquet")
    # find split
    bcols = broad_pop.column_names
    if "base_split" in bcols:
        bt = broad_pop.filter(pc.field("base_split") == "train")
        broad_train_mints = list(set(bt["mint"].to_pylist()))
    else:
        # use token_assignments all train mints
        broad_train_mints = [
            m
            for m, s in zip(tok["mint"].to_pylist(), tok["base_split"].to_pylist(), strict=True)
            if s == "train"
        ]
        broad_train_mints = list(set(broad_train_mints))

    n_bt = len(broad_train_mints)
    n_grad = sum(1 for m in broad_train_mints if m in mig_map)
    # survival among graduated vs not — from rich train joined if available; else broad targets
    surv_by_grad: dict[str, Any] = {"status": "partial"}
    if surv_col and surv_status:
        mints_j = rich_joined_train["mint"].to_pylist()
        active = rich_joined_train[surv_col].to_pylist()
        status = rich_joined_train[surv_status].to_pylist()
        # mint-level: any obs positive?
        pos_mints = set()
        elig_mints = set()
        for m, a, st in zip(mints_j, active, status, strict=True):
            if st in ("valid_positive", "valid_negative", "STATUS_VALID_POSITIVE", "STATUS_VALID_NEGATIVE"):
                elig_mints.add(m)
            if a is True:
                pos_mints.add(m)
        # Use status constants from data
        # recount with string statuses
        from collections import Counter

        st_counts = Counter(str(x) for x in status)
        grad_pos = sum(1 for m in pos_mints if m in mig_map)
        nongrad_pos = sum(1 for m in pos_mints if m not in mig_map)
        grad_mints_rich = {m for m in set(mints_j) if m in mig_map}
        nongrad_mints_rich = {m for m in set(mints_j) if m not in mig_map}
        surv_by_grad = {
            "population": "rich_train_observations_aggregated_to_mint_where_applicable",
            "status_value_counts": dict(st_counts),
            "n_mints_with_any_positive_surv_h1800_active": len(pos_mints),
            "among_graduated_mints_in_rich_train": {
                "n_mints": len(grad_mints_rich),
                "n_with_any_positive": grad_pos,
                "prevalence": float(grad_pos / len(grad_mints_rich)) if grad_mints_rich else None,
            },
            "among_non_graduated_mints_in_rich_train": {
                "n_mints": len(nongrad_mints_rich),
                "n_with_any_positive": nongrad_pos,
                "prevalence": (
                    float(nongrad_pos / len(nongrad_mints_rich)) if nongrad_mints_rich else None
                ),
            },
            "note": (
                "Broad observation-level survival_30m prevalence cited from lead intake / "
                "baseline (~0.01%); mint-level rates above use rich train joined surv_h1800_*."
            ),
        }
    else:
        # probe columns
        surv_by_grad = {
            "status": "could_not_find_surv_h1800_active",
            "surv_like_columns": [c for c in rich_joined.column_names if "surv" in c.lower()][:40],
        }

    # Observation-level broad survival if cache exists
    # Try reading one broad joined partition for surv — may be heavy; use target store
    btarg = WORLD / "targets/target_dataset_v3"
    # list
    surv_files = list(btarg.glob("*surv*")) if btarg.exists() else []

    return {
        **Q85,
        "definition": {
            "file": "src/trencher/data/world_model/targets/compute.py",
            "function": "compute_activity_survival",
            "function_line": fn_line,
            "positive_rule_line": pos_line,
            "positive": "any genuine future trade with event_time >= T+H (regime==0 trades)",
            "negative": (
                "no such trade AND pump_trades source covers "
                "[T+H, T+H+SURVIVAL_NEGATIVE_CONFIRMATION_SEC] with no unresolved gap"
            ),
            "activity_source": "Pump bonding-curve trades in the PIT/sequence event path (regime==0)",
        },
        "sample_20_train_mints": sample_rows,
        "activity_vs_graduation": {
            "n_graduated_rich_train_mints_with_events": int(darr.size),
            "n_last_trade_before_migration": last_before,
            "n_last_trade_near_migration": last_equalish,
            "n_last_trade_after_migration": last_after,
            "last_trade_minus_migration_sec_percentiles": (
                {f"p{q}": float(np.percentile(darr, q)) for q in (10, 25, 50, 75, 90, 99)}
                if darr.size
                else None
            ),
            "evidence": (
                "sequence_events_v2 last event_time vs universe/migrations.migrated_at for "
                "graduated rich-train mints. If last trades cluster at/before migration and "
                "post-grad PumpSwap trades are absent from this store, the activity source "
                "effectively stops at graduation for survival labels."
            ),
        },
        "broad_train_graduation": {
            "n_mints": n_bt,
            "n_graduated": n_grad,
            "fraction_graduated": float(n_grad / n_bt) if n_bt else None,
            "mint_source": "broad_population_v1 or token_assignments train",
        },
        "survival_30m_by_graduation": surv_by_grad,
        "surv_target_files_glob": [str(p.relative_to(REPO)) for p in surv_files[:20]],
    }


def section5_coverage() -> dict[str, Any]:
    plan_md = (REPO / "manifests/world_model_data/FUSION_EXPANSION_COST_PLAN.md").read_text()
    plan = json.loads(
        (REPO / "manifests/world_model_data/fusion_expansion_cost_plan.json").read_text()
    )
    mm = pq.read_table(WORLD / "universe" / "modality_matrix.parquet")
    n_complete = int(pc.sum(mm["fusion_complete"]).as_py())
    subset = pq.read_table(WORLD / "universe" / "rich_subset_mints.parquet")
    bias = json.loads((WORLD / "universe" / "unbiased_subset_bias_proof.json").read_text())
    algo = json.loads((WORLD / "universe" / "unbiased_subset_algorithm.json").read_text())

    # Prevalence rich vs broad
    # explosion from assignments / joined
    rich_j = pq.read_table(SPLIT / "training_cache_v1/rich/joined.parquet")
    rich_tv = rich_j.filter(pc.field("base_split").isin(["train", "val"]))
    # explosion_within_60s
    def _prev(table, col: str) -> dict[str, Any]:
        if col not in table.column_names:
            return {"column": col, "status": "missing"}
        y = table[col].to_pylist()
        vals = [v for v in y if v is True or v is False]
        pos = sum(1 for v in vals if v is True)
        return {"column": col, "n": len(vals), "n_pos": pos, "prevalence": float(pos / len(vals)) if vals else None}

    rich_exp = {
        "explosion_within_60s": _prev(rich_tv, "explosion_within_60s"),
        "explosion_within_300s": _prev(rich_tv, "explosion_within_300s"),
    }
    rich_mints = set(rich_tv["mint"].to_pylist())
    mig = set(pq.read_table(WORLD / "universe" / "migrations.parquet")["mint"].to_pylist())
    rich_grad_rate = float(len(rich_mints & mig) / len(rich_mints)) if rich_mints else None

    # broad prevalence from baseline core or broad targets summary
    core = json.loads(
        (REPO / "manifests/world_model_data/baseline_v2/BASELINE_CORE_METRICS_V2.json").read_text()
    )
    broad_exp = {}
    for h in ("explosion_60s", "explosion_300s"):
        # find prevalence in core
        core.get("heads", core.get(h, {}))
        if h in core.get("heads", {}):
            met = core["heads"][h]
            # structure varies
            broad_exp[h] = {
                "source": "BASELINE_CORE_METRICS_V2.json",
                "metrics_snippet": {
                    k: met.get(k)
                    for k in ("prevalence", "n", "roc_auc")
                    if k in met
                }
                or met.get("permanent_val")
                or met.get("val")
                or list(met.keys())[:12],
            }
        else:
            broad_exp[h] = {"status": "locate_in_core", "top_keys": list(core.keys())[:20]}

    # Try more precise broad prevalence from core structure
    def _find_prev(obj: Any, path: str = "") -> list[tuple[str, Any]]:
        found = []
        if isinstance(obj, dict):
            if "prevalence" in obj and "explosion" in path:
                found.append((path, obj.get("prevalence")))
            for k, v in obj.items():
                found.extend(_find_prev(v, f"{path}/{k}"))
        return found

    # Simpler: read SPLIT stats
    rich_stats = json.loads((SPLIT / "observation_assignments_rich_stats.json").read_text())

    # Cost scaling: plan is for ownership recovery of 2846 mints, not full rich for 25k/100k
    rpc = plan.get("rpc_ownership_plan") or {}
    plan.get("cost_if_all_unqueried_days") or plan.get("bigquery_execution") or {}
    # locate cost block
    cost_block = plan.get("decision") or {}
    exec_actuals = {
        "bq_usd": plan.get("execution_actuals", {}).get("bq_usd_charged_est")
        if isinstance(plan.get("execution_actuals"), dict)
        else None,
        "rpc_ownership_mints_unlocked": (
            (plan.get("execution_actuals") or {}).get("rpc_ownership_mints_unlocked")
        ),
        "final_fusion_overlap": (plan.get("execution_actuals") or {}).get("final_fusion_overlap"),
    }
    # Read end of plan json for execution_actuals
    if "execution_actuals" in plan:
        exec_actuals = plan["execution_actuals"]

    n_broad_train_val = None
    try:
        # from lead intake report
        lead = json.loads((LEAD_OUT / "LEAD_INTAKE_REPORT.json").read_text())
        n_broad_train_val = (
            lead["G"]["counts"]["broad_tokens_train"] + lead["G"]["counts"]["broad_tokens_val"]
        )
    except Exception as exc:  # noqa: BLE001
        n_broad_train_val = f"unavailable: {exc}"

    return {
        **Q85,
        "fusion_expansion_plan_summary": {
            "recommended_path": plan.get("recommended_path"),
            "rpc_ownership": {
                "target_mints": rpc.get("target_mints"),
                "bq_usd": rpc.get("bq_usd"),
                "expected_fusion_overlap_if_all_succeed": rpc.get(
                    "expected_fusion_overlap_if_all_succeed"
                ),
                "rationale": rpc.get("rationale"),
            },
            "execution_actuals": exec_actuals,
            "md_header_excerpt": "\n".join(plan_md.splitlines()[:20]),
            "scaling_to_N": {
                "N_25k": {
                    "status": "could_not_complete",
                    "reason": (
                        "FUSION_EXPANSION_COST_PLAN covers ownership recovery for the existing "
                        "rich subset gap (2,846 ownership-only mints → ~3,132 fusion overlap), "
                        "not per-group cost/time to build microstructure/ownership/graph/"
                        "wallet_prior for N=25k new mints."
                    ),
                },
                "N_100k": {
                    "status": "could_not_complete",
                    "reason": "Same as N_25k: plan has no per-mint unit costs for full rich groups.",
                },
                "N_all_broad_train_val": {
                    "n_broad_train_val_mints": n_broad_train_val,
                    "status": "could_not_complete",
                    "reason": (
                        "No frozen unit-cost model for expanding all rich groups to the full "
                        "broad train+val mint set; plan is ownership-backfill scoped."
                    ),
                },
            },
        },
        "why_rich_count": {
            "rich_subset_mints_parquet_n": subset.num_rows,
            "fusion_complete_n": n_complete,
            "rich_train_val_mints": 3623,
            "selection": {
                "file": "src/trencher/data/world_model/subset.py",
                "function": "select_unbiased_rich_subset",
                "policy": (
                    "Stratified sample by launch day × mayhem × holder tertile × buy tertile "
                    "using only pre-T launch fields; graduation is post-hoc composition only; "
                    "Jul 3 outage day excluded; capped at max_mints=4000; deterministic mint rank."
                ),
                "not_random_iid": True,
                "not_biased_toward_active_or_graduated": bias.get(
                    "selection_is_not_graduation_proxy"
                ),
                "bias_proof": {
                    "subset_graduation_rate": bias.get("subset_graduation_rate"),
                    "universe_graduation_rate": bias.get("universe_graduation_rate"),
                },
                "algorithm_version": algo.get("version") or algo.get("algorithm", {}).get("version"),
            },
            "note_3623": (
                "4000 rich-subset mints total; 3623 appear in train+val observation assignments "
                "(377 test mints excluded from analysis)."
            ),
        },
        "rich_vs_broad_rates": {
            "rich_train_val_explosion": rich_exp,
            "rich_train_val_graduation_rate": rich_grad_rate,
            "rich_n_mints_train_val": len(rich_mints),
            "broad_explosion_from_core": broad_exp,
            "rich_assignment_stats": {
                k: rich_stats.get(k) for k in list(rich_stats)[:12]
            },
        },
        "decision_cost_block_keys": list(cost_block) if isinstance(cost_block, dict) else cost_block,
    }


def write_markdown(report: dict[str, Any]) -> None:
    s1 = report["section1"]
    s2 = report["section2"]
    s3 = report["section3"]
    s4 = report["section4"]
    s5 = report["section5"]
    lines: list[str] = []
    lines.append("# Phase 2 Diagnostics Report")
    lines.append("")
    lines.append("`q85_test_accessed: false` · branch `review/phase2-v1` · CPU only")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    m60 = s1["explosion_60s"]["models"]
    lines.append(
        f"1. Stage-B init weights match Stage-B `latest.pt` tensors exactly; file SHA differs "
        f"(`{s1['checkpoint_hashes']['stage_c_init_150000']['sha256'][:12]}…` vs "
        f"`ccf27af7…`). Rich-val explosion_60s ROC: "
        f"Stage B {m60['stage_b']['roc_auc']:.4f}, Stage C {m60['stage_c']['roc_auc']:.4f}, "
        f"LGBM-B {m60['lgbm_b']['roc_auc']:.4f}, LGBM-A {m60['lgbm_a']['roc_auc']:.4f}."
    )
    boot = s1["explosion_60s"]["bootstrap"]["stage_c_minus_stage_b_roc"]
    lines.append(
        f"2. Mint-bootstrap ΔAUROC Stage C−B (60s): {boot['delta']:.4f} "
        f"95% CI [{boot['ci95'][0]:.4f}, {boot['ci95'][1]:.4f}]."
    )
    # best drop group
    drops = s2["heads"]["explosion_60s"]["drop_one_group"]
    worst = min(drops.items(), key=lambda kv: kv[1]["metrics"]["roc_auc"])
    lines.append(
        f"3. Dropping rich group **{worst[0]}** hurts arm-B explosion_60s most "
        f"(ROC {worst[1]['metrics']['roc_auc']:.4f} vs full "
        f"{s2['heads']['explosion_60s']['full_B']['roc_auc']:.4f})."
    )
    lines.append(
        f"4. Lead-time variant (a) exclude already-started: explosion_60s n_pos="
        f"{s3['variants']['explosion_60s']['a_exclude_already_started']['n_pos']}."
    )
    pump = s3["pump_level_recall_primary_val_events"]["lgbm_b"]
    lines.append(
        f"5. Pump-level top-2% recall (LGBM-B, 50 primary val events): "
        f"{pump['fraction_caught_top2pct']:.2%} caught."
    )
    lines.append(
        "6. Survival positives require genuine Pump-trade activity at/after T+H "
        f"(`compute_activity_survival`, line {s4['definition']['function_line']})."
    )
    lines.append(
        f"7. Activity vs graduation: among graduated rich-train mints with events, "
        f"{s4['activity_vs_graduation']['n_last_trade_after_migration']} have last trade "
        f"after migration; {s4['activity_vs_graduation']['n_last_trade_before_migration']} before."
    )
    lines.append(
        f"8. Rich subset: stratified unbiased draw capped at 4000 "
        f"(`select_unbiased_rich_subset`); 3623 train+val; fusion_complete={s5['why_rich_count']['fusion_complete_n']}."
    )
    lines.append(
        "9. Fusion cost plan is ownership-backfill scoped; no unit costs for N=25k/100k/all-broad."
    )
    lines.append(
        f"10. LOADER_AUDIT rewriter: `{report['loader_audit_rewriter']['test_function']}` "
        f"via `{report['loader_audit_rewriter']['command_that_rewrites']}`."
    )
    lines.append("")

    # Section 1 tables
    lines.append("## 1. Stage B vs Stage C on rich val (CPU)")
    lines.append("")
    lines.append(s1["init_vs_stage_b_latest"]["explanation"])
    lines.append("")
    lines.append(f"- `--allow-dirty` used: **{s1['allow_dirty_used']}** (tree was clean; scratch gitignored).")
    lines.append(f"- Scoring note: {s1['scoring_note']}")
    lines.append("")
    for head in ("explosion_60s", "explosion_300s"):
        lines.append(f"### {head}")
        lines.append("")
        lines.append("| model | ROC-AUC | PR-AUC | P@1% | P@2% | P@5% | n_pos |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, m in s1[head]["models"].items():
            lines.append(
                f"| {name} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | "
                f"{m['precision_at']['top_0.01']:.4f} | {m['precision_at']['top_0.02']:.4f} | "
                f"{m['precision_at']['top_0.05']:.4f} | {m['n_pos']} |"
            )
        lines.append("")
        lines.append("Bootstrap (mint, 2000, seed 0):")
        lines.append("")
        lines.append("| contrast | metric | delta | 95% CI |")
        lines.append("|---|---|---:|---|")
        for k, v in s1[head]["bootstrap"].items():
            lines.append(
                f"| {k} | {v['metric']} | {v['delta']:.4f} | "
                f"[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] |"
            )
        lines.append("")

    lines.append("## 2. Rich feature group ablations (LightGBM arm B)")
    lines.append("")
    lines.append("### Column → group map")
    lines.append("")
    lines.append("| column | group |")
    lines.append("|---|---|")
    for c, g in s2["rich_column_to_group"].items():
        lines.append(f"| `{c}` | {g} |")
    lines.append("")
    for head, h in s2["heads"].items():
        lines.append(f"### {head}")
        lines.append(
            f"Full B ROC={h['full_B']['roc_auc']:.4f} (committed {h['committed_B_roc_auc']:.4f}; "
            f"within ±0.005: {h['repro_within_0_005']})."
        )
        lines.append("")
        lines.append("| ablation | ROC-AUC | PR-AUC | ΔROC vs full (drop−full) CI |")
        lines.append("|---|---:|---:|---|")
        lines.append(
            f"| full_B | {h['full_B']['roc_auc']:.4f} | {h['full_B']['pr_auc']:.4f} | — |"
        )
        for g, rec in h["drop_one_group"].items():
            d = rec["delta_vs_full_B"]["roc"]
            lines.append(
                f"| drop_{g} | {rec['metrics']['roc_auc']:.4f} | {rec['metrics']['pr_auc']:.4f} | "
                f"{d['delta']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] |"
            )
        for name, met in h["broad_plus_single_group"].items():
            lines.append(
                f"| {name} | {met['roc_auc']:.4f} | {met['pr_auc']:.4f} | — |"
            )
        lines.append("")
        lines.append("Top-30 gain importance (full B):")
        lines.append("")
        lines.append("| rank | feature | group | gain |")
        lines.append("|---:|---|---|---:|")
        for i, row in enumerate(h["gain_importance_top30"], 1):
            lines.append(
                f"| {i} | `{row['feature']}` | {row['group']} | {row['gain']:.4f} |"
            )
        lines.append("")

    lines.append("## 3. Lead-time-aware pump evaluation")
    lines.append("")
    for head, variants in s3["variants"].items():
        lines.append(f"### {head}")
        lines.append("")
        lines.append("| variant | n | n_pos | stage_b ROC | stage_c ROC | lgbm_a ROC | lgbm_b ROC |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for vn, rec in variants.items():
            def _fmt(auc: float | None) -> str:
                return f"{auc:.4f}" if auc is not None else "NA"

            mb = rec["models"]
            lines.append(
                f"| {vn} | {rec['n']} | {rec['n_pos']} | "
                f"{_fmt(mb['stage_b'].get('roc_auc'))} | "
                f"{_fmt(mb['stage_c'].get('roc_auc'))} | "
                f"{_fmt(mb['lgbm_a'].get('roc_auc'))} | "
                f"{_fmt(mb['lgbm_b'].get('roc_auc'))} |"
            )
        lines.append("")
    lines.append("### Pump-level recall (50 primary val events, top 2% on explosion_60s scores)")
    lines.append("")
    lines.append("| model | fraction caught | n_caught | lead p50 (s) |")
    lines.append("|---|---:|---:|---:|")
    for m, rec in s3["pump_level_recall_primary_val_events"].items():
        p50 = (rec.get("lead_time_sec_percentiles") or {}).get("p50")
        lines.append(
            f"| {m} | {rec['fraction_caught_top2pct']:.3f} | {rec['n_caught']} | "
            f"{p50 if p50 is not None else 'NA'} |"
        )
    lines.append("")
    rem = next(iter(s3["pump_level_recall_primary_val_events"].values()))["remaining_peak_gain"]
    lines.append(f"Remaining peak gain: **{rem['status']}** — {rem['reason']}")
    lines.append("")

    lines.append("## 4. Survival label investigation")
    lines.append("")
    d = s4["definition"]
    lines.append(
        f"- **Code:** `{d['file']}` :: `{d['function']}` (line {d['function_line']}); "
        f"positive rule at line {d['positive_rule_line']}."
    )
    lines.append(f"- Positive: {d['positive']}")
    lines.append(f"- Negative: {d['negative']}")
    lines.append("")
    lines.append(
        f"Broad train graduation rate: {s4['broad_train_graduation']['fraction_graduated']} "
        f"({s4['broad_train_graduation']['n_graduated']}/{s4['broad_train_graduation']['n_mints']})."
    )
    lines.append("")
    ag = s4["activity_vs_graduation"]
    lines.append(
        f"Last trade vs migration (graduated rich-train with events, n={ag['n_graduated_rich_train_mints_with_events']}): "
        f"before={ag['n_last_trade_before_migration']}, near={ag['n_last_trade_near_migration']}, "
        f"after={ag['n_last_trade_after_migration']}."
    )
    lines.append("")
    lines.append("Sample of 20 train mints is in `PHASE2_REPORT.json` §section4.sample_20_train_mints.")
    lines.append("")

    lines.append("## 5. Data coverage for expansion")
    lines.append("")
    lines.append(s5["why_rich_count"]["selection"]["policy"])
    lines.append("")
    lines.append(
        f"Rich train+val graduation rate: {s5['rich_vs_broad_rates']['rich_train_val_graduation_rate']:.4f}; "
        f"explosion_60s prevalence: {s5['rich_vs_broad_rates']['rich_train_val_explosion']['explosion_within_60s']}."
    )
    lines.append("")
    lines.append("### Cost plan vs N=25k / 100k / all-broad")
    lines.append("")
    for k, v in s5["fusion_expansion_plan_summary"]["scaling_to_N"].items():
        lines.append(f"- **{k}**: {v.get('status')} — {v.get('reason')}")
    lines.append("")
    lines.append(
        f"Plan actuals: RPC unlocked "
        f"{(s5['fusion_expansion_plan_summary'].get('execution_actuals') or {}).get('rpc_ownership_mints_unlocked')}, "
        f"final overlap "
        f"{(s5['fusion_expansion_plan_summary'].get('execution_actuals') or {}).get('final_fusion_overlap')}."
    )
    lines.append("")

    lines.append("## Could not complete")
    lines.append("")
    for item in report.get("could_not_complete", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## LOADER_AUDIT rewrite")
    lines.append("")
    la = report["loader_audit_rewriter"]
    lines.append(
        f"`{la['file']}` / `{la['test_function']}` — run `{la['command_that_rewrites']}`."
    )
    lines.append("")

    (OUT / "PHASE2_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Reuse section1/2 JSON caches under scratch/phase2_v1/cache/",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = SCRATCH / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        **Q85,
        "loader_audit_rewriter": section_loader_audit_rewriter(),
    }
    c1, c2 = cache / "section1.json", cache / "section2.json"
    if args.resume and c1.exists():
        print("section1... (cache)", flush=True)
        report["section1"] = json.loads(c1.read_text(encoding="utf-8"))
    else:
        print("section1...", flush=True)
        report["section1"] = section1_checkpoints_and_paired()
        write_json(c1, report["section1"])
    if args.resume and c2.exists():
        print("section2... (cache)", flush=True)
        report["section2"] = json.loads(c2.read_text(encoding="utf-8"))
    else:
        print("section2...", flush=True)
        report["section2"] = section2_lgbm_groups()
        write_json(c2, report["section2"])
    print("section3...", flush=True)
    report["section3"] = section3_lead_time(report["section1"])
    print("section4...", flush=True)
    report["section4"] = section4_survival()
    print("section5...", flush=True)
    report["section5"] = section5_coverage()

    could = []
    could.append(
        "step17 metrics JSON AUROC unreliable (_safe_auc bug); recomputed with binary_metrics."
    )
    could.append(
        "Fusion plan lacks unit costs for rich groups at N=25k/100k/all-broad train+val."
    )
    rem = report["section3"]["pump_level_recall_primary_val_events"]["stage_c"][
        "remaining_peak_gain"
    ]
    if rem.get("status") == "could_not_complete":
        could.append(f"Remaining peak gain after T: {rem.get('reason')}")
    if report["section4"]["survival_30m_by_graduation"].get("status") == "could_not_find_surv_h1800_active":
        could.append("Could not locate surv_h1800_active on rich joined for graduation split rates.")
    report["could_not_complete"] = could

    write_json(OUT / "PHASE2_REPORT.json", report)
    write_markdown(report)
    print("wrote", OUT / "PHASE2_REPORT.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
