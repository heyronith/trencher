"""RICH-additive ablation: broad-only vs broad+rich on the same rich observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from trencher.baselines.dataset import load_joined_matrices
from trencher.baselines.feature_spec import (
    all_feature_names,
    broad_feature_names,
    rich_additive_feature_names,
)
from trencher.baselines.metrics import binary_metrics, regression_metrics
from trencher.baselines.targets import (
    CORE_TARGET_SUITE,
    extract_binary_labels,
    extract_regression_labels,
)
from trencher.baselines.train_trees import train_lightgbm_head
from trencher.data.world_model.io import ensure_dir, utc_now, write_json

ABLATION_HEADS = (
    "explosion_60s",
    "explosion_300s",
    "barrier_u25_l15_5m",
    "barrier_u100_l30_1h",
    "return_5m",
    "return_1h",
)


def _mask_rich(meta: list[dict[str, Any]]) -> np.ndarray:
    return np.array([bool(m.get("in_rich_subset")) for m in meta], dtype=bool)


def run_rich_ablation_v2(
    *,
    world_root: Path,
    split_root: Path | None = None,
    artifacts_root: Path,
    seed: int = 0,
    max_partitions: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Compare broad-only vs broad+rich features on identical rich rows."""
    world_root = Path(world_root)
    split_root = Path(split_root or world_root / "splits" / "split_v1")
    out_dir = ensure_dir(Path(artifacts_root) / "rich_ablation_v2")

    broad_names = broad_feature_names()
    rich_names = all_feature_names(include_rich=True)
    assert len(broad_names) == 124
    assert len(rich_names) == 124 + len(rich_additive_feature_names())

    # Prefer feature cache that includes rich columns when present; otherwise
    # broad cache still allows broad-only arm (rich arm skipped with note).
    train_b = load_joined_matrices(
        world_root=world_root,
        split_root=split_root,
        split="train",
        feature_names=broad_names,
        max_partitions=max_partitions or (2 if smoke else None),
        include_rich_features=False,
    )
    val_b = load_joined_matrices(
        world_root=world_root,
        split_root=split_root,
        split="val",
        feature_names=broad_names,
        max_partitions=max_partitions or (2 if smoke else None),
        include_rich_features=False,
    )
    train_r = load_joined_matrices(
        world_root=world_root,
        split_root=split_root,
        split="train",
        feature_names=rich_names,
        max_partitions=max_partitions or (2 if smoke else None),
        include_rich_features=True,
    )
    val_r = load_joined_matrices(
        world_root=world_root,
        split_root=split_root,
        split="val",
        feature_names=rich_names,
        max_partitions=max_partitions or (2 if smoke else None),
        include_rich_features=True,
    )

    tr_mask = _mask_rich(train_b["meta"])
    va_mask = _mask_rich(val_b["meta"])
    # Align rich masks if lengths match
    if train_r["n"] == train_b["n"]:
        tr_mask = tr_mask & _mask_rich(train_r["meta"])
    if val_r["n"] == val_b["n"]:
        va_mask = va_mask & _mask_rich(val_r["meta"])

    payload: dict[str, Any] = {
        "ablation_version": "BASELINE_RICH_ABLATION_V2",
        "generated_at": utc_now(),
        "feature_dims": {
            "broad": len(broad_names),
            "broad_plus_rich": len(rich_names),
            "rich_additive": len(rich_additive_feature_names()),
        },
        "rich_train_n": int(tr_mask.sum()),
        "rich_val_n": int(va_mask.sum()),
        "heads": {},
        "note": "Same rich observation ids for arms A (broad-only) and B (broad+rich).",
    }

    specs = [s for s in CORE_TARGET_SUITE if s.head_id in ABLATION_HEADS]
    for spec in specs:
        if spec.kind == "binary":
            y_tr, elig_tr, _ = extract_binary_labels(train_b["targets"], spec)
            y_va, elig_va, _ = extract_binary_labels(val_b["targets"], spec)
        elif spec.kind == "regression":
            y_tr, elig_tr, _ = extract_regression_labels(train_b["targets"], spec)
            y_va, elig_va, _ = extract_regression_labels(val_b["targets"], spec)
        else:
            # barriers: reuse binary upper-first via multiclass skipped in tiny ablation
            continue

        tr = tr_mask & elig_tr
        va = va_mask & elig_va
        if tr.sum() < 30 or va.sum() < 10:
            payload["heads"][spec.head_id] = {
                "skipped": True,
                "train_n": int(tr.sum()),
                "val_n": int(va.sum()),
            }
            continue

        arms = {}
        for arm, Xtr_src, Xva_src in (
            ("A_broad_only", train_b["X"][:, : len(broad_names)], val_b["X"][:, : len(broad_names)]),
            (
                "B_broad_plus_rich",
                train_r["X"] if train_r["n"] == train_b["n"] else train_b["X"],
                val_r["X"] if val_r["n"] == val_b["n"] else val_b["X"],
            ),
        ):
            if arm.startswith("B") and train_r["feature_names"] == train_b["feature_names"]:
                arms[arm] = {"skipped": True, "reason": "rich feature columns unavailable in cache"}
                continue
            art = train_lightgbm_head(
                X_train=Xtr_src[tr],
                y_train=y_tr[tr],
                X_val=Xva_src[va],
                y_val=y_va[va],
                kind="binary" if spec.kind == "binary" else "regression",
                out_dir=ensure_dir(out_dir / spec.head_id / arm),
                tag=f"rich_ablation {arm} {spec.head_id}",
                num_boost_round=80 if smoke else 200,
                early_stopping_rounds=30,
                seed=seed,
                use_gpu=False,
            )
            pred = art["raw_val_pred"]
            if spec.kind == "binary":
                m = binary_metrics(y_va[va], pred)
            else:
                m = regression_metrics(y_va[va], pred)
            arms[arm] = {"metrics": m, "runtime_sec": art["runtime_sec"]}

        uplift = {}
        if (
            "A_broad_only" in arms
            and "B_broad_plus_rich" in arms
            and "metrics" in arms["A_broad_only"]
            and "metrics" in arms["B_broad_plus_rich"]
        ):
            a = arms["A_broad_only"]["metrics"]
            b = arms["B_broad_plus_rich"]["metrics"]
            for k in ("pr_auc", "roc_auc", "mae", "rmse", "spearman"):
                if a.get(k) is not None and b.get(k) is not None:
                    uplift[k] = float(b[k] - a[k])
        payload["heads"][spec.head_id] = {
            "train_n": int(tr.sum()),
            "val_n": int(va.sum()),
            "arms": arms,
            "uplift_B_minus_A": uplift,
        }

    write_json(out_dir / "BASELINE_RICH_ABLATION_V2.json", payload)
    return payload
