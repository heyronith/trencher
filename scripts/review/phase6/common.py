"""Shared paths and helpers for Phase-6."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.review.lead_intake_lib import (  # noqa: E402
    BOOT_SEED,
    N_BOOT,
    binary_auc_ap,
    topk_precision_recall,
    write_json,
)
from scripts.review.phase4.fast_boot import mint_bootstrap_delta  # noqa: E402
from scripts.review.phase5.common import metrics_block as _p5_metrics_block  # noqa: E402

WORLD = REPO / "data" / "external" / "world_model"
SPLIT = WORLD / "splits" / "split_v1"
SCRATCH = REPO / "scratch" / "phase6_paper_v1"
OUT = REPO / "manifests" / "world_model_data" / "evaluation" / "phase6_paper_v1"
PHASE4_PREDS = REPO / "scratch" / "phase4_clean_v1" / "preds" / "trees_broad_permanent_val.parquet"
PHASE5_SCRATCH = REPO / "scratch" / "phase5_trees_v1"
BROAD_EVENTS = WORLD / "broad" / "broad_targets_v1" / "explosion_events_v1.parquet"
Q85 = {"q85_test_accessed": False}
TOP_FRACS = [0.001, 0.005, 0.01, 0.02]
SEED = 0
HEADS = ("explosion_60s", "explosion_300s")
VARIANTS = ("all", "a_exclude_already_started", "c_onset_ge_5s")
DEFINITIONS = {
    "primary": "explosion_primary_v1",
    "conservative": "explosion_conservative_v1",
    "aggressive": "explosion_aggressive_v1",
}


def ensure_dirs() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "cache").mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)


def cached_json(name: str, fn):  # noqa: ANN001
    path = SCRATCH / "cache" / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    val = fn()
    write_json(path, val)
    return val


def cutoffs() -> dict[str, Any]:
    return json.loads((SPLIT / "cutoffs.json").read_text(encoding="utf-8"))


def val_window() -> tuple[float, float]:
    c = cutoffs()
    return float(c["quantiles_unix"]["q70"]), float(c["quantiles_unix"]["q85"])


def metrics_block(y: np.ndarray, s: np.ndarray) -> dict[str, Any]:
    return _p5_metrics_block(y, s)


def feature_groups(names: list[str]) -> dict[str, list[int]]:
    """Map feature-group name -> column indices for ablation / SHAP rollup."""
    groups: dict[str, list[int]] = {
        "age": [],
        "masks": [],
        "event_flow": [],
        "curve": [],
        "creator": [],
        "global_context": [],
        "other": [],
    }
    for i, n in enumerate(names):
        if n.startswith("token_age"):
            groups["age"].append(i)
        elif n.startswith("mask_"):
            groups["masks"].append(i)
        elif n.startswith("creator_"):
            groups["creator"].append(i)
        elif n.startswith(("global_", "daily_network_")):
            groups["global_context"].append(i)
        elif any(
            x in n
            for x in (
                "curve_",
                "latest_curve",
                "latest_price",
                "latest_v_",
                "latest_is_buy",
                "reserve_delta",
            )
        ):
            groups["curve"].append(i)
        elif any(
            x in n
            for x in (
                "event_count",
                "buy_count",
                "sell_count",
                "buy_fraction",
                "buy_sol",
                "sell_sol",
                "net_sol",
                "flow_imbalance",
                "sol_mean",
                "sol_std",
                "sol_max",
                "interarrival",
                "burst",
                "intensity",
                "price_return",
                "price_slope",
                "realized_vol",
                "drawdown",
                "price_accel",
            )
        ):
            groups["event_flow"].append(i)
        else:
            groups["other"].append(i)
    return groups


__all__ = [
    "BOOT_SEED",
    "BROAD_EVENTS",
    "DEFINITIONS",
    "HEADS",
    "N_BOOT",
    "OUT",
    "PHASE4_PREDS",
    "PHASE5_SCRATCH",
    "Q85",
    "REPO",
    "SCRATCH",
    "SEED",
    "SPLIT",
    "TOP_FRACS",
    "VARIANTS",
    "WORLD",
    "binary_auc_ap",
    "cached_json",
    "cutoffs",
    "ensure_dirs",
    "feature_groups",
    "metrics_block",
    "mint_bootstrap_delta",
    "topk_precision_recall",
    "val_window",
    "write_json",
]
