"""Shared paths for Phase-7."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.review.lead_intake_lib import write_json  # noqa: E402
from scripts.review.phase5.common import metrics_block as _metrics_block  # noqa: E402
from scripts.review.phase6.boot import mint_bootstrap_delta  # noqa: E402
from scripts.review.phase6.common import (  # noqa: E402
    PHASE4_PREDS,
    PHASE5_SCRATCH,
    SPLIT,
    WORLD,
)

SCRATCH = REPO / "scratch" / "phase7_pretest_v1"
OUT = REPO / "manifests" / "world_model_data" / "evaluation" / "phase7_pretest_v1"
PROTOCOL_V2 = REPO / "manifests" / "world_model_data" / "evaluation" / "TEST_PROTOCOL_V2.md"
TEST_ONCE_DIR = REPO / "manifests" / "world_model_data" / "evaluation" / "test_once_v1"
STAGE_B_PREDS = (
    REPO / "scratch" / "phase4_clean_v1" / "preds" / "stage_b_broad_permanent_val.parquet"
)
Q85 = {"q85_test_accessed": False}
SEED = 0
TOP_FRACS = [0.001, 0.005, 0.01, 0.02]
# Preregistration commit that first introduced TEST_PROTOCOL_V2.md
V2_PREREG_SHA = "3299ee8d9b05286bbac9f9e1a7dc085e8d0de349"
AGE_CURVE_FEATURES = [
    "token_age_seconds",
    "token_age_log1p",
    "latest_v_tok",
    "latest_v_sol",
    "latest_price",
    "latest_curve_pct",
]


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


def metrics_block(y: np.ndarray, s: np.ndarray) -> dict[str, Any]:
    return _metrics_block(y, s)


def core_xgb_fixed(head_id: str) -> int:
    core = json.loads(
        (
            REPO / "manifests/world_model_data/baseline_v2/BASELINE_CORE_METRICS_V2.json"
        ).read_text(encoding="utf-8")
    )
    return int(core["xgboost"]["heads"][head_id]["fixed_iterations"])


__all__ = [
    "AGE_CURVE_FEATURES",
    "OUT",
    "PHASE4_PREDS",
    "PHASE5_SCRATCH",
    "PROTOCOL_V2",
    "Q85",
    "REPO",
    "SCRATCH",
    "SEED",
    "SPLIT",
    "STAGE_B_PREDS",
    "TEST_ONCE_DIR",
    "TOP_FRACS",
    "V2_PREREG_SHA",
    "WORLD",
    "cached_json",
    "core_xgb_fixed",
    "ensure_dirs",
    "metrics_block",
    "mint_bootstrap_delta",
    "write_json",
]
