"""Shared paths for Phase-8 TEST pipeline (TEST still locked)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.review.lead_intake_lib import write_json  # noqa: E402
from scripts.review.phase7.common import (  # noqa: E402
    AGE_CURVE_FEATURES,
    PHASE4_PREDS,
    PROTOCOL_V2,
    STAGE_B_PREDS,
    V2_PREREG_SHA,
)

WORLD = REPO / "data" / "external" / "world_model"
SPLIT = WORLD / "splits" / "split_v1"
SCRATCH = REPO / "scratch" / "phase8_testpipe_v1"
MODELS_DIR = REPO / "scratch" / "test_models_v2"
OUT = REPO / "manifests" / "world_model_data" / "evaluation" / "phase8_testpipe_v1"
MANIFEST_PATH = (
    REPO / "manifests" / "world_model_data" / "evaluation" / "TEST_MODEL_MANIFEST_V2.json"
)
TEST_ONCE_DIR = REPO / "manifests" / "world_model_data" / "evaluation" / "test_once_v1"
STAGE_B_CKPT_SRC = REPO / "scratch" / "phase2_v1" / "ckpts" / "stage_b_latest.pt"
STAGE_B_SHA256 = "ccf27af79bf178f095c70a84e48ad2e573bae8d5cfcfdde9d32fbb0a1f7aa208"
Q85 = {"q85_test_accessed": False}
SEED = 0
HEADS = ("explosion_60s", "explosion_300s")
TWOX_FIXED_ITERS = 64  # phase7 median WF
AGE_CURVE_FIXED = {"explosion_60s": 50, "explosion_300s": 50}
LOGISTIC_C = 0.01  # phase6/7 winner


def ensure_dirs() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "cache").mkdir(parents=True, exist_ok=True)
    (SCRATCH / "dry_run_val").mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_ids(ids: Any) -> str:
    """Stable hash of observation_id array (training rows identity)."""
    import numpy as np

    arr = np.asarray(ids, dtype=object)
    payload = b"\n".join(str(x).encode("utf-8") for x in arr.tolist())
    return sha256_bytes(payload)


def core_fixed(family: str, head_id: str) -> int:
    core = json.loads(
        (
            REPO / "manifests/world_model_data/baseline_v2/BASELINE_CORE_METRICS_V2.json"
        ).read_text(encoding="utf-8")
    )
    return int(core[family]["heads"][head_id]["fixed_iterations"])


def cached_json(name: str, fn):  # noqa: ANN001
    path = SCRATCH / "cache" / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    val = fn()
    write_json(path, val)
    return val


__all__ = [
    "AGE_CURVE_FEATURES",
    "AGE_CURVE_FIXED",
    "HEADS",
    "LOGISTIC_C",
    "MANIFEST_PATH",
    "MODELS_DIR",
    "OUT",
    "PHASE4_PREDS",
    "PROTOCOL_V2",
    "Q85",
    "REPO",
    "SCRATCH",
    "SEED",
    "SPLIT",
    "STAGE_B_CKPT_SRC",
    "STAGE_B_PREDS",
    "STAGE_B_SHA256",
    "TEST_ONCE_DIR",
    "TWOX_FIXED_ITERS",
    "V2_PREREG_SHA",
    "WORLD",
    "cached_json",
    "core_fixed",
    "ensure_dirs",
    "sha256_file",
    "sha256_ids",
    "write_json",
]
