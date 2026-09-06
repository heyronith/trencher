#!/usr/bin/env bash
# Launch three-phase world-model training from a staged directory.
# Does NOT embed credentials. Review before running on a paid GPU.
set -euo pipefail

STAGE="${TRENCHER_LOCAL_STAGE:-/content/trencher_stage}"
CONFIG="${TRENCHER_WM_CONFIG:-${STAGE}/configs/world_model/world_model_candidate_v1.yaml}"
PHASE="${TRENCHER_WM_PHASE:-all}"
DEVICE="${TRENCHER_WM_DEVICE:-cuda}"
CKPT_DIR="${TRENCHER_WM_CKPT_DIR:-${STAGE}/checkpoints}"

echo "[launch_training] STAGE=${STAGE}"
echo "[launch_training] CONFIG=${CONFIG}"
echo "[launch_training] PHASE=${PHASE}"
echo "[launch_training] DEVICE=${DEVICE}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  echo "Run scripts/colab/stage_from_gcs.py first." >&2
  exit 1
fi

mkdir -p "${CKPT_DIR}"

# Entry point is intentionally explicit — production runners should wrap this
# with their scheduler. This script never downloads credentials.
python - <<'PY'
import os
from pathlib import Path

import torch

from trencher.world_model.checkpoint import save_checkpoint
from trencher.world_model.config import load_world_model_config
from trencher.world_model.model import WorldModel, count_parameters
from trencher.world_model.progress import ProgressEmitter
from trencher.world_model.trainer import ThreePhaseTrainer

config_path = os.environ.get("TRENCHER_WM_CONFIG") or (
    Path(os.environ.get("TRENCHER_LOCAL_STAGE", "/content/trencher_stage"))
    / "configs/world_model/world_model_candidate_v1.yaml"
)
device = os.environ.get("TRENCHER_WM_DEVICE", "cuda")
if device == "cuda" and not torch.cuda.is_available():
    print("CUDA requested but unavailable; refusing silent CPU fallback for production launch.")
    raise SystemExit(2)

cfg = load_world_model_config(config_path)
model = WorldModel(cfg)
print(f"parameters={count_parameters(model):,}")
print(f"config_hash={cfg.config_hash()}")
print("Launch scaffold ready. Wire real DataLoaders before full A100 runs.")
# Intentionally do not start multi-day training from this stub without loaders.
ckpt_dir = Path(os.environ.get("TRENCHER_WM_CKPT_DIR", "checkpoints"))
ckpt_dir.mkdir(parents=True, exist_ok=True)
save_checkpoint(ckpt_dir / "init.pt", model=model, step=0, phase="ssl", config=cfg)
print(f"wrote init checkpoint to {ckpt_dir / 'init.pt'}")
PY
