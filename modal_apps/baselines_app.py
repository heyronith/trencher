"""Modal app: baseline feature cache + tree/survival training jobs.

Usage:
  modal run modal_apps/baselines_app.py::sync_essential_data
  modal run modal_apps/baselines_app.py::build_features
  modal run modal_apps/baselines_app.py::train_lightgbm --smoke False
  modal run modal_apps/baselines_app.py::train_xgboost
  modal run modal_apps/baselines_app.py::train_catboost
  modal run modal_apps/baselines_app.py::train_survival
  modal run modal_apps/baselines_app.py::publish_results
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "trencher-baselines-v1"
VOL_NAME = "trencher-world-model-v1"
ARTIFACTS_VOL = "trencher-baseline-artifacts-v1"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1")
    .pip_install(
        "numpy",
        "pyarrow",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "catboost",
        "lifelines",
        "click",
    )
    .env({"PYTHONPATH": "/root/trencher/src"})
)

# Package source into image
image = image.add_local_dir(
    str(Path(__file__).resolve().parents[1] / "src" / "trencher"),
    remote_path="/root/trencher/src/trencher",
)

app = modal.App(APP_NAME)
world_vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
art_vol = modal.Volume.from_name(ARTIFACTS_VOL, create_if_missing=True)

WORLD = Path("/data/world_model")
SPLIT = WORLD / "splits" / "split_v1"
ART = Path("/artifacts/baselines")


@app.function(image=image, volumes={"/data/world_model": world_vol}, timeout=60 * 60)
def sync_note() -> str:
    """Document required volume layout (upload via modal volume put)."""
    return (
        "Populate volume trencher-world-model-v1 with:\n"
        "  splits/split_v1/\n"
        "  broad/broad_sequence_events_v1/\n"
        "  broad/broad_sequence_creator_v1/\n"
        "  broad/broad_observation_sequence_refs_v1/\n"
        "  pit/global/\n"
        "  global_context/\n"
        "From repo root:\n"
        "  modal volume put trencher-world-model-v1 data/external/world_model/splits /splits\n"
        "  modal volume put trencher-world-model-v1 data/external/world_model/broad /broad\n"
        "  ..."
    )


@app.function(
    image=image,
    volumes={"/data/world_model": world_vol, "/artifacts/baselines": art_vol},
    timeout=60 * 60 * 6,
    cpu=8,
    memory=65536,
)
def build_features(max_partitions: int | None = None, include_rich: bool = False) -> dict:
    from trencher.baselines.feature_cache import build_baseline_features_v1
    from trencher.baselines.progress import ProgressEmitter

    progress = ProgressEmitter(
        "Baseline feature cache",
        status_path=ART / "feature_cache_progress.json",
    )
    ART.mkdir(parents=True, exist_ok=True)
    meta = build_baseline_features_v1(
        world_root=WORLD,
        split_root=SPLIT,
        max_partitions=max_partitions,
        progress=progress,
        include_rich=include_rich,
    )
    world_vol.commit()
    art_vol.commit()
    return meta


def _train_family(family: str, *, smoke: bool, use_gpu: bool) -> dict:
    from trencher.baselines.train_trees import run_family_core_suite

    ART.mkdir(parents=True, exist_ok=True)
    return run_family_core_suite(
        family=family,
        world_root=WORLD,
        split_root=SPLIT,
        artifacts_root=ART,
        smoke=smoke,
        use_gpu=use_gpu,
        max_partitions=2 if smoke else None,
    )


@app.function(
    image=image,
    volumes={"/data/world_model": world_vol, "/artifacts/baselines": art_vol},
    timeout=60 * 60 * 8,
    gpu="A10G",
    memory=65536,
)
def train_lightgbm(smoke: bool = False) -> dict:
    out = _train_family("lightgbm", smoke=smoke, use_gpu=True)
    art_vol.commit()
    return out


@app.function(
    image=image,
    volumes={"/data/world_model": world_vol, "/artifacts/baselines": art_vol},
    timeout=60 * 60 * 8,
    gpu="A10G",
    memory=65536,
)
def train_xgboost(smoke: bool = False) -> dict:
    out = _train_family("xgboost", smoke=smoke, use_gpu=True)
    art_vol.commit()
    return out


@app.function(
    image=image,
    volumes={"/data/world_model": world_vol, "/artifacts/baselines": art_vol},
    timeout=60 * 60 * 8,
    gpu="A10G",
    memory=65536,
)
def train_catboost(smoke: bool = False) -> dict:
    out = _train_family("catboost", smoke=smoke, use_gpu=True)
    art_vol.commit()
    return out


@app.function(
    image=image,
    volumes={"/data/world_model": world_vol, "/artifacts/baselines": art_vol},
    timeout=60 * 60 * 4,
    cpu=8,
    memory=65536,
)
def train_survival(smoke: bool = False) -> dict:
    from trencher.baselines.survival import run_survival_suite

    ART.mkdir(parents=True, exist_ok=True)
    out = run_survival_suite(
        world_root=WORLD,
        split_root=SPLIT,
        artifacts_root=ART,
        smoke=smoke,
        max_partitions=2 if smoke else None,
    )
    art_vol.commit()
    return out


@app.function(
    image=image,
    volumes={"/data/world_model": world_vol, "/artifacts/baselines": art_vol},
    timeout=60 * 30,
)
def publish_results() -> dict:
    import json

    from trencher.baselines.audit import publish_baseline_artifacts

    feat_man = {}
    mp = SPLIT / "baseline_features_v1" / "manifest.json"
    if mp.exists():
        feat_man = json.loads(mp.read_text())
    out = publish_baseline_artifacts(
        manifest_dir=ART / "manifests",
        artifacts_root=ART,
        feature_manifest=feat_man,
        modal_usage={"app": APP_NAME, "volume": VOL_NAME, "artifacts_volume": ARTIFACTS_VOL},
    )
    art_vol.commit()
    return {"audit": out["audit"]["audit_version"], "feature_n": out["audit"]["feature_n"]}


@app.local_entrypoint()
def main(smoke: bool = True) -> None:
    print(sync_note.remote())
    print("build_features…")
    print(build_features.remote(max_partitions=2 if smoke else None))
    print("train_lightgbm…")
    print(train_lightgbm.remote(smoke=smoke))
    print("train_xgboost…")
    print(train_xgboost.remote(smoke=smoke))
    print("train_catboost…")
    print(train_catboost.remote(smoke=smoke))
    print("train_survival…")
    print(train_survival.remote(smoke=smoke))
    print(publish_results.remote())
