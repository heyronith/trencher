"""Modal app: baseline feature cache + tree/survival training jobs (V2).

Usage:
  modal run modal_apps/baselines_app.py::build_features --population broad --resume
  modal run modal_apps/baselines_app.py::build_features --population rich --resume
  modal run modal_apps/baselines_app.py::purge_stale_feature_partitions --population broad
  modal run modal_apps/baselines_app.py::train_lightgbm --no-smoke
  modal run modal_apps/baselines_app.py::train_xgboost --no-smoke
  modal run modal_apps/baselines_app.py::train_catboost --no-smoke
  modal run modal_apps/baselines_app.py::train_survival --no-smoke
  modal run modal_apps/baselines_app.py::run_rich_ablation --no-smoke
  modal run modal_apps/baselines_app.py::publish_results_v2
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
        "requests",
        "pyyaml",
        "tqdm",
        "rich",
        "tenacity",
    )
    .env({"PYTHONPATH": "/root/trencher/src"})
)

image = image.add_local_dir(
    str(Path(__file__).resolve().parents[1] / "src" / "trencher"),
    remote_path="/root/trencher/src/trencher",
    ignore=["**/__pycache__/**", "**/*.pyc", "**/*.pyo", "**/.DS_Store"],
)

app = modal.App(APP_NAME)
world_vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
art_vol = modal.Volume.from_name(ARTIFACTS_VOL, create_if_missing=True)

WORLD = Path("/data/world_model")
SPLIT = WORLD / "splits" / "split_v1"
ART = Path("/artifacts/baselines")


@app.function(image=image, volumes={"/data/world_model": world_vol}, timeout=60 * 60)
def sync_note() -> str:
    return (
        "Populate volume trencher-world-model-v1 with splits/split_v1, broad/*, "
        "sequences/sequence_dataset_v2/*, pit/, global_context/, "
        "broad/broad_targets_v3/observation_targets for impact overlays. "
        "Do not use graph/edges_wallet_token lifetime aggregates for rich features."
    )


@app.function(
    image=image,
    volumes={"/data/world_model": world_vol, "/artifacts/baselines": art_vol},
    timeout=60 * 60 * 12,
    cpu=8,
    memory=65536,
)
def build_features(
    max_partitions: int | None = None,
    population: str = "broad",
    include_rich: bool | None = None,
    resume: bool = True,
) -> dict:
    from trencher.baselines.feature_cache import build_baseline_features_v1
    from trencher.baselines.progress import ProgressEmitter

    if include_rich is True:
        population = "rich"
    elif include_rich is False:
        population = "broad"
    if population not in ("broad", "rich"):
        raise ValueError(f"population must be broad|rich, got {population!r}")

    progress = ProgressEmitter(
        f"Baseline feature cache ({population})",
        status_path=ART / f"feature_cache_{population}_progress.json",
    )
    ART.mkdir(parents=True, exist_ok=True)
    meta = build_baseline_features_v1(
        world_root=WORLD,
        split_root=SPLIT,
        max_partitions=max_partitions,
        progress=progress,
        population=population,  # type: ignore[arg-type]
        resume=resume,
    )
    world_vol.commit()
    art_vol.commit()
    return meta


@app.function(
    image=image,
    volumes={"/data/world_model": world_vol, "/artifacts/baselines": art_vol},
    timeout=60 * 60,
    cpu=2,
    memory=8192,
)
def purge_stale_feature_partitions(population: str = "broad") -> dict:
    """Remove feature parquet dirs so stale rich columns cannot be resumed as broad."""
    import shutil

    root = SPLIT / "baseline_features_v1" / population / "by_mint_partition"
    removed = 0
    if root.exists():
        for p in root.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
                removed += 1
    world_vol.commit()
    return {"population": population, "removed_dirs": removed, "path": str(root)}


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
        protocol_version="v2",
    )


@app.function(
    image=image,
    volumes={"/data/world_model": world_vol, "/artifacts/baselines": art_vol},
    timeout=60 * 60 * 12,
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
    timeout=60 * 60 * 12,
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
    timeout=60 * 60 * 12,
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
    timeout=60 * 60 * 6,
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
    timeout=60 * 60 * 4,
    cpu=8,
    memory=65536,
)
def run_rich_ablation(smoke: bool = False) -> dict:
    from trencher.baselines.rich_ablation import run_rich_ablation_v2

    ART.mkdir(parents=True, exist_ok=True)
    out = run_rich_ablation_v2(
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
def publish_results_v2() -> dict:
    import json

    from trencher.baselines.audit_v2 import publish_baseline_artifacts_v2

    feat_man = {}
    for cand in (
        SPLIT / "baseline_features_v1" / "broad" / "manifest.json",
        SPLIT / "baseline_features_v1" / "manifest.json",
    ):
        if cand.exists():
            feat_man = json.loads(cand.read_text())
            break
    out = publish_baseline_artifacts_v2(
        manifest_dir=ART / "manifests_v2",
        artifacts_root=ART,
        feature_manifest=feat_man,
        modal_usage={
            "app": APP_NAME,
            "volume": VOL_NAME,
            "artifacts_volume": ARTIFACTS_VOL,
            "label": "final_full_population_validation_control",
        },
    )
    art_vol.commit()
    return {
        "audit": out["audit"]["audit_version"],
        "feature_n": out["audit"]["feature_n"],
        "feature_rows": out["audit"].get("feature_rows"),
    }


@app.function(
    image=image,
    volumes={"/data/world_model": world_vol, "/artifacts/baselines": art_vol},
    timeout=60 * 30,
)
def publish_results() -> dict:
    """Back-compat alias for publish_results_v2."""
    return publish_results_v2.local()


@app.local_entrypoint()
def main(smoke: bool = True) -> None:
    print(sync_note.remote())
    print("purge stale broad feature partitions…")
    print(purge_stale_feature_partitions.remote(population="broad"))
    print("build_features broad…")
    print(
        build_features.remote(
            max_partitions=2 if smoke else None, population="broad", resume=True
        )
    )
    print(train_lightgbm.remote(smoke=smoke))
    print(train_xgboost.remote(smoke=smoke))
    print(train_catboost.remote(smoke=smoke))
    print(train_survival.remote(smoke=smoke))
    print("build_features rich…")
    print(
        build_features.remote(
            max_partitions=2 if smoke else None, population="rich", resume=True
        )
    )
    print(run_rich_ablation.remote(smoke=smoke))
    print(publish_results_v2.remote())
