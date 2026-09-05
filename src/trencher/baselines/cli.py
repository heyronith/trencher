"""CLI for causal tabular baselines."""

from __future__ import annotations

from pathlib import Path

import click

from trencher.baselines.split_guard import PermanentTestAccessError, refuse_test_split


@click.group()
def baselines() -> None:
    """Causal tabular baseline features + Modal training controls."""


@baselines.command("refuse-test-check")
@click.option("--split", required=True)
def refuse_test_check(split: str) -> None:
    """Hard guard smoke: must error on split=test."""
    try:
        refuse_test_split(split)
    except PermanentTestAccessError as exc:
        click.echo(f"REFUSED: {exc}")
        raise SystemExit(2) from exc
    click.echo(f"ok split={split}")


@baselines.command("build-features")
@click.option("--world-root", type=click.Path(path_type=Path), default=Path("data/external/world_model"))
@click.option("--max-partitions", type=int, default=None)
@click.option("--smoke", is_flag=True, default=False)
def build_features_cmd(world_root: Path, max_partitions: int | None, smoke: bool) -> None:
    """Local tiny/smoke feature build only (full runs on Modal)."""
    from trencher.baselines.feature_cache import build_baseline_features_v1
    from trencher.baselines.progress import ProgressEmitter

    if not smoke and max_partitions is None:
        click.echo(
            "Refusing full local feature build. Use --smoke or --max-partitions, "
            "or run on Modal: modal run modal_apps/baselines_app.py::build_features",
            err=True,
        )
        raise SystemExit(1)
    mp = max_partitions if max_partitions is not None else (2 if smoke else None)
    meta = build_baseline_features_v1(
        world_root=world_root,
        max_partitions=mp,
        progress=ProgressEmitter("Baseline feature cache"),
    )
    click.echo(meta)


@baselines.command("publish-contract")
@click.option("--manifest-dir", type=click.Path(path_type=Path), default=Path("manifests/world_model_data"))
def publish_contract_cmd(manifest_dir: Path) -> None:
    from trencher.baselines.audit import build_baseline_contract_v1, publish_baseline_artifacts
    from trencher.data.world_model.io import write_json

    contract = build_baseline_contract_v1()
    write_json(manifest_dir / "BASELINE_CONTRACT_V1.json", contract)
    publish_baseline_artifacts(
        manifest_dir=manifest_dir,
        artifacts_root=Path("artifacts/baselines"),
        feature_manifest={"feature_n": len(contract["feature_contract"]["broad_features"])},
    )
    click.echo("published BASELINE_CONTRACT_V1 / AUDIT / RESULTS stubs")


@baselines.command("train-smoke")
@click.option("--world-root", type=click.Path(path_type=Path), default=Path("data/external/world_model"))
@click.option("--family", type=click.Choice(["lightgbm", "xgboost", "catboost", "survival"]), default="lightgbm")
def train_smoke_cmd(world_root: Path, family: str) -> None:
    """Tiny local smoke (2 partitions). Full fits must run on Modal."""
    from trencher.baselines.feature_cache import build_baseline_features_v1
    from trencher.baselines.progress import ProgressEmitter
    from trencher.baselines.survival import run_survival_suite
    from trencher.baselines.train_trees import run_family_core_suite

    # Ensure tiny feature cache exists
    build_baseline_features_v1(
        world_root=world_root,
        max_partitions=2,
        progress=ProgressEmitter("Baseline feature cache"),
    )
    art = Path("artifacts/baselines")
    if family == "survival":
        out = run_survival_suite(
            world_root=world_root,
            artifacts_root=art,
            smoke=True,
            max_partitions=2,
        )
    else:
        out = run_family_core_suite(
            family=family,
            world_root=world_root,
            artifacts_root=art,
            smoke=True,
            use_gpu=False,
            max_partitions=2,
        )
    click.echo({k: out.get(k) for k in ("family", "train_n", "val_n", "heads")})
