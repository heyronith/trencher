"""Build STEP17_PREFLIGHT.json (fail-closed lineage + proposed GPU runs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trencher.world_model.config import load_world_model_config
from trencher.world_model.evaluation import STEP17_CONTRACT, STEP17_DURABLE_PREFIX
from trencher.world_model.evaluation.ablations import (
    CONFIG_HASH,
    ablation_launch_commands,
    scientific_ablation_configs,
)
from trencher.world_model.evaluation.baselines import compare_world_model_to_baselines
from trencher.world_model.evaluation.gru_control import gru_control_plan
from trencher.world_model.evaluation.lineage import (
    source_models_provenance,
    stage_b_lineage,
    stage_c_lineage,
)
from trencher.world_model.evaluation.masking import diagnostic_configs
from trencher.world_model.evaluation.modes import assert_step17_mode, provenance_q85_flag
from trencher.world_model.evaluation.provenance import (
    discover_evaluation_runtime,
    evaluation_runtime_fields,
    write_json_with_sidecar,
)
from trencher.world_model.split_guard import PermanentTestAccessError


def _q85_refusal_proof() -> dict[str, Any]:
    proof: dict[str, Any] = {"aliases_tested": [], "all_refused": True}
    for alias in ("q85_test", "test", "permanent_test"):
        try:
            assert_step17_mode(alias)
            proof["all_refused"] = False
            proof["aliases_tested"].append({"alias": alias, "refused": False})
        except PermanentTestAccessError as exc:
            proof["aliases_tested"].append(
                {"alias": alias, "refused": True, "error": str(exc)}
            )
    allowed = assert_step17_mode("permanent_val_eval")
    proof["permanent_val_eval_allowed"] = allowed == "permanent_val_eval"
    proof.update(provenance_q85_flag(accessed=False))
    if not proof["all_refused"] or not proof["permanent_val_eval_allowed"]:
        raise RuntimeError("q85 refusal proof failed")
    return proof


def build_preflight(
    *,
    repo_root: Path | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root else Path.cwd()
    cfg_path = root / "configs/world_model/world_model_candidate_v2.yaml"
    cfg = load_world_model_config(cfg_path)
    if cfg.config_hash() != CONFIG_HASH:
        raise RuntimeError(
            f"frozen config hash mismatch: got={cfg.config_hash()} want={CONFIG_HASH}"
        )

    runtime = discover_evaluation_runtime(repo_root=root, allow_dirty=allow_dirty)
    stage_b = stage_b_lineage()
    stage_c = stage_c_lineage()
    source_models = source_models_provenance()
    if stage_b["config_hash"] != stage_c["config_hash"]:
        raise RuntimeError("Stage-B/C config hash mismatch")
    if stage_c["checkpoint_sha256"] != source_models["stage_c"]["checkpoint_sha256"]:
        raise RuntimeError("Stage-C checkpoint identity drift")
    # Evaluator runtime must NOT be required equal to Stage-C training SHA.
    if runtime["git_sha"] == stage_c["training_git_sha"]:
        # Equality is allowed when evaluating from that historical checkout, but
        # it is never *required*. Record the observation only.
        runtime_note = (
            "evaluation_runtime_git_sha currently equals Stage-C training_git_sha; "
            "this is incidental, not a requirement"
        )
    else:
        runtime_note = (
            "evaluation_runtime_git_sha differs from Stage-C training_git_sha "
            "(expected for Step-17 implementation commits)"
        )

    baseline_cmp = compare_world_model_to_baselines(
        {"explosion": {"60": {"auroc": None, "support": None}, "300": {"auroc": None}}}
    )

    ablations = scientific_ablation_configs()
    commands = ablation_launch_commands()
    retrain = [c for c in commands if c.get("action") == "rich_finetune_from_stage_b"]

    payload: dict[str, Any] = {
        "contract": STEP17_CONTRACT,
        "status": "ready_for_gpu_decision",
        "config_hash": CONFIG_HASH,
        "durable_prefix": STEP17_DURABLE_PREFIX,
        **evaluation_runtime_fields(runtime),
        "evaluation_runtime_policy": (
            "Discovered at artifact generation / execution via git rev-parse HEAD. "
            "Never pinned to Stage-B or Stage-C source-training SHAs."
        ),
        "evaluation_runtime_note": runtime_note,
        "source_models": source_models,
        "stage_b": stage_b,
        "stage_c": stage_c,
        "permanent_validation_population_for_paired_eval": "rich",
        "paired_eval_blocker": stage_b["paired_blocker"],
        "stage_b_rich_inference_command": (
            "uv run python -m trencher.world_model.evaluation.step17 infer "
            "--checkpoint /content/trencher_stage/checkpoints/stage_b_broad/latest.pt "
            "--expected-checkpoint-sha256 "
            f"{stage_b['checkpoint_sha256']} "
            "--config /content/trencher/configs/world_model/world_model_candidate_v2.yaml "
            "--world-root /content/trencher_stage/world_model "
            "--split-root /content/trencher_stage/world_model/splits/split_v1 "
            "--population rich --phase rich --mode permanent_val_eval "
            "--run-id stage_b_rich_permanent_val "
            "--out-dir /content/trencher_stage/evaluation/step17/stage_b_rich_permanent_val"
        ),
        "inference_time_diagnostics": {
            "label": "inference_time_modality_mask_diagnostic",
            "configs": diagnostic_configs(),
            "results": None,
            "note": (
                "Not executed in the editor task (requires GPU + restored rich corpus). "
                "Run after Stage-C checkpoint restore to prioritize retraining ablations. "
                "Evaluator records evaluation_runtime_git_sha from git HEAD internally."
            ),
            "command": (
                "uv run python -m trencher.world_model.evaluation.step17 diagnose-masks "
                "--checkpoint /content/trencher_stage/checkpoints/stage_c_rich/latest.pt "
                f"--expected-checkpoint-sha256 {stage_c['checkpoint_sha256']} "
                "--world-root /content/trencher_stage/world_model "
                "--split-root /content/trencher_stage/world_model/splits/split_v1 "
                "--out-dir /content/trencher_stage/evaluation/step17/diagnostics"
            ),
        },
        "scientific_ablations": ablations,
        "proposed_ablation_gpu_runs": retrain,
        "ablation_launch_commands": commands,
        "expected_gpu_requirement_per_ablation": {
            "accelerator": "1x NVIDIA A100 40GB",
            "steps": 25000,
            "batch_size": 64,
            "expected_wallclock_hours": 1.5,
            "expected_artifact_location": f"{STEP17_DURABLE_PREFIX}/<run_id>/",
        },
        "simple_temporal_control": gru_control_plan(),
        "baseline_comparison_policy": baseline_cmp,
        "q85_refusal_proof": _q85_refusal_proof(),
        "stop_gate": {
            "do_not_provision_gpu": True,
            "do_not_train_ablations": True,
            "do_not_train_gru_control": True,
            "do_not_touch_q85_test": True,
            "next_decision": (
                "Whether all four scientific retraining ablations are justified, "
                "or diagnostics allow reducing the GPU experiment set."
            ),
        },
    }
    payload.update(provenance_q85_flag(accessed=False))
    return payload


def write_preflight(
    path: Path,
    *,
    repo_root: Path | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    payload = build_preflight(repo_root=repo_root, allow_dirty=allow_dirty)
    write_json_with_sidecar(Path(path), payload)
    return payload
