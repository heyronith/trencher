"""Scientific Stage-C modality ablation configurations and launch commands."""

from __future__ import annotations

from typing import Any

from trencher.world_model.evaluation import STEP17_DURABLE_PREFIX

STAGE_B_CKPT_URI = (
    "gs://trencher-world-model-v1-ronithworks-20260907/world_model_freeze_v1/"
    "runs/stage_b_broad/latest.pt"
)
STAGE_B_CKPT_SHA256 = "ccf27af79bf178f095c70a84e48ad2e573bae8d5cfcfdde9d32fbb0a1f7aa208"
STAGE_C_CKPT_URI = (
    "gs://trencher-world-model-v1-ronithworks-20260907/world_model_freeze_v1/"
    "runs/stage_c_rich/latest.pt"
)
STAGE_C_CKPT_SHA256 = "11bf593e10e5dcc08915fc0fa2300ce988592f660b3608a5024c43121051d839"
RUNTIME_SHA = "b7abd1de786bf7845b98a69752d05fb695e7d170"
CONFIG_HASH = "c2e7c0e51ef91df87333615f5e6cd66e0baed891fbabcb907c5f5fa55a6d8945"
RICH_STEPS = 25000


def scientific_ablation_configs() -> list[dict[str, Any]]:
    """Prespecified scientific ablations. ``stage_c_full`` reuses frozen Stage C."""
    return [
        {
            "run_id": "stage_c_full",
            "omit": [],
            "retrain": False,
            "reuse_checkpoint_uri": STAGE_C_CKPT_URI,
            "reuse_checkpoint_sha256": STAGE_C_CKPT_SHA256,
            "note": "Do not retrain; reuse frozen Stage-C checkpoint",
        },
        {
            "run_id": "stage_c_no_graph",
            "omit": ["graph"],
            "retrain": True,
            "starting_checkpoint_uri": STAGE_B_CKPT_URI,
            "starting_checkpoint_sha256": STAGE_B_CKPT_SHA256,
            "rich_steps": RICH_STEPS,
        },
        {
            "run_id": "stage_c_no_wallet_prior",
            "omit": ["wallet_prior"],
            "retrain": True,
            "starting_checkpoint_uri": STAGE_B_CKPT_URI,
            "starting_checkpoint_sha256": STAGE_B_CKPT_SHA256,
            "rich_steps": RICH_STEPS,
        },
        {
            "run_id": "stage_c_no_ownership",
            "omit": ["ownership"],
            "retrain": True,
            "starting_checkpoint_uri": STAGE_B_CKPT_URI,
            "starting_checkpoint_sha256": STAGE_B_CKPT_SHA256,
            "rich_steps": RICH_STEPS,
        },
        {
            "run_id": "stage_c_no_microstructure",
            "omit": ["microstructure"],
            "retrain": True,
            "starting_checkpoint_uri": STAGE_B_CKPT_URI,
            "starting_checkpoint_sha256": STAGE_B_CKPT_SHA256,
            "rich_steps": RICH_STEPS,
        },
    ]


def ablation_launch_env(cfg: dict[str, Any]) -> dict[str, str]:
    """Exact env for one scientific ablation rich fine-tune (not auto-launched)."""
    omit = ",".join(cfg.get("omit") or [])
    run_id = str(cfg["run_id"])
    return {
        "TRENCHER_LOCAL_STAGE": "/content/trencher_stage",
        "TRENCHER_WORLD_ROOT": "/content/trencher_stage/world_model",
        "TRENCHER_SPLIT_ROOT": "/content/trencher_stage/world_model/splits/split_v1",
        "TRENCHER_WM_CONFIG": (
            "/content/trencher/configs/world_model/world_model_candidate_v2.yaml"
        ),
        "TRENCHER_WM_PHASE": "rich",
        "TRENCHER_WM_MODE": "final_train",
        "TRENCHER_WM_POPULATION": "rich",
        "TRENCHER_WM_DEVICE": "cuda",
        "TRENCHER_WM_CKPT_DIR": f"/content/trencher_stage/checkpoints/{run_id}",
        "TRENCHER_WM_RESUME": "/content/trencher_stage/checkpoints/stage_b_broad/latest.pt",
        "TRENCHER_WM_MAX_STEPS": str(cfg.get("rich_steps", RICH_STEPS)),
        "TRENCHER_WM_NUM_WORKERS": "2",
        "TRENCHER_WM_EXPECT_EPOCH_BATCHES": "566",
        "TRENCHER_WM_RUN_ID": run_id,
        "TRENCHER_GIT_SHA": RUNTIME_SHA,
        "TRENCHER_GCS_URI": (
            "gs://trencher-world-model-v1-ronithworks-20260907/world_model_freeze_v1"
        ),
        "TRENCHER_GCS_TRANSPORT": "python",
        "TRENCHER_WM_DURABLE": "1",
        "TRENCHER_WM_OMIT_MODALITIES": omit,
        "PYTHONUNBUFFERED": "1",
        "GOOGLE_CLOUD_PROJECT": "gen-lang-client-0390594165",
    }


def ablation_launch_commands() -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for cfg in scientific_ablation_configs():
        if not cfg.get("retrain"):
            commands.append(
                {
                    "run_id": cfg["run_id"],
                    "action": "reuse_frozen_stage_c",
                    "checkpoint_uri": cfg["reuse_checkpoint_uri"],
                    "checkpoint_sha256": cfg["reuse_checkpoint_sha256"],
                    "durable_eval_prefix": f"{STEP17_DURABLE_PREFIX}/{cfg['run_id']}/",
                }
            )
            continue
        env = ablation_launch_env(cfg)
        export_lines = "\n".join(f"export {k}={v!s}" for k, v in env.items())
        cmd = (
            f"{export_lines}\n"
            f"mkdir -p \"$TRENCHER_WM_CKPT_DIR\"\n"
            f"# restore Stage-B latest.pt into local resume path before launch\n"
            f"nohup bash /content/trencher/scripts/colab/launch_training.sh "
            f">/content/trencher_stage/{cfg['run_id']}_train.log 2>&1 &\n"
            f"echo $! >/content/trencher_stage/{cfg['run_id']}_train.pid"
        )
        commands.append(
            {
                "run_id": cfg["run_id"],
                "action": "rich_finetune_from_stage_b",
                "omit": cfg["omit"],
                "steps": cfg.get("rich_steps", RICH_STEPS),
                "starting_checkpoint_uri": cfg["starting_checkpoint_uri"],
                "starting_checkpoint_sha256": cfg["starting_checkpoint_sha256"],
                "expected_gpu": "1x A100 40GB",
                "expected_wallclock_hours": 1.5,
                "durable_prefix": f"{STEP17_DURABLE_PREFIX}/{cfg['run_id']}/",
                "env": env,
                "shell": cmd,
            }
        )
    return commands
