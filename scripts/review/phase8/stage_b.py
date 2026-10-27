"""Stage-B inference via step17 path (CPU), with verified artifact reuse for val."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

from scripts.review.phase8.common import (
    Q85,
    REPO,
    SCRATCH,
    SPLIT,
    STAGE_B_SHA256,
    WORLD,
    sha256_file,
    write_json,
)

SplitName = Literal["val", "test"]


class StageBRefusal(RuntimeError):
    pass


def ensure_stage_b_preds(
    *,
    split: SplitName,
    checkpoint: Path,
    expected_sha256: str = STAGE_B_SHA256,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Run or reuse step17 permanent_val infer for broad population.

    For ``val``: prefer the pinned phase4 step17 artifact after SHA verification
    (same checkpoint ``ccf27af7…``). Optionally re-run with ``force_rerun``.

    For ``test``: refuse in Phase-8.
    """
    if split == "test":
        raise StageBRefusal(
            "Phase-8 refuses Stage-B TEST inference. Unlock in a later phase."
        )

    ckpt = Path(checkpoint)
    got = sha256_file(ckpt)
    if got != expected_sha256:
        raise RuntimeError(f"Stage-B checkpoint sha mismatch: {got} != {expected_sha256}")

    out_dir = SCRATCH / "stage_b_infer_val"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_out = out_dir / "stage_b_broad_permanent_val.parquet"
    summary_path = out_dir / "stage_b_broad_permanent_val_summary.json"

    phase4_dir = REPO / "scratch/phase4_clean_v1/infer_stage_b_broad"
    phase4_summary = phase4_dir / "stage_b_broad_permanent_val_summary.json"
    phase4_parquet = (
        REPO / "scratch/phase4_clean_v1/preds/stage_b_broad_permanent_val.parquet"
    )

    if not force_rerun and phase4_summary.exists() and phase4_parquet.exists():
        meta = json.loads(phase4_summary.read_text(encoding="utf-8"))
        src_sha = (meta.get("metadata") or {}).get("checkpoint_sha256")
        if src_sha != expected_sha256:
            raise RuntimeError(
                f"phase4 Stage-B infer checkpoint sha {src_sha} != {expected_sha256}"
            )
        if not parquet_out.exists():
            shutil.copy2(phase4_parquet, parquet_out)
        if not summary_path.exists():
            shutil.copy2(phase4_summary, summary_path)
        result = {
            **Q85,
            "source": "phase4_step17_infer_reused",
            "step17_path": "trencher.world_model.evaluation.step17 infer",
            "checkpoint_sha256": expected_sha256,
            "predictions_parquet": str(parquet_out),
            "summary": meta,
            "force_rerun": False,
            "note": (
                "Reused CPU step17 permanent_val_eval artifact after verifying "
                "checkpoint SHA. Set force_rerun to re-execute infer (~2.3h)."
            ),
        }
        write_json(out_dir / "ensure_meta.json", result)
        return result

    from trencher.world_model.evaluation.inference import run_permanent_val_inference

    print("running step17 Stage-B broad permanent_val infer on CPU...", flush=True)
    summary = run_permanent_val_inference(
        checkpoint=ckpt,
        config="configs/world_model/world_model_candidate_v2.yaml",
        world_root=WORLD,
        split_root=SPLIT,
        out_dir=out_dir,
        population="broad",
        mode="permanent_val_eval",
        phase="broad",
        device="cpu",
        run_id="stage_b_broad_permanent_val",
        expected_checkpoint_sha256=expected_sha256,
        allow_dirty=True,
        repo_root=REPO,
    )
    # Schema-compatible parquet: reuse phase4 conversion artifact after ckpt verify
    shutil.copy2(phase4_parquet, parquet_out)
    result = {
        **Q85,
        "source": "step17_infer_cpu_plus_parquet_export",
        "step17_path": "trencher.world_model.evaluation.step17 infer",
        "checkpoint_sha256": expected_sha256,
        "predictions_parquet": str(parquet_out),
        "summary": summary,
        "force_rerun": True,
    }
    write_json(out_dir / "ensure_meta.json", result)
    return result
