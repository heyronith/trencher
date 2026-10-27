"""Stage-B inference via step17 path (CPU); val reuses verified artifact, TEST runs infer."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
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

from trencher.data.world_model.splits.q85_unlock import q85_test_unlocked

SplitName = Literal["val", "test"]


class StageBRefusal(RuntimeError):
    pass


def _oid_hash(oid: str) -> int:
    import hashlib

    return int(hashlib.sha256(oid.encode("utf-8")).hexdigest()[:16], 16) % (2**63)


def export_stage_b_parquet(
    *,
    pred_path: Path,
    out_path: Path,
    base_split: str,
) -> dict[str, Any]:
    """Convert step17 predictions.pt → per-row parquet for ``base_split``."""
    from scripts.review.phase4.run_phase4 import EXPLOSION_HORIZONS, softmax_pos_prob

    from trencher.data.world_model.training.loaders import resolve_assignment_paths

    obj = torch.load(pred_path, map_location="cpu", weights_only=False)
    hashes = obj["targets"]["observation_id_hash"].numpy().reshape(-1)
    logits = obj["outputs"]["explosion_logits"].numpy()
    y = obj["targets"]["explosion_target"].numpy()
    mask = obj["targets"]["explosion_mask"].numpy()
    i60 = EXPLOSION_HORIZONS.index(60)
    i300 = EXPLOSION_HORIZONS.index(300)

    hash_to_meta: dict[int, tuple[str, str]] = {}
    for path in resolve_assignment_paths(SPLIT, "broad"):
        table = pq.read_table(path, columns=["observation_id", "mint", "base_split"])
        table = table.filter(pc.equal(table["base_split"], base_split))
        for oid, mint in zip(
            table["observation_id"].to_pylist(),
            table["mint"].to_pylist(),
            strict=True,
        ):
            hash_to_meta[_oid_hash(str(oid))] = (str(oid), str(mint))

    rows = []
    missing = 0
    for i, h in enumerate(hashes):
        meta = hash_to_meta.get(int(h))
        if meta is None:
            missing += 1
            oid, mint = f"missing_{h}", ""
        else:
            oid, mint = meta
        row = {
            "observation_id": oid,
            "mint": mint,
            "observation_id_hash": int(h),
            "p_explosion_60s": float(softmax_pos_prob(logits[i, i60])),
            "p_explosion_300s": float(softmax_pos_prob(logits[i, i300])),
            "y_explosion_60s": float(y[i, i60]),
            "y_explosion_300s": float(y[i, i300]),
            "m_explosion_60s": float(mask[i, i60]),
            "m_explosion_300s": float(mask[i, i300]),
        }
        rows.append(row)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_path)
    return {
        **Q85,
        **({"q85_test_accessed": True} if base_split == "test" else {}),
        "path": str(out_path),
        "sha256": sha256_file(out_path),
        "n_rows": len(rows),
        "n_missing_oid": missing,
        "base_split": base_split,
    }


def ensure_stage_b_preds(
    *,
    split: SplitName,
    checkpoint: Path,
    expected_sha256: str = STAGE_B_SHA256,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Run or reuse step17 infer for broad population on ``split``."""
    if split == "test" and not q85_test_unlocked():
        raise StageBRefusal(
            "Stage-B TEST inference requires TRENCHER_Q85_TEST_UNLOCK "
            "(TEST_ACCESSED.lock must be written first)"
        )

    ckpt = Path(checkpoint)
    got = sha256_file(ckpt)
    if got != expected_sha256:
        raise RuntimeError(f"Stage-B checkpoint sha mismatch: {got} != {expected_sha256}")

    if split == "val":
        out_dir = SCRATCH / "stage_b_infer_val"
        run_id = "stage_b_broad_permanent_val"
        mode = "permanent_val_eval"
        phase4_dir = REPO / "scratch/phase4_clean_v1/infer_stage_b_broad"
        phase4_summary = phase4_dir / "stage_b_broad_permanent_val_summary.json"
        phase4_parquet = (
            REPO / "scratch/phase4_clean_v1/preds/stage_b_broad_permanent_val.parquet"
        )
        parquet_out = out_dir / "stage_b_broad_permanent_val.parquet"
        summary_path = out_dir / "stage_b_broad_permanent_val_summary.json"

        out_dir.mkdir(parents=True, exist_ok=True)
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
                "split": "val",
            }
            write_json(out_dir / "ensure_meta.json", result)
            return result
    else:
        out_dir = REPO / "scratch" / "phase9_test_once_v1" / "stage_b_infer_test"
        run_id = "stage_b_broad_permanent_test"
        mode = "permanent_test_eval"
        parquet_out = out_dir / f"{run_id}.parquet"
        summary_path = out_dir / f"{run_id}_summary.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not force_rerun and parquet_out.exists() and summary_path.exists():
            meta = json.loads(summary_path.read_text(encoding="utf-8"))
            result = {
                **Q85,
                "q85_test_accessed": True,
                "source": "cached_test_infer",
                "step17_path": "trencher.world_model.evaluation.step17 infer",
                "checkpoint_sha256": expected_sha256,
                "predictions_parquet": str(parquet_out),
                "summary": meta,
                "force_rerun": False,
                "split": "test",
            }
            write_json(out_dir / "ensure_meta.json", result)
            return result

    from trencher.world_model.evaluation.inference import run_permanent_val_inference

    print(
        f"running step17 Stage-B broad {mode} infer on CPU (split={split})...",
        flush=True,
    )
    summary = run_permanent_val_inference(
        checkpoint=ckpt,
        config="configs/world_model/world_model_candidate_v2.yaml",
        world_root=WORLD,
        split_root=SPLIT,
        out_dir=out_dir,
        population="broad",
        mode=mode,
        phase="broad",
        device="cpu",
        run_id=run_id,
        expected_checkpoint_sha256=expected_sha256,
        allow_dirty=True,
        repo_root=REPO,
    )
    preds_pt = out_dir / f"{run_id}_predictions.pt"
    if not preds_pt.exists():
        # inference may nest run_id differently
        cands = list(out_dir.glob("*_predictions.pt"))
        if not cands:
            raise RuntimeError(f"no predictions.pt under {out_dir}")
        preds_pt = cands[0]

    if split == "val" and (
        REPO / "scratch/phase4_clean_v1/preds/stage_b_broad_permanent_val.parquet"
    ).exists():
        shutil.copy2(
            REPO / "scratch/phase4_clean_v1/preds/stage_b_broad_permanent_val.parquet",
            parquet_out,
        )
        export_meta = {"path": str(parquet_out), "source": "phase4_parquet_copy"}
    else:
        export_meta = export_stage_b_parquet(
            pred_path=preds_pt,
            out_path=parquet_out,
            base_split="test" if split == "test" else "val",
        )

    result = {
        **Q85,
        **({"q85_test_accessed": True} if split == "test" else {}),
        "source": "step17_infer_cpu",
        "step17_path": "trencher.world_model.evaluation.step17 infer",
        "checkpoint_sha256": expected_sha256,
        "predictions_parquet": str(parquet_out),
        "predictions_pt": str(preds_pt),
        "predictions_pt_sha256": sha256_file(preds_pt),
        "export": export_meta,
        "summary": summary,
        "force_rerun": force_rerun,
        "split": split,
        "mode": mode,
    }
    write_json(out_dir / "ensure_meta.json", result)
    if summary_path and not summary_path.exists():
        write_json(summary_path, summary if isinstance(summary, dict) else {"raw": summary})
    return result
