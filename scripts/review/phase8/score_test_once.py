"""Full split-agnostic V2 scorer. Default ``--split val``. TEST path gated; never run here."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scripts.review.phase7.score_test_once import (
    ScorerRefusal,
    check_test_once_dir_empty,
    verify_preregistration,
)
from scripts.review.phase8.common import (
    HEADS,
    MANIFEST_PATH,
    PROTOCOL_V2,
    Q85,
    REPO,
    SCRATCH,
    TEST_ONCE_DIR,
    V2_PREREG_SHA,
    ensure_dirs,
    write_json,
)
from scripts.review.phase8.features import (
    FeatureBuildRefusal,
    ensure_broad_features,
    load_split_bundle,
)
from scripts.review.phase8.metrics_v2 import (
    _align_stage_b,
    load_side_fields_for_oids,
    pump_recall_primary,
    score_head_bundle,
    score_twox,
)
from scripts.review.phase8.predict import (
    load_manifest,
    load_stage_b_val_preds,
    predict_all_tree_models,
    predict_twox,
)
from scripts.review.phase8.stage_b import StageBRefusal, ensure_stage_b_preds

SplitName = Literal["val", "test"]


def _peak_rss_mb() -> float:
    # macOS ru_maxrss is bytes; Linux is KB
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def run_split_pipeline(
    *,
    split: SplitName,
    out_dir: Path,
    manifest_path: Path = MANIFEST_PATH,
    run_bootstrap: bool = True,
    force_stage_b_rerun: bool = False,
) -> dict[str, Any]:
    """Features → frozen models → every V2 metric for ``split`` (val only in Phase-8)."""
    if split == "test":
        raise ScorerRefusal(
            "Phase-8 build: refusing --split test. Unlock only after val proof "
            "in a later dedicated phase."
        )

    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)

    print("ensure broad features...", flush=True)
    feat_meta = ensure_broad_features(split=split)

    print("ensure Stage-B preds via step17 path...", flush=True)
    sb_meta = ensure_stage_b_preds(
        split=split,
        checkpoint=REPO / manifest["models"]["stage_b_broad"]["path"],
        expected_sha256=manifest["models"]["stage_b_broad"]["sha256"],
        force_rerun=force_stage_b_rerun,
    )
    sb = load_stage_b_val_preds(manifest)

    report: dict[str, Any] = {
        **Q85,
        "split": split,
        "manifest_path": str(manifest_path),
        "feature_meta": feat_meta,
        "stage_b_meta": sb_meta,
        "by_head": {},
        "twox": None,
        "model_hashes": {
            mid: {"sha256": m["sha256"], "path": m["path"]}
            for mid, m in manifest["models"].items()
        },
    }

    # Side fields once from first head oids
    side_cache: dict[str, Any] | None = None

    for head_id in HEADS:
        print(f"=== scoring {head_id} ===", flush=True)
        bundle = load_split_bundle(head_id, split)
        if side_cache is None:
            print("loading side fields (already_started / tte)...", flush=True)
            side_cache = load_side_fields_for_oids(bundle["observation_id"])
        scores = predict_all_tree_models(
            manifest=manifest,
            head_id=head_id,
            X=np.asarray(bundle["X"]),
            feature_names=list(bundle["feature_names"]),
        )
        scores["stage_b"] = _align_stage_b(bundle["observation_id"], sb, head_id)
        head_report = score_head_bundle(
            head_id=head_id,
            y=bundle["y"],
            elig=bundle["elig"],
            scores=scores,
            mints=bundle["mint"],
            times=bundle["times"],
            oids=bundle["observation_id"],
            X=np.asarray(bundle["X"]),
            feature_names=list(bundle["feature_names"]),
            already=side_cache["already_started"],
            tte=side_cache["tte_sec"],
            run_bootstrap=run_bootstrap,
        )
        # Pump recall on primary (xgb champion)
        head_report["pump_level_recall_primary"] = pump_recall_primary(
            scores=scores["xgb_champion"],
            mints=bundle["mint"],
            times=bundle["times"],
            elig=bundle["elig"],
        )
        report["by_head"][head_id] = head_report
        write_json(out_dir / f"head_{head_id}.json", head_report)

    # 2× head
    print("=== scoring 2× head ===", flush=True)
    bundle300 = load_split_bundle("explosion_300s", split)
    twox_scores = predict_twox(manifest=manifest, X=np.asarray(bundle300["X"]))
    report["twox"] = score_twox(
        scores=twox_scores,
        oids=bundle300["observation_id"],
        y300=bundle300["y"],
        elig300=bundle300["elig"],
        mints=bundle300["mint"],
        times=bundle300["times"],
    )
    write_json(out_dir / "twox.json", report["twox"])

    report["runtime_sec"] = time.time() - t0
    report["peak_rss_mb"] = _peak_rss_mb()
    write_json(out_dir / "SCORE_SPLIT_REPORT.json", report)
    print(
        f"done split={split} runtime_sec={report['runtime_sec']:.1f} "
        f"peak_rss_mb={report['peak_rss_mb']:.1f}",
        flush=True,
    )
    return report


def run_test_once(
    *,
    protocol: Path,
    prereg_sha: str,
    out_dir: Path,
    split: SplitName = "val",
    allow_dirty: bool = False,
    run_bootstrap: bool = True,
) -> dict[str, Any]:
    """Entry with Phase-7 gates for test; val bypasses lock/TEST IO."""
    protocol = Path(protocol)
    out_dir = Path(out_dir)

    if split == "val":
        # Val dry-run / full pipeline: no TEST lock, allow dirty for development
        return run_split_pipeline(
            split="val",
            out_dir=out_dir,
            run_bootstrap=run_bootstrap,
        )

    # ---- TEST path (Phase-8 still refuses after gates) ----
    meta = verify_preregistration(
        protocol_path=protocol,
        prereg_sha=prereg_sha,
        allow_dirty=allow_dirty,
    )
    check_test_once_dir_empty(out_dir)
    # Would write lock BEFORE reading TEST — but we refuse before any TEST IO.
    raise ScorerRefusal(
        "Phase-8: TEST gates verified in code paths, but refusing to write "
        f"TEST_ACCESSED.lock or read q85+ TEST (prereg={meta['preregistration_sha']})."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_V2)
    parser.add_argument("--preregistration-sha", default=V2_PREREG_SHA)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default: scratch dry_run_val or test_once_v1)",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--force-stage-b-rerun", action="store_true")
    args = parser.parse_args(argv)
    ensure_dirs()
    out = args.out
    if out is None:
        out = SCRATCH / "dry_run_val" if args.split == "val" else TEST_ONCE_DIR
    try:
        if args.split == "val":
            result = run_split_pipeline(
                split="val",
                out_dir=out,
                run_bootstrap=not args.skip_bootstrap,
                force_stage_b_rerun=args.force_stage_b_rerun,
            )
        else:
            result = run_test_once(
                protocol=args.protocol,
                prereg_sha=args.preregistration_sha,
                out_dir=out,
                split="test",
                allow_dirty=args.allow_dirty,
            )
    except (ScorerRefusal, FeatureBuildRefusal, StageBRefusal) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "split": result.get("split"),
                "runtime_sec": result.get("runtime_sec"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
