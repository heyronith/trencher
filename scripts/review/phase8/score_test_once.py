"""Full split-agnostic V2 scorer. Default ``--split val``.

TEST path: verify_preregistration → check_test_once_dir_empty → write_access_lock
→ set_q85_test_unlock → pipeline. Any TEST IO before the lock is a bug.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
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
    write_access_lock,
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
    sha256_file,
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
    load_stage_b_preds,
    predict_all_tree_models,
    predict_twox,
)
from scripts.review.phase8.stage_b import StageBRefusal, ensure_stage_b_preds

from trencher.data.world_model.splits.q85_unlock import (
    clear_q85_test_unlock,
    set_q85_test_unlock,
)

SplitName = Literal["val", "test"]


def _peak_rss_mb() -> float:
    # macOS ru_maxrss is bytes; Linux is KB
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def run_split_pipeline(
    *,
    split: SplitName,
    out_dir: Path,
    manifest_path: Path = MANIFEST_PATH,
    run_bootstrap: bool = True,
    force_stage_b_rerun: bool = False,
) -> dict[str, Any]:
    """Features → frozen models → every V2 metric for ``split``."""
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
    sb = load_stage_b_preds(
        manifest,
        split=split,
        preds_parquet=Path(sb_meta["predictions_parquet"]),
    )

    report: dict[str, Any] = {
        **Q85,
        **({"q85_test_accessed": True} if split == "test" else {}),
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

    side_cache: dict[str, Any] | None = None
    pred_scratch = (
        REPO / "scratch" / "phase9_test_once_v1" / "preds"
        if split == "test"
        else SCRATCH / "dry_run_val" / "preds"
    )
    pred_scratch.mkdir(parents=True, exist_ok=True)
    pred_artifacts: dict[str, Any] = {}

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
        # Persist large per-row preds under gitignored scratch
        for name, arr in scores.items():
            p = pred_scratch / f"{split}_{head_id}_{name}.npy"
            np.save(p, np.asarray(arr, dtype=np.float64))
            pred_artifacts[f"{head_id}/{name}"] = {
                "path": str(p.relative_to(REPO)),
                "sha256": sha256_file(p),
                "n": int(np.asarray(arr).shape[0]),
            }
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
            split=split,
        )
        head_report["pump_level_recall_primary"] = pump_recall_primary(
            scores=scores["xgb_champion"],
            mints=bundle["mint"],
            times=bundle["times"],
            elig=bundle["elig"],
            split=split,
        )
        report["by_head"][head_id] = head_report
        write_json(out_dir / f"head_{head_id}.json", head_report)

    print("=== scoring 2× head ===", flush=True)
    bundle300 = load_split_bundle("explosion_300s", split)
    twox_scores = predict_twox(manifest=manifest, X=np.asarray(bundle300["X"]))
    p_twox = pred_scratch / f"{split}_twox_xgb.npy"
    np.save(p_twox, np.asarray(twox_scores, dtype=np.float64))
    pred_artifacts["twox/xgb_champion"] = {
        "path": str(p_twox.relative_to(REPO)),
        "sha256": sha256_file(p_twox),
        "n": int(np.asarray(twox_scores).shape[0]),
    }
    report["twox"] = score_twox(
        scores=twox_scores,
        oids=bundle300["observation_id"],
        y300=bundle300["y"],
        elig300=bundle300["elig"],
        mints=bundle300["mint"],
        times=bundle300["times"],
        split=split,
    )
    write_json(out_dir / "twox.json", report["twox"])

    report["pred_artifacts"] = pred_artifacts
    report["runtime_sec"] = time.time() - t0
    report["peak_rss_mb"] = _peak_rss_mb()
    write_json(out_dir / "SCORE_SPLIT_REPORT.json", report)
    print(
        f"done split={split} runtime_sec={report['runtime_sec']:.1f} "
        f"peak_rss_mb={report['peak_rss_mb']:.1f}",
        flush=True,
    )
    return report


def _fmt_ci(block: dict[str, Any] | None) -> str:
    if not block:
        return "n/a"
    delta = block.get("delta")
    ci = block.get("ci95") or [None, None]
    if delta is None:
        return "n/a"
    lo, hi = ci[0], ci[1]
    if lo is None or hi is None:
        return f"{delta:.6f}"
    return f"{delta:.6f} [{lo:.6f}, {hi:.6f}]"


def build_test_once_report(
    *,
    scored: dict[str, Any],
    prereg_meta: dict[str, Any],
    val_scored: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Assemble TEST_ONCE_REPORT.{json,md} from a scored TEST pipeline result."""
    head_sha = prereg_meta.get("head_sha") or _git_head()
    report: dict[str, Any] = {
        "q85_test_accessed": True,
        "preregistration_sha": prereg_meta.get("preregistration_sha", V2_PREREG_SHA),
        "head_sha": head_sha,
        "protocol_match": prereg_meta.get("protocol_match"),
        "split": "test",
        "model_hashes": scored.get("model_hashes"),
        "feature_meta": scored.get("feature_meta"),
        "stage_b_meta": {
            k: scored.get("stage_b_meta", {}).get(k)
            for k in (
                "source",
                "checkpoint_sha256",
                "predictions_parquet",
                "predictions_pt_sha256",
                "split",
                "mode",
            )
        },
        "pred_artifacts": scored.get("pred_artifacts"),
        "by_head": scored.get("by_head"),
        "twox": scored.get("twox"),
        "runtime_sec": scored.get("runtime_sec"),
        "peak_rss_mb": scored.get("peak_rss_mb"),
        "val_comparison": None,
    }

    # Side-by-side champion / baselines vs permanent val (reporting only)
    if val_scored and val_scored.get("by_head"):
        cmp: dict[str, Any] = {}
        for head_id in HEADS:
            th = (scored.get("by_head") or {}).get(head_id) or {}
            vh = (val_scored.get("by_head") or {}).get(head_id) or {}
            t_all = ((th.get("variants") or {}).get("all") or {}).get("models") or {}
            v_all = ((vh.get("variants") or {}).get("all") or {}).get("models") or {}
            row: dict[str, Any] = {}
            for mid in (
                "xgb_champion",
                "lgbm_tuned",
                "logistic_l2",
                "stage_b",
                "age_curve_xgb",
            ):
                row[mid] = {
                    "val_roc": (v_all.get(mid) or {}).get("roc_auc"),
                    "val_pr": (v_all.get(mid) or {}).get("pr_auc"),
                    "test_roc": (t_all.get(mid) or {}).get("roc_auc"),
                    "test_pr": (t_all.get(mid) or {}).get("pr_auc"),
                }
            cmp[head_id] = row
        report["val_comparison"] = cmp

    lines = [
        "# TEST_ONCE_REPORT",
        "",
        "## Provenance",
        "",
        "- q85_test_accessed: **true**",
        f"- preregistration_sha: `{report['preregistration_sha']}`",
        f"- head_sha: `{head_sha}`",
        f"- protocol_match: {report['protocol_match']}",
        f"- runtime_sec: {report['runtime_sec']}",
        f"- peak_rss_mb: {report['peak_rss_mb']}",
        "",
        "## Model hashes",
        "",
        "| model_id | sha256 |",
        "|---|---|",
    ]
    for mid, m in sorted((report.get("model_hashes") or {}).items()):
        lines.append(f"| {mid} | `{m['sha256']}` |")

    lines += ["", "## Heads (all variants)", ""]
    for head_id in HEADS:
        hr = (scored.get("by_head") or {}).get(head_id) or {}
        lines.append(f"### {head_id}")
        lines.append("")
        lines.append(f"event_window: {hr.get('event_window')}")
        lines.append("")
        for mode, block in (hr.get("variants") or {}).items():
            lines.append(f"#### variant `{mode}` (n={block.get('n')}, n_pos={block.get('n_pos')})")
            lines.append("")
            lines.append("| model | ROC-AUC | PR-AUC | P@1% |")
            lines.append("|---|---:|---:|---:|")
            for mid, met in (block.get("models") or {}).items():
                p1 = (met.get("precision_at") or {}).get("top_0.01")
                lines.append(
                    f"| {mid} | {met.get('roc_auc')} | {met.get('pr_auc')} | {p1} |"
                )
            lines.append("")
            if mode == "all" and block.get("deltas_vs_champion"):
                lines.append("Mint-bootstrap Δ (champion − other):")
                lines.append("")
                for dk, dv in block["deltas_vs_champion"].items():
                    lines.append(
                        f"- {dk} ROC: {_fmt_ci(dv.get('roc_auc'))}; "
                        f"PR: {_fmt_ci(dv.get('pr_auc'))}"
                    )
                lines.append("")

        strata = hr.get("strata") or {}
        lines.append("#### Strata (champion + baselines)")
        lines.append("")
        def _champ_roc(sm: dict[str, Any]) -> Any:
            models = sm.get("models") or {}
            return (models.get("xgb_champion") or {}).get("roc_auc")

        for trig, sm in (strata.get("by_trigger") or {}).items():
            if not isinstance(sm, dict):
                continue
            lines.append(
                f"- trigger={trig}: n={sm.get('n')} champ_roc={_champ_roc(sm)}"
            )
        for b, sm in (strata.get("by_age_bucket") or {}).items():
            if not isinstance(sm, dict):
                continue
            lines.append(f"- age={b}: n={sm.get('n')} champ_roc={_champ_roc(sm)}")
        for ak, sm in (strata.get("active_only") or {}).items():
            if not isinstance(sm, dict):
                continue
            lines.append(f"- active={ak}: n={sm.get('n')} champ_roc={_champ_roc(sm)}")
        lines.append("")
        if hr.get("alert_feed"):
            af = hr["alert_feed"]
            lines.append(
                f"Alert feed: lift_vs_base={af.get('lift_vs_base')} "
                f"mean_frac={af.get('mean_frac_explode_300s')} "
                f"base={af.get('base_rate')} hours={af.get('n_hours_in_window')}"
            )
            lines.append("")
        if hr.get("pump_level_recall_primary"):
            pump_json = json.dumps(
                hr["pump_level_recall_primary"], default=str
            )[:500]
            lines.append(f"Pump-level recall (primary): {pump_json}")
            lines.append("")

    tw = scored.get("twox") or {}
    lines += [
        "## 2× head (`within_300s` ∧ `reaches_plus_100pct`)",
        "",
        f"- n_elig={tw.get('n_elig')} n_pos={tw.get('n_pos')} prevalence={tw.get('prevalence')}",
        f"- ROC={((tw.get('metrics') or {}).get('roc_auc'))} "
        f"PR={((tw.get('metrics') or {}).get('pr_auc'))}",
        f"- alert lift={((tw.get('alert_feed') or {}).get('lift_vs_base'))}",
        f"- n_primary_events_mag_ge_1={tw.get('n_primary_events_mag_ge_1')}",
        "",
    ]

    if report.get("val_comparison"):
        lines += [
            "## Val vs TEST (reporting only; no re-selection)",
            "",
            "| head | model | val ROC | val PR | TEST ROC | TEST PR |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for head_id, row in report["val_comparison"].items():
            for mid, m in row.items():
                lines.append(
                    f"| {head_id} | {mid} | {m.get('val_roc')} | {m.get('val_pr')} | "
                    f"{m.get('test_roc')} | {m.get('test_pr')} |"
                )
        lines.append("")

    lines += [
        "## Scratch pred artifacts (gitignored)",
        "",
        "| key | path | sha256 | n |",
        "|---|---|---|---:|",
    ]
    for k, a in sorted((report.get("pred_artifacts") or {}).items()):
        lines.append(f"| {k} | `{a['path']}` | `{a['sha256']}` | {a['n']} |")
    lines.append("")

    return report, "\n".join(lines) + "\n"


def run_test_once(
    *,
    protocol: Path,
    prereg_sha: str,
    out_dir: Path,
    split: SplitName = "val",
    allow_dirty: bool = False,
    run_bootstrap: bool = True,
    force_stage_b_rerun: bool = False,
) -> dict[str, Any]:
    """Entry with Phase-7 gates for test; val bypasses lock/TEST IO."""
    protocol = Path(protocol)
    out_dir = Path(out_dir)

    if split == "val":
        return run_split_pipeline(
            split="val",
            out_dir=out_dir,
            run_bootstrap=run_bootstrap,
            force_stage_b_rerun=force_stage_b_rerun,
        )

    # ---- TEST path: gates → lock → unlock → score (once) ----
    meta = verify_preregistration(
        protocol_path=protocol,
        prereg_sha=prereg_sha,
        allow_dirty=allow_dirty,
    )
    check_test_once_dir_empty(out_dir)
    lock = write_access_lock(out_dir, head_sha=meta["head_sha"])
    set_q85_test_unlock(lock)
    try:
        scored = run_split_pipeline(
            split="test",
            out_dir=out_dir,
            run_bootstrap=run_bootstrap,
            force_stage_b_rerun=force_stage_b_rerun,
        )
    except Exception:
        # Do not delete the lock; do not silently rerun.
        clear_q85_test_unlock()
        raise

    val_path = SCRATCH / "dry_run_val" / "SCORE_SPLIT_REPORT.json"
    val_scored = None
    if val_path.exists():
        val_scored = json.loads(val_path.read_text(encoding="utf-8"))

    report, md = build_test_once_report(
        scored=scored, prereg_meta=meta, val_scored=val_scored
    )
    write_json(out_dir / "TEST_ONCE_REPORT.json", report)
    (out_dir / "TEST_ONCE_REPORT.md").write_text(md, encoding="utf-8")
    clear_q85_test_unlock()
    return report


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
                run_bootstrap=not args.skip_bootstrap,
                force_stage_b_rerun=args.force_stage_b_rerun,
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
                "split": result.get("split") or args.split,
                "runtime_sec": result.get("runtime_sec"),
                "q85_test_accessed": result.get("q85_test_accessed"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
