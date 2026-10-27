#!/usr/bin/env python3
# ruff: noqa: E501
"""Phase-8 orchestrator: freeze models → val pipeline proof → TEST counts → report."""

from __future__ import annotations

import argparse
import json
from typing import Any

from scripts.review.phase8.common import (
    MANIFEST_PATH,
    OUT,
    Q85,
    REPO,
    SCRATCH,
    V2_PREREG_SHA,
    cached_json,
    ensure_dirs,
    write_json,
)
from scripts.review.phase8.freeze_models import build_manifest
from scripts.review.phase8.score_test_once import run_split_pipeline
from scripts.review.phase8.test_pop import test_population_from_metadata


def _load_phase_refs() -> dict[str, Any]:
    p6 = json.loads(
        (
            REPO
            / "manifests/world_model_data/evaluation/phase6_paper_v1/PHASE6_REPORT.json"
        ).read_text(encoding="utf-8")
    )
    p7 = json.loads(
        (
            REPO
            / "manifests/world_model_data/evaluation/phase7_pretest_v1/PHASE7_REPORT.json"
        ).read_text(encoding="utf-8")
    )
    p5_60 = json.loads(
        (REPO / "scratch/phase5_trees_v1/cache/score_lightgbm_explosion_60s.json").read_text()
    )["permanent_val_metrics"]
    p5_300 = json.loads(
        (REPO / "scratch/phase5_trees_v1/cache/score_lightgbm_explosion_300s.json").read_text()
    )["permanent_val_metrics"]
    return {"p6": p6, "p7": p7, "p5_lgbm": {"explosion_60s": p5_60, "explosion_300s": p5_300}}


def _row(
    metric: str, phase_val: float | None, dry_val: float | None, tol: float = 1e-6
) -> dict[str, Any]:
    if phase_val is None or dry_val is None:
        diff = None
        ok = False
    else:
        diff = abs(float(dry_val) - float(phase_val))
        ok = diff <= tol
    return {
        "metric": metric,
        "phase_report_value": phase_val,
        "dry_run_value": dry_val,
        "abs_diff": diff,
        "pass": ok,
    }


def build_reproduction_table(
    *, scored: dict[str, Any], refs: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p6 = refs["p6"]
    p7 = refs["p7"]

    # XGB champion from phase6 baselines
    for head in ("explosion_60s", "explosion_300s"):
        champ = (
            p6["section2_baselines"]["by_head"][head]["baselines"]["xgb_champion_frozen_v2"][
                "variants"
            ]["all"]
        )
        got = scored["by_head"][head]["variants"]["all"]["models"]["xgb_champion"]
        rows.append(_row(f"{head}/xgb_champion/roc_auc", champ["roc_auc"], got["roc_auc"]))
        rows.append(_row(f"{head}/xgb_champion/pr_auc", champ["pr_auc"], got["pr_auc"]))

        log = p6["section2_baselines"]["by_head"][head]["baselines"]["logistic_l2"]["variants"][
            "all"
        ]
        got_l = scored["by_head"][head]["variants"]["all"]["models"]["logistic_l2"]
        rows.append(_row(f"{head}/logistic_l2/roc_auc", log["roc_auc"], got_l["roc_auc"]))
        rows.append(_row(f"{head}/logistic_l2/pr_auc", log["pr_auc"], got_l["pr_auc"]))

        # LGBM from phase5
        p5 = refs["p5_lgbm"][head]
        got_lgb = scored["by_head"][head]["variants"]["all"]["models"]["lgbm_tuned"]
        rows.append(_row(f"{head}/lgbm_tuned/roc_auc", p5["roc_auc"], got_lgb["roc_auc"]))
        rows.append(_row(f"{head}/lgbm_tuned/pr_auc", p5["pr_auc"], got_lgb["pr_auc"]))

        # age_curve from phase7
        ac = p7["section2_strata"]["by_head"][head]["overall"]["models"]["age_curve_xgb"]
        got_ac = scored["by_head"][head]["variants"]["all"]["models"]["age_curve_xgb"]
        rows.append(_row(f"{head}/age_curve_xgb/roc_auc", ac["roc_auc"], got_ac["roc_auc"]))
        rows.append(_row(f"{head}/age_curve_xgb/pr_auc", ac["pr_auc"], got_ac["pr_auc"]))

        # stage_b from phase7 strata overall
        sb = p7["section2_strata"]["by_head"][head]["overall"]["models"]["stage_b"]
        got_sb = scored["by_head"][head]["variants"]["all"]["models"]["stage_b"]
        rows.append(_row(f"{head}/stage_b/roc_auc", sb["roc_auc"], got_sb["roc_auc"]))
        rows.append(_row(f"{head}/stage_b/pr_auc", sb["pr_auc"], got_sb["pr_auc"]))

        # Strata P@1% sample: fixed_age xgb
        st_ref = p7["section2_strata"]["by_head"][head]["strata"]["by_trigger"]["fixed_age"]
        st_got = scored["by_head"][head]["strata"]["by_trigger"]["fixed_age"]
        rows.append(
            _row(
                f"{head}/strata/fixed_age/n",
                float(st_ref["n"]),
                float(st_got["n"]),
            )
        )
        rows.append(
            _row(
                f"{head}/strata/fixed_age/xgb_champion/roc_auc",
                st_ref["models"]["xgb_champion"]["roc_auc"],
                st_got["models"]["xgb_champion"]["roc_auc"],
            )
        )

    # 2× from phase7
    tw = p7["section1_twox"]
    got_tw = scored["twox"]
    rows.append(
        _row("twox/n_pos", float(tw["counts"]["val"]["n_pos"]), float(got_tw["n_pos"]))
    )
    rows.append(
        _row(
            "twox/roc_auc",
            tw["permanent_val_metrics"]["roc_auc"],
            got_tw["metrics"]["roc_auc"],
        )
    )
    rows.append(
        _row(
            "twox/pr_auc",
            tw["permanent_val_metrics"]["pr_auc"],
            got_tw["metrics"]["pr_auc"],
        )
    )
    rows.append(
        _row(
            "twox/alert_lift",
            tw["alert_feed"]["lift_vs_base"],
            got_tw["alert_feed"]["lift_vs_base"],
        )
    )
    t1_ref = tw["pump_level_recall_magnitude_ge_1"]["by_top_frac"]["top_0.01"][
        "fraction_caught"
    ]
    t1_got = got_tw["pump_level_recall_magnitude_ge_1"]["by_top_frac"]["top_0.01"][
        "fraction_caught"
    ]
    rows.append(_row("twox/pump_recall_top1pct", t1_ref, t1_got))

    # Alert feed phase6 xgb
    alert_p6 = (p6.get("section1_pump_eval") or {}).get("alert_feed_hourly") or {}
    if "xgb" in alert_p6 and scored["by_head"]["explosion_300s"].get("alert_feed"):
        rows.append(
            _row(
                "explosion_300s/alert_lift",
                alert_p6["xgb"]["lift_vs_base"],
                scored["by_head"]["explosion_300s"]["alert_feed"]["lift_vs_base"],
            )
        )

    return rows


def build_report(
    *,
    manifest: dict[str, Any],
    scored: dict[str, Any],
    repro: list[dict[str, Any]],
    test_pop: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    n_pass = sum(1 for r in repro if r["pass"])
    n_fail = sum(1 for r in repro if not r["pass"])
    summary = [
        f"Frozen {manifest['n_models']} models under scratch/test_models_v2/; "
        f"all val_reproduction_max_abs_diff ≤ 1e-6.",
        f"Full val pipeline runtime_sec={scored.get('runtime_sec'):.1f}, "
        f"peak_rss_mb={scored.get('peak_rss_mb'):.1f}.",
        f"Reproduction table: {n_pass} pass / {n_fail} fail (tol 1e-6).",
        f"TEST population (metadata only): mints={test_pop['n_test_mints']} "
        f"obs={test_pop['n_test_observations']} "
        f"launch=[{test_pop['launch_time_range_unix']['min']}, "
        f"{test_pop['launch_time_range_unix']['max']}].",
        f"V2 prereg SHA={V2_PREREG_SHA}. q85_test_accessed: false. --split test not run.",
    ]
    report = {
        **Q85,
        "summary_lines": summary,
        "v2_preregistration_sha": V2_PREREG_SHA,
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO)),
        "manifest": {
            "n_models": manifest["n_models"],
            "model_ids": list(manifest["models"].keys()),
            "all_val_reproduction_pass": manifest.get("all_val_reproduction_pass"),
        },
        "val_pipeline": {
            "runtime_sec": scored.get("runtime_sec"),
            "peak_rss_mb": scored.get("peak_rss_mb"),
            "stage_b_meta": scored.get("stage_b_meta"),
            "feature_meta": scored.get("feature_meta"),
        },
        "reproduction_table": repro,
        "reproduction_all_pass": n_fail == 0,
        "test_population_metadata": test_pop,
        "scored_val_summary": {
            "by_head_keys": list((scored.get("by_head") or {}).keys()),
            "twox_n_pos": (scored.get("twox") or {}).get("n_pos"),
            "twox_roc": ((scored.get("twox") or {}).get("metrics") or {}).get("roc_auc"),
        },
    }
    lines = [
        "# PHASE8_REPORT",
        "",
        "## Summary",
        "",
        *[f"{i}. {s}" for i, s in enumerate(summary, 1)],
        "",
        "## 1. Frozen models",
        "",
        f"Manifest: `{MANIFEST_PATH.relative_to(REPO)}`",
        "",
        "| model_id | sha256 | fixed_iters | val_max_abs_diff |",
        "|---|---|---:|---:|",
    ]
    for mid, m in sorted(manifest["models"].items()):
        lines.append(
            f"| {mid} | `{m['sha256'][:12]}…` | {m.get('fixed_iterations')} | "
            f"{m.get('val_reproduction_max_abs_diff')} |"
        )
    lines += [
        "",
        "## 2. Val pipeline reproduction (tol 1e-6)",
        "",
        f"runtime_sec={scored.get('runtime_sec')} peak_rss_mb={scored.get('peak_rss_mb')}",
        "",
        "| metric | phase-report value | dry-run value | abs diff | pass |",
        "|---|---:|---:|---:|:---:|",
    ]
    for r in repro:
        lines.append(
            f"| {r['metric']} | {r['phase_report_value']} | {r['dry_run_value']} | "
            f"{r['abs_diff']} | {'PASS' if r['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        f"**All pass:** {n_fail == 0} ({n_pass}/{len(repro)})",
        "",
        "## 3. TEST population (metadata only)",
        "",
        f"- n_test_mints: {test_pop['n_test_mints']}",
        f"- n_test_observations: {test_pop['n_test_observations']}",
        f"- launch_time_range_unix: {test_pop['launch_time_range_unix']}",
        f"- q85_unix: {test_pop['q85_unix']}",
        "",
        f"Source: {test_pop['source']}",
        "",
        "## Notes",
        "",
        "- q85_test_accessed: false",
        "- `--split test` was not run",
        "- TEST_PROTOCOL_V2.md was not edited",
        "",
    ]
    return report, "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-freeze", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()
    ensure_dirs()

    if args.skip_freeze and MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        manifest = cached_json("manifest_build", build_manifest)
        # cached_json may wrap; build_manifest also writes MANIFEST_PATH
        if "models" not in manifest:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    dry_dir = SCRATCH / "dry_run_val"
    if args.skip_score and (dry_dir / "SCORE_SPLIT_REPORT.json").exists():
        scored = json.loads((dry_dir / "SCORE_SPLIT_REPORT.json").read_text())
    else:
        print("running full val pipeline...", flush=True)
        scored = run_split_pipeline(
            split="val",
            out_dir=dry_dir,
            run_bootstrap=not args.skip_bootstrap,
        )

    refs = _load_phase_refs()
    repro = build_reproduction_table(scored=scored, refs=refs)
    test_pop = test_population_from_metadata()
    write_json(SCRATCH / "cache" / "test_pop.json", test_pop)

    report, md = build_report(
        manifest=manifest, scored=scored, repro=repro, test_pop=test_pop
    )
    write_json(OUT / "PHASE8_REPORT.json", report)
    (OUT / "PHASE8_REPORT.md").write_text(md, encoding="utf-8")
    # Ensure manifest is committed path
    if not MANIFEST_PATH.exists():
        write_json(MANIFEST_PATH, manifest)
    print(f"wrote {OUT / 'PHASE8_REPORT.md'}", flush=True)
    for line in report["summary_lines"]:
        print(f"  • {line}", flush=True)
    if not report["reproduction_all_pass"]:
        fails = [r for r in repro if not r["pass"]]
        print(f"FAIL {len(fails)} reproduction rows", flush=True)
        for r in fails[:20]:
            print(f"  FAIL {r}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
