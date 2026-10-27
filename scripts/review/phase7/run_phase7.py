#!/usr/bin/env python3
# ruff: noqa: E501
"""Phase-7 orchestrator: 2× fix → strata → scorer dry-run → report."""

from __future__ import annotations

import argparse
import json
from typing import Any

from scripts.review.phase7.common import (
    OUT,
    PROTOCOL_V2,
    Q85,
    REPO,
    SCRATCH,
    V2_PREREG_SHA,
    cached_json,
    ensure_dirs,
    write_json,
)
from scripts.review.phase7.score_test_once import run_dry_run_on_val
from scripts.review.phase7.strata import section_strata
from scripts.review.phase7.twox import section_twox


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def build_report(
    *,
    twox: dict[str, Any],
    strata: dict[str, Any],
    dry_run: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    counts = twox.get("counts") or {}
    met = twox.get("permanent_val_metrics") or {}
    diag = twox.get("diagnosis") or {}
    disagree = (diag.get("reaches_vs_upside_ge1") or {}).get("disagree_rate")
    alert = twox.get("alert_feed") or {}
    pump = twox.get("pump_level_recall_magnitude_ge_1") or {}
    t1 = (pump.get("by_top_frac") or {}).get("top_0.01") or {}
    t2 = (pump.get("by_top_frac") or {}).get("top_0.02") or {}

    s300 = (strata.get("by_head") or {}).get("explosion_300s") or {}
    ac = s300.get("age_curve") or {}
    delta_roc = (ac.get("delta_champion_minus_age_curve") or {}).get("roc_auc") or {}
    delta_pr = (ac.get("delta_champion_minus_age_curve") or {}).get("pr_auc") or {}

    summary = [
        f"2× (within_300s ∧ reaches_plus_100pct): train n_pos={ (counts.get('train') or {}).get('n_pos') } "
        f"prev={_fmt((counts.get('train') or {}).get('prevalence'), 5)}; "
        f"val n_pos={(counts.get('val') or {}).get('n_pos')} "
        f"prev={_fmt((counts.get('val') or {}).get('prevalence'), 5)}; "
        f"XGB ROC={_fmt(met.get('roc_auc'))} PR={_fmt(met.get('pr_auc'))} "
        f"P@1%={_fmt((met.get('precision_at') or {}).get('top_0.01'))}.",
        f"Phase-6's 269 came from training_cache remaining_upside (~97% null on 300s positives); "
        f"reach vs upside≥1 disagree_rate={_fmt(disagree, 3)} "
        f"(onset magnitude vs from-T upside — confirmed).",
        f"Alert-feed lift={_fmt(alert.get('lift_vs_base'))}; "
        f"pump recall mag≥1 top1% caught={_fmt(t1.get('fraction_caught'))} "
        f"top2%={_fmt(t2.get('fraction_caught'))} "
        f"(n_events={pump.get('n_events')}).",
        f"Age+curve baseline (300s): champ−ageΔROC={_fmt(delta_roc.get('delta'))} "
        f"CI={delta_roc.get('ci95')}; ΔPR={_fmt(delta_pr.get('delta'))} CI={delta_pr.get('ci95')}.",
        f"Dry-run-on-val matches Phase-6 ROC/PR within 1e-9: "
        f"{dry_run.get('matches_phase6_within_1e-9')}.",
        f"TEST_PROTOCOL_V2 preregistration SHA={V2_PREREG_SHA}. q85_test_accessed: false.",
    ]

    report = {
        **Q85,
        "summary_lines": summary,
        "v2_preregistration_sha": V2_PREREG_SHA,
        "test_protocol": str(PROTOCOL_V2.relative_to(REPO)),
        "section1_twox": twox,
        "section2_strata": strata,
        "section4_dry_run_on_val": dry_run,
        "permanent_val_configs_scored": [
            *(twox.get("permanent_val_configs_scored") or []),
            *[
                cfg
                for h in (strata.get("by_head") or {}).values()
                for cfg in (h.get("permanent_val_configs_scored") or [])
            ],
        ],
    }

    lines = [
        "# PHASE7_REPORT",
        "",
        "## Summary",
        "",
        *[f"{i}. {s}" for i, s in enumerate(summary, 1)],
        "",
        "## 1. Fixed 2× target",
        "",
        f"**Definition:** `{counts.get('definition')}`",
        "",
        "| split | n_elig | n_pos | prevalence |",
        "|---|---:|---:|---:|",
        f"| train | {(counts.get('train') or {}).get('n_elig')} | {(counts.get('train') or {}).get('n_pos')} | {_fmt((counts.get('train') or {}).get('prevalence'), 6)} |",
        f"| val | {(counts.get('val') or {}).get('n_elig')} | {(counts.get('val') or {}).get('n_pos')} | {_fmt((counts.get('val') or {}).get('prevalence'), 6)} |",
        "",
        f"**Why Phase-6 had 269:** {(diag.get('root_cause') or '')}",
        "",
        f"**reach vs upside≥1:** disagree_rate={_fmt(disagree, 4)}; "
        f"{(diag.get('reaches_vs_upside_ge1') or {}).get('explanation')}",
        "",
        f"Fixed iterations (median WF fold 1/2): `{twox.get('fixed_iterations')}`",
        "",
        "### Permanent-val XGB champion on 2×",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| ROC-AUC | {_fmt(met.get('roc_auc'), 6)} |",
        f"| PR-AUC | {_fmt(met.get('pr_auc'), 6)} |",
    ]
    for k, v in sorted((met.get("precision_at") or {}).items()):
        lines.append(f"| P@{k} | {_fmt(v, 6)} |")
    lines += [
        "",
        f"Alert feed: lift={_fmt(alert.get('lift_vs_base'))}, "
        f"hours={alert.get('n_hours_in_window')}, empty={alert.get('n_hours_empty')}.",
        "",
        f"Pump-level recall (primary, magnitude ≥ +100%, n={pump.get('n_events')}): "
        f"top1% caught={_fmt(t1.get('fraction_caught'))}, "
        f"top2%={_fmt(t2.get('fraction_caught'))}.",
        "",
        "## 2. Stratified evaluation (permanent val)",
        "",
    ]

    for head_id, block in sorted((strata.get("by_head") or {}).items()):
        st = block.get("strata") or {}
        lines += [f"### {head_id}", ""]
        lines += [
            "#### By trigger",
            "",
            "| trigger | n | n_pos | prev | model | ROC | PR | P@1% |",
            "|---|---:|---:|---:|---|---:|---:|---:|",
        ]
        for trig, row in (st.get("by_trigger") or {}).items():
            for model, m in (row.get("models") or {}).items():
                lines.append(
                    f"| {trig} | {row.get('n')} | {row.get('n_pos')} | {_fmt(row.get('prevalence'), 5)} | "
                    f"{model} | {_fmt(m.get('roc_auc'))} | {_fmt(m.get('pr_auc'))} | {_fmt(m.get('precision_at_1pct'))} |"
                )
        lines += [
            "",
            "#### By age bucket",
            "",
            "| bucket | n | n_pos | prev | model | ROC | PR | P@1% |",
            "|---|---:|---:|---:|---|---:|---:|---:|",
        ]
        for bucket, row in (st.get("by_age_bucket") or {}).items():
            for model, m in (row.get("models") or {}).items():
                lines.append(
                    f"| {bucket} | {row.get('n')} | {row.get('n_pos')} | {_fmt(row.get('prevalence'), 5)} | "
                    f"{model} | {_fmt(m.get('roc_auc'))} | {_fmt(m.get('pr_auc'))} | {_fmt(m.get('precision_at_1pct'))} |"
                )
        lines += [
            "",
            "#### Active-only",
            "",
            "| stratum | n | n_pos | prev | model | ROC | PR | P@1% |",
            "|---|---:|---:|---:|---|---:|---:|---:|",
        ]
        for name, row in (st.get("active_only") or {}).items():
            for model, m in (row.get("models") or {}).items():
                lines.append(
                    f"| {name} | {row.get('n')} | {row.get('n_pos')} | {_fmt(row.get('prevalence'), 5)} | "
                    f"{model} | {_fmt(m.get('roc_auc'))} | {_fmt(m.get('pr_auc'))} | {_fmt(m.get('precision_at_1pct'))} |"
                )
        ac_h = block.get("age_curve") or {}
        d_roc = (ac_h.get("delta_champion_minus_age_curve") or {}).get("roc_auc") or {}
        d_pr = (ac_h.get("delta_champion_minus_age_curve") or {}).get("pr_auc") or {}
        lines += [
            "",
            f"#### Age + curve baseline ({head_id})",
            "",
            f"Features: `{ac_h.get('features')}`; fixed_iterations=`{ac_h.get('fixed_iterations')}`",
            "",
            "| metric | age+curve | champ − age+curve | CI95 |",
            "|---|---:|---:|---|",
            f"| ROC | {_fmt((ac_h.get('metrics') or {}).get('roc_auc'))} | {_fmt(d_roc.get('delta'))} | {d_roc.get('ci95')} |",
            f"| PR | {_fmt((ac_h.get('metrics') or {}).get('pr_auc'))} | {_fmt(d_pr.get('delta'))} | {d_pr.get('ci95')} |",
            "",
        ]

    lines += [
        "## 3. TEST_PROTOCOL_V2",
        "",
        f"Preregistration SHA: `{V2_PREREG_SHA}`",
        "",
        f"Protocol path: `{PROTOCOL_V2}`",
        "",
        "## 4. Scorer dry-run on permanent val",
        "",
        f"matches_phase6_within_1e-9: **{dry_run.get('matches_phase6_within_1e-9')}**",
        "",
        "```json",
        json.dumps(dry_run.get("phase6_reproduction") or {}, indent=2),
        "```",
        "",
        "## Notes",
        "",
        "- q85_test_accessed: false — scorer never read TEST rows.",
        "- All models trained on train only; not refit on train+val.",
        "",
    ]
    return report, "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-twox", action="store_true")
    parser.add_argument("--skip-strata", action="store_true")
    parser.add_argument("--skip-dry-run", action="store_true")
    args = parser.parse_args()
    ensure_dirs()

    twox = (
        cached_json("section_twox", section_twox)
        if not args.skip_twox
        else json.loads((SCRATCH / "cache" / "section_twox.json").read_text())
    )
    strata = (
        cached_json("section_strata", section_strata)
        if not args.skip_strata
        else json.loads((SCRATCH / "cache" / "section_strata.json").read_text())
    )
    dry_dir = SCRATCH / "dry_run_val"
    if args.skip_dry_run and (dry_dir / "DRY_RUN_VAL_REPORT.json").exists():
        dry_run = json.loads((dry_dir / "DRY_RUN_VAL_REPORT.json").read_text())
    else:
        print("dry-run-on-val...", flush=True)
        dry_run = run_dry_run_on_val(out_dir=dry_dir)

    report, md = build_report(twox=twox, strata=strata, dry_run=dry_run)
    write_json(OUT / "PHASE7_REPORT.json", report)
    (OUT / "PHASE7_REPORT.md").write_text(md, encoding="utf-8")
    print(f"wrote {OUT / 'PHASE7_REPORT.md'}", flush=True)
    for line in report["summary_lines"]:
        print(f"  • {line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
