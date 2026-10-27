#!/usr/bin/env python3
# ruff: noqa: E501
"""Phase-6 orchestrator: pump fix → baselines → labels → stability → economics → report."""

from __future__ import annotations

import argparse
import json
from typing import Any

from scripts.review.phase6.common import (
    OUT,
    Q85,
    SCRATCH,
    cached_json,
    ensure_dirs,
    write_json,
)


def build_report(
    *,
    pump: dict[str, Any],
    baselines: dict[str, Any],
    labels: dict[str, Any],
    stability: dict[str, Any],
    economics: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    # Champion vs baselines headline
    beat = []
    head = "explosion_300s"
    b300 = ((baselines.get("by_head") or {}).get(head) or {}).get("baselines") or {}
    champ = b300.get("xgb_champion_frozen_v2", {}).get("variants", {}).get("all", {})
    for name, block in b300.items():
        if name == "xgb_champion_frozen_v2":
            continue
        v = (block.get("variants") or {}).get("all") or {}
        d = (v.get("delta_champion_minus_baseline") or {}).get("pr_auc") or {}
        beat.append(
            {
                "baseline": name,
                "champ_pr": champ.get("pr_auc"),
                "base_pr": v.get("pr_auc"),
                "delta_pr": d.get("delta"),
                "ci95": d.get("ci95"),
                "champ_beats": (
                    d.get("ci95") is not None
                    and d["ci95"][0] is not None
                    and d["ci95"][0] > 0
                ),
            }
        )

    # Pump recall primary xgb 60s top 1%/2%
    prim = (pump.get("pump_level_recall") or {}).get("xgb_explosion_60s_primary") or {}
    t1 = (prim.get("by_top_frac") or {}).get("top_0.01") or {}
    t2 = (prim.get("by_top_frac") or {}).get("top_0.02") or {}

    # 2x target
    twox = (labels.get("targets") or {}).get("explosion_300s_and_upside_ge_1") or {}

    # Daily stability
    days = (stability.get("permanent_val_by_day") or {}).get("xgb_explosion_300s") or []
    day_rocs = [d["roc_auc"] for d in days if d.get("roc_auc") is not None]

    summary = [
        "(1) Champion vs simple baselines (explosion_300s all): "
        + (
            "beats all with PR CI>0"
            if beat and all(b.get("champ_beats") for b in beat)
            else "see table — "
            + ", ".join(
                f"{b['baseline']} ΔPR={b['delta_pr']} beat={b['champ_beats']}"
                for b in beat
            )
        ),
        f"(2) Broad primary pumps in val onset window: "
        f"n={pump.get('event_counts_onset_in_val_window', {}).get('primary')}; "
        f"XGB@60s top1% caught={t1.get('fraction_caught')} "
        f"lead_p50={(t1.get('lead_time_sec_percentiles') or {}).get('p50')}; "
        f"top2% caught={t2.get('fraction_caught')} "
        f"lead_p50={(t2.get('lead_time_sec_percentiles') or {}).get('p50')}.",
        f"(3) 2× target (300s & upside≥1): n_pos={twox.get('n_pos')} "
        f"prev={twox.get('prevalence')} ROC={(twox.get('metrics') or {}).get('roc_auc')} "
        f"PR={(twox.get('metrics') or {}).get('pr_auc')}; "
        f"daily XGB@300s ROC range "
        f"[{min(day_rocs):.3f},{max(day_rocs):.3f}]" if day_rocs else "n/a",
        f"Alert feed XGB: lift="
        f"{(pump.get('alert_feed_hourly') or {}).get('xgb', {}).get('lift_vs_base')} "
        f"hours={((pump.get('alert_feed_hourly') or {}).get('xgb') or {}).get('n_hours_in_window')} "
        f"empty={((pump.get('alert_feed_hourly') or {}).get('xgb') or {}).get('n_hours_empty')}.",
        "TEST_PROTOCOL_V1.md committed before any TEST access.",
        "q85_test_accessed: false",
    ]

    report = {
        **Q85,
        "summary_lines": summary,
        "section1_pump_eval": pump,
        "section2_baselines": baselines,
        "section3_labels": labels,
        "section4_stability": stability,
        "section5_economics": economics,
        "champion_vs_baselines_300s": beat,
        "test_protocol": "manifests/world_model_data/evaluation/TEST_PROTOCOL_V1.md",
    }

    lines = [
        "# PHASE6_REPORT",
        "",
        "## Summary",
        *[f"{i}. {s}" for i, s in enumerate(summary, 1)],
        "",
        "## 1. Fixed pump-level evaluation (broad)",
        "",
        f"Event counts (onset in [q70,q85)): `{pump.get('event_counts_onset_in_val_window')}`",
        "",
        f"Phase5 bug note: `{pump.get('note')}`",
        "",
        "### Pump recall (XGB)",
        "",
        "| defn | head | top | fraction_caught | lead_p50 | FA/caught |",
        "|---|---|---|---:|---:|---:|",
    ]
    for key, rec in sorted((pump.get("pump_level_recall") or {}).items()):
        if not key.startswith("xgb_"):
            continue
        for frac, b in (rec.get("by_top_frac") or {}).items():
            parts = key.split("_")
            # xgb_explosion_60s_primary
            defn = parts[-1]
            head = "_".join(parts[1:-1])
            p50 = (b.get("lead_time_sec_percentiles") or {}).get("p50")
            lines.append(
                f"| {defn} | {head} | {frac} | {b.get('fraction_caught')} | {p50} | "
                f"{b.get('false_alerts_per_caught_event')} |"
            )

    lines += [
        "",
        "### Alert feed",
        "",
        "| model | mean_frac | base | lift | hours | empty |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fam, a in (pump.get("alert_feed_hourly") or {}).items():
        lines.append(
            f"| {fam} | {a.get('mean_frac_explode_300s')} | {a.get('base_rate_eligible')} | "
            f"{a.get('lift_vs_base')} | {a.get('n_hours_in_window')} | {a.get('n_hours_empty')} |"
        )

    lines += ["", "## 2. Simple baselines vs XGB champion", ""]
    for head_id, block in (baselines.get("by_head") or {}).items():
        lines += [f"### {head_id}", "", "| model | ROC | PR | P@1% | ΔPR champ−base CI | alert lift |", "|---|---:|---:|---:|---|---:|"]
        for name, b in (block.get("baselines") or {}).items():
            v = (b.get("variants") or {}).get("all") or {}
            d = (v.get("delta_champion_minus_baseline") or {}).get("pr_auc") or {}
            af = b.get("alert_feed") or {}
            lines.append(
                f"| {name} | {v.get('roc_auc')} | {v.get('pr_auc')} | "
                f"{(v.get('precision_at') or {}).get('top_0.01')} | "
                f"{d.get('delta')} {d.get('ci95')} | {af.get('lift_vs_base')} |"
            )
        lines.append("")
        lines.append("Group ablation / TreeSHAP: see JSON.")
        shap = block.get("treeshap") or {}
        if shap.get("top20"):
            lines += ["", "Top-10 SHAP:", ""]
            for row in shap["top20"][:10]:
                lines.append(f"- {row['feature']}: {row['mean_abs_shap']:.6f}")
        lines.append("")

    lines += ["", "## 3. Label sensitivity / 2× target", ""]
    lines.append(f"Data check: `{json.dumps(labels.get('data_check'), indent=2)[:2500]}`")
    lines += ["", "| target | n_pos | prevalence | ROC | PR |", "|---|---:|---:|---:|---:|"]
    for name, t in (labels.get("targets") or {}).items():
        m = t.get("metrics") or {}
        lines.append(
            f"| {name} | {t.get('n_pos')} | {t.get('prevalence')} | {m.get('roc_auc')} | {m.get('pr_auc')} |"
        )

    lines += ["", "## 4. Stability", ""]
    for head, folds in (stability.get("wf_folds") or {}).items():
        lines.append(f"### {head}")
        for fid, rec in folds.items():
            lines.append(
                f"- {fid}: ROC={rec.get('roc_auc')} PR={rec.get('pr_auc')} n_pos={rec.get('n_pos')}"
            )
        lines.append("")
    lines.append("Daily permanent-val metrics: see JSON `permanent_val_by_day`.")

    lines += [
        "",
        "## 5. Economic context (descriptive)",
        "",
        f"```json\n{json.dumps(economics, indent=2)[:3000]}\n```",
        "",
        "## 6. Test protocol",
        "",
        "See `manifests/world_model_data/evaluation/TEST_PROTOCOL_V1.md` (preregistration; no TEST executed).",
        "",
        "---",
        "q85_test_accessed: false",
        "",
    ]
    return report, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=[
            "pump",
            "baselines",
            "labels",
            "stability",
            "economics",
            "report",
            "all",
        ],
        default="all",
    )
    args = parser.parse_args(argv)
    ensure_dirs()
    cache = SCRATCH / "cache"

    if args.only in ("pump", "all"):
        from scripts.review.phase6.pump_eval import section_pump_eval

        print("=== section1 pump ===", flush=True)
        cached_json("section_pump_eval", section_pump_eval)

    if args.only in ("baselines", "all"):
        from scripts.review.phase6.baselines import section_baselines

        print("=== section2 baselines ===", flush=True)
        cached_json("section_baselines", section_baselines)

    if args.only in ("labels", "all"):
        from scripts.review.phase6.labels import section_labels

        print("=== section3 labels ===", flush=True)
        cached_json("section_labels", section_labels)

    if args.only in ("stability", "all"):
        from scripts.review.phase6.stability import section_stability

        print("=== section4 stability ===", flush=True)
        cached_json("section_stability", section_stability)

    if args.only in ("economics", "all"):
        from scripts.review.phase6.economics import section_economics

        print("=== section5 economics ===", flush=True)
        cached_json("section_economics", section_economics)

    if args.only in ("report", "all"):
        def _load(name: str) -> dict[str, Any]:
            p = cache / f"{name}.json"
            if not p.exists():
                return {**Q85, "missing": True}
            return json.loads(p.read_text(encoding="utf-8"))

        report, md = build_report(
            pump=_load("section_pump_eval"),
            baselines=_load("section_baselines"),
            labels=_load("section_labels"),
            stability=_load("section_stability"),
            economics=_load("section_economics"),
        )
        write_json(OUT / "PHASE6_REPORT.json", report)
        (OUT / "PHASE6_REPORT.md").write_text(md + "\n", encoding="utf-8")
        print(f"wrote {OUT / 'PHASE6_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
