#!/usr/bin/env python3
# ruff: noqa: E501
"""Phase-5 orchestrator: sanity → pump eval → tune → history → report."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from scripts.review.phase5.common import (
    OUT,
    Q85,
    SCRATCH,
    cached_json,
    ensure_dirs,
    write_json,
)


def _rpc_configured() -> bool:
    return bool(os.environ.get("HELIUS_API_KEY") or os.environ.get("SOLANA_RPC_URL"))


def section_rpc() -> dict[str, Any]:
    if not _rpc_configured():
        return {
            **Q85,
            "status": "skipped",
            "reason": "Neither HELIUS_API_KEY nor SOLANA_RPC_URL is set",
        }
    # Would call phase4 refetch_micro — not reached when unset.
    from scripts.review.phase4.refetch_micro import fetch_if_configured

    return {**Q85, "status": "started", "fetch": fetch_if_configured()}


def build_report(
    *,
    sanity: dict[str, Any],
    pump: dict[str, Any],
    tune: dict[str, Any],
    history: dict[str, Any],
    rpc: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    # Champion alert-feed headline from XGB (stronger in phase4)
    alert_xgb = (pump.get("alert_feed_hourly") or {}).get("xgb") or {}
    pump_xgb_60 = (pump.get("pump_level_recall") or {}).get("xgb_explosion_60s") or {}
    top1 = (pump_xgb_60.get("by_top_frac") or {}).get("top_0.01") or {}
    top2 = (pump_xgb_60.get("by_top_frac") or {}).get("top_0.02") or {}

    # Tuning help?
    tune_help = []
    for key, block in (tune.get("by_family_head") or {}).items():
        pv = block.get("permanent_val") or {}
        d = (pv.get("delta_tuned_minus_frozen_phase4") or {}).get("pr_auc") or {}
        tune_help.append(
            {
                "key": key,
                "delta_pr": d.get("delta"),
                "ci95": d.get("ci95"),
                "tuned_pr": (pv.get("permanent_val_metrics") or {}).get("pr_auc"),
                "tuned_roc": (pv.get("permanent_val_metrics") or {}).get("roc_auc"),
            }
        )

    # Learning curve rising?
    rising_notes = []
    for key, cur in ((history.get("learning_curve") or {}).get("curves") or {}).items():
        rising_notes.append(
            {
                "key": key,
                "pr_rising": cur.get("pr_auc_still_rising_at_100pct"),
                "roc_rising": cur.get("roc_auc_still_rising_at_100pct"),
            }
        )

    n_trades_a = (sanity.get("arms") or {}).get("explosion_60s", {}).get("A_plus_n_trades", {})
    n_trades_base = (sanity.get("arms") or {}).get("explosion_60s", {}).get("A", {})

    summary_lines = [
        f"(1) Champion alert-feed (XGB top-10/hour deduped mint): mean frac explode@300s="
        f"{alert_xgb.get('mean_frac_explode_300s')} vs base={alert_xgb.get('base_rate_eligible')}; "
        f"pump recall top1% caught={top1.get('fraction_caught')} "
        f"lead_p50={(top1.get('lead_time_sec_percentiles') or {}).get('p50')}s; "
        f"top2% caught={top2.get('fraction_caught')} "
        f"FA/caught={top2.get('false_alerts_per_caught_pump')}.",
        "(2) Tuning: "
        + (
            "; ".join(
                f"{t['key']} ΔPR={t['delta_pr']} CI={t['ci95']}" for t in tune_help
            )
            if tune_help
            else "not run"
        ),
        "(3) More history: "
        + (
            "; ".join(
                f"{r['key']} PR rising@100%={r['pr_rising']}" for r in rising_notes
            )
            if rising_notes
            else "curve pending"
        ),
        f"(4) Clean micro: {rpc.get('status')} — {rpc.get('reason', rpc.get('status'))}.",
        f"Sanity n_trades: now injected; A ROC60={n_trades_base.get('roc_auc')} "
        f"A+n_trades ROC60={n_trades_a.get('roc_auc')} "
        f"(n_feat A={n_trades_base.get('n_features')}, "
        f"A+n={n_trades_a.get('n_features')}).",
        f"Determinism: {sanity.get('determinism_check')}.",
        "q85_test_accessed: false",
    ]

    report = {
        **Q85,
        "summary_lines": summary_lines,
        "section1_sanity": sanity,
        "section2_pump_eval": pump,
        "section3_tune": tune,
        "section4_history": history,
        "section5_rpc": rpc,
        "permanent_val_configs_scored_log": tune.get("permanent_val_configs_scored", [])
        + [{"source": "phase4_frozen_v2_preds", "path": str(pump.get("champion_source"))}],
    }

    # Markdown
    lines = [
        "# PHASE5_REPORT",
        "",
        "## Summary",
        *[f"{i}. {s}" for i, s in enumerate(summary_lines, 1)],
        "",
        "## 1. Sanity checks",
        "",
        f"**n_trades root cause:** {(sanity.get('n_trades_diagnosis') or {}).get('root_cause')}",
        "",
        "### n_trades_before_T distribution (rich)",
        "",
        "```json",
        json.dumps(sanity.get("n_trades_distribution"), indent=2)[:2000],
        "```",
        "",
        "### Re-run rich arms (deterministic)",
        "",
        "| head | arm | n_features | ROC-AUC | PR-AUC |",
        "|---|---|---:|---:|---:|",
    ]
    for head, arms in (sanity.get("arms") or {}).items():
        for arm, m in arms.items():
            lines.append(
                f"| {head} | {arm} | {m.get('n_features')} | {m.get('roc_auc'):.4f} | {m.get('pr_auc'):.4f} |"
            )
    lines += [
        "",
        "### Determinism",
        "",
        "```json",
        json.dumps(sanity.get("determinism_check"), indent=2),
        "```",
        "",
        "### Phase3 vs Phase4 graph/wallet 300s",
        "",
        "```json",
        json.dumps(sanity.get("phase3_vs_phase4_graph_wallet"), indent=2)[:2500],
        "```",
        "",
        "## 2. Pump-level evaluation (broad permanent val, frozen V2 trees)",
        "",
        f"TTE overlay join: `{json.dumps(pump.get('tte_overlay_join'))}`",
        "",
        f"already_started join: `{json.dumps(pump.get('already_started_join'))}`",
        "",
        "### Metrics by variant",
        "",
        "| head | model | variant | ROC | PR | P@0.1% | P@0.5% | P@1% | P@2% |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for head, models in (pump.get("metrics") or {}).items():
        for fam, variants in models.items():
            for vname, m in variants.items():
                pa = m.get("precision_at") or {}
                lines.append(
                    f"| {head} | {fam} | {vname} | {m.get('roc_auc'):.4f} | {m.get('pr_auc'):.4f} | "
                    f"{pa.get('top_0.001', float('nan')):.4f} | {pa.get('top_0.005', float('nan')):.4f} | "
                    f"{pa.get('top_0.01', float('nan')):.4f} | {pa.get('top_0.02', float('nan')):.4f} |"
                )

    lines += ["", "### Pump-level recall", ""]
    for key, rec in (pump.get("pump_level_recall") or {}).items():
        lines.append(f"**{key}** (n_events={rec.get('n_primary_val_events')})")
        lines.append("")
        lines.append("| top | fraction_caught | lead_p50 | FA/caught |")
        lines.append("|---|---:|---:|---:|")
        for frac, b in (rec.get("by_top_frac") or {}).items():
            p50 = (b.get("lead_time_sec_percentiles") or {}).get("p50")
            lines.append(
                f"| {frac} | {b.get('fraction_caught')} | {p50} | {b.get('false_alerts_per_caught_pump')} |"
            )
        lines.append("")

    lines += [
        "### Alert feed (top-10/hour, mint-deduped)",
        "",
        "| model | mean_frac@300s | base_rate | lift | n_hours |",
        "|---|---:|---:|---:|---:|",
    ]
    for fam, a in (pump.get("alert_feed_hourly") or {}).items():
        lines.append(
            f"| {fam} | {a.get('mean_frac_explode_300s')} | {a.get('base_rate_eligible')} | "
            f"{a.get('lift_vs_base')} | {a.get('n_hours')} |"
        )

    lines += ["", "## 3. WF tuning", ""]
    for key, block in (tune.get("by_family_head") or {}).items():
        sel = block.get("selection") or {}
        pv = block.get("permanent_val") or {}
        met = pv.get("permanent_val_metrics") or {}
        dpr = (pv.get("delta_tuned_minus_frozen_phase4") or {}).get("pr_auc") or {}
        droc = (pv.get("delta_tuned_minus_frozen_phase4") or {}).get("roc_auc") or {}
        lines += [
            f"### {key}",
            "",
            f"- Best WF mean PR-AUC: **{sel.get('mean_pr_auc_wf')}**",
            f"- Config: `{json.dumps(sel.get('best_config'))}`",
            f"- Fixed iters (median best): **{sel.get('median_best_iterations')}**",
            f"- Permanent val ROC/PR: **{met.get('roc_auc')}** / **{met.get('pr_auc')}**",
            f"- ΔPR vs frozen V2 (tuned−frozen): {dpr.get('delta')} CI {dpr.get('ci95')}",
            f"- ΔROC vs frozen V2: {droc.get('delta')} CI {droc.get('ci95')}",
            "",
        ]
    lines.append(
        f"Permanent-val configs scored log: "
        f"`{json.dumps(tune.get('permanent_val_configs_scored'), indent=2)}`"
    )

    lines += ["", "## 4. History lever", ""]
    cov = history.get("coverage") or {}
    lines += [
        "### Coverage",
        f"- Freeze: `{json.dumps(cov.get('freeze'))}`",
        f"- Pumpfun trades: `{json.dumps((cov.get('external_corpora') or {}).get('pumpfun_memecoin_corpus'))}`",
        f"- RED-PUMP: `{(cov.get('external_corpora') or {}).get('red_pump_2026')}`",
        f"- RED-COHORT: `{(cov.get('external_corpora') or {}).get('red_cohort_2026')}`",
        f"- Solarchive: `{(cov.get('external_corpora') or {}).get('solarchive')}`",
        f"- Dune: `{json.dumps(cov.get('dune'))}`",
        "",
        "### Cost to extend",
        "```json",
        json.dumps(history.get("cost_to_extend"), indent=2)[:3000],
        "```",
        "",
        "### Learning curve",
        "",
    ]
    for key, cur in ((history.get("learning_curve") or {}).get("curves") or {}).items():
        lines.append(f"**{key}** — PR rising@100%={cur.get('pr_auc_still_rising_at_100pct')}")
        lines.append("")
        lines.append("| fraction | n_train | ROC | PR |")
        lines.append("|---:|---:|---:|---:|")
        for p in cur.get("points") or []:
            lines.append(
                f"| {p['fraction']} | {p['n_train_eligible']} | {p['roc_auc']:.4f} | {p['pr_auc']:.4f} |"
            )
        lines.append("")

    lines += [
        "## 5. Causal micro refetch",
        "",
        f"Status: **{rpc.get('status')}** — {rpc.get('reason', '')}",
        "",
        "---",
        "q85_test_accessed: false",
        "",
    ]
    md = "\n".join(lines)
    return report, md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["sanity", "pump", "tune", "history", "rpc", "report", "all"],
        default="all",
    )
    args = parser.parse_args(argv)
    ensure_dirs()

    sanity = pump = tune = history = rpc = None
    cache = SCRATCH / "cache"

    if args.only in ("sanity", "all"):
        from scripts.review.phase5.sanity import section_sanity

        print("=== section1 sanity ===", flush=True)
        sanity = cached_json("section_sanity", section_sanity)

    if args.only in ("pump", "all"):
        from scripts.review.phase5.pump_eval import section_pump_eval

        print("=== section2 pump eval ===", flush=True)
        pump = cached_json("section_pump_eval", section_pump_eval)

    if args.only in ("tune", "all"):
        from scripts.review.phase5.tune import section_tune

        print("=== section3 tune ===", flush=True)
        tune = cached_json("section_tune", section_tune)

    if args.only in ("history", "all"):
        from scripts.review.phase5.history import section_history

        if tune is None and (cache / "section_tune.json").exists():
            tune = json.loads((cache / "section_tune.json").read_text(encoding="utf-8"))
        print("=== section4 history ===", flush=True)
        history = cached_json(
            "section_history", lambda: section_history(tune)
        )

    if args.only in ("rpc", "all"):
        print("=== section5 rpc ===", flush=True)
        rpc = cached_json("section_rpc", section_rpc)

    if args.only in ("report", "all"):
        def _load(name: str) -> dict[str, Any]:
            p = cache / f"{name}.json"
            if not p.exists():
                return {**Q85, "missing": True}
            return json.loads(p.read_text(encoding="utf-8"))

        sanity = sanity or _load("section_sanity")
        pump = pump or _load("section_pump_eval")
        tune = tune or _load("section_tune")
        history = history or _load("section_history")
        rpc = rpc or _load("section_rpc")
        report, md = build_report(
            sanity=sanity, pump=pump, tune=tune, history=history, rpc=rpc
        )
        write_json(OUT / "PHASE5_REPORT.json", report)
        (OUT / "PHASE5_REPORT.md").write_text(md + "\n", encoding="utf-8")
        print(f"wrote {OUT / 'PHASE5_REPORT.md'}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
