#!/usr/bin/env python3
"""Generate every table and figure in the paper from committed evaluation reports.

No number in tables/ or figures/ is typed by hand: each value is read from a
JSON report under manifests/world_model_data/evaluation/ (or, for the detector
and feature list, from the source code that defines them).

Usage (from the repository root):
    python paper/scripts/make_assets.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("TRENCHER_REPO", PAPER.parent)).resolve()
EV = REPO / "manifests/world_model_data/evaluation"
WMD = REPO / "manifests/world_model_data"
TAB = PAPER / "tables"
FIG = PAPER / "figures"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "src"))


def load(rel: str) -> dict:
    return json.loads((EV / rel).read_text(encoding="utf-8"))


TEST = load("test_once_v1/TEST_ONCE_REPORT.json")
P3 = load("phase3_leak_v1/PHASE3_REPORT.json")
P5 = load("phase5_trees_v1/PHASE5_REPORT.json")
P6 = load("phase6_paper_v1/PHASE6_REPORT.json")
P7 = load("phase7_pretest_v1/PHASE7_REPORT.json")
SPLIT = json.loads((WMD / "SPLIT_AUDIT_V1.json").read_text(encoding="utf-8"))
BSUP = json.loads((WMD / "BROAD_SUPERVISION_AUDIT_V3.json").read_text(encoding="utf-8"))

MODELS = [
    ("xgb_champion", "XGBoost (champion)"),
    ("lgbm_tuned", "LightGBM (tuned)"),
    ("logistic_l2", "Logistic regression (L2)"),
    ("stage_b", "Transformer"),
    ("age_curve_xgb", "Age + curve XGBoost"),
]
HEADS = [("explosion_60s", "60 s"), ("explosion_300s", "300 s")]


def f3(x: float) -> str:
    return f"{x:.3f}"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}\\%"


def ci(d: dict, digits: int = 3) -> str:
    lo, hi = d["ci95"]
    return f"{d['delta']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]".replace("-", "$-$")


def intc(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", "{,}")


def write(name: str, body: str) -> None:
    (TAB / name).write_text(body.strip() + "\n", encoding="utf-8")


def bold_max(vals: list[float], digits: int = 3) -> list[str]:
    m = max(vals)
    return [
        (f"\\textbf{{{v:.{digits}f}}}" if abs(v - m) < 1e-12 else f"{v:.{digits}f}") for v in vals
    ]


# --------------------------------------------------------------------- Table 1
def table_data() -> None:
    tok = SPLIT["token_counts"]["broad_event_curve"]
    obs = SPLIT["observation_counts"]["broad"]
    pop = BSUP["population_counts"]
    targ = BSUP["target_counts"]
    cut = SPLIT["cutoffs"]["quantiles_iso"]
    rng = SPLIT["cutoffs"]
    ev = P6["section1_pump_eval"]["event_counts_onset_in_val_window"]["primary"]
    ev_test = TEST["by_head"]["explosion_60s"]["pump_level_recall_primary"]["n_events"]
    val_rate = {h: P6["section2_baselines"]["by_head"][h]["baselines"]["logistic_l2"]["variants"]["all"]["prevalence"] for h, _ in HEADS}
    test_rate = {h: TEST["by_head"][h]["variants"]["all"]["prevalence"] for h, _ in HEADS}

    def day(s: str) -> str:
        return datetime.fromisoformat(s).strftime("%-d %b %H:%M")

    body = rf"""
\begin{{tabular}}{{lrrr}}
\toprule
 & Train & Validation & Test \\
\midrule
First launch (2026, UTC) & {day(rng['launch_min_iso'])} & {day(cut['q70'])} & {day(cut['q85'])} \\
Last launch (2026, UTC) & {day(cut['q70'])} & {day(cut['q85'])} & {day(rng['launch_max_iso'])} \\
Tokens with $\geq$1 trade & {intc(tok['train'])} & {intc(tok['val'])} & {intc(tok['test'])} \\
Observations & {intc(obs['train'])} & {intc(obs['val'])} & {intc(obs['test'])} \\
Primary surges with onset in window & -- & {intc(ev)} & {intc(ev_test)} \\
Base rate, onset within 60 s & -- & {pct(val_rate['explosion_60s'], 2)} & {pct(test_rate['explosion_60s'], 2)} \\
Base rate, onset within 300 s & -- & {pct(val_rate['explosion_300s'], 2)} & {pct(test_rate['explosion_300s'], 2)} \\
\bottomrule
\end{{tabular}}
"""
    write("tab_data.tex", body)
    facts = {
        "launch_population": pop["launch_population"],
        "event_curve_population": pop["event_curve_population"],
        "primary_explosion_events": targ["primary_explosion_events"],
        "observations_total": SPLIT["observation_counts"]["broad_total"],
    }
    (TAB / "facts_data.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")


# --------------------------------------------------------------------- Table 2
def table_main_test() -> None:
    rows = []
    for h, hname in HEADS:
        var = TEST["by_head"][h]["variants"]["all"]
        rocs = [var["models"][m]["roc_auc"] for m, _ in MODELS]
        prs = [var["models"][m]["pr_auc"] for m, _ in MODELS]
        rb, pb = bold_max(rocs), bold_max(prs)

        rows.append(rf"\multicolumn{{8}}{{l}}{{\emph{{Onset within {hname}; base rate {pct(var['prevalence'], 2)}}}}} \\")
        for i, (m, name) in enumerate(MODELS):
            mm = var["models"][m]
            pa = mm["precision_at"]
            vr = TEST["val_comparison"][h][m]
            vroc, vpr = vr["val_roc"], vr["val_pr"]
            rows.append(
                f"{name} & {rb[i]} & {pb[i]} & {pct(pa['top_0.001'])} & {pct(pa['top_0.005'])} & "
                f"{pct(pa['top_0.01'])} & {pct(pa['top_0.02'])} & {f3(vroc)} / {f3(vpr)} \\\\"
            )
        rows.append(r"\addlinespace")
    body = (
        r"""
\begin{tabular}{lccccccc}
\toprule
 & \multicolumn{6}{c}{Test week (preregistered, scored once)} & Validation \\
\cmidrule(lr){2-7}\cmidrule(lr){8-8}
Model & ROC-AUC & PR-AUC & P@0.1\% & P@0.5\% & P@1\% & P@2\% & ROC / PR \\
\midrule
"""
        + "\n".join(rows[:-1])
        + r"""
\bottomrule
\end{tabular}
"""
    )
    write("tab_main_test.tex", body)


def table_deltas() -> None:
    rows = []
    for h, hname in HEADS:
        d = TEST["by_head"][h]["variants"]["all"]["deltas_vs_champion"]
        for key, name in [
            ("champion_minus_logistic_l2", "Logistic regression"),
            ("champion_minus_stage_b", "Transformer"),
            ("champion_minus_age_curve_xgb", "Age + curve XGBoost"),
        ]:
            rows.append(f"{hname} & {name} & {ci(d[key]['roc_auc'], 4)} & {ci(d[key]['pr_auc'], 4)} \\\\")
        rows.append(r"\addlinespace")
    n_mints = TEST["by_head"]["explosion_60s"]["variants"]["all"]["deltas_vs_champion"]["champion_minus_logistic_l2"]["roc_auc"]["n_mints"]
    body = (
        r"""
\begin{tabular}{llcc}
\toprule
Horizon & Champion minus & $\Delta$ROC-AUC [95\% CI] & $\Delta$PR-AUC [95\% CI] \\
\midrule
"""
        + "\n".join(rows[:-1])
        + r"""
\bottomrule
\end{tabular}
"""
    )
    write("tab_deltas.tex", body)
    (TAB / "facts_deltas.json").write_text(json.dumps({"n_mints_bootstrap": n_mints}), encoding="utf-8")


# --------------------------------------------------------------------- Table 3
STRATA = [
    ("by_trigger", "fixed_age", "Fixed-age observations"),
    ("by_trigger", "event_count", "Trade-count observations"),
    ("by_age_bucket", "[0,30s)", "Age $<$ 30 s"),
    ("by_age_bucket", "[30s,2m)", "Age 30 s -- 2 min"),
    ("by_age_bucket", "[2m,10m)", "Age 2 -- 10 min"),
    ("by_age_bucket", "[10m,1h)", "Age 10 min -- 1 h"),
    ("active_only", "ge1_trade_last_60s_approx", "Active: $\\geq$1 trade in last 60 s"),
    ("active_only", "ge5_trades_last_60s_approx", "Active: $\\geq$5 trades in last 60 s"),
]


def table_strata(head: str, fname: str) -> None:
    st = TEST["by_head"][head]["strata"]
    rows = []
    for grp, key, name in STRATA:
        s = st[grp][key]
        rocs = [s["models"][m]["roc_auc"] for m, _ in MODELS]
        rb = bold_max(rocs)
        rows.append(f"{name} & {intc(s['n'])} & {pct(s['prevalence'], 2)} & " + " & ".join(rb) + " \\\\")
    old = st["by_age_bucket"][">=1h"]
    body = (
        r"""
\begin{tabular}{lrr""" + "c" * len(MODELS) + r"""}
\toprule
 & & & \multicolumn{""" + str(len(MODELS)) + r"""}{c}{ROC-AUC} \\
\cmidrule(lr){4-""" + str(3 + len(MODELS)) + r"""}
Stratum & $n$ & Prevalence & XGB & LGBM & Logistic & Transf. & Age+curve \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
"""
    )
    write(fname, body)
    (TAB / f"facts_{fname}.json").write_text(json.dumps({"ge1h_n": old["n"], "ge1h_n_pos": old.get("n_pos")}), encoding="utf-8")


# --------------------------------------------------------------------- Table 4
def table_recall() -> None:
    rows = []
    for h, hname in HEADS:
        pr = TEST["by_head"][h]["pump_level_recall_primary"]
        for k, kname in [("top_0.01", "Top 1\\%"), ("top_0.02", "Top 2\\%")]:
            x = pr["by_top_frac"][k]
            lt = x["lead_time_sec_percentiles"]
            rows.append(
                f"{hname} & {kname} & {pct(x['fraction_caught'])} & {intc(x['n_caught'])} & "
                f"{lt['p10']:.1f} & {lt['p25']:.1f} & {lt['p50']:.1f} & {lt['p75']:.1f} & {lt['p90']:.1f} & "
                f"{x['false_alerts_per_caught_event']:.1f} \\\\"
            )
    n_ev = TEST["by_head"]["explosion_60s"]["pump_level_recall_primary"]["n_events"]
    body = (
        r"""
\begin{tabular}{llrrrrrrrr}
\toprule
 & & & & \multicolumn{5}{c}{Lead time (s), percentiles} & False alerts \\
\cmidrule(lr){5-9}
Scores & Threshold & Caught & $n$ caught & p10 & p25 & p50 & p75 & p90 & per catch \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
"""
    )
    write("tab_recall.tex", body)

    # variants + alert feed
    vrows = []
    for h, hname in HEADS:
        for v, vname in [("all", "All observations"), ("a_exclude_already_started", "(a) Already-started removed"), ("c_onset_ge_5s", "(c) $\\geq$5 s warning required")]:
            vv = TEST["by_head"][h]["variants"][v]
            m = vv["models"]["xgb_champion"]
            vrows.append(
                f"{hname} & {vname} & {intc(vv['n_pos'])} & {pct(vv['prevalence'], 2)} & {f3(m['roc_auc'])} & {f3(m['pr_auc'])} & {pct(m['precision_at']['top_0.01'])} \\\\"
            )
    af = TEST["by_head"]["explosion_300s"]["alert_feed"]
    body2 = (
        r"""
\begin{tabular}{llrrccc}
\toprule
Horizon & Evaluation set & Positives & Prevalence & ROC-AUC & PR-AUC & P@1\% \\
\midrule
"""
        + "\n".join(vrows)
        + r"""
\bottomrule
\end{tabular}
"""
    )
    write("tab_variants.tex", body2)
    facts = {
        "n_events_test": n_ev,
        "alert_feed_test_300s": {k: af[k] for k in ("mean_frac_explode_300s", "base_rate_eligible", "lift_vs_base", "n_hours_in_window", "n_hours_empty", "n_hours_with_observations")},
    }
    (TAB / "facts_recall.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")


# --------------------------------------------------------------------- Table 5
def table_labels() -> None:
    t = P6["section3_labels"]["targets"]
    rows = []
    for key, name in [
        ("explosion_60s_conservative", "Conservative, 60 s"),
        ("explosion_60s_primary_retrain", "Primary, 60 s"),
        ("explosion_60s_aggressive", "Aggressive, 60 s"),
    ]:
        m = t[key]["metrics"]
        rows.append(f"Validation & {name} & {intc(m['n_pos'])} & {pct(m['prevalence'], 2)} & {f3(m['roc_auc'])} & {f3(m['pr_auc'])} \\\\")
    tv = P7["section1_twox"]["permanent_val_metrics"]
    rows.append(f"Validation & 2$\\times$ surge within 300 s & {intc(tv['n_pos'])} & {pct(tv['prevalence'], 2)} & {f3(tv['roc_auc'])} & {f3(tv['pr_auc'])} \\\\")
    tw = TEST["twox"]
    tm = tw["metrics"]
    rows.append(f"Test & 2$\\times$ surge within 300 s & {intc(tm['n_pos'])} & {pct(tm['prevalence'], 2)} & {f3(tm['roc_auc'])} & {f3(tm['pr_auc'])} \\\\")
    body = (
        r"""
\begin{tabular}{llrrcc}
\toprule
Split & Target & Positives & Prevalence & ROC-AUC & PR-AUC \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
"""
    )
    write("tab_labels.tex", body)
    rec = P7["section1_twox"]["pump_level_recall_magnitude_ge_1"]
    facts = {
        "twox_val_recall": {k: rec["by_top_frac"][k]["fraction_caught"] for k in rec["by_top_frac"]},
        "twox_val_events": rec.get("n_events"),
        "twox_val_alert_lift": P7["section1_twox"]["alert_feed"]["lift_vs_base"],
        "twox_test_alert": {k: tw["alert_feed"][k] for k in ("lift_vs_base", "mean_frac_explode_300s", "base_rate_eligible")},
    }
    (TAB / "facts_labels.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")


# --------------------------------------------------------------------- Table 7 (val baselines)
def table_baselines_val() -> None:
    names = [
        ("volume_spike_z", "Volume-spike z-score (rule)"),
        ("net_inflow", "Net SOL inflow, 30 s (rule)"),
        ("curve_progress_rate", "Curve-progress rate"),
        ("best_single_feature", "Best single feature"),
        ("mlp_2x256", "MLP (2$\\times$256)"),
        ("logistic_l2", "Logistic regression (L2)"),
        ("xgb_champion_frozen_v2", "XGBoost (champion)"),
    ]
    DPR = {}
    for r in P6["champion_vs_baselines_300s"]:
        DPR[("explosion_300s", r["baseline"])] = {"delta": r["delta_pr"], "ci95": r["ci95"]}
    rows = []
    for h, hname in HEADS:
        bl = P6["section2_baselines"]["by_head"][h]["baselines"]
        rows.append(rf"\multicolumn{{6}}{{l}}{{\emph{{Onset within {hname}}}}} \\")
        for k, name in names:
            b = bl[k]
            m = b["variants"]["all"]
            dpr = DPR.get((h, k))
            dtxt = ci(dpr, 3) if dpr else "--"
            rows.append(f"{name} & {f3(m['roc_auc'])} & {f3(m['pr_auc'])} & {pct(m['precision_at']['top_0.01'])} & {b['alert_feed']['lift_vs_base']:.1f} & {dtxt} \\\\")
        rows.append(r"\addlinespace")
    body = (
        r"""
\begin{tabular}{lccccc}
\toprule
Model & ROC-AUC & PR-AUC & P@1\% & Alert lift & Champion $-$ model, $\Delta$PR [95\% CI] \\
\midrule
"""
        + "\n".join(rows[:-1])
        + r"""
\bottomrule
\end{tabular}
"""
    )
    write("tab_baselines_val.tex", body)


# --------------------------------------------------------------------- Table 6 (economics)
def table_econ() -> None:
    e = P6["section5_economics"]
    rows = []
    for key, name in [("all_eligible", "All eligible observations"), ("top1_alerts", "Champion top 1\\%")]:
        x = e[key]
        rows.append(
            f"{name} & {intc(x['h300_simple_return']['n'])} & {pct(x['h300_simple_return']['median'])} & {pct(x['h300_simple_return']['mean'])} & "
            f"{pct(x['h300_mfe']['median'])} & {pct(x['h300_mae']['median'])} \\\\"
        )
    body = (
        r"""
\begin{tabular}{lrcccc}
\toprule
 & & \multicolumn{2}{c}{5-min return} & Median & Median \\
\cmidrule(lr){3-4}
Observations & $n$ & Median & Mean & MFE & MAE \\
\midrule
"""
        + "\n".join(rows).replace("-", "$-$").replace("$-$$-$", "--")
        + r"""
\bottomrule
\end{tabular}
"""
    )
    write("tab_econ.tex", body)


# --------------------------------------------------------------------- Table 6b (leak)
def table_leak() -> None:
    s1 = P3["section1"]
    h = "explosion_60s"
    sf = {k: v["roc_auc"] for k, v in s1["heads"][h]["single_feature_auroc_val"].items()}
    lv = s1["lgbm_variants"][h]
    clean = P3["section3"]["lgbm_clean_only"][h]
    rows = [
        f"Broad features only (arm A) & {f3(clean['arm_A_roc_auc'])} & {f3(clean['arm_A_pr_auc'])} \\\\",
        f"Broad + all rich features (arm B) & {f3(lv['full_B']['roc_auc'])} & {f3(lv['full_B']['pr_auc'])} \\\\",
        f"Arm B + leak canary (share of fetched txs before $T$) & {f3(lv['d_plus_frac_sel_before_canary']['roc_auc'])} & {f3(lv['d_plus_frac_sel_before_canary']['pr_auc'])} \\\\",
        f"Arm B without microstructure & {f3(lv['b_minus_all_microstructure']['roc_auc'])} & {f3(lv['b_minus_all_microstructure']['pr_auc'])} \\\\",
        f"Broad + causally selected rich sources only & {f3(clean['roc_auc'])} & {f3(clean['pr_auc'])} \\\\",
    ]
    singles = [
        ("micro_n", "\\texttt{micro\\_n} alone"),
        ("largest_after_T", "``Largest trade is after $T$'' alone"),
        ("n_trades_before_T", "Causal trade count before $T$ alone"),
    ]
    srows = [f"{name} & {f3(sf[k])} & -- \\\\" for k, name in singles]
    body = (
        r"""
\begin{tabular}{lcc}
\toprule
Model or feature (multimodal subset, validation tokens) & ROC-AUC & PR-AUC \\
\midrule
"""
        + "\n".join(rows)
        + "\n\\midrule\n"
        + "\n".join(srows)
        + r"""
\bottomrule
\end{tabular}
"""
    )
    write("tab_leak.tex", body)
    s4 = P3["section4"]
    facts = {
        "stage_c_mask_baseline_roc60": s4["runs"]["baseline"]["metrics_recomputed_binary_metrics"]["60"]["roc_auc"],
        "stage_c_mask_no_micro_roc60": s4["drops_vs_baseline"]["no_microstructure"]["60"]["masked_roc"],
        "safe_auc_buggy_60": P3["section5"]["by_horizon"]["60"]["buggy_safe_auc"],
        "safe_auc_true_60": P3["section5"]["by_horizon"]["60"]["binary_metrics_roc_auc"],
        "match_rate": s1["match_n_sel_before_vs_micro_n"],
        "clean_vs_A_roc_ci": clean["vs_arm_A_roc"]["ci95"],
        "n_val_obs": clean["vs_arm_A_roc"]["n"],
        "n_val_mints": clean["vs_arm_A_roc"]["n_mints"],
    }
    (TAB / "facts_leak.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")


# --------------------------------------------------------------------- Appendix tables
def table_detector() -> None:
    # Parse the frozen definitions straight from source (avoids importing the package).
    import ast
    from types import SimpleNamespace

    src = (REPO / "src/trencher/data/world_model/targets/explosion.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    consts = {"PRIMARY_DEFINITION_ID": "explosion_primary_v1"}
    defs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ExplosionDefinition":
            kw = {}
            for k in node.keywords:
                kw[k.arg] = eval(compile(ast.Expression(k.value), "<def>", "eval"), {"math": math, **consts})
            defs[kw["label"]] = SimpleNamespace(**kw)
    assert set(defs) == {"conservative", "primary", "aggressive"}, defs.keys()
    order = ["conservative", "primary", "aggressive"]
    spec = [
        ("Window (s)", lambda d: f"{d.window_sec:g}"),
        ("Baseline (s)", lambda d: f"{d.baseline_sec:g}"),
        ("Confirmation window (s)", lambda d: f"{d.confirm_sec:g}"),
        ("Minimum price rise in window", lambda d: pct(math.exp(d.min_log_price_disp) - 1, 0)),
        ("Trade-rate multiple vs.\\ baseline", lambda d: f"{d.min_intensity_mult:g}$\\times$"),
        ("Minimum distinct buyers", lambda d: f"{d.min_unique_buyers}"),
        ("Buy-volume multiple vs.\\ baseline", lambda d: f"{d.min_buy_sol_mult:g}$\\times$"),
        ("Minimum trades in window", lambda d: f"{d.min_trades_in_window}"),
        ("Minimum subsequent rise", lambda d: pct(d.min_subsequent_mfe, 0)),
        ("Peak search after onset (s)", lambda d: f"{d.peak_search_sec:g}"),
        ("Exhaustion drawdown from peak", lambda d: pct(d.exhaustion_drawdown, 0)),
    ]
    rows = [f"{n} & " + " & ".join(fn(defs[o]) for o in order) + " \\\\" for n, fn in spec]
    body = (
        r"""
\begin{tabular}{lccc}
\toprule
Parameter & Conservative & Primary & Aggressive \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
"""
    )
    write("tab_detector.tex", body)


def table_features() -> None:
    from trencher.baselines.feature_spec import broad_feature_names

    names = broad_feature_names()
    groups: dict[str, list[str]] = {}
    for n in names:
        if n.startswith("token_age"):
            g = "Token age"
        elif n.startswith("mask_"):
            g = "Missing-data indicators"
        elif n.startswith("creator_"):
            g = "Creator history"
        elif n.startswith(("global_", "daily_network_")):
            g = "Market context"
        elif any(x in n for x in ("curve_", "latest_curve", "latest_price", "latest_v_", "latest_is_buy", "reserve_delta")):
            g = "Curve state"
        else:
            g = "Event flow"
        groups.setdefault(g, []).append(n)
    windowed = sorted({n.rsplit("_w", 1)[0] for n in names if n.rsplit("_w", 1)[-1] in {"10", "30", "120", "full"}}, key=names.index if False else str)
    rows = []
    for g in ["Event flow", "Curve state", "Market context", "Creator history", "Token age", "Missing-data indicators"]:
        items = groups[g]
        base = []
        for n in items:
            stem = n.rsplit("_w", 1)
            b = stem[0] if len(stem) == 2 and stem[1] in {"10", "30", "120", "full"} else n
            if b not in base:
                base.append(b)
        txt = ", ".join("\\texttt{" + b.replace("_", "\\_") + "}" for b in base)
        rows.append(f"{g} & {len(items)} & {txt} \\\\")
    body = (
        r"""
\begin{tabularx}{\linewidth}{lrX}
\toprule
Group & $n$ & Features (windowed features appear once per window: 10 s, 30 s, 120 s and full history) \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabularx}
"""
    )
    write("tab_features.tex", body)
    (TAB / "facts_features.json").write_text(json.dumps({"n": len(names), "groups": {k: len(v) for k, v in groups.items()}, "windowed_stems": windowed}, indent=2), encoding="utf-8")


def table_val_strata() -> None:
    st = P7["section2_strata"]["by_head"]["explosion_60s"]["strata"]
    rows = []
    act = st["active_only"]
    for key, name in [("ge1_trade_last_60s_approx", "Active: $\\geq$1 trade in last 60 s"), ("ge5_trades_last_60s_approx", "Active: $\\geq$5 trades in last 60 s")]:
        s = act[key]
        mods = s["models"]
        rows.append(f"{name} & " + " & ".join(f3(mods[m]["roc_auc"]) for m in ("xgb_champion", "logistic_l2", "stage_b")) + " \\\\")
    body = (
        r"""
\begin{tabular}{lccc}
\toprule
Validation week, 60 s & XGBoost & Logistic & Transformer \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
"""
    )
    write("tab_val_strata.tex", body)


# --------------------------------------------------------------------- Figures
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
C1, C2, C3 = "#1f4e79", "#c55a11", "#7f7f7f"


def fig_timeline() -> None:
    q = {k: datetime.fromisoformat(v) for k, v in SPLIT["cutoffs"]["quantiles_iso"].items()}
    lo = datetime.fromisoformat(SPLIT["cutoffs"]["launch_min_iso"])
    hi = datetime.fromisoformat(SPLIT["cutoffs"]["launch_max_iso"])
    fig, ax = plt.subplots(figsize=(6.6, 1.55))
    rows = [
        ("Fold 1", [(lo, q["q40"], C3, "train"), (q["q40"], q["q55"], C2, "select")]),
        ("Fold 2", [(lo, q["q55"], C3, "train"), (q["q55"], q["q70"], C2, "select")]),
        ("Final", [(lo, q["q70"], C1, "train"), (q["q70"], q["q85"], C2, "validation"), (q["q85"], hi, "#2e7d32", "test (locked)")]),
    ]
    import matplotlib.dates as mdates

    for i, (lab, segs) in enumerate(rows):
        y = len(rows) - i
        for a, b, c, t in segs:
            ax.barh(y, mdates.date2num(b) - mdates.date2num(a), left=mdates.date2num(a), height=0.6, color=c, alpha=0.85)
            ax.text(mdates.date2num(a) + (mdates.date2num(b) - mdates.date2num(a)) / 2, y, t, ha="center", va="center", color="white", fontsize=7)
        ax.text(mdates.date2num(lo) - 0.4, y, lab, ha="right", va="center", fontsize=7.5)
    outage = datetime(2026, 7, 3, tzinfo=UTC)
    ax.axvspan(mdates.date2num(outage), mdates.date2num(outage) + 1, color="black", alpha=0.12, lw=0)
    ax.text(mdates.date2num(outage) + 0.5, 3.55, "no data\n(3 Jul)", ha="center", va="bottom", fontsize=6)
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(0.4, 4.2)
    ax.set_xlabel("Token launch time (2026, UTC)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_timeline.pdf")
    plt.close(fig)


def fig_shap() -> None:
    ts = P6["section2_baselines"]["by_head"]["explosion_300s"]["treeshap"]
    g = ts["mean_abs_shap_by_group"]
    label = {"age": "Token age", "curve": "Curve state", "event_flow": "Event flow", "global_context": "Market context", "creator": "Creator history", "masks": "Missing-data ind."}
    items = sorted([(label[k], v) for k, v in g.items() if k in label], key=lambda x: x[1])
    top = ts["top20"][:15][::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1, 1.25]})
    a.barh([i[0] for i in items], [i[1] for i in items], color=C1)
    a.set_xlabel("Mean |SHAP| (sum over group)")
    a.set_title("(a) By feature group", fontsize=8)
    b.barh([t["feature"] for t in top], [t["mean_abs_shap"] for t in top], color=C2)
    b.set_xlabel("Mean |SHAP|")
    b.set_title("(b) Top 15 features", fontsize=8)
    b.tick_params(axis="y", labelsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_shap.pdf")
    plt.close(fig)
    (TAB / "facts_shap.json").write_text(json.dumps({"head": ts["head_id"], "groups": g, "sample_n": ts["sample_n"]}, indent=2), encoding="utf-8")


def fig_stability() -> None:
    d = P6["section4_stability"]["permanent_val_by_day"]
    lc = P5["section4_history"]["learning_curve"]["curves"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.2))
    for key, name, c in [("xgb_explosion_60s", "60 s", C1), ("xgb_explosion_300s", "300 s", C2)]:
        pts = sorted(d[key], key=lambda p: p["day_unix"])
        days = list(range(pts[0]["day_unix"], pts[-1]["day_unix"] + 1))
        by = {p["day_unix"]: p["roc_auc"] for p in pts}
        x = [datetime.fromtimestamp(dd * 86400, tz=UTC) for dd in days]
        y = [by.get(dd, float("nan")) for dd in days]  # gap = no data (3 Jul outage)
        a.plot(x, y, marker="o", ms=3, color=c, label=name)
    import matplotlib.dates as mdates

    a.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    a.set_ylabel("ROC-AUC")
    a.set_title("(a) Champion, by day of validation week", fontsize=8)
    a.legend(frameon=False, fontsize=7)
    for key, name, c in [("lightgbm/explosion_60s", "LightGBM", C1), ("xgboost/explosion_60s", "XGBoost", C2)]:
        pts = lc[key]["points"]
        b.plot([100 * p["fraction"] for p in pts], [p["pr_auc"] for p in pts], marker="o", ms=3, color=c, label=name)
    b.set_xlabel("Most recent share of training period used (%)")
    b.set_ylabel("Validation PR-AUC (60 s)")
    b.set_title("(b) Learning curve", fontsize=8)
    b.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_stability.pdf")
    plt.close(fig)
    days = {k: [(p["day_unix"], p["roc_auc"], p["n_pos"]) for p in d[k]] for k in ("xgb_explosion_60s", "xgb_explosion_300s")}
    (TAB / "facts_stability.json").write_text(json.dumps({"days": days, "learning_curve": {k: [(p["fraction"], p["roc_auc"], p["pr_auc"]) for p in v["points"]] for k, v in lc.items()}}, indent=2), encoding="utf-8")


def main() -> None:
    table_data()
    table_main_test()
    table_deltas()
    table_strata("explosion_60s", "tab_strata_60.tex")
    table_strata("explosion_300s", "tab_strata_300.tex")
    table_recall()
    table_labels()
    table_baselines_val()
    table_econ()
    table_leak()
    table_detector()
    table_features()
    table_val_strata()
    fig_timeline()
    fig_shap()
    fig_stability()
    print("assets written to", TAB, "and", FIG)


if __name__ == "__main__":
    main()
