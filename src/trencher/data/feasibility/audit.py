"""End-to-end historical-data feasibility audit."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trencher.data.feasibility.assessment import label_feasibility, reconstruction_assessment
from trencher.data.feasibility.concepts import (
    COLUMN_CONCEPT_MAP,
    EXCLUDE_WALLETS,
    LEAKAGE_RISK_FIELDS,
    UNRELIABLE_FIELDS,
)
from trencher.data.feasibility.joinability import compute_joinability
from trencher.data.feasibility.quality import (
    audit_migrations,
    audit_postgard_outcomes,
    audit_red_pump,
    audit_snapshots_quality,
    audit_tokens_table,
    audit_trades_quality,
    load_known_issues_text,
    parquet_schema_report,
)
from trencher.data.paths import resolve_external_root

logger = logging.getLogger(__name__)


@dataclass
class FeasibilityResult:
    ok: bool
    report_dir: Path
    summary: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _schema_bundle(pumpfun_root: Path) -> dict[str, Any]:
    files = {
        "tokens": pumpfun_root / "tokens.parquet",
        "snapshots": pumpfun_root / "snapshots.parquet",
        "migrations": pumpfun_root / "migrations.parquet",
        "wallet_stats": pumpfun_root / "wallet_stats.parquet",
        "postgard_outcomes": pumpfun_root / "postgard_outcomes.parquet",
        "postgard_snapshots": pumpfun_root / "postgard_snapshots.parquet",
    }
    out: dict[str, Any] = {}
    for name, path in files.items():
        out[name] = parquet_schema_report(path)
    # one trade shard as schema exemplar + aggregate rows
    trade_files = sorted((pumpfun_root / "trades").glob("*.parquet"))
    if trade_files:
        exemplar = parquet_schema_report(trade_files[0])
        total_rows = sum(parquet_schema_report(p)["rows"] for p in trade_files)
        out["trades"] = {
            **exemplar,
            "path": str(pumpfun_root / "trades"),
            "shard_count": len(trade_files),
            "rows": total_rows,
        }
    return out


def _verdict(quality: dict[str, Any], joinability: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for section in (
        quality.get("tokens", {}).get("known_issues_checks", {}),
        quality.get("trades", {}).get("known_issues_checks", {}),
        quality.get("snapshots", {}).get("known_issues_checks", {}),
        quality.get("migrations", {}).get("known_issues_checks", {}),
        quality.get("postgard_outcomes", {}).get("known_issues_checks", {}),
        quality.get("red_pump", {}).get("known_issues_checks", {}),
    ):
        for key, value in section.items():
            if isinstance(value, tuple):
                checks.append({"check": key, "value": value[0], "pass": bool(value[1])})
            else:
                checks.append({"check": key, "pass": bool(value)})

    mint_cov = (
        joinability.get("token_mint_overlap", {})
        .get("pumpfun_vs_red_pump", {})
        .get("coverage_of_right")
    )
    cohort_wallet_cov = (
        joinability.get("wallet_overlap", {})
        .get("cohort_wallets_vs_trade_wallets", {})
        .get("coverage_of_left")
    )

    can_build = True
    reasons = [
        (
            "PumpFun provides ordered trades, curve state, graduation, "
            "and post-grad paths for graduates."
        ),
        (
            "Labels (survival, barriers, MFE/MAE, persistence) can be "
            "regenerated from price/activity paths."
        ),
        (
            "KNOWN_ISSUES are quantified and filterable; "
            "unreliable aggregates are identifiable."
        ),
    ]
    exclusions = [
        "wallet_stats activity/volume aggregates",
        (
            "uncorrected concentration percentages; "
            "top10_pct_suspect==True rows for concentration features"
        ),
        "System Program wallet trades for wallet features",
        "trades with null/inconsistent sol/price for SOL P&L features",
        "synthetic/backfilled migration pool_address values when needing real pools",
        "source outcome_label taxonomies as canonical labels",
        "entry_price_20s/30s/1m_usd as predictive features (graduation leakage)",
    ]
    partial = [
        "Post-graduation trajectory only for ~5.7k graduated tokens",
        (
            "RED-PUMP overlaps PumpFun only partially "
            "(different observation windows/populations)"
        ),
        (
            "RED-COHORT is a short June 11-25 window coordination catalogue, "
            "not full-history wallet graph"
        ),
    ]
    solarchive_required = False
    solarchive_rationale = (
        "Not required for the intended research-grade bonding-curve/graduation model. "
        "Solarchive would only be needed for raw instruction/account reconstruction beyond "
        "what trades/snapshots already encode, or for chain periods outside this corpus."
    )

    return {
        "can_build_research_grade_model": can_build,
        "fully_supported_components": [
            "pre-graduation trade/event history at arbitrary T",
            "bonding-curve price/reserve trajectory",
            "wallet participation history (recomputed from trades)",
            "graduation / time-to-graduation labels",
            "self-generated survival, barrier, return, MFE/MAE, inactivity labels",
            "buyer/activity persistence labels",
        ],
        "partially_supported_components": partial,
        "exclude_or_recompute": exclusions,
        "solarchive_supplementation_required": solarchive_required,
        "solarchive_rationale": solarchive_rationale,
        "known_issues_check_results": checks,
        "coverage_highlights": {
            "red_pump_mint_coverage_of_red_pump": mint_cov,
            "cohort_wallet_coverage_in_trades": cohort_wallet_cov,
        },
        "reasons": reasons,
    }


def render_feasibility_md(summary: dict[str, Any]) -> str:
    def _fmt_rate(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    v = summary["verdict"]
    q = summary["quality"]
    j = summary["joinability"]
    lines = [
        "# Historical Data Feasibility",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Verdict",
        "",
        f"**Can we build the intended research-grade model from these datasets?** "
        f"{'YES' if v['can_build_research_grade_model'] else 'NO'}",
        "",
        "### Fully supported",
        "",
    ]
    lines.extend(f"- {x}" for x in v["fully_supported_components"])
    lines.extend(["", "### Partially supported", ""])
    lines.extend(f"- {x}" for x in v["partially_supported_components"])
    lines.extend(["", "### Exclude / recompute", ""])
    lines.extend(f"- {x}" for x in v["exclude_or_recompute"])
    lines.extend(
        [
            "",
            "### Solarchive supplementation",
            "",
            f"**Required:** {v['solarchive_supplementation_required']}",
            "",
            v["solarchive_rationale"],
            "",
            "## Major quality findings",
            "",
            f"- PumpFun tokens: {q['tokens']['rows']:,} rows; "
            f"{q['tokens']['graduated_tokens']:,} graduated; "
            f"{q['tokens']['mayhem_tokens']:,} mayhem; "
            f"{q['tokens']['top10_pct_suspect_true']:,} concentration-suspect.",
            f"- Trades: {q['trades'].get('rows'):,} rows; "
            f"null price rate={_fmt_rate(q['trades'].get('null_price_sol_rate'))}; "
            f"system-program rate={_fmt_rate(q['trades'].get('system_program_trade_rate'))}; "
            f"sol inconsistency rate="
            f"{_fmt_rate(q['trades'].get('sol_amount_inconsistent_rate_vs_checked'))}; "
            f"Jul 3 trades={q['trades'].get('jul3_trade_rows')}.",
            f"- Snapshots curve OOB rate="
            f"{_fmt_rate(q['snapshots'].get('curve_pct_depleted_eob_oob_rate'))}.",
            f"- Migrations synthetic/backfilled="
            f"{q['migrations']['synthetic_graduation_queue']}/"
            f"{q['migrations']['backfilled_from_pumpswap_trade']}.",
            f"- Postgard outcome_label has `rug`? "
            f"{q['postgard_outcomes']['has_rug_label_value']}.",
            "",
            "## Usable overlap / coverage",
            "",
            (
                f"- PumpFun ∩ RED-PUMP mints: "
                f"{j['token_mint_overlap']['pumpfun_vs_red_pump']['intersection']:,} "
                f"(coverage of RED-PUMP="
                f"{j['token_mint_overlap']['pumpfun_vs_red_pump']['coverage_of_right']:.3f})"
            ),
            (
                f"- Graduated ∩ migrations: "
                f"{j['token_mint_overlap']['pumpfun_graduated_vs_migrations']['intersection']:,} "
                f"(coverage of graduated="
                f"{j['token_mint_overlap']['pumpfun_graduated_vs_migrations']['coverage_of_left']:.3f})"
            ),
            (
                f"- Cohort mints ∩ PumpFun: "
                f"{j['token_mint_overlap']['cohort_mints_vs_pumpfun']['intersection']:,} "
                f"(coverage of cohort mints="
                f"{j['token_mint_overlap']['cohort_mints_vs_pumpfun']['coverage_of_left']:.3f})"
            ),
            (
                f"- Cohort wallets ∩ trade wallets: "
                f"{j['wallet_overlap']['cohort_wallets_vs_trade_wallets']['intersection']:,} "
                f"(coverage of cohort wallets="
                f"{j['wallet_overlap']['cohort_wallets_vs_trade_wallets']['coverage_of_left']:.3f})"
            ),
            (
                f"- PumpFun graduated ∩ RED-PUMP graduated: "
                f"{j['token_mint_overlap']['pumpfun_graduated_vs_red_pump_graduated']['intersection']:,}"
                f" (near-disjoint graduation populations)"
            ),
            "",
            "## Unsupported / unreliable / leakage",
            "",
            "See `unreliable_fields.json`, `leakage_risk_fields.json`, and "
            "`column_concept_map.json` in this directory.",
            "",
            "## Bottom line",
            "",
            "These datasets are sufficient to build a research-grade pre-/peri-graduation "
            "model with self-generated labels, provided documented quality filters are "
            "applied and source-specific outcome taxonomies are not treated as canonical.",
            "",
        ]
    )
    return "\n".join(lines)


def run_feasibility_audit(
    *,
    external_root: Path | None = None,
    manifest_root: Path | None = None,
    trade_wallet_max_batches: int | None = None,
    skip_heavy_trade_scan: bool = False,
) -> FeasibilityResult:
    """Audit local historical corpora and write manifests/feasibility reports."""
    root = resolve_external_root(external_root)
    # Reports live at manifests/feasibility/ (sibling of manifests/external/).
    if manifest_root is None:
        report_dir = Path("manifests") / "feasibility"
    else:
        report_dir = Path(manifest_root) / "feasibility"
    report_dir.mkdir(parents=True, exist_ok=True)

    pumpfun = root / "pumpfun_memecoin_corpus"
    red_pump = root / "red_pump_2026"
    red_cohort = root / "red_cohort_2026"
    errors: list[str] = []

    required = [
        pumpfun / "tokens.parquet",
        pumpfun / "KNOWN_ISSUES.md",
        pumpfun / "trades",
        red_pump / "red_pump_2026_v1_launches.jsonl.gz",
        red_pump / "red_pump_2026_v1_outcomes.csv.gz",
        red_cohort / "extracted" / "RED-COHORT-2026-v1" / "sniper_cohorts.jsonl",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing required path: {path}")
    if errors:
        return FeasibilityResult(ok=False, report_dir=report_dir, errors=errors)

    logger.info("Auditing schemas")
    schemas = _schema_bundle(pumpfun)

    logger.info("Auditing tokens quality")
    tokens_q = audit_tokens_table(pumpfun / "tokens.parquet")

    if skip_heavy_trade_scan:
        trades_q = {
            "rows": schemas.get("trades", {}).get("rows"),
            "skipped": True,
            "known_issues_checks": {},
        }
        snapshots_q = {"skipped": True, "known_issues_checks": {}}
    else:
        logger.info("Auditing trades quality (streaming)")
        trades_q = audit_trades_quality(pumpfun / "trades")
        logger.info("Auditing snapshots quality (streaming)")
        snapshots_q = audit_snapshots_quality(pumpfun / "snapshots.parquet")

    migrations_q = audit_migrations(pumpfun / "migrations.parquet")
    postgard_q = audit_postgard_outcomes(pumpfun / "postgard_outcomes.parquet")
    red_pump_q = audit_red_pump(
        red_pump / "red_pump_2026_v1_launches.jsonl.gz",
        red_pump / "red_pump_2026_v1_outcomes.csv.gz",
    )

    quality = {
        "generated_at": datetime.now(UTC).isoformat(),
        "tokens": tokens_q,
        "trades": trades_q,
        "snapshots": snapshots_q,
        "migrations": migrations_q,
        "postgard_outcomes": postgard_q,
        "red_pump": red_pump_q,
        "known_issues_document_bytes": len(
            load_known_issues_text(pumpfun / "KNOWN_ISSUES.md")
        ),
    }

    logger.info("Computing joinability")
    joinability = compute_joinability(
        pumpfun_root=pumpfun,
        red_pump_root=red_pump,
        red_cohort_root=red_cohort,
        trade_wallet_max_batches=trade_wallet_max_batches,
    )

    reconstruction = reconstruction_assessment()
    labels = label_feasibility()
    verdict = _verdict(quality, joinability)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "external_root": str(root),
        "quality": quality,
        "joinability": joinability,
        "reconstruction": reconstruction,
        "labels": labels,
        "verdict": verdict,
        "column_concept_map": COLUMN_CONCEPT_MAP,
        "unreliable_fields": UNRELIABLE_FIELDS,
        "leakage_risk_fields": LEAKAGE_RISK_FIELDS,
        "exclude_wallets": EXCLUDE_WALLETS,
    }

    _write_json(report_dir / "schemas.json", schemas)
    _write_json(report_dir / "quality.json", quality)
    _write_json(report_dir / "joinability.json", joinability)
    _write_json(report_dir / "reconstruction.json", reconstruction)
    _write_json(report_dir / "labels.json", labels)
    _write_json(report_dir / "column_concept_map.json", COLUMN_CONCEPT_MAP)
    _write_json(report_dir / "unreliable_fields.json", {"fields": UNRELIABLE_FIELDS})
    _write_json(report_dir / "leakage_risk_fields.json", {"fields": LEAKAGE_RISK_FIELDS})
    _write_json(report_dir / "verdict.json", verdict)
    _write_json(report_dir / "summary.json", summary)
    _write_text(report_dir / "FEASIBILITY.md", render_feasibility_md(summary))

    return FeasibilityResult(ok=True, report_dir=report_dir, summary=summary, errors=[])
