"""Diagnose fusion modality gaps and plan ownership expansion."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.constants import (
    BQ_FREE_TIER_BYTES,
    BQ_HEADROOM_RESERVE_BYTES,
    CORPUS_END,
    CORPUS_START,
)
from trencher.data.world_model.io import ensure_dir, utc_now, write_json

# Google Cloud BigQuery on-demand query pricing (US regions), $/TiB processed.
BQ_ON_DEMAND_USD_PER_TIB = 6.25
FUSION_TARGET = 2000


def _mint_set(path: Path, column: str = "mint") -> set[str]:
    if not path.exists() or pq.ParquetFile(path).metadata.num_rows == 0:
        return set()
    return set(pq.read_table(path, columns=[column]).column(0).to_pylist()) - {None}


def build_modality_matrix(world_root: Path) -> dict[str, Any]:
    subset = pq.read_table(world_root / "universe" / "rich_subset_mints.parquet")
    mints = subset.column("mint").to_pylist()
    S = set(mints)
    event = _mint_set(world_root / "execution" / "execution_curve_states.parquet") & S
    graph = _mint_set(world_root / "graph" / "edges_wallet_token.parquet", "dst") & S
    graph |= _mint_set(world_root / "graph" / "edges_creator_token.parquet", "dst") & S
    own = _mint_set(world_root / "bigquery" / "token_transfers_unbiased_days.parquet") & S
    own |= _mint_set(world_root / "bigquery" / "token_transfers_rich.parquet") & S
    rpc_own = world_root / "graph" / "edges_ownership_rpc_token_balances.parquet"
    own |= _mint_set(rpc_own) & S
    rt_path = world_root / "microstructure" / "rpc_transactions.parquet"
    micro: set[str] = set()
    if rt_path.exists() and pq.ParquetFile(rt_path).metadata.num_rows:
        rt = pq.read_table(rt_path, columns=["mint", "found"])
        micro = {
            m
            for m, f in zip(
                rt.column("mint").to_pylist(),
                rt.column("found").to_pylist(),
                strict=True,
            )
            if m and f
        } & S
    market_ok = True
    rows = []
    for i, mint in enumerate(mints):
        row = {
            "mint": mint,
            "launch_day": subset.column("launch_day")[i].as_py(),
            "launch_regime": subset.column("launch_regime")[i].as_py(),
            "graduated": bool(subset.column("graduated")[i].as_py()),
            "has_event_curve": mint in event,
            "has_graph": mint in graph,
            "has_ownership_spl": mint in own,
            "has_microstructure": mint in micro,
            "has_execution_state": mint in event,
            "has_market_context": market_ok,
        }
        row["fusion_complete"] = (
            row["has_event_curve"]
            and row["has_graph"]
            and row["has_ownership_spl"]
            and row["has_microstructure"]
            and row["has_execution_state"]
            and row["has_market_context"]
        )
        miss = [
            k
            for k, ok in [
                ("ownership", row["has_ownership_spl"]),
                ("microstructure", row["has_microstructure"]),
                ("graph", row["has_graph"]),
                ("execution", row["has_execution_state"]),
                ("event_curve", row["has_event_curve"]),
                ("market", row["has_market_context"]),
            ]
            if not ok
        ]
        row["missing"] = "+".join(miss)
        rows.append(row)

    out_dir = ensure_dir(world_root / "universe")
    pq.write_table(pa.Table.from_pylist(rows), out_dir / "modality_matrix.parquet")
    blockers = Counter(r["missing"] or "complete" for r in rows)
    need_own_only = [
        r
        for r in rows
        if (not r["fusion_complete"])
        and (not r["has_ownership_spl"])
        and r["has_event_curve"]
        and r["has_graph"]
        and r["has_microstructure"]
        and r["has_execution_state"]
        and r["has_market_context"]
    ]
    complete = sum(1 for r in rows if r["fusion_complete"])
    diagnosis = {
        "generated_at": utc_now(),
        "subset_n": len(rows),
        "fusion_complete": complete,
        "gap_to_target": max(0, FUSION_TARGET - complete),
        "fusion_target": FUSION_TARGET,
        "coverage": {
            "event_curve": sum(r["has_event_curve"] for r in rows),
            "graph": sum(r["has_graph"] for r in rows),
            "ownership_spl": sum(r["has_ownership_spl"] for r in rows),
            "microstructure": sum(r["has_microstructure"] for r in rows),
            "execution": sum(r["has_execution_state"] for r in rows),
            "market": sum(r["has_market_context"] for r in rows),
        },
        "blocker_combos": blockers.most_common(),
        "need_ownership_only": len(need_own_only),
        "limiting_modality": "ownership_spl"
        if len(need_own_only) >= (FUSION_TARGET - complete)
        else "mixed",
        "smallest_acquisition_set": {
            "strategy": (
                "Recover genuine SPL ownership for need_ownership_only mints "
                "via archival RPC token-balance parses (preferred) and/or "
                "partition-pruned BigQuery Token Transfers on unqueried days."
            ),
            "mints": len(need_own_only),
            "note": (
                "Launch-day BQ partitions already scanned; residual BQ value is "
                "post-launch transfer days and is expected to unlock far fewer "
                "than 1.7k tokens alone."
            ),
        },
    }
    return {"rows": rows, "diagnosis": diagnosis, "need_own_only": need_own_only}


def _existing_tt_days(world_root: Path) -> set[str]:
    return {
        p.name[len("token_transfers_") : -len(".parquet")]
        for p in (world_root / "bigquery").glob("token_transfers_20*.parquet")
    }


def _corpus_days() -> list[date]:
    out = []
    d = CORPUS_START
    while d <= CORPUS_END:
        if d.isoformat() != "2026-07-03":
            out.append(d)
        d += timedelta(days=1)
    return out


def estimate_unlocks_for_day(
    day: str,
    need_rows: list[dict[str, Any]],
    historical_post_only_rate: float = 0.008,
) -> dict[str, float | int]:
    """Conservative expected new ownership mints from a post-launch day scan."""
    eligible = [r for r in need_rows if r["launch_day"] <= day]
    # Prior extracts yielded ~25 post-only uniques across many days (~0.8%/day of eligible).
    expected = min(len(eligible), int(round(len(eligible) * historical_post_only_rate)))
    conservative = min(len(eligible), max(0, expected // 2))
    optimistic = min(len(eligible), max(expected, int(round(len(eligible) * 0.02))))
    return {
        "eligible": len(eligible),
        "expected_base": expected,
        "expected_conservative": conservative,
        "expected_optimistic": optimistic,
    }


def build_cost_plan(
    *,
    world_root: Path,
    manifest_dir: Path,
    dry_run_candidates: list[dict[str, Any]],
    lifetime_bytes_billed: int,
    diagnosis: dict[str, Any],
    need_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    remaining_free = max(0, BQ_FREE_TIER_BYTES - lifetime_bytes_billed)
    remaining_safe = max(0, remaining_free - BQ_HEADROOM_RESERVE_BYTES)
    complete = int(diagnosis["fusion_complete"])
    gap = max(0, FUSION_TARGET - complete)

    ranked = []
    for cand in dry_run_candidates:
        if cand.get("kind") != "day_partition":
            continue
        day = cand["day"]
        unlock = estimate_unlocks_for_day(day, need_rows)
        gib = float(cand["estimated_gib"])
        base = int(unlock["expected_base"])
        tokens_per_gib = (base / gib) if gib > 0 else 0.0
        ranked.append(
            {
                **cand,
                **unlock,
                "expected_tokens_unlocked_per_gib_base": round(tokens_per_gib, 4),
                "sql": cand.get("sql"),
            }
        )
    ranked.sort(
        key=lambda r: (
            -float(r["expected_tokens_unlocked_per_gib_base"]),
            r["estimated_bytes"],
        )
    )

    # Greedy plan until projected overlap hits target (base case), keep emergency headroom.
    plan_rows = []
    cum_bytes = 0
    cum_unlock = 0
    for row in ranked:
        if cum_unlock >= gap:
            break
        plan_rows.append(
            {
                **{k: row[k] for k in row if k != "sql"},
                "sql_sha_preview": row.get("name"),
                "include_in_execution_plan": True,
                "cumulative_projected_bytes": cum_bytes + int(row["estimated_bytes"]),
                "cumulative_projected_unlock_base": cum_unlock
                + int(row["expected_base"]),
                "cumulative_projected_overlap_base": complete
                + cum_unlock
                + int(row["expected_base"]),
            }
        )
        cum_bytes += int(row["estimated_bytes"])
        cum_unlock += int(row["expected_base"])

    # Even scanning ALL unqueried days is unlikely to hit 2000.
    all_unq_bytes = sum(int(r["estimated_bytes"]) for r in ranked)
    all_unq_unlock_base = sum(int(r["expected_base"]) for r in ranked)
    all_unq_unlock_cons = sum(int(r["expected_conservative"]) for r in ranked)
    all_unq_unlock_opt = sum(int(r["expected_optimistic"]) for r in ranked)

    full = next((c for c in dry_run_candidates if c.get("kind") == "full_corpus_window"), None)

    def cost_from_additional(additional_bytes: int) -> dict[str, float | int]:
        paid = max(0, additional_bytes - remaining_free)
        paid_tib = paid / (1024**4)
        return {
            "additional_bytes": additional_bytes,
            "additional_gib": round(additional_bytes / (1024**3), 3),
            "paid_bytes": paid,
            "paid_gib": round(paid / (1024**3), 3),
            "paid_tib": round(paid_tib, 4),
            "usd": round(paid_tib * BQ_ON_DEMAND_USD_PER_TIB, 2),
        }

    rpc_path = {
        "preferred": True,
        "rationale": (
            "2,846 tokens already have event+graph+micro+execution and lack only "
            "ownership. Launch-day BQ partitions were already scanned. Re-parsing "
            "archival getTransaction pre/post token balances for existing Pump "
            "trade signatures yields genuine SPL ownership edges without BQ cost."
        ),
        "target_mints": len(need_rows),
        "expected_fusion_overlap_if_all_succeed": complete + len(need_rows),
        "bq_usd": 0.0,
    }

    bq_all_days_cost = cost_from_additional(all_unq_bytes)
    full_cost = cost_from_additional(int(full["estimated_bytes"])) if full else None

    # Base projected to reach 2000 via BQ alone is infeasible at observed rates.
    projected_overlap_bq_all_days_base = complete + all_unq_unlock_base
    bytes_to_target_bq = None
    if projected_overlap_bq_all_days_base >= FUSION_TARGET:
        # find minimal prefix
        b = 0
        u = 0
        for row in ranked:
            b += int(row["estimated_bytes"])
            u += int(row["expected_base"])
            if complete + u >= FUSION_TARGET:
                bytes_to_target_bq = b
                break
    else:
        bytes_to_target_bq = None  # cannot reach via these partitions at base rate

    plan = {
        "generated_at": utc_now(),
        "pricing_assumption_usd_per_tib": BQ_ON_DEMAND_USD_PER_TIB,
        "pricing_note": (
            "BigQuery on-demand query processing list price for US regions "
            "($6.25/TiB). Dry-runs do not bill processing bytes."
        ),
        "current_bigquery_monthly_bytes_consumed": lifetime_bytes_billed,
        "current_bigquery_monthly_gib_consumed": round(lifetime_bytes_billed / (1024**3), 3),
        "free_tier_bytes": BQ_FREE_TIER_BYTES,
        "remaining_free_tier_bytes": remaining_free,
        "remaining_free_tier_gib": round(remaining_free / (1024**3), 3),
        "emergency_headroom_reserve_bytes": BQ_HEADROOM_RESERVE_BYTES,
        "remaining_safe_bytes_after_reserve": remaining_safe,
        "fusion_target": FUSION_TARGET,
        "current_fusion_overlap": complete,
        "gap_to_target": gap,
        "diagnosis_summary": {
            "limiting_modality": diagnosis.get("limiting_modality"),
            "need_ownership_only": diagnosis.get("need_ownership_only"),
            "blocker_combos": diagnosis.get("blocker_combos"),
        },
        "recommended_path": "archival_rpc_token_balances_then_selective_bq",
        "rpc_ownership_plan": rpc_path,
        "bigquery_candidates_ranked_by_tokens_per_gib": [
            {k: v for k, v in r.items() if k != "sql"} for r in ranked
        ],
        "bigquery_partition_execution_plan": plan_rows,
        "totals": {
            "unqueried_days_count": len(ranked),
            "unqueried_days_projected_bytes": all_unq_bytes,
            "unqueried_days_projected_gib": round(all_unq_bytes / (1024**3), 3),
            "expected_unlock_conservative": all_unq_unlock_cons,
            "expected_unlock_base": all_unq_unlock_base,
            "expected_unlock_optimistic": all_unq_unlock_opt,
            "projected_overlap_if_all_unqueried_days_conservative": complete
            + all_unq_unlock_cons,
            "projected_overlap_if_all_unqueried_days_base": projected_overlap_bq_all_days_base,
            "projected_overlap_if_all_unqueried_days_optimistic": complete
            + all_unq_unlock_opt,
            "cost_if_all_unqueried_days": bq_all_days_cost,
            "full_window_candidate": (
                {
                    "estimated_gib": full.get("estimated_gib"),
                    "estimated_bytes": full.get("estimated_bytes"),
                    "cost": full_cost,
                    "note": (
                        "Dry-run bytes equal prior graduated full-window scan; "
                        "mostly re-reads already-extracted partitions; low "
                        "marginal tokens/GiB vs RPC ownership parse."
                    ),
                }
                if full
                else None
            ),
            "projected_additional_bytes_required_to_reach_2000_via_bq_only": bytes_to_target_bq,
            "bq_alone_can_reach_2000_at_base_rate": bytes_to_target_bq is not None,
            "projected_paid_usd_if_rpc_path": 0.0,
            "projected_final_fusion_overlap_if_rpc_path": complete + len(need_rows),
            "execution_rule": (
                "Auto-execute if projected paid BQ processing USD <= 50; "
                "otherwise stop and report. Prefer RPC ownership first."
            ),
            "auto_execute_bq": False,
            "auto_execute_reason": (
                "BQ unqueried-day scans are cheap in USD but cannot reach ~2000 "
                "overlap at observed Token Transfers hit rates; execute RPC "
                "token-balance ownership recovery first (projected $0 BQ)."
            ),
        },
        "decision": {
            "proceed_with_rpc_ownership": True,
            "proceed_with_bq_before_review": False,
            "stop_for_cost_review": False,
            "projected_bq_usd_if_forced_all_unqueried_days": bq_all_days_cost["usd"],
            "under_50_usd_threshold": bq_all_days_cost["usd"] <= 50,
        },
    }
    ensure_dir(manifest_dir)
    write_json(manifest_dir / "fusion_expansion_cost_plan.json", plan)
    (manifest_dir / "FUSION_EXPANSION_COST_PLAN.md").write_text(
        render_cost_plan_md(plan), encoding="utf-8"
    )
    return plan


def render_cost_plan_md(plan: dict[str, Any]) -> str:
    t = plan["totals"]
    lines = [
        "# Fusion Expansion Cost Plan",
        "",
        f"Generated at: `{plan['generated_at']}`",
        "",
        "## BigQuery account state",
        "",
        f"- Bytes already consumed this month: "
        f"**{plan['current_bigquery_monthly_gib_consumed']} GiB**",
        f"- Free-tier remaining: **{plan['remaining_free_tier_gib']} GiB**",
        f"- Emergency reserve: "
        f"**{round(plan['emergency_headroom_reserve_bytes']/(1024**3),1)} GiB**",
        f"- On-demand price assumption: **${plan['pricing_assumption_usd_per_tib']}/TiB**",
        "",
        "## Fusion target",
        "",
        f"- Current multimodal overlap: **{plan['current_fusion_overlap']}**",
        f"- Target: **{plan['fusion_target']}**",
        f"- Gap: **{plan['gap_to_target']}**",
        f"- Limiting modality: **{plan['diagnosis_summary']['limiting_modality']}** "
        f"({plan['diagnosis_summary']['need_ownership_only']} tokens need ownership only)",
        "",
        "## Recommended path (non-BigQuery first)",
        "",
        plan["rpc_ownership_plan"]["rationale"],
        "",
        f"- Expected overlap if RPC ownership succeeds for all targets: "
        f"**{plan['rpc_ownership_plan']['expected_fusion_overlap_if_all_succeed']}**",
        "- Projected BQ processing $: **$0.00**",
        "",
        "## BigQuery-only alternative (dry-run ranked)",
        "",
        f"- Unqueried partitions: **{t['unqueried_days_count']}** days / "
        f"**{t['unqueried_days_projected_gib']} GiB**",
        f"- Expected new ownership mints (conservative / base / optimistic): "
        f"**{t['expected_unlock_conservative']} / {t['expected_unlock_base']} / "
        f"{t['expected_unlock_optimistic']}**",
        f"- Projected overlap after all unqueried days: "
        f"**{t['projected_overlap_if_all_unqueried_days_conservative']} / "
        f"{t['projected_overlap_if_all_unqueried_days_base']} / "
        f"{t['projected_overlap_if_all_unqueried_days_optimistic']}**",
        f"- Paid bytes if executed: **{t['cost_if_all_unqueried_days']['paid_gib']} GiB**",
        f"- Projected paid USD: **${t['cost_if_all_unqueried_days']['usd']}**",
        f"- BQ alone reaches 2000 at base rate: "
        f"**{t['bq_alone_can_reach_2000_at_base_rate']}**",
        "",
        "### Partition ranking (tokens unlocked / GiB, base)",
        "",
        "| day | est GiB | eligible | unlock base | tokens/GiB | cum overlap |",
        "| --- | ------: | -------: | ----------: | ---------: | ----------: |",
    ]
    cum_u = plan["current_fusion_overlap"]
    for row in plan["bigquery_candidates_ranked_by_tokens_per_gib"]:
        cum_u += int(row["expected_base"])
        lines.append(
            f"| {row['day']} | {row['estimated_gib']} | {row['eligible']} | "
            f"{row['expected_base']} | {row['expected_tokens_unlocked_per_gib_base']} | "
            f"{cum_u} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Proceed with RPC ownership: **{plan['decision']['proceed_with_rpc_ownership']}**",
            f"- Proceed with BQ before review: "
            f"**{plan['decision']['proceed_with_bq_before_review']}**",
            f"- Forced-all-days BQ USD (under $50?): "
            f"**${plan['decision']['projected_bq_usd_if_forced_all_unqueried_days']}** "
            f"({plan['decision']['under_50_usd_threshold']})",
            "",
            plan["totals"]["auto_execute_reason"],
            "",
            "Dry-runs do not incur BigQuery data-processing charges.",
            "",
        ]
    )
    return "\n".join(lines)
