"""Outcome-independent rich-subset selection for fusion training."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from datetime import UTC, date, datetime
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.constants import (
    CORPUS_END,
    CORPUS_START,
    REPRESENTATIVE_BQ_DAYS,
)
from trencher.data.world_model.io import file_provenance, utc_now, write_json

# Launch-day partitions used both for date spread and BQ day extracts.
# Covers start, Jun 9-10 regime, mid, pre-outage, post-outage, end. Jul 3 excluded.
REPRESENTATIVE_LAUNCH_DAYS: tuple[str, ...] = REPRESENTATIVE_BQ_DAYS

OUTAGE_DAY = "2026-07-03"
SELECTION_SALT = "trencher-unbiased-v1"
SELECTION_FEATURES = (
    "mint",
    "detected_at",
    "is_mayhem_mode",
    "initial_holder_count",
    "initial_buy_sol",
)
FORBIDDEN_SELECTION_FIELDS = (
    "graduated_at",
    "seconds_to_graduation",
    "peak_market_cap_sol",
    "peak_market_cap_at",
    "entry_price_usd",
    "entry_price_20s_usd",
    "entry_price_30s_usd",
    "entry_price_1m_usd",
    "trade_count",
    "is_zombie",
    "is_training_ready",
    "pool_address",
    "migrated_at",
)


def launch_regime(day: str) -> str:
    """Market-regime label from launch calendar (KNOWN_ISSUES breaks)."""
    if day <= "2026-06-08":
        return "jun5_8_pre_holdercap"
    if day <= "2026-06-10":
        return "jun9_10_holdercap"
    if day <= "2026-07-02":
        return "jun11_jul2_pre_outage"
    if day == OUTAGE_DAY:
        return "jul3_outage"
    return "jul4_14_post_outage"


def _day_key(ts: Any) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        d = ts.astimezone(UTC).date() if ts.tzinfo else ts.date()
        return d.isoformat()
    if isinstance(ts, date):
        return ts.isoformat()
    text = str(ts)
    return text[:10] if len(text) >= 10 else None


def _mint_rank(mint: str) -> int:
    digest = hashlib.sha256(f"{SELECTION_SALT}:{mint}".encode()).hexdigest()
    return int(digest[:16], 16)


def _tertile(values: list[float | None]) -> dict[int, int]:
    """Map index -> tertile 0/1/2 using non-null empirical cuts (deterministic)."""
    present = sorted(v for v in values if v is not None)
    if not present:
        return {i: 0 for i in range(len(values))}
    n = len(present)
    c1 = present[(n - 1) // 3]
    c2 = present[(2 * (n - 1)) // 3]
    out: dict[int, int] = {}
    for i, v in enumerate(values):
        if v is None:
            out[i] = 1
        elif v <= c1:
            out[i] = 0
        elif v <= c2:
            out[i] = 1
        else:
            out[i] = 2
    return out


def _two_proportion_z(
    s1: int, n1: int, s2: int, n2: int
) -> dict[str, float | None]:
    """Two-proportion z-test (subset vs universe graduation)."""
    if n1 == 0 or n2 == 0:
        return {"z": None, "p_approx": None, "diff": None}
    p1 = s1 / n1
    p2 = s2 / n2
    p = (s1 + s2) / (n1 + n2)
    denom = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0.0
    z = (p1 - p2) / denom if denom else 0.0
    # Abramowitz-Stegun-ish two-sided p from |z|
    abs_z = abs(z)
    p_approx = math.erfc(abs_z / math.sqrt(2.0))
    return {"z": z, "p_approx": p_approx, "diff": p1 - p2}


def select_unbiased_rich_subset(
    tokens_path: Any,
    *,
    max_mints: int,
    representative_days: tuple[str, ...] | None = REPRESENTATIVE_LAUNCH_DAYS,
) -> dict[str, Any]:
    """Select tokens using only pre-T launch fields. Graduation is post-hoc only.

    ``representative_days`` is recorded for BQ targeting; inclusion itself uses
    every corpus launch day except the Jul 3 outage.
    """
    table = pq.read_table(
        tokens_path,
        columns=[
            "mint",
            "detected_at",
            "creator",
            "is_mayhem_mode",
            "initial_holder_count",
            "initial_buy_sol",
            "graduated_at",  # post-hoc composition only; never used for inclusion
        ],
    )
    mints = table.column("mint").to_pylist()
    detected = table.column("detected_at").to_pylist()
    creators = table.column("creator").to_pylist()
    mayhem = table.column("is_mayhem_mode").to_pylist()
    holders = table.column("initial_holder_count").to_pylist()
    buy_sol = table.column("initial_buy_sol").to_pylist()
    graduated = table.column("graduated_at").to_pylist()

    n = len(mints)
    universe_grad = sum(1 for g in graduated if g is not None)
    holder_t = _tertile([float(x) if x is not None else None for x in holders])
    buy_t = _tertile([float(x) if x is not None else None for x in buy_sol])

    eligible_idx: list[int] = []
    for i, ts in enumerate(detected):
        day = _day_key(ts)
        if not day or day == OUTAGE_DAY:
            continue
        if day < CORPUS_START.isoformat() or day > CORPUS_END.isoformat():
            continue
        if representative_days and day not in representative_days:
            continue
        if not mints[i]:
            continue
        eligible_idx.append(i)

    # Stratum: day × mayhem × holder tertile × buy tertile
    buckets: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for i in eligible_idx:
        day = _day_key(detected[i])
        mflag = 1 if mayhem[i] else 0
        key = (day, mflag, holder_t[i], buy_t[i])
        buckets[key].append(i)
    for key in buckets:
        buckets[key].sort(key=lambda i: _mint_rank(mints[i]))

    n_eligible = len(eligible_idx)
    target = min(max_mints, n_eligible)
    # Proportional allocation with at least 1 from each non-empty stratum.
    chosen: list[int] = []
    remaining_slots = target
    strata = sorted(buckets)
    base_alloc: dict[tuple[Any, ...], int] = {}
    for key in strata:
        share = max(1, round(target * len(buckets[key]) / n_eligible)) if n_eligible else 0
        take = min(share, len(buckets[key]), remaining_slots)
        base_alloc[key] = take
        remaining_slots -= take
        if remaining_slots <= 0:
            break
    # Fill leftover from largest strata in rank order
    if remaining_slots > 0:
        for key in sorted(strata, key=lambda k: -len(buckets[k])):
            extra = min(remaining_slots, len(buckets[key]) - base_alloc.get(key, 0))
            if extra <= 0:
                continue
            base_alloc[key] = base_alloc.get(key, 0) + extra
            remaining_slots -= extra
            if remaining_slots <= 0:
                break

    for key, take in base_alloc.items():
        chosen.extend(buckets[key][:take])
    chosen = chosen[:target]
    chosen.sort(key=lambda i: _mint_rank(mints[i]))

    rows = []
    for i in chosen:
        day = _day_key(detected[i]) or ""
        g = graduated[i]
        rows.append(
            {
                "mint": mints[i],
                "creator": creators[i],
                "detected_at": detected[i],
                "launch_day": day,
                "launch_regime": launch_regime(day),
                "is_mayhem_mode": bool(mayhem[i]),
                "initial_holder_count": holders[i],
                "initial_buy_sol": buy_sol[i],
                "holder_tertile": holder_t[i],
                "buy_sol_tertile": buy_t[i],
                "graduated": g is not None,  # label, not a selection input
                "observation_time_t": detected[i],
                "selection_salt": SELECTION_SALT,
            }
        )

    subset_n = len(rows)
    subset_grad = sum(1 for r in rows if r["graduated"])
    day_counts: dict[str, int] = defaultdict(int)
    regime_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        day_counts[r["launch_day"]] += 1
        regime_counts[r["launch_regime"]] += 1

    test = _two_proportion_z(subset_grad, subset_n or 1, universe_grad, n)
    # Fail the "is enrichment a graduation proxy" test if rates differ a lot.
    # Selection itself must not track graduation; |rate diff| < 2pp or p>0.01.
    univ_rate = universe_grad / n if n else 0.0
    sub_rate = subset_grad / subset_n if subset_n else 0.0
    not_grad_proxy = abs(sub_rate - univ_rate) < 0.02 or (
        test["p_approx"] is not None and test["p_approx"] > 0.01
    )

    algorithm = {
        "version": SELECTION_SALT,
        "features_used": list(SELECTION_FEATURES),
        "forbidden_fields": list(FORBIDDEN_SELECTION_FIELDS),
        "outage_day_excluded": OUTAGE_DAY,
        "representative_launch_days": list(representative_days or ()),
        "inclusion_days": (
            "representative corpus days spanning regimes (Jul 3 excluded)"
            if representative_days
            else "all corpus launch days except Jul 3 outage"
        ),
        "strata": "launch_day x is_mayhem_mode x holder_tertile x buy_sol_tertile",
        "ranking": "sha256(salt:mint) lexicographic within stratum",
        "observation_time_t": "token.detected_at (first observation)",
        "notes": [
            "Graduation/migration/postgrad/future returns are not selection inputs.",
            "graduated flag is attached post-hoc for composition audit only.",
            "Representative days span corpus start/end and documented regime breaks.",
        ],
    }

    proof = {
        "universe_tokens": n,
        "universe_graduated": universe_grad,
        "universe_graduation_rate": univ_rate,
        "subset_tokens": subset_n,
        "subset_graduated": subset_grad,
        "subset_graduation_rate": sub_rate,
        "subset_non_graduated": subset_n - subset_grad,
        "two_proportion_z": test,
        "selection_is_not_graduation_proxy": bool(not_grad_proxy),
        "mayhem_rate_universe": sum(1 for x in mayhem if x) / n if n else None,
        "mayhem_rate_subset": (
            sum(1 for r in rows if r["is_mayhem_mode"]) / subset_n if subset_n else None
        ),
        "launch_day_counts": dict(sorted(day_counts.items())),
        "launch_regime_counts": dict(regime_counts),
        "jul3_included": any(r["launch_day"] == OUTAGE_DAY for r in rows),
    }

    return {
        "rows": rows,
        "algorithm": algorithm,
        "bias_proof": proof,
        "generated_at": utc_now(),
    }


def write_unbiased_subset(result: dict[str, Any], out_dir: Any) -> dict[str, Any]:
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    path = out_dir / "rich_subset_mints.parquet"
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), path)
    else:
        pq.write_table(pa.table({"mint": pa.array([], pa.string())}), path)
    write_json(out_dir / "unbiased_subset_algorithm.json", result["algorithm"])
    write_json(out_dir / "unbiased_subset_bias_proof.json", result["bias_proof"])
    return {
        "path": file_provenance(path, source="pumpfun.tokens.launch_fields_only"),
        "mint_count": len(rows),
        "quality_flag": "outcome_independent_stratified",
        "bias_proof": result["bias_proof"],
        "algorithm": result["algorithm"],
    }
