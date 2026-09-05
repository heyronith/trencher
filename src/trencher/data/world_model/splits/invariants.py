"""Leakage / purge invariants for chronological splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trencher.data.world_model.splits.cutoffs import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
from trencher.data.world_model.splits.purge import (
    SURVIVAL_HEAD,
    bounded_window_eligible,
    eligibility_column,
    survival_confirmation_eligible,
)
from trencher.data.world_model.targets.status import (
    EXPLOSION_IMMINENCE_HORIZONS_SEC,
    HORIZONS_SEC,
    SURVIVAL_HORIZONS_SEC,
)


def verify_token_disjointness(token_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sets = {SPLIT_TRAIN: set(), SPLIT_VAL: set(), SPLIT_TEST: set()}
    for r in token_rows:
        sets[r["base_split"]].add(r["mint"])
    overlaps = {
        "train_val": sorted(sets[SPLIT_TRAIN] & sets[SPLIT_VAL])[:5],
        "train_test": sorted(sets[SPLIT_TRAIN] & sets[SPLIT_TEST])[:5],
        "val_test": sorted(sets[SPLIT_VAL] & sets[SPLIT_TEST])[:5],
    }
    ok = all(len(v) == 0 for v in overlaps.values())
    return {
        "passed": ok,
        "counts": {k: len(v) for k, v in sets.items()},
        "overlap_samples": overlaps,
    }


def verify_chronological_ordering(
    token_rows: list[dict[str, Any]], *, q70: float, q85: float
) -> dict[str, Any]:
    by = {SPLIT_TRAIN: [], SPLIT_VAL: [], SPLIT_TEST: []}
    for r in token_rows:
        by[r["base_split"]].append(float(r["detected_at_unix"]))
    mx_tr = max(by[SPLIT_TRAIN]) if by[SPLIT_TRAIN] else None
    mn_va = min(by[SPLIT_VAL]) if by[SPLIT_VAL] else None
    mx_va = max(by[SPLIT_VAL]) if by[SPLIT_VAL] else None
    mn_te = min(by[SPLIT_TEST]) if by[SPLIT_TEST] else None
    # Tie-safe via assign rules: train t<q70, val q70<=t<q85, test t>=q85.
    ok = True
    reasons: list[str] = []
    train_val_ok = mx_tr is None or mn_va is None or (mx_tr < q70 and mn_va >= q70)
    if not train_val_ok:
        ok = False
        reasons.append("train/val ordering")
    val_test_ok = mx_va is None or mn_te is None or (mx_va < q85 and mn_te >= q85)
    if not val_test_ok:
        ok = False
        reasons.append("val/test ordering")
    return {
        "passed": ok,
        "max_train_launch": mx_tr,
        "min_val_launch": mn_va,
        "max_val_launch": mx_va,
        "min_test_launch": mn_te,
        "q70": q70,
        "q85": q85,
        "reasons": reasons,
    }


def verify_walkforward_excludes_test(
    token_rows: list[dict[str, Any]], walkforward: dict[str, Any]
) -> dict[str, Any]:
    test_mints = {r["mint"] for r in token_rows if r["base_split"] == SPLIT_TEST}
    violations = 0
    for r in token_rows:
        if r["mint"] not in test_mints:
            continue
        for fold in walkforward["folds"]:
            role = r.get(f"{fold['fold_id']}_role")
            if role in {"train", "val"}:
                violations += 1
    return {"passed": violations == 0, "test_tokens_in_folds": violations}


def sample_verify_eligibility_strict(
    assignment_dir: Path,
    *,
    q70: float,
    q85: float,
    test_open_end: float,
    max_parts: int = 3,
) -> dict[str, Any]:
    """Spot-check: no eligible bounded/survival flag violates strict inequalities."""
    parts = sorted(assignment_dir.glob("part-*.parquet"))[:max_parts]
    if assignment_dir.is_file():
        parts = [assignment_dir]
    violations = 0
    checked = 0
    for path in parts:
        for row in pq.read_table(path).to_pylist():
            T = float(row["observation_time_t"])
            split = row["base_split"]
            if split == SPLIT_TRAIN:
                end = q70
            elif split == SPLIT_VAL:
                end = q85
            else:
                end = test_open_end
            if not row.get("obs_in_split_interval"):
                # all head flags must be false
                for h in HORIZONS_SEC:
                    col = eligibility_column("returns", h)
                    if row.get(col):
                        violations += 1
                checked += 1
                continue
            for h in HORIZONS_SEC:
                col = eligibility_column("returns", h)
                if row.get(col) and not bounded_window_eligible(T, h, end):
                    violations += 1
            for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
                col = eligibility_column("explosion_imminence", h)
                if row.get(col) and not bounded_window_eligible(T, h, end):
                    violations += 1
            for h in SURVIVAL_HORIZONS_SEC:
                col = eligibility_column(SURVIVAL_HEAD, h)
                if row.get(col) and not survival_confirmation_eligible(T, h, end):
                    violations += 1
            checked += 1
    return {
        "passed": violations == 0,
        "rows_checked": checked,
        "violations": violations,
        "parts_sampled": len(parts),
    }


def run_all_invariants(
    *,
    token_rows: list[dict[str, Any]],
    cutoffs: dict[str, Any],
    walkforward: dict[str, Any],
    broad_obs_dir: Path,
    rich_obs_path: Path,
) -> dict[str, Any]:
    q70 = float(cutoffs["quantiles_unix"]["q70"])
    q85 = float(cutoffs["quantiles_unix"]["q85"])
    test_open_end = float(cutoffs["launch_max_unix"]) + 86400.0 + 1800.0 + 1.0
    results = {
        "token_disjoint": verify_token_disjointness(token_rows),
        "chronological_ordering": verify_chronological_ordering(
            token_rows, q70=q70, q85=q85
        ),
        "walkforward_excludes_test": verify_walkforward_excludes_test(
            token_rows, walkforward
        ),
        "broad_eligibility_strict": sample_verify_eligibility_strict(
            broad_obs_dir, q70=q70, q85=q85, test_open_end=test_open_end
        ),
        "rich_eligibility_strict": sample_verify_eligibility_strict(
            rich_obs_path, q70=q70, q85=q85, test_open_end=test_open_end
        ),
    }
    results["all_passed"] = all(v.get("passed") for v in results.values())
    return results
