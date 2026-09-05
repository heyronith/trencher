"""Core baseline target suite + label extraction from slim joined cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

TaskKind = Literal["binary", "multiclass", "regression", "quantile", "survival", "tte"]


@dataclass(frozen=True)
class TargetSpec:
    head_id: str
    kind: TaskKind
    family: str
    label_col: str | None
    status_col: str | None = None
    eligible_col: str | None = None
    # TTE
    duration_col: str | None = None
    event_col: str | None = None
    at_risk_col: str | None = None
    # barrier
    outcome_map: dict[str, int] | None = None
    description: str = ""


CORE_TARGET_SUITE: tuple[TargetSpec, ...] = (
    TargetSpec(
        "explosion_60s",
        "binary",
        "ignition",
        "explosion_within_60s",
        "explosion_within_60s_status",
        "eligible_explosion_imminence_h60",
        description="explosion within 60s",
    ),
    TargetSpec(
        "explosion_300s",
        "binary",
        "ignition",
        "explosion_within_300s",
        "explosion_within_300s_status",
        "eligible_explosion_imminence_h300",
        description="explosion within 300s",
    ),
    TargetSpec(
        "barrier_u25_l15_5m",
        "multiclass",
        "barriers",
        "h300_barrier_u25_l15_outcome",
        None,
        "eligible_barriers_h300",
        outcome_map={"upper_first": 0, "lower_first": 1, "neither": 2},
        description="+25% before -15% @ 5m",
    ),
    TargetSpec(
        "barrier_u100_l30_1h",
        "multiclass",
        "barriers",
        "h3600_barrier_u100_l30_outcome",
        None,
        "eligible_barriers_h3600",
        outcome_map={"upper_first": 0, "lower_first": 1, "neither": 2},
        description="+100% before -30% @ 1h",
    ),
    TargetSpec(
        "return_5m",
        "regression",
        "returns",
        "h300_simple_return",
        "h300_status",
        "eligible_returns_h300",
        description="5m simple return",
    ),
    TargetSpec(
        "return_1h",
        "regression",
        "returns",
        "h3600_simple_return",
        "h3600_status",
        "eligible_returns_h3600",
        description="1h simple return",
    ),
    TargetSpec(
        "survival_30m",
        "binary",
        "survival",
        "surv_h1800_active",
        "surv_h1800_status",
        "eligible_survival_h1800",
        description="active/survival @ 30m",
    ),
    TargetSpec(
        "survival_24h",
        "binary",
        "survival",
        "surv_h86400_active",
        "surv_h86400_status",
        "eligible_survival_h86400",
        description="active/survival @ 24h",
    ),
    TargetSpec(
        "explosion_tte",
        "tte",
        "tte",
        None,
        None,
        "eligible_explosion_tte",
        duration_col="explosion_tte_duration_sec",
        event_col="explosion_tte_event_observed",
        at_risk_col="explosion_tte_at_risk",
        description="explosion TTE",
    ),
    TargetSpec(
        "graduation_tte",
        "tte",
        "tte",
        None,
        None,
        "eligible_graduation_tte",
        duration_col="graduation_tte_duration_sec",
        event_col="graduation_tte_event_observed",
        at_risk_col="graduation_tte_at_risk",
        description="graduation TTE (sparse / event-only)",
    ),
)


RESOLVED_BINARY_STATUSES = frozenset({"valid_positive", "valid_negative"})


def extract_binary_labels(
    rows: list[dict[str, Any]], spec: TargetSpec
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return y (0/1), eligible mask, counts. Censored/unavailable excluded from eligible."""
    y = np.zeros(len(rows), dtype=np.float64)
    elig = np.zeros(len(rows), dtype=bool)
    counts = {"n": len(rows), "eligible": 0, "pos": 0, "neg": 0, "excluded": 0}
    for i, r in enumerate(rows):
        if spec.eligible_col and r.get(spec.eligible_col) is False:
            counts["excluded"] += 1
            continue
        st = r.get(spec.status_col) if spec.status_col else None
        val = r.get(spec.label_col) if spec.label_col else None
        if st is not None and st not in RESOLVED_BINARY_STATUSES:
            counts["excluded"] += 1
            continue
        if val is None:
            counts["excluded"] += 1
            continue
        label = 1.0 if bool(val) else 0.0
        # valid_positive/negative consistency
        if st == "valid_positive":
            label = 1.0
        elif st == "valid_negative":
            label = 0.0
        y[i] = label
        elig[i] = True
        counts["eligible"] += 1
        if label >= 0.5:
            counts["pos"] += 1
        else:
            counts["neg"] += 1
    return y, elig, counts


def extract_regression_labels(
    rows: list[dict[str, Any]], spec: TargetSpec
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    y = np.full(len(rows), np.nan, dtype=np.float64)
    elig = np.zeros(len(rows), dtype=bool)
    counts = {"n": len(rows), "eligible": 0, "excluded": 0}
    for i, r in enumerate(rows):
        if spec.eligible_col and r.get(spec.eligible_col) is False:
            counts["excluded"] += 1
            continue
        st = r.get(spec.status_col) if spec.status_col else None
        val = r.get(spec.label_col)
        if val is None or (st is not None and st not in RESOLVED_BINARY_STATUSES):
            counts["excluded"] += 1
            continue
        if not np.isfinite(float(val)):
            counts["excluded"] += 1
            continue
        y[i] = float(val)
        elig[i] = True
        counts["eligible"] += 1
    return y, elig, counts


def extract_barrier_labels(
    rows: list[dict[str, Any]], spec: TargetSpec
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    mapping = spec.outcome_map or {}
    y = np.full(len(rows), -1, dtype=np.int64)
    elig = np.zeros(len(rows), dtype=bool)
    counts = {"n": len(rows), "eligible": 0, "excluded": 0, "by_class": {}}
    for i, r in enumerate(rows):
        if spec.eligible_col and r.get(spec.eligible_col) is False:
            counts["excluded"] += 1
            continue
        outcome = r.get(spec.label_col)
        if outcome not in mapping:
            counts["excluded"] += 1
            continue
        y[i] = int(mapping[str(outcome)])
        elig[i] = True
        counts["eligible"] += 1
        counts["by_class"][str(outcome)] = counts["by_class"].get(str(outcome), 0) + 1
    return y, elig, counts


def extract_tte_labels(
    rows: list[dict[str, Any]], spec: TargetSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """duration, event_observed, at_risk mask. Never treat censor as negative event."""
    duration = np.full(len(rows), np.nan, dtype=np.float64)
    event = np.zeros(len(rows), dtype=np.float64)
    at_risk = np.zeros(len(rows), dtype=bool)
    counts = {
        "n": len(rows),
        "at_risk": 0,
        "events": 0,
        "censored": 0,
        "excluded": 0,
    }
    for i, r in enumerate(rows):
        if spec.eligible_col and r.get(spec.eligible_col) is False:
            counts["excluded"] += 1
            continue
        ar = r.get(spec.at_risk_col)
        # also accept nested effective fields from live overlay aliases
        if ar is None:
            ar = r.get(f"{spec.head_id}_at_risk")
        if not ar:
            counts["excluded"] += 1
            continue
        dur = r.get(spec.duration_col)
        ev = r.get(spec.event_col)
        if dur is None or not np.isfinite(float(dur)) or float(dur) < 0:
            counts["excluded"] += 1
            continue
        duration[i] = float(dur)
        event[i] = 1.0 if bool(ev) else 0.0
        at_risk[i] = True
        counts["at_risk"] += 1
        if event[i] >= 0.5:
            counts["events"] += 1
        else:
            counts["censored"] += 1
    return duration, event, at_risk, counts


def core_suite_manifest() -> list[dict[str, Any]]:
    return [
        {
            "head_id": t.head_id,
            "kind": t.kind,
            "family": t.family,
            "description": t.description,
            "label_col": t.label_col,
            "status_col": t.status_col,
            "eligible_col": t.eligible_col,
        }
        for t in CORE_TARGET_SUITE
    ]
