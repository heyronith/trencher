"""Head-specific split-boundary purge / eligibility."""

from __future__ import annotations

from typing import Any

from trencher.data.world_model.targets.status import (
    EXPLOSION_IMMINENCE_HORIZONS_SEC,
    HORIZONS_SEC,
    SURVIVAL_HORIZONS_SEC,
    SURVIVAL_NEGATIVE_CONFIRMATION_SEC,
)

# Head families that share a bounded horizon window ending at T+H.
BOUNDED_HEAD_FAMILIES: tuple[str, ...] = (
    "returns",
    "barriers",
    "mfe_mae",
    "persistence",
    "explosion_imminence",
    "graduation_binary",
    "execution",
    "future_path",
)

SURVIVAL_HEAD = "survival"
GRADUATION_TTE_HEAD = "graduation_tte"
EXPLOSION_TTE_HEAD = "explosion_tte"


def eligibility_column(head: str, horizon_sec: int | None = None) -> str:
    if horizon_sec is None:
        return f"eligible_{head}"
    return f"eligible_{head}_h{int(horizon_sec)}"


def all_eligibility_columns() -> list[str]:
    cols: list[str] = []
    for head in BOUNDED_HEAD_FAMILIES:
        for h in HORIZONS_SEC:
            if head == "explosion_imminence" and h not in EXPLOSION_IMMINENCE_HORIZONS_SEC:
                continue
            cols.append(eligibility_column(head, h))
    for h in SURVIVAL_HORIZONS_SEC:
        cols.append(eligibility_column(SURVIVAL_HEAD, h))
    cols.append(eligibility_column(GRADUATION_TTE_HEAD))
    cols.append(eligibility_column(EXPLOSION_TTE_HEAD))
    return cols


def obs_in_split_interval(
    observation_time_t: float, *, split_start: float, split_end: float
) -> bool:
    """State time must lie in [split_start, split_end)."""
    t = float(observation_time_t)
    return split_start <= t < split_end


def bounded_window_eligible(
    observation_time_t: float, horizon_sec: float, split_end: float
) -> bool:
    """Strict: T + H < split_end (event at exactly split_end belongs to next period)."""
    return float(observation_time_t) + float(horizon_sec) < float(split_end)


def survival_confirmation_eligible(
    observation_time_t: float,
    horizon_sec: float,
    split_end: float,
    *,
    confirmation_sec: float = SURVIVAL_NEGATIVE_CONFIRMATION_SEC,
) -> bool:
    """Strict: T + H + confirmation < split_end."""
    return (
        float(observation_time_t) + float(horizon_sec) + float(confirmation_sec)
        < float(split_end)
    )


def compute_eligibility_flags(
    observation_time_t: float,
    *,
    split_start: float,
    split_end: float,
) -> dict[str, bool]:
    """Return head-specific eligibility flags for one observation.

    If the observation state itself is outside the split interval, all heads
    are ineligible (late observation of an earlier token).
    """
    out: dict[str, bool] = {}
    in_interval = obs_in_split_interval(
        observation_time_t, split_start=split_start, split_end=split_end
    )
    if not in_interval:
        for col in all_eligibility_columns():
            out[col] = False
        out["obs_in_split_interval"] = False
        return out

    out["obs_in_split_interval"] = True
    T = float(observation_time_t)
    for head in BOUNDED_HEAD_FAMILIES:
        horizons = (
            EXPLOSION_IMMINENCE_HORIZONS_SEC
            if head == "explosion_imminence"
            else HORIZONS_SEC
        )
        for h in horizons:
            out[eligibility_column(head, h)] = bounded_window_eligible(T, h, split_end)

    for h in SURVIVAL_HORIZONS_SEC:
        out[eligibility_column(SURVIVAL_HEAD, h)] = survival_confirmation_eligible(
            T, h, split_end
        )

    # Unbounded TTE: eligible to *supervise* only while obs is in-interval;
    # unresolved events remain censored (never forced negative). Flag marks
    # that split-boundary does not auto-invalidate the TTE head's censoring.
    out[eligibility_column(GRADUATION_TTE_HEAD)] = True
    out[eligibility_column(EXPLOSION_TTE_HEAD)] = True
    return out


def purge_reason_summary() -> dict[str, Any]:
    return {
        "bounded_rule": "T + H < split_end",
        "survival_confirmation_rule": (
            f"T + H + {int(SURVIVAL_NEGATIVE_CONFIRMATION_SEC)} < split_end"
        ),
        "obs_interval_rule": "split_start <= T < split_end",
        "unbounded_tte_policy": (
            "preserve legitimate censoring; never invent negatives at boundary"
        ),
        "head_families": list(BOUNDED_HEAD_FAMILIES)
        + [SURVIVAL_HEAD, GRADUATION_TTE_HEAD, EXPLOSION_TTE_HEAD],
    }
