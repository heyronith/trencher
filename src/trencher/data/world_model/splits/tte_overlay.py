"""Split-aware administrative censoring for graduation/explosion TTE.

Does not rewrite frozen target stores. Training views must never observe an
event at or after the split boundary.
"""

from __future__ import annotations

from typing import Any

from trencher.data.world_model.splits.cutoffs import SPLIT_TRAIN, SPLIT_VAL
from trencher.data.world_model.targets.status import (
    CORPUS_END_UNIX,
    STATUS_RIGHT_CENSORED,
    STATUS_VALID_NEGATIVE,
    STATUS_VALID_POSITIVE,
)

ADMIN_STATUS = STATUS_RIGHT_CENSORED


def admin_boundary_unix(
    base_split: str, *, q70: float, q85: float, corpus_end: float = CORPUS_END_UNIX
) -> float:
    """Exclusive end used for administrative TTE censoring."""
    if base_split == SPLIT_TRAIN:
        return float(q70)
    if base_split == SPLIT_VAL:
        return float(q85)
    # Test: corpus source bound (not an open future beyond the freeze).
    return float(corpus_end)


def _effective_tte_fields(
    *,
    observation_time_t: float,
    split_end: float,
    event_observed: bool,
    event_time_unix: float | None,
    raw_tte_sec: float | None,
    raw_status: str | None,
    already_occurred: bool = False,
) -> dict[str, Any]:
    """Apply split-aware TTE overlay for one head."""
    T = float(observation_time_t)
    end = float(split_end)
    out: dict[str, Any] = {
        "effective_event_observed": False,
        "effective_tte_sec": None,
        "effective_censoring_status": None,
        "administrative_censor_time": None,
        "split_boundary_censored": False,
        "raw_status": raw_status,
        "raw_tte_sec": raw_tte_sec,
        "raw_event_observed": bool(event_observed),
    }

    if already_occurred:
        # Event already known at/before T — no future leakage.
        out["effective_event_observed"] = True
        out["effective_tte_sec"] = None
        out["effective_censoring_status"] = raw_status or STATUS_VALID_NEGATIVE
        return out

    if event_observed and event_time_unix is not None:
        et = float(event_time_unix)
        if et < end:
            # Strictly before boundary → may remain observed.
            out["effective_event_observed"] = True
            out["effective_tte_sec"] = float(et - T) if raw_tte_sec is None else float(raw_tte_sec)
            out["effective_censoring_status"] = STATUS_VALID_POSITIVE
            return out
        # At or after split_end → hide future event; admin right-censor.
        out["effective_event_observed"] = False
        out["effective_tte_sec"] = end - T
        out["effective_censoring_status"] = ADMIN_STATUS
        out["administrative_censor_time"] = end
        out["split_boundary_censored"] = True
        return out

    # Unresolved: censor at earliest legitimate source/split boundary.
    # Never invent a negative from administrative censoring alone.
    if raw_status == STATUS_VALID_NEGATIVE:
        # Frozen store already resolved a genuine negative with coverage.
        out["effective_event_observed"] = False
        out["effective_tte_sec"] = None
        out["effective_censoring_status"] = STATUS_VALID_NEGATIVE
        return out

    out["effective_event_observed"] = False
    out["effective_tte_sec"] = end - T
    out["effective_censoring_status"] = ADMIN_STATUS
    out["administrative_censor_time"] = end
    out["split_boundary_censored"] = True
    return out


def apply_graduation_tte_overlay(
    *,
    observation_time_t: float,
    split_end: float,
    time_to_graduation_sec: float | None,
    time_to_graduation_status: str | None,
    graduated_at_or_before_t: bool | None = None,
) -> dict[str, Any]:
    already = bool(graduated_at_or_before_t)
    event_observed = (
        time_to_graduation_status == STATUS_VALID_POSITIVE
        and time_to_graduation_sec is not None
    )
    event_time = None
    if event_observed and time_to_graduation_sec is not None:
        event_time = float(observation_time_t) + float(time_to_graduation_sec)
    fields = _effective_tte_fields(
        observation_time_t=observation_time_t,
        split_end=split_end,
        event_observed=event_observed,
        event_time_unix=event_time,
        raw_tte_sec=time_to_graduation_sec,
        raw_status=time_to_graduation_status,
        already_occurred=already,
    )
    return {f"graduation_tte_{k}": v for k, v in fields.items()}


def apply_explosion_tte_overlay(
    *,
    observation_time_t: float,
    split_end: float,
    time_to_explosion_sec: float | None,
    explosion_event_observed: bool | None,
    explosion_already_started_at_t: bool | None = None,
    explosion_right_censored: bool | None = None,
) -> dict[str, Any]:
    already = bool(explosion_already_started_at_t)
    event_observed = (
        bool(explosion_event_observed)
        and time_to_explosion_sec is not None
        and not already
    )
    event_time = None
    if event_observed and time_to_explosion_sec is not None:
        event_time = float(observation_time_t) + float(time_to_explosion_sec)
    raw_status = None
    if already:
        raw_status = STATUS_VALID_NEGATIVE  # already started → TTE not a future event
    elif event_observed:
        raw_status = STATUS_VALID_POSITIVE
    elif explosion_right_censored:
        raw_status = STATUS_RIGHT_CENSORED
    else:
        raw_status = STATUS_RIGHT_CENSORED
    fields = _effective_tte_fields(
        observation_time_t=observation_time_t,
        split_end=split_end,
        event_observed=event_observed,
        event_time_unix=event_time,
        raw_tte_sec=time_to_explosion_sec,
        raw_status=raw_status,
        already_occurred=already,
    )
    return {f"explosion_tte_{k}": v for k, v in fields.items()}


def apply_tte_overlays_for_row(
    target_row: dict[str, Any],
    *,
    observation_time_t: float,
    base_split: str,
    q70: float,
    q85: float,
    corpus_end: float = CORPUS_END_UNIX,
) -> dict[str, Any]:
    split_end = admin_boundary_unix(
        base_split, q70=q70, q85=q85, corpus_end=corpus_end
    )
    out: dict[str, Any] = {
        "observation_id": target_row.get("observation_id"),
        "mint": target_row.get("mint"),
        "observation_time_t": float(observation_time_t),
        "base_split": base_split,
        "admin_boundary_unix": split_end,
    }
    out.update(
        apply_graduation_tte_overlay(
            observation_time_t=observation_time_t,
            split_end=split_end,
            time_to_graduation_sec=target_row.get("time_to_graduation_sec"),
            time_to_graduation_status=target_row.get("time_to_graduation_status"),
            graduated_at_or_before_t=target_row.get("graduated_at_or_before_t"),
        )
    )
    out.update(
        apply_explosion_tte_overlay(
            observation_time_t=observation_time_t,
            split_end=split_end,
            time_to_explosion_sec=target_row.get("time_to_explosion_sec"),
            explosion_event_observed=target_row.get("explosion_event_observed"),
            explosion_already_started_at_t=target_row.get(
                "explosion_already_started_at_t"
            ),
            explosion_right_censored=target_row.get("explosion_right_censored"),
        )
    )
    return out
