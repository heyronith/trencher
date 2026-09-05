"""Split-aware administrative censoring for graduation/explosion TTE.

Does not rewrite frozen target stores. Training views must never observe an
event at or after the split boundary.

Training representation (per at-risk observation):
  duration_sec, event_observed, censored, at_risk, effective_status,
  split_boundary_censored
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
STATUS_LEFT_TRUNCATED = "left_truncated"


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


def _pack(
    *,
    duration_sec: float | None,
    event_observed: bool,
    censored: bool,
    at_risk: bool,
    effective_status: str | None,
    split_boundary_censored: bool,
    administrative_censor_time: float | None,
    raw_status: str | None,
    raw_tte_sec: float | None,
    raw_event_observed: bool,
) -> dict[str, Any]:
    """Canonical TTE training-view fields (+ backward-compatible aliases)."""
    return {
        "duration_sec": duration_sec,
        "event_observed": bool(event_observed),
        "censored": bool(censored),
        "at_risk": bool(at_risk),
        "effective_status": effective_status,
        "split_boundary_censored": bool(split_boundary_censored),
        "administrative_censor_time": administrative_censor_time,
        # Aliases used by persisted overlays / older call sites
        "effective_event_observed": bool(event_observed),
        "effective_tte_sec": duration_sec,
        "effective_censoring_status": effective_status,
        "raw_status": raw_status,
        "raw_tte_sec": raw_tte_sec,
        "raw_event_observed": bool(raw_event_observed),
    }


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

    if already_occurred:
        # Event already known at/before T — left-truncated / current-state.
        # Not a newly observed *future* event; exclude from at-risk TTE risk set.
        return _pack(
            duration_sec=None,
            event_observed=False,
            censored=False,
            at_risk=False,
            effective_status=STATUS_LEFT_TRUNCATED,
            split_boundary_censored=False,
            administrative_censor_time=None,
            raw_status=raw_status,
            raw_tte_sec=raw_tte_sec,
            raw_event_observed=bool(event_observed),
        )

    if event_observed and event_time_unix is not None:
        et = float(event_time_unix)
        if et < end:
            # Strictly before boundary → may remain observed.
            dur = float(et - T) if raw_tte_sec is None else float(raw_tte_sec)
            return _pack(
                duration_sec=dur,
                event_observed=True,
                censored=False,
                at_risk=True,
                effective_status=STATUS_VALID_POSITIVE,
                split_boundary_censored=False,
                administrative_censor_time=None,
                raw_status=raw_status,
                raw_tte_sec=raw_tte_sec,
                raw_event_observed=True,
            )
        # At or after split_end → hide future event; admin right-censor.
        # Retain censor duration; never convert to a negative.
        return _pack(
            duration_sec=float(end - T),
            event_observed=False,
            censored=True,
            at_risk=True,
            effective_status=ADMIN_STATUS,
            split_boundary_censored=True,
            administrative_censor_time=end,
            raw_status=raw_status,
            raw_tte_sec=raw_tte_sec,
            raw_event_observed=True,
        )

    # Unresolved: censor at earliest legitimate source/split boundary.
    # Never invent a negative from administrative censoring alone.
    if raw_status == STATUS_VALID_NEGATIVE:
        # Frozen store already resolved a genuine negative with coverage.
        return _pack(
            duration_sec=None,
            event_observed=False,
            censored=False,
            at_risk=True,
            effective_status=STATUS_VALID_NEGATIVE,
            split_boundary_censored=False,
            administrative_censor_time=None,
            raw_status=raw_status,
            raw_tte_sec=raw_tte_sec,
            raw_event_observed=False,
        )

    return _pack(
        duration_sec=float(end - T),
        event_observed=False,
        censored=True,
        at_risk=True,
        effective_status=ADMIN_STATUS,
        split_boundary_censored=True,
        administrative_censor_time=end,
        raw_status=raw_status,
        raw_tte_sec=raw_tte_sec,
        raw_event_observed=False,
    )


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
        and not already
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
    if already:
        raw_status = STATUS_LEFT_TRUNCATED
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


def normalize_overlay_row(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Map persisted overlay columns → canonical training fields."""
    # Prefer new names; fall back to V2 aliases.
    duration = row.get(f"{prefix}_duration_sec")
    if duration is None:
        duration = row.get(f"{prefix}_effective_tte_sec")
    event_obs = row.get(f"{prefix}_event_observed")
    if event_obs is None:
        event_obs = row.get(f"{prefix}_effective_event_observed")
    censored = row.get(f"{prefix}_censored")
    if censored is None:
        censored = bool(row.get(f"{prefix}_split_boundary_censored")) or (
            row.get(f"{prefix}_effective_censoring_status") == ADMIN_STATUS
            and not bool(event_obs)
        )
    at_risk = row.get(f"{prefix}_at_risk")
    if at_risk is None:
        status = row.get(f"{prefix}_effective_status") or row.get(
            f"{prefix}_effective_censoring_status"
        )
        at_risk = status != STATUS_LEFT_TRUNCATED
    status = row.get(f"{prefix}_effective_status") or row.get(
        f"{prefix}_effective_censoring_status"
    )
    return {
        "duration_sec": duration,
        "event_observed": bool(event_obs) if event_obs is not None else False,
        "censored": bool(censored),
        "at_risk": bool(at_risk),
        "effective_status": status,
        "split_boundary_censored": bool(row.get(f"{prefix}_split_boundary_censored")),
        "administrative_censor_time": row.get(f"{prefix}_administrative_censor_time"),
        "raw_status": row.get(f"{prefix}_raw_status"),
        "raw_tte_sec": row.get(f"{prefix}_raw_tte_sec"),
        "raw_event_observed": bool(row.get(f"{prefix}_raw_event_observed")),
    }
