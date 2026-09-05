"""Split- and source-aware administrative censoring for graduation/explosion TTE.

Does not rewrite frozen target stores. Training views must never observe an
event at or after the split boundary.

Unresolved TTE uses the earliest legitimate censor boundary from the frozen
source-coverage registry — never fabricate long survival from split_end alone
when continuous coverage is unproven.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from trencher.data.world_model.splits.cutoffs import SPLIT_TRAIN, SPLIT_VAL
from trencher.data.world_model.targets.coverage import (
    SourceCoverage,
    SourceCoverageRegistry,
    earliest_continuous_censor_endpoint,
    load_source_coverage_registry,
)
from trencher.data.world_model.targets.status import (
    CORPUS_END_UNIX,
    STATUS_MODALITY_UNAVAILABLE,
    STATUS_RIGHT_CENSORED,
    STATUS_SOURCE_BOUNDARY_UNAVAILABLE,
    STATUS_VALID_NEGATIVE,
    STATUS_VALID_POSITIVE,
)

ADMIN_STATUS = STATUS_RIGHT_CENSORED
STATUS_LEFT_TRUNCATED = "left_truncated"
STATUS_UNAVAILABLE = STATUS_MODALITY_UNAVAILABLE


def admin_boundary_unix(
    base_split: str, *, q70: float, q85: float, corpus_end: float = CORPUS_END_UNIX
) -> float:
    """Exclusive end used for administrative TTE censoring."""
    if base_split == SPLIT_TRAIN:
        return float(q70)
    if base_split == SPLIT_VAL:
        return float(q85)
    return float(corpus_end)


@lru_cache(maxsize=4)
def default_coverage_registry(world_root: str | None = None) -> SourceCoverageRegistry:
    """Load frozen coverage registry (preferred paths, then rebuild)."""
    roots: list[Path] = []
    if world_root:
        roots.append(Path(world_root))
    roots.append(Path("data/external/world_model"))
    candidates = [
        "targets/target_dataset_v3/target_source_coverage_v2.json",
        "broad/broad_targets_v3/target_source_coverage_v2.json",
    ]
    for root in roots:
        for rel in candidates:
            path = root / rel
            if path.exists():
                return load_source_coverage_registry(path)
    # Last resort: empty migrations + empty pump (forces unavailable censoring)
    return SourceCoverageRegistry(
        pump_trades=SourceCoverage(
            source_id="pumpfun_trades",
            supports_absence_negatives=False,
            coverage_confidence="insufficient",
        ),
        migrations=SourceCoverage(
            source_id="migrations",
            supports_absence_negatives=False,
            coverage_confidence="insufficient",
        ),
        postgrad_snapshots=SourceCoverage(source_id="postgrad_snapshots"),
        ownership=SourceCoverage(source_id="ownership"),
    )


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
    return {
        "duration_sec": duration_sec,
        "event_observed": bool(event_observed),
        "censored": bool(censored),
        "at_risk": bool(at_risk),
        "effective_status": effective_status,
        "split_boundary_censored": bool(split_boundary_censored),
        "administrative_censor_time": administrative_censor_time,
        "effective_event_observed": bool(event_observed),
        "effective_tte_sec": duration_sec,
        "effective_censoring_status": effective_status,
        "raw_status": raw_status,
        "raw_tte_sec": raw_tte_sec,
        "raw_event_observed": bool(raw_event_observed),
    }


def _unavailable(
    *,
    raw_status: str | None,
    raw_tte_sec: float | None,
    raw_event_observed: bool,
    effective_status: str = STATUS_UNAVAILABLE,
) -> dict[str, Any]:
    return _pack(
        duration_sec=None,
        event_observed=False,
        censored=False,
        at_risk=False,
        effective_status=effective_status,
        split_boundary_censored=False,
        administrative_censor_time=None,
        raw_status=raw_status,
        raw_tte_sec=raw_tte_sec,
        raw_event_observed=raw_event_observed,
    )


def _censor_at(
    *,
    observation_time_t: float,
    endpoint: float,
    split_end: float,
    raw_status: str | None,
    raw_tte_sec: float | None,
    raw_event_observed: bool,
) -> dict[str, Any]:
    T = float(observation_time_t)
    end = float(endpoint)
    return _pack(
        duration_sec=float(end - T),
        event_observed=False,
        censored=True,
        at_risk=True,
        effective_status=ADMIN_STATUS,
        split_boundary_censored=end >= float(split_end) - 1e-9,
        administrative_censor_time=end,
        raw_status=raw_status,
        raw_tte_sec=raw_tte_sec,
        raw_event_observed=raw_event_observed,
    )


def _observed_event(
    *,
    observation_time_t: float,
    split_end: float,
    event_time_unix: float,
    raw_tte_sec: float | None,
    raw_status: str | None,
    censor_source: SourceCoverage | None,
) -> dict[str, Any]:
    """Future event relative to T; hide if at/after split_end."""
    T = float(observation_time_t)
    et = float(event_time_unix)
    end = float(split_end)
    if et < end:
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
    # Hide post-boundary known event. Prefer earliest continuous source/outage
    # cap when available; always at least admin-censor at split_end so the
    # future event is never leaked into the training view.
    ep = end
    if censor_source is not None:
        src_ep = earliest_continuous_censor_endpoint(
            censor_source, T, split_end=end
        )
        if src_ep is not None:
            ep = min(end, src_ep)
    return _censor_at(
        observation_time_t=T,
        endpoint=ep,
        split_end=end,
        raw_status=raw_status,
        raw_tte_sec=raw_tte_sec,
        raw_event_observed=True,
    )


def apply_graduation_tte_overlay(
    *,
    observation_time_t: float,
    split_end: float,
    time_to_graduation_sec: float | None,
    time_to_graduation_status: str | None,
    graduated_at_or_before_t: bool | None = None,
    coverage: SourceCoverageRegistry | None = None,
) -> dict[str, Any]:
    """Graduation TTE with migration source-coverage semantics."""
    T = float(observation_time_t)
    end = float(split_end)
    cov = (coverage or default_coverage_registry()).migrations
    already = bool(graduated_at_or_before_t)
    raw_status = time_to_graduation_status

    if already:
        return {
            f"graduation_tte_{k}": v
            for k, v in _pack(
                duration_sec=None,
                event_observed=False,
                censored=False,
                at_risk=False,
                effective_status=STATUS_LEFT_TRUNCATED,
                split_boundary_censored=False,
                administrative_censor_time=None,
                raw_status=raw_status,
                raw_tte_sec=time_to_graduation_sec,
                raw_event_observed=bool(
                    time_to_graduation_status == STATUS_VALID_POSITIVE
                ),
            ).items()
        }

    event_observed = (
        time_to_graduation_status == STATUS_VALID_POSITIVE
        and time_to_graduation_sec is not None
    )
    if event_observed and time_to_graduation_sec is not None:
        event_time = T + float(time_to_graduation_sec)
        # Known future migration: may remain an event before boundary.
        # Do not require absence coverage to observe a positive.
        fields = _observed_event(
            observation_time_t=T,
            split_end=end,
            event_time_unix=event_time,
            raw_tte_sec=time_to_graduation_sec,
            raw_status=raw_status,
            censor_source=None,  # hide at split_end when past boundary
        )
        return {f"graduation_tte_{k}": v for k, v in fields.items()}

    if raw_status == STATUS_VALID_NEGATIVE:
        # Frozen store resolved a genuine negative with coverage.
        fields = _pack(
            duration_sec=None,
            event_observed=False,
            censored=False,
            at_risk=True,
            effective_status=STATUS_VALID_NEGATIVE,
            split_boundary_censored=False,
            administrative_censor_time=None,
            raw_status=raw_status,
            raw_tte_sec=time_to_graduation_sec,
            raw_event_observed=False,
        )
        return {f"graduation_tte_{k}": v for k, v in fields.items()}

    # Unresolved / right_censored: only censor if migrations support continuous
    # absence coverage (frozen registry: they do not).
    ep = earliest_continuous_censor_endpoint(cov, T, split_end=end)
    if ep is None:
        fields = _unavailable(
            raw_status=raw_status or STATUS_RIGHT_CENSORED,
            raw_tte_sec=time_to_graduation_sec,
            raw_event_observed=False,
            effective_status=STATUS_SOURCE_BOUNDARY_UNAVAILABLE
            if not cov.supports_absence_negatives
            else STATUS_UNAVAILABLE,
        )
        return {f"graduation_tte_{k}": v for k, v in fields.items()}
    fields = _censor_at(
        observation_time_t=T,
        endpoint=ep,
        split_end=end,
        raw_status=raw_status,
        raw_tte_sec=time_to_graduation_sec,
        raw_event_observed=False,
    )
    return {f"graduation_tte_{k}": v for k, v in fields.items()}


def _explosion_modality_unavailable(target_hints: dict[str, Any] | None) -> bool:
    if not target_hints:
        return False
    if target_hints.get("explosion_primary_exists") is False:
        # explicit false means evaluated; None/missing checked via statuses
        pass
    if target_hints.get("explosion_primary_exists") is None and any(
        target_hints.get(f"explosion_within_{h}s_status") == STATUS_MODALITY_UNAVAILABLE
        for h in (10, 30, 60, 120, 300)
    ):
        return True
    st60 = target_hints.get("explosion_within_60s_status")
    if st60 == STATUS_MODALITY_UNAVAILABLE:
        return True
    return bool(target_hints.get("explosion_modality_unavailable"))


def apply_explosion_tte_overlay(
    *,
    observation_time_t: float,
    split_end: float,
    time_to_explosion_sec: float | None,
    explosion_event_observed: bool | None,
    explosion_already_started_at_t: bool | None = None,
    explosion_right_censored: bool | None = None,
    coverage: SourceCoverageRegistry | None = None,
    modality_unavailable: bool | None = None,
    target_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Explosion TTE with Pump trade continuous-coverage censoring."""
    T = float(observation_time_t)
    end = float(split_end)
    pump = (coverage or default_coverage_registry()).pump_trades
    already = bool(explosion_already_started_at_t)

    if modality_unavailable is None:
        modality_unavailable = _explosion_modality_unavailable(target_hints)

    if already:
        fields = _pack(
            duration_sec=None,
            event_observed=False,
            censored=False,
            at_risk=False,
            effective_status=STATUS_LEFT_TRUNCATED,
            split_boundary_censored=False,
            administrative_censor_time=None,
            raw_status=STATUS_LEFT_TRUNCATED,
            raw_tte_sec=time_to_explosion_sec,
            raw_event_observed=bool(explosion_event_observed),
        )
        return {f"explosion_tte_{k}": v for k, v in fields.items()}

    if modality_unavailable:
        fields = _unavailable(
            raw_status=STATUS_MODALITY_UNAVAILABLE,
            raw_tte_sec=time_to_explosion_sec,
            raw_event_observed=bool(explosion_event_observed),
        )
        return {f"explosion_tte_{k}": v for k, v in fields.items()}

    event_observed = (
        bool(explosion_event_observed)
        and time_to_explosion_sec is not None
        and not already
    )
    if event_observed and time_to_explosion_sec is not None:
        event_time = T + float(time_to_explosion_sec)
        fields = _observed_event(
            observation_time_t=T,
            split_end=end,
            event_time_unix=event_time,
            raw_tte_sec=time_to_explosion_sec,
            raw_status=STATUS_VALID_POSITIVE,
            censor_source=pump,
        )
        return {f"explosion_tte_{k}": v for k, v in fields.items()}

    # Evaluated no-explosion or unresolved: censor only while Pump coverage continuous
    if explosion_right_censored or not event_observed:
        ep = earliest_continuous_censor_endpoint(pump, T, split_end=end)
        if ep is None:
            fields = _unavailable(
                raw_status=STATUS_RIGHT_CENSORED
                if explosion_right_censored
                else STATUS_SOURCE_BOUNDARY_UNAVAILABLE,
                raw_tte_sec=time_to_explosion_sec,
                raw_event_observed=False,
            )
            return {f"explosion_tte_{k}": v for k, v in fields.items()}
        fields = _censor_at(
            observation_time_t=T,
            endpoint=ep,
            split_end=end,
            raw_status=STATUS_RIGHT_CENSORED,
            raw_tte_sec=time_to_explosion_sec,
            raw_event_observed=False,
        )
        return {f"explosion_tte_{k}": v for k, v in fields.items()}

    fields = _unavailable(
        raw_status=STATUS_RIGHT_CENSORED,
        raw_tte_sec=time_to_explosion_sec,
        raw_event_observed=False,
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
    coverage: SourceCoverageRegistry | None = None,
    world_root: Path | str | None = None,
) -> dict[str, Any]:
    split_end = admin_boundary_unix(
        base_split, q70=q70, q85=q85, corpus_end=corpus_end
    )
    cov = coverage
    if cov is None and world_root is not None:
        cov = default_coverage_registry(str(world_root))
    if cov is None:
        cov = default_coverage_registry()
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
            coverage=cov,
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
            coverage=cov,
            target_hints=target_row,
        )
    )
    return out


def normalize_overlay_row(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Map persisted overlay columns → canonical training fields."""
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
    status = row.get(f"{prefix}_effective_status") or row.get(
        f"{prefix}_effective_censoring_status"
    )
    if at_risk is None:
        at_risk = status not in (
            STATUS_LEFT_TRUNCATED,
            STATUS_UNAVAILABLE,
            STATUS_MODALITY_UNAVAILABLE,
            STATUS_SOURCE_BOUNDARY_UNAVAILABLE,
        )
    return {
        "duration_sec": duration,
        "event_observed": bool(event_obs) if event_obs is not None else False,
        "censored": bool(censored) if censored is not None else False,
        "at_risk": bool(at_risk),
        "effective_status": status,
        "split_boundary_censored": bool(row.get(f"{prefix}_split_boundary_censored")),
        "administrative_censor_time": row.get(f"{prefix}_administrative_censor_time"),
        "raw_status": row.get(f"{prefix}_raw_status"),
        "raw_tte_sec": row.get(f"{prefix}_raw_tte_sec"),
        "raw_event_observed": bool(row.get(f"{prefix}_raw_event_observed")),
    }
