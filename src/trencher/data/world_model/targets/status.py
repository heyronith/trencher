"""Common target-status contract — never encode unknown/censored as zero."""

from __future__ import annotations

from typing import Any

# Canonical status codes for every target family.
STATUS_VALID_POSITIVE = "valid_positive"
STATUS_VALID_NEGATIVE = "valid_negative"
STATUS_RIGHT_CENSORED = "right_censored"
STATUS_MODALITY_UNAVAILABLE = "modality_unavailable"
STATUS_SOURCE_BOUNDARY_UNAVAILABLE = "source_boundary_unavailable"
STATUS_MIGRATION_REGIME_UNAVAILABLE = "migration_regime_unavailable"
STATUS_KNOWN_SOURCE_OUTAGE = "known_source_outage"
STATUS_REFERENCE_PRICE_UNAVAILABLE = "reference_price_unavailable"
STATUS_NO_EVENT = "no_event"

ALL_STATUSES = (
    STATUS_VALID_POSITIVE,
    STATUS_VALID_NEGATIVE,
    STATUS_RIGHT_CENSORED,
    STATUS_MODALITY_UNAVAILABLE,
    STATUS_SOURCE_BOUNDARY_UNAVAILABLE,
    STATUS_MIGRATION_REGIME_UNAVAILABLE,
    STATUS_KNOWN_SOURCE_OUTAGE,
    STATUS_REFERENCE_PRICE_UNAVAILABLE,
    STATUS_NO_EVENT,
)

# Barrier first-hit outcomes (distinct from status).
BARRIER_UPPER_FIRST = "upper_first"
BARRIER_LOWER_FIRST = "lower_first"
BARRIER_NEITHER = "neither"
BARRIER_CENSORED = "censored"
BARRIER_UNAVAILABLE = "unavailable"

HORIZONS_SEC: tuple[int, ...] = (
    10,
    30,
    60,
    120,
    300,
    600,
    1800,
    3600,
    21600,
    86400,
)

SURVIVAL_HORIZONS_SEC: tuple[int, ...] = (
    30,
    60,
    120,
    300,
    600,
    1800,
    3600,
    21600,
    86400,
)

EXPLOSION_IMMINENCE_HORIZONS_SEC: tuple[int, ...] = (10, 30, 60, 120, 300)

BARRIER_PAIRS: tuple[tuple[float, float, str], ...] = (
    (0.10, -0.10, "u10_l10"),
    (0.25, -0.15, "u25_l15"),
    (0.50, -0.20, "u50_l20"),
    (1.00, -0.30, "u100_l30"),
)

POSITION_SIZES_SOL: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0)

INACTIVITY_CONFIRMATION_GAP_SEC = 1800.0  # 30 minutes

# July 3 2026 collection outage (UTC day).
OUTAGE_START_UNIX = 1783036800.0  # 2026-07-03 00:00:00 UTC
OUTAGE_END_UNIX = 1783123200.0  # 2026-07-04 00:00:00 UTC

CORPUS_START_UNIX = 1780646400.0  # 2026-06-05 00:00:00 UTC
CORPUS_END_UNIX = 1784073600.0  # 2026-07-15 00:00:00 UTC (end of Jul 14)


def status_contract() -> dict[str, Any]:
    return {
        "contract_id": "target_status_v1",
        "rule": "Never encode unknown/censored/unavailable as numeric zero.",
        "statuses": list(ALL_STATUSES),
        "barrier_outcomes": [
            BARRIER_UPPER_FIRST,
            BARRIER_LOWER_FIRST,
            BARRIER_NEITHER,
            BARRIER_CENSORED,
            BARRIER_UNAVAILABLE,
        ],
        "horizons_sec": list(HORIZONS_SEC),
        "survival_horizons_sec": list(SURVIVAL_HORIZONS_SEC),
        "explosion_imminence_horizons_sec": list(EXPLOSION_IMMINENCE_HORIZONS_SEC),
        "barrier_pairs": [
            {"upper": u, "lower": lo, "id": i} for u, lo, i in BARRIER_PAIRS
        ],
        "inactivity_confirmation_gap_sec": INACTIVITY_CONFIRMATION_GAP_SEC,
        "outage_window_unix": {
            "start": OUTAGE_START_UNIX,
            "end": OUTAGE_END_UNIX,
            "day": "2026-07-03",
        },
    }


def crosses_outage(t0: float, t1: float) -> bool:
    """True if [t0, t1] overlaps the known Jul 3 collection outage."""
    return t0 < OUTAGE_END_UNIX and t1 > OUTAGE_START_UNIX
