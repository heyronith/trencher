"""Centralized LABEL-ONLY source coverage registry (v2 evidence model).

Absence-based targets must use these source intervals — never a mint's own
last event time. Coverage intervals are annotated with basis/evidence/
confidence so first/last event timestamps are not silently treated as
proven continuous collection coverage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trencher.data.world_model.io import utc_now, write_json
from trencher.data.world_model.targets.status import (
    CORPUS_END_UNIX,
    CORPUS_START_UNIX,
    OUTAGE_END_UNIX,
    OUTAGE_START_UNIX,
    STATUS_KNOWN_SOURCE_OUTAGE,
    STATUS_RIGHT_CENSORED,
    STATUS_SOURCE_BOUNDARY_UNAVAILABLE,
    STATUS_VALID_NEGATIVE,
    crosses_outage,
)
from trencher.data.world_model.targets.timeutil import to_unix

COVERAGE_MANIFEST_VERSION = "target_source_coverage_v2"

# Confidence levels for absence-based negatives.
CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
CONFIDENCE_INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class TimeInterval:
    start: float
    end: float

    def contains(self, t: float) -> bool:
        return self.start <= t <= self.end

    def covers_range(self, t0: float, t1: float) -> bool:
        return self.start <= t0 and t1 <= self.end

    def overlaps(self, t0: float, t1: float) -> bool:
        return t0 < self.end and t1 > self.start


@dataclass
class SourceCoverage:
    source_id: str
    intervals: list[TimeInterval] = field(default_factory=list)
    known_gaps: list[TimeInterval] = field(default_factory=list)
    notes: str = ""
    coverage_basis: str = ""
    coverage_evidence: list[str] = field(default_factory=list)
    coverage_confidence: str = CONFIDENCE_INSUFFICIENT
    supports_absence_negatives: bool = False
    event_span: dict[str, float | None] | None = None

    def covers_through(self, t: float) -> bool:
        return any(iv.contains(t) for iv in self.intervals)

    def covers_window(self, t0: float, t1: float) -> bool:
        if t1 < t0:
            return False
        if not self.supports_absence_negatives or not self.intervals:
            return False
        if any(iv.covers_range(t0, t1) for iv in self.intervals):
            return True
        merged = _merge_intervals(self.intervals)
        return any(iv.covers_range(t0, t1) for iv in merged)

    def intersects_known_gap(self, t0: float, t1: float) -> bool:
        return any(g.overlaps(t0, t1) for g in self.known_gaps)

    def absence_status(
        self,
        t0: float,
        t1: float,
        *,
        resolved_by_later_observation: bool = False,
    ) -> str:
        """Status when no event was observed in/through a window.

        Never returns valid_positive. valid_negative only when the source
        fully covers the confirmation window with sufficient confidence.
        """
        if not self.supports_absence_negatives:
            # Cannot prove continuous collection coverage → censor absences.
            if self.intersects_known_gap(t0, t1) and not resolved_by_later_observation:
                return STATUS_KNOWN_SOURCE_OUTAGE
            return STATUS_RIGHT_CENSORED

        if self.covers_window(t0, t1):
            return STATUS_VALID_NEGATIVE
        if self.intersects_known_gap(t0, t1) and not resolved_by_later_observation:
            return STATUS_KNOWN_SOURCE_OUTAGE
        if (
            crosses_outage(t0, t1)
            and self.known_gaps
            and not resolved_by_later_observation
            and any(
                g.start == OUTAGE_START_UNIX and g.end == OUTAGE_END_UNIX
                for g in self.known_gaps
            )
        ):
            # Only treat global Jul 3 as outage when this source declares that gap.
            return STATUS_KNOWN_SOURCE_OUTAGE
        last_end = max((iv.end for iv in self.intervals), default=None)
        if last_end is not None and t1 > last_end:
            return STATUS_SOURCE_BOUNDARY_UNAVAILABLE
        return STATUS_RIGHT_CENSORED

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "coverage_basis": self.coverage_basis,
            "coverage_evidence": list(self.coverage_evidence),
            "coverage_confidence": self.coverage_confidence,
            "supports_absence_negatives": self.supports_absence_negatives,
            "intervals": [asdict(iv) for iv in self.intervals],
            "known_gaps": [asdict(g) for g in self.known_gaps],
            "event_span": self.event_span,
            "notes": self.notes,
        }


def _merge_intervals(intervals: list[TimeInterval]) -> list[TimeInterval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv.start)
    out = [ordered[0]]
    for iv in ordered[1:]:
        prev = out[-1]
        if iv.start <= prev.end:
            out[-1] = TimeInterval(prev.start, max(prev.end, iv.end))
        else:
            out.append(iv)
    return out


@dataclass
class SourceCoverageRegistry:
    pump_trades: SourceCoverage
    migrations: SourceCoverage
    postgrad_snapshots: SourceCoverage
    ownership: SourceCoverage
    generated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": COVERAGE_MANIFEST_VERSION,
            "generated_at": self.generated_at,
            "corpus_window_unix": {
                "start": CORPUS_START_UNIX,
                "end": CORPUS_END_UNIX,
            },
            "outage_window_unix": {
                "start": OUTAGE_START_UNIX,
                "end": OUTAGE_END_UNIX,
                "day": "2026-07-03",
                "applies_to": ["pumpfun_trades"],
                "note": (
                    "KNOWN_ISSUES documents Jul 3 as a websocket trade-collection "
                    "outage (zero trades). Do not automatically inherit onto "
                    "migrations or post-grad without local evidence."
                ),
            },
            "sources": {
                "pump_trades": self.pump_trades.to_dict(),
                "migrations": self.migrations.to_dict(),
                "postgrad_snapshots": self.postgrad_snapshots.to_dict(),
                "ownership": self.ownership.to_dict(),
            },
            "policy": {
                "absence_targets_use_source_coverage_not_mint_last_event": True,
                "event_min_max_is_not_collection_coverage": True,
                "jul3_outage_never_creates_death_labels": True,
                "synthetic_pool_address_does_not_invalidate_migrated_at": True,
            },
        }


def _jul3_gap() -> TimeInterval:
    return TimeInterval(OUTAGE_START_UNIX, OUTAGE_END_UNIX)


def _split_around_outage(start: float, end: float) -> list[TimeInterval]:
    gap = _jul3_gap()
    if end <= start:
        return []
    if end <= gap.start or start >= gap.end:
        return [TimeInterval(start, end)]
    out: list[TimeInterval] = []
    if start < gap.start:
        out.append(TimeInterval(start, gap.start))
    if end > gap.end:
        out.append(TimeInterval(gap.end, end))
    return out


def _event_span(times: list[float]) -> dict[str, float | None]:
    if not times:
        return {"min": None, "max": None, "n": 0}
    return {"min": float(min(times)), "max": float(max(times)), "n": len(times)}


def load_source_coverage_registry(path: Path) -> SourceCoverageRegistry:
    """Load frozen ``target_source_coverage_v2`` JSON (no rebuild)."""
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sources = payload.get("sources") or {}

    def _cov(key: str) -> SourceCoverage:
        d = sources.get(key) or {}
        return SourceCoverage(
            source_id=str(d.get("source_id") or key),
            intervals=[
                TimeInterval(float(iv["start"]), float(iv["end"]))
                for iv in (d.get("intervals") or [])
            ],
            known_gaps=[
                TimeInterval(float(g["start"]), float(g["end"]))
                for g in (d.get("known_gaps") or [])
            ],
            notes=str(d.get("notes") or ""),
            coverage_basis=str(d.get("coverage_basis") or ""),
            coverage_evidence=list(d.get("coverage_evidence") or []),
            coverage_confidence=str(
                d.get("coverage_confidence") or CONFIDENCE_INSUFFICIENT
            ),
            supports_absence_negatives=bool(d.get("supports_absence_negatives")),
            event_span=d.get("event_span"),
        )

    return SourceCoverageRegistry(
        pump_trades=_cov("pump_trades"),
        migrations=_cov("migrations"),
        postgrad_snapshots=_cov("postgrad_snapshots"),
        ownership=_cov("ownership"),
        generated_at=str(payload.get("generated_at") or utc_now()),
    )


def earliest_continuous_censor_endpoint(
    cov: SourceCoverage,
    observation_time_t: float,
    *,
    split_end: float,
) -> float | None:
    """Earliest legitimate continuous censor time from T, or None if unavailable.

    Caps follow-up at the earliest of: split boundary, end of the coverage
    interval containing T, or the start of a known gap after T.
    """
    T = float(observation_time_t)
    end_cap = float(split_end)
    if not cov.supports_absence_negatives or not cov.intervals:
        return None
    containing = [iv for iv in cov.intervals if iv.start <= T < iv.end]
    if not containing:
        # Inclusive end edge: T exactly at interval.end is not continuous interior
        containing = [iv for iv in cov.intervals if iv.contains(T) and iv.end > T]
    if not containing:
        return None
    iv = containing[0]
    endpoint = min(end_cap, float(iv.end))
    for g in cov.known_gaps:
        if T < g.start < endpoint:
            endpoint = float(g.start)
    if endpoint <= T:
        return None
    return endpoint


def build_source_coverage_registry(
    *,
    world_root: Path,
    pump_root: Path | None = None,
) -> SourceCoverageRegistry:
    """Derive evidence-annotated coverage from frozen local documentation."""
    pump_root = pump_root or Path("data/external/pumpfun_memecoin_corpus")
    gap = _jul3_gap()
    known_issues = pump_root / "KNOWN_ISSUES.md"
    known_issues_exists = known_issues.exists()

    # --- Pump trades: high confidence ---
    trade_start = CORPUS_START_UNIX
    trade_end = CORPUS_END_UNIX
    buckets = world_root / "pit" / "global" / "trade_second_buckets.parquet"
    bucket_rows = 0
    if buckets.exists() and pq.ParquetFile(buckets).metadata.num_rows:
        secs = pq.read_table(buckets, columns=["second_unix"]).column(0).to_pylist()
        if secs:
            trade_start = float(min(secs))
            trade_end = float(max(secs)) + 1.0
            bucket_rows = len(secs)

    pump_trades = SourceCoverage(
        source_id="pumpfun_trades",
        intervals=_split_around_outage(trade_start, trade_end),
        known_gaps=[gap],
        coverage_basis=(
            "frozen_global_trade_second_buckets_plus_documented_jul3_outage"
        ),
        coverage_evidence=[
            str(buckets) if buckets.exists() else "corpus_window_fallback",
            f"trade_second_bucket_rows={bucket_rows}",
            "KNOWN_ISSUES.md §3.4/§5.5 Jul 3 websocket collection outage (zero trades)",
            "feasibility/reconstruction.json caveats: Jul 3 2026 collection outage",
            f"corpus_window={CORPUS_START_UNIX}..{CORPUS_END_UNIX}",
        ],
        coverage_confidence=CONFIDENCE_HIGH,
        supports_absence_negatives=True,
        event_span={"min": trade_start, "max": trade_end, "n": bucket_rows},
        notes=(
            "Pump trade collection coverage from frozen global second buckets. "
            "Jul 3 gap is documented specifically for trades."
        ),
    )

    # --- Migrations: do NOT treat min/max migrated_at as continuous coverage ---
    mig_times: list[float] = []
    mig_source_path = None
    for path in (
        world_root / "pit" / "global" / "migrations_timeline.parquet",
        world_root / "universe" / "migrations.parquet",
        pump_root / "migrations.parquet",
    ):
        if not path.exists():
            continue
        cols = pq.read_schema(path).names
        tcol = (
            "event_time_unix"
            if "event_time_unix" in cols
            else ("migrated_at" if "migrated_at" in cols else None)
        )
        if tcol is None:
            continue
        raw = [
            to_unix(x)
            for x in pq.read_table(path, columns=[tcol]).column(0).to_pylist()
        ]
        mig_times = [t for t in raw if t is not None]
        mig_source_path = str(path)
        break

    mig_evidence = [
        mig_source_path or "migrations_path_missing",
        "KNOWN_ISSUES.md §4.55: pool_address may be synthetic/backfilled — "
        "does not invalidate migrated_at as an event time",
        "feasibility/reconstruction.json: migrations.migrated_at is a graduation "
        "evidence field; pool_address partially synthetic is a separate caveat",
        "No local documentation proves continuous migration-event collection "
        "coverage equivalent to the trade websocket stream",
        "Jul 3 outage is documented for trades, not migrations — not inherited",
    ]
    if known_issues_exists:
        mig_evidence.append(f"reviewed:{known_issues}")

    migrations = SourceCoverage(
        source_id="migrations",
        # Empty intervals: continuous collection coverage not proven.
        intervals=[],
        known_gaps=[],
        coverage_basis="event_presence_only_continuous_collection_unproven",
        coverage_evidence=mig_evidence,
        coverage_confidence=CONFIDENCE_INSUFFICIENT,
        supports_absence_negatives=False,
        event_span=_event_span(mig_times),
        notes=(
            "migrated_at event timestamps remain valid event evidence. "
            "Absence-based graduation negatives are censored unless a later "
            "observed migration proves a shorter-horizon negative/positive. "
            "Min/max migrated_at is recorded as event_span only, not coverage."
        ),
    )

    # --- Post-grad snapshots: event span; do not inherit Jul 3 trade gap ---
    pg_times: list[float] = []
    pg_path = pump_root / "postgard_snapshots.parquet"
    if pg_path.exists():
        raw = [
            to_unix(x)
            for x in pq.read_table(pg_path, columns=["snapshot_time"])
            .column(0)
            .to_pylist()
        ]
        pg_times = [t for t in raw if t is not None]

    postgrad = SourceCoverage(
        source_id="postgard_snapshots",
        intervals=[],
        known_gaps=[],
        coverage_basis="graduated_subset_snapshot_event_span_not_full_collection",
        coverage_evidence=[
            str(pg_path) if pg_path.exists() else "postgard_snapshots_missing",
            "KNOWN_ISSUES.md: postgard_snapshots covers graduated subset (~5701 mints), "
            "not full launch universe",
            "Jul 3 trade websocket outage not documented as applying to postgard "
            "snapshot collection — not inherited",
            "feasibility/reconstruction.json: post_graduation_trajectory = partial",
        ],
        coverage_confidence=CONFIDENCE_LOW,
        supports_absence_negatives=False,
        event_span=_event_span(pg_times),
        notes=(
            "Post-grad marks remain usable where snapshots exist. "
            "Do not treat event min/max as proven continuous coverage, and do "
            "not fabricate a Jul 3 gap without local evidence."
        ),
    )

    ownership = SourceCoverage(
        source_id="ownership_evidence",
        intervals=[],
        known_gaps=[],
        coverage_basis="mint_level_sparse_evidence_flags_only",
        coverage_evidence=[
            str(world_root / "universe" / "ownership_evidence.parquet"),
            "No continuous time-indexed holder ledger for most mints in freeze",
        ],
        coverage_confidence=CONFIDENCE_INSUFFICIENT,
        supports_absence_negatives=False,
        event_span=None,
        notes="Ownership targets remain sparse by design.",
    )

    return SourceCoverageRegistry(
        pump_trades=pump_trades,
        migrations=migrations,
        postgrad_snapshots=postgrad,
        ownership=ownership,
    )


def write_coverage_manifest(
    registry: SourceCoverageRegistry, path: Path
) -> dict[str, Any]:
    payload = registry.to_dict()
    write_json(path, payload)
    return payload
