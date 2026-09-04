"""Centralized LABEL-ONLY source coverage registry.

Absence-based targets (survival, persistence zeros, graduation negatives)
must use these source intervals — never a mint's own last event time.
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

COVERAGE_MANIFEST_VERSION = "target_source_coverage_v1"


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

    def covers_through(self, t: float) -> bool:
        """True if some interval includes time t (source observes at t)."""
        return any(iv.contains(t) for iv in self.intervals)

    def covers_window(self, t0: float, t1: float) -> bool:
        """True if [t0, t1] lies entirely inside the union of coverage intervals.

        Gaps (including Jul 3 outage) mean the window is not fully covered.
        """
        if t1 < t0:
            return False
        # Walk coverage: every point in [t0,t1] must fall in some interval.
        # Conservative: require a single interval that covers the whole window,
        # or a chain of contiguous intervals without gaps inside [t0,t1].
        if any(iv.covers_range(t0, t1) for iv in self.intervals):
            return True
        # Merge intervals and check gap-free coverage of [t0,t1].
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
        fully covers the confirmation window so absence is observable.
        """
        if self.covers_window(t0, t1):
            return STATUS_VALID_NEGATIVE
        if self.intersects_known_gap(t0, t1) and not resolved_by_later_observation:
            return STATUS_KNOWN_SOURCE_OUTAGE
        if crosses_outage(t0, t1) and not resolved_by_later_observation:
            return STATUS_KNOWN_SOURCE_OUTAGE
        # Prefer source-boundary when t1 is past the last covered instant.
        last_end = max((iv.end for iv in self.intervals), default=None)
        if last_end is not None and t1 > last_end:
            return STATUS_SOURCE_BOUNDARY_UNAVAILABLE
        return STATUS_RIGHT_CENSORED

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "intervals": [asdict(iv) for iv in self.intervals],
            "known_gaps": [asdict(g) for g in self.known_gaps],
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
    """All label-relevant source coverage planes."""

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
            },
            "sources": {
                "pump_trades": self.pump_trades.to_dict(),
                "migrations": self.migrations.to_dict(),
                "postgrad_snapshots": self.postgrad_snapshots.to_dict(),
                "ownership": self.ownership.to_dict(),
            },
            "policy": {
                "absence_targets_use_source_coverage_not_mint_last_event": True,
                "jul3_outage_never_creates_death_or_graduation_negative": True,
            },
        }


def _jul3_gap() -> TimeInterval:
    return TimeInterval(OUTAGE_START_UNIX, OUTAGE_END_UNIX)


def _split_around_outage(start: float, end: float) -> list[TimeInterval]:
    """Coverage intervals excluding the Jul 3 collection outage."""
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


def build_source_coverage_registry(
    *,
    world_root: Path,
    pump_root: Path | None = None,
) -> SourceCoverageRegistry:
    """Derive coverage from frozen corpus / world-model indexes."""
    pump_root = pump_root or Path("data/external/pumpfun_memecoin_corpus")
    gap = _jul3_gap()

    trade_start = CORPUS_START_UNIX
    trade_end = CORPUS_END_UNIX
    buckets = world_root / "pit" / "global" / "trade_second_buckets.parquet"
    if buckets.exists() and pq.ParquetFile(buckets).metadata.num_rows:
        secs = pq.read_table(buckets, columns=["second_unix"]).column(0).to_pylist()
        if secs:
            trade_start = float(min(secs))
            # Buckets are second starts; coverage extends through end of last second.
            trade_end = float(max(secs)) + 1.0

    pump_trades = SourceCoverage(
        source_id="pumpfun_trades",
        intervals=_split_around_outage(trade_start, trade_end),
        known_gaps=[gap],
        notes=(
            "Pump trade corpus coverage from frozen second buckets / corpus window. "
            "Jul 3 2026 is a known collection outage (zero trades)."
        ),
    )

    mig_start = CORPUS_START_UNIX
    mig_end = CORPUS_END_UNIX
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
        times = [
            to_unix(x)
            for x in pq.read_table(path, columns=[tcol]).column(0).to_pylist()
        ]
        times = [t for t in times if t is not None]
        if times:
            mig_start = float(min(times))
            mig_end = float(max(times))
            break

    migrations = SourceCoverage(
        source_id="migrations",
        intervals=_split_around_outage(mig_start, mig_end),
        known_gaps=[gap],
        notes=(
            "Migration-event source coverage from frozen migrations timeline. "
            "Graduation negatives require migration source covering through T+H, "
            "not the token's own trade activity."
        ),
    )

    pg_start = CORPUS_START_UNIX
    pg_end = CORPUS_END_UNIX
    pg_path = pump_root / "postgard_snapshots.parquet"
    if pg_path.exists():
        times = [
            to_unix(x)
            for x in pq.read_table(pg_path, columns=["snapshot_time"])
            .column(0)
            .to_pylist()
        ]
        times = [t for t in times if t is not None]
        if times:
            pg_start = float(min(times))
            pg_end = float(max(times))

    postgrad = SourceCoverage(
        source_id="postgard_snapshots",
        intervals=_split_around_outage(pg_start, pg_end),
        known_gaps=[gap],
        notes="Genuine post-grad snapshot coverage only; not fabricated continuity.",
    )

    ownership = SourceCoverage(
        source_id="ownership_evidence",
        intervals=[],  # mint-level sparse; no continuous time series in freeze
        known_gaps=[gap],
        notes=(
            "Ownership targets remain sparse; no continuous holder ledger path "
            "for most mints in the freeze."
        ),
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
