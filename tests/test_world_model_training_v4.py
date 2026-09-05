"""V4 training-readiness acceptance tests."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.splits.tte_overlay import (
    STATUS_LEFT_TRUNCATED,
    apply_explosion_tte_overlay,
    apply_graduation_tte_overlay,
)
from trencher.data.world_model.targets.coverage import (
    SourceCoverage,
    SourceCoverageRegistry,
    TimeInterval,
)
from trencher.data.world_model.targets.coverage import (
    earliest_continuous_censor_endpoint as cov_ep,
)
from trencher.data.world_model.targets.status import (
    OUTAGE_END_UNIX,
    OUTAGE_START_UNIX,
    STATUS_MODALITY_UNAVAILABLE,
    STATUS_RIGHT_CENSORED,
    STATUS_SOURCE_BOUNDARY_UNAVAILABLE,
    STATUS_VALID_POSITIVE,
)
from trencher.data.world_model.training.global_batch import (
    GlobalContextReader,
    last_completed_utc_day,
)
from trencher.data.world_model.training.schemas import (
    continuous_channel_names,
    modality_schema_manifest,
)
from trencher.data.world_model.training.transforms import (
    fit_sequence_transforms_from_unique_rows,
)
from trencher.data.world_model.training.vocabs import (
    EDGE_TYPE_VOCAB,
    edge_type_id,
    event_kind_id,
)


def _pump_cov(*, start: float, end: float, gap: bool = True) -> SourceCoverageRegistry:
    intervals = [TimeInterval(start, end)]
    gaps = []
    if gap and start < OUTAGE_START_UNIX < end:
        intervals = [
            TimeInterval(start, OUTAGE_START_UNIX),
            TimeInterval(OUTAGE_END_UNIX, end),
        ]
        gaps = [TimeInterval(OUTAGE_START_UNIX, OUTAGE_END_UNIX)]
    return SourceCoverageRegistry(
        pump_trades=SourceCoverage(
            source_id="pumpfun_trades",
            intervals=intervals,
            known_gaps=gaps,
            supports_absence_negatives=True,
            coverage_confidence="high",
        ),
        migrations=SourceCoverage(
            source_id="migrations",
            supports_absence_negatives=False,
            coverage_confidence="insufficient",
        ),
        postgrad_snapshots=SourceCoverage(source_id="postgrad"),
        ownership=SourceCoverage(source_id="ownership"),
    )


def test_last_completed_utc_day_never_current() -> None:
    # Mid-day
    T = datetime(2026, 6, 15, 12, 0, tzinfo=UTC).timestamp()
    assert last_completed_utc_day(T) == "2026-06-14"
    # Exactly midnight of D → still D incomplete → D-1
    T0 = datetime(2026, 6, 15, 0, 0, tzinfo=UTC).timestamp()
    assert last_completed_utc_day(T0) == "2026-06-14"


def test_same_day_future_global_mutation_bit_identical(tmp_path: Path) -> None:
    world = tmp_path / "world"
    (world / "pit" / "global").mkdir(parents=True)
    (world / "global_context").mkdir(parents=True)
    # Empty timelines
    pq.write_table(
        pa.table({"event_time_unix": pa.array([], type=pa.float64())}),
        world / "pit" / "global" / "launches_timeline.parquet",
    )
    pq.write_table(
        pa.table({"event_time_unix": pa.array([], type=pa.float64())}),
        world / "pit" / "global" / "migrations_timeline.parquet",
    )
    # Second buckets around T
    T = datetime(2026, 6, 15, 12, 0, tzinfo=UTC).timestamp()
    last_sec = int(T) - 1
    secs = list(range(last_sec - 10, last_sec + 1))
    pq.write_table(
        pa.table(
            {
                "second_unix": secs,
                "trades": [1.0] * len(secs),
                "buys": [1.0] * len(secs),
                "sells": [0.0] * len(secs),
                "buy_sol": [1.0] * len(secs),
                "sell_sol": [0.0] * len(secs),
                "unique_mints": [1.0] * len(secs),
                "unique_wallets": [1.0] * len(secs),
            }
        ),
        world / "pit" / "global" / "trade_second_buckets.parquet",
    )
    day_d = datetime.fromtimestamp(T, tz=UTC).strftime("%Y-%m-%d")
    day_prev = last_completed_utc_day(T)
    net_path = world / "global_context" / "daily_pump_network_context.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "day": day_prev,
                    "launches": 10.0,
                    "graduations": 1.0,
                    "trades": 100.0,
                    "buy_volume_sol": 50.0,
                    "graduation_rate": 0.1,
                },
                {
                    "day": day_d,
                    "launches": 999.0,
                    "graduations": 99.0,
                    "trades": 9999.0,
                    "buy_volume_sol": 9999.0,
                    "graduation_rate": 0.99,
                },
            ]
        ),
        net_path,
    )
    reader = GlobalContextReader(world, last_n_seconds=8)
    a = reader.load_for_t(T)
    # Mutate same-day D record
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "day": day_prev,
                    "launches": 10.0,
                    "graduations": 1.0,
                    "trades": 100.0,
                    "buy_volume_sol": 50.0,
                    "graduation_rate": 0.1,
                },
                {
                    "day": day_d,
                    "launches": 1.0,
                    "graduations": 0.0,
                    "trades": 1.0,
                    "buy_volume_sol": 1.0,
                    "graduation_rate": 0.0,
                },
            ]
        ),
        net_path,
    )
    # Clear caches
    from trencher.data.world_model.training import global_batch as gb

    gb._load_daily_network.cache_clear()
    reader2 = GlobalContextReader(world, last_n_seconds=8)
    b = reader2.load_for_t(T)
    assert a["global_scalars"]["daily_network_launches"] == 10.0
    assert b["global_scalars"]["daily_network_launches"] == 10.0
    assert np.array_equal(a["global"], b["global"])
    assert a["global_scalars"] == b["global_scalars"]
    assert "pit_last_completed_second" not in a["global_scalars"]
    assert "pit_last_completed_second" in (a.get("debug_meta") or {})


def test_no_raw_unix_in_predictive_schema() -> None:
    man = modality_schema_manifest()
    blob = str(man).lower()
    assert "pit_last_completed_second" not in blob
    for ch in continuous_channel_names():
        assert "unix" not in ch


def test_deterministic_categoricals_across_hashseed() -> None:
    # Same-process baseline
    assert edge_type_id("creator_token") == float(EDGE_TYPE_VOCAB["creator_token"])
    assert edge_type_id("wallet_token_buy") == 2.0
    assert edge_type_id("totally_unknown_xyz") == 0.0
    assert event_kind_id("rpc_pre_post_token_balance") == 1.0
    code = (
        "from trencher.data.world_model.training.vocabs import edge_type_id, event_kind_id\n"
        "print(edge_type_id('creator_token'), edge_type_id('native_sol_funding'), "
        "edge_type_id('nope'), event_kind_id('bq_spl_transfer'))\n"
    )
    env1 = {**os.environ, "PYTHONHASHSEED": "0"}
    env2 = {**os.environ, "PYTHONHASHSEED": "999"}
    out1 = subprocess.check_output([sys.executable, "-c", code], env=env1, text=True)
    out2 = subprocess.check_output([sys.executable, "-c", code], env=env2, text=True)
    assert out1 == out2
    assert out1.strip() == "1.0 5.0 0.0 2.0"


def test_post_q70_train_mint_events_do_not_affect_transforms(tmp_path: Path) -> None:
    q70 = 1000.0
    rows = [
        {"mint": "m1", "event_time_unix": 100.0, "price": 1.0},
        {"mint": "m1", "event_time_unix": 200.0, "price": 2.0},
        {"mint": "m1", "event_time_unix": 1500.0, "price": 999.0},  # post-q70
    ]
    path = tmp_path / "seq.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    fit1 = fit_sequence_transforms_from_unique_rows(
        sequence_parts=[path],
        train_mints={"m1"},
        numeric_columns=["price"],
        max_event_time=q70,
    )
    # Mutate post-q70
    rows[2]["price"] = -1e9
    pq.write_table(pa.Table.from_pylist(rows), path)
    fit2 = fit_sequence_transforms_from_unique_rows(
        sequence_parts=[path],
        train_mints={"m1"},
        numeric_columns=["price"],
        max_event_time=q70,
    )
    assert abs(fit1["columns"]["price"]["mean"] - fit2["columns"]["price"]["mean"]) < 1e-12
    assert abs(fit1["columns"]["price"]["mean"] - 1.5) < 1e-9


def test_transform_coverage_lists_exemptions() -> None:
    man = modality_schema_manifest()
    ex = man["transform_exemptions"]
    assert "rel_dt" in ex["relative_time_raw"]
    assert "decimals" in ex["bounded_ordinals"]
    assert "categorical_ids" in ex["never_normalize"]
    # Every continuous consumed channel is either transformable or exempt
    cont = set(continuous_channel_names())
    assert "rel_dt" not in cont
    assert "decimals" not in cont


def test_explosion_modality_unavailable_at_risk_zero() -> None:
    ov = apply_explosion_tte_overlay(
        observation_time_t=1782000000.0,
        split_end=1782869641.0,
        time_to_explosion_sec=None,
        explosion_event_observed=False,
        modality_unavailable=True,
    )
    assert ov["explosion_tte_at_risk"] is False
    assert ov["explosion_tte_duration_sec"] is None
    assert ov["explosion_tte_effective_status"] == STATUS_MODALITY_UNAVAILABLE


def test_outage_truncates_explosion_censor() -> None:
    cov = _pump_cov(start=1780650748.0, end=1784041375.0, gap=True)
    T = OUTAGE_START_UNIX - 100.0
    split_end = OUTAGE_END_UNIX + 10_000.0
    ov = apply_explosion_tte_overlay(
        observation_time_t=T,
        split_end=split_end,
        time_to_explosion_sec=None,
        explosion_event_observed=False,
        explosion_right_censored=True,
        coverage=cov,
    )
    assert ov["explosion_tte_at_risk"] is True
    assert ov["explosion_tte_censored"] is True
    assert abs(ov["explosion_tte_duration_sec"] - 100.0) < 1e-6
    assert ov["explosion_tte_administrative_censor_time"] == OUTAGE_START_UNIX


def test_unresolved_graduation_no_fabricated_split_duration() -> None:
    cov = _pump_cov(start=1780650748.0, end=1784041375.0)
    T = 1782000000.0
    split_end = 1782869641.0
    ov = apply_graduation_tte_overlay(
        observation_time_t=T,
        split_end=split_end,
        time_to_graduation_sec=None,
        time_to_graduation_status=STATUS_RIGHT_CENSORED,
        coverage=cov,
    )
    assert ov["graduation_tte_at_risk"] is False
    assert ov["graduation_tte_duration_sec"] is None
    assert ov["graduation_tte_effective_status"] == STATUS_SOURCE_BOUNDARY_UNAVAILABLE
    # Must not equal fabricated split_end - T
    assert ov["graduation_tte_duration_sec"] != (split_end - T)


def test_graduation_observed_before_boundary_still_event() -> None:
    ov = apply_graduation_tte_overlay(
        observation_time_t=1000.0,
        split_end=5000.0,
        time_to_graduation_sec=100.0,
        time_to_graduation_status=STATUS_VALID_POSITIVE,
    )
    assert ov["graduation_tte_event_observed"] is True
    assert ov["graduation_tte_at_risk"] is True


def test_already_started_left_truncated() -> None:
    ov = apply_explosion_tte_overlay(
        observation_time_t=1000.0,
        split_end=2000.0,
        time_to_explosion_sec=None,
        explosion_event_observed=True,
        explosion_already_started_at_t=True,
    )
    assert ov["explosion_tte_at_risk"] is False
    assert ov["explosion_tte_effective_status"] == STATUS_LEFT_TRUNCATED


def test_earliest_censor_endpoint_helper() -> None:
    cov = SourceCoverage(
        source_id="pump",
        intervals=[
            TimeInterval(100.0, OUTAGE_START_UNIX),
            TimeInterval(OUTAGE_END_UNIX, 1784041375.0),
        ],
        known_gaps=[TimeInterval(OUTAGE_START_UNIX, OUTAGE_END_UNIX)],
        supports_absence_negatives=True,
    )
    ep = cov_ep(cov, OUTAGE_START_UNIX - 50.0, split_end=1784041375.0)
    assert ep == OUTAGE_START_UNIX
