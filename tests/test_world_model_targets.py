"""Label-only target engine v3 tests — survival confirmation, coverage, isolation."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.feasibility.concepts import SYSTEM_PROGRAM_WALLET
from trencher.data.world_model.pit import model_state_digest
from trencher.data.world_model.targets.build import (
    assert_input_modules_do_not_import_targets,
    build_target_dataset_v3,
)
from trencher.data.world_model.targets.compute import (
    compute_activity_survival,
    compute_explosion_obs,
    compute_graduation,
    compute_persistence,
    compute_return_barrier_mfe,
    reference_price_at_t,
)
from trencher.data.world_model.targets.coverage import (
    SourceCoverage,
    SourceCoverageRegistry,
    TimeInterval,
)
from trencher.data.world_model.targets.explosion import (
    PRIMARY_DEFINITION_ID,
    detect_explosions_for_mint,
)
from trencher.data.world_model.targets.status import (
    BARRIER_NEITHER,
    BARRIER_UPPER_FIRST,
    OUTAGE_END_UNIX,
    OUTAGE_START_UNIX,
    STATUS_KNOWN_SOURCE_OUTAGE,
    STATUS_REFERENCE_PRICE_UNAVAILABLE,
    STATUS_RIGHT_CENSORED,
    STATUS_SOURCE_BOUNDARY_UNAVAILABLE,
    STATUS_VALID_NEGATIVE,
    STATUS_VALID_POSITIVE,
    SURVIVAL_HORIZONS_SEC,
    SURVIVAL_NEGATIVE_CONFIRMATION_SEC,
)


def _cov(
    *,
    trade_start: float = 0.0,
    trade_end: float = 1e12,
    mig_start: float | None = None,
    mig_end: float | None = None,
    with_outage_gap: bool = False,
    mig_supports_absence: bool = True,
) -> SourceCoverageRegistry:
    gaps = (
        [TimeInterval(OUTAGE_START_UNIX, OUTAGE_END_UNIX)] if with_outage_gap else []
    )
    intervals = [TimeInterval(trade_start, trade_end)]
    if with_outage_gap and trade_start < OUTAGE_START_UNIX < trade_end:
        intervals = [
            TimeInterval(trade_start, OUTAGE_START_UNIX),
            TimeInterval(OUTAGE_END_UNIX, trade_end),
        ]
    ms = mig_start if mig_start is not None else trade_start
    me = mig_end if mig_end is not None else trade_end
    mig_iv = [TimeInterval(ms, me)] if mig_supports_absence else []
    if mig_supports_absence and with_outage_gap and ms < OUTAGE_START_UNIX < me:
        mig_iv = [
            TimeInterval(ms, OUTAGE_START_UNIX),
            TimeInterval(OUTAGE_END_UNIX, me),
        ]
    return SourceCoverageRegistry(
        pump_trades=SourceCoverage(
            "pump",
            intervals=intervals,
            known_gaps=gaps,
            supports_absence_negatives=True,
            coverage_confidence="high",
            coverage_basis="test_fixture",
        ),
        migrations=SourceCoverage(
            "mig",
            intervals=mig_iv,
            known_gaps=gaps if mig_supports_absence else [],
            supports_absence_negatives=mig_supports_absence,
            coverage_confidence="high" if mig_supports_absence else "insufficient",
            coverage_basis="test_fixture",
        ),
        postgrad_snapshots=SourceCoverage(
            "pg",
            intervals=[],
            known_gaps=[],
            supports_absence_negatives=False,
            coverage_confidence="insufficient",
        ),
        ownership=SourceCoverage(
            "own",
            intervals=[],
            known_gaps=[],
            supports_absence_negatives=False,
        ),
    )


def _arr_from_trades(
    times: list[float],
    prices: list[float],
    *,
    is_buy: list[bool] | None = None,
    wallets: list[str] | None = None,
    sol: list[float] | None = None,
    curve: list[float] | None = None,
) -> dict[str, np.ndarray]:
    n = len(times)
    is_buy = is_buy or [True] * n
    wallets = wallets or [f"w{i}" for i in range(n)]
    sol = sol or [1.0] * n
    curve = curve or [float(i) for i in range(n)]
    return {
        "t": np.asarray(times, dtype=np.float64),
        "price": np.asarray(prices, dtype=np.float64),
        "regime": np.zeros(n, dtype=np.int8),
        "is_buy": np.asarray([1 if b else 0 for b in is_buy], dtype=np.int8),
        "sol": np.asarray(sol, dtype=np.float64),
        "token": np.asarray([1e6] * n, dtype=np.float64),
        "wallet": np.asarray(wallets, dtype=object),
        "v_sol": np.asarray([30.0 + i for i in range(n)], dtype=np.float64),
        "v_tok": np.asarray([1e9 - i * 1e6 for i in range(n)], dtype=np.float64),
        "curve": np.asarray(curve, dtype=np.float64),
        "price_source": np.asarray(["trade_price_sol"] * n, dtype=object),
        "path_idx": np.arange(n, dtype=np.int64),
        "usable_price": np.ones(n, dtype=bool),
    }


def test_input_modules_do_not_import_targets() -> None:
    proof = assert_input_modules_do_not_import_targets()
    assert proof["ok"], proof["violations"]


def test_survival_trade_at_t5_coverage_through_h_plus_30m_is_negative() -> None:
    """Trade at T+5s only; coverage through H+30m → negative at H."""
    T = 1_000_000.0
    H = 30
    times = [T - 10, T + 5]
    arr = _arr_from_trades(times, [1.0, 1.0])
    cov = _cov(
        trade_start=T - 100,
        trade_end=T + H + SURVIVAL_NEGATIVE_CONFIRMATION_SEC + 60,
    )
    out = compute_activity_survival(arr, T=T, coverage=cov)
    assert out["surv_h30_status"] == STATUS_VALID_NEGATIVE
    assert out["surv_h30_active"] is False
    # Longer horizons need longer confirmation coverage; with only H=30 covered → censored.
    assert out["surv_h86400_status"] != STATUS_VALID_NEGATIVE


def test_survival_source_ends_at_h_plus_5s_censored() -> None:
    T = 1_000_000.0
    H = 30
    arr = _arr_from_trades([T - 1], [1.0])
    cov = _cov(trade_start=T - 100, trade_end=T + H + 5)
    out = compute_activity_survival(arr, T=T, coverage=cov)
    assert out["surv_h30_status"] in (
        STATUS_RIGHT_CENSORED,
        STATUS_SOURCE_BOUNDARY_UNAVAILABLE,
    )
    assert out["surv_h30_status"] != STATUS_VALID_NEGATIVE
    assert out["surv_h30_active"] is None


def test_survival_source_ends_exactly_at_h_censored() -> None:
    T = 1_000_000.0
    H = 30
    arr = _arr_from_trades([T - 1], [1.0])
    cov = _cov(trade_start=T - 100, trade_end=T + H)
    out = compute_activity_survival(arr, T=T, coverage=cov)
    assert out["surv_h30_status"] in (
        STATUS_RIGHT_CENSORED,
        STATUS_SOURCE_BOUNDARY_UNAVAILABLE,
    )
    assert out["surv_h30_status"] != STATUS_VALID_NEGATIVE


def test_survival_trade_at_h_plus_1h_positive() -> None:
    T = 1_000_000.0
    H = 30
    arr = _arr_from_trades([T - 1, T + H + 3600], [1.0, 1.1])
    cov = _cov(trade_start=T - 100, trade_end=T + 200_000)
    out = compute_activity_survival(arr, T=T, coverage=cov)
    assert out["surv_h30_status"] == STATUS_VALID_POSITIVE
    assert out["surv_h30_active"] is True


def test_survival_outage_before_h_confirmation_after_can_be_negative() -> None:
    """Outage entirely before H; [H, H+confirmation] fully after outage → negative."""
    T = OUTAGE_END_UNIX - 100
    H = 300  # T+H = OUTAGE_END + 200
    arr = _arr_from_trades([T - 1], [1.0])
    cov = _cov(
        trade_start=OUTAGE_START_UNIX - 10_000,
        trade_end=OUTAGE_END_UNIX + 100_000,
        with_outage_gap=True,
    )
    out = compute_activity_survival(arr, T=T, coverage=cov)
    assert out[f"surv_h{H}_status"] == STATUS_VALID_NEGATIVE
    assert out[f"surv_h{H}_active"] is False


def test_survival_outage_intersects_confirmation_is_outage() -> None:
    T = OUTAGE_START_UNIX - 10
    arr = _arr_from_trades([T - 1], [1.0])
    cov = _cov(
        trade_start=OUTAGE_START_UNIX - 1000,
        trade_end=OUTAGE_END_UNIX + 100_000,
        with_outage_gap=True,
    )
    out = compute_activity_survival(arr, T=T, coverage=cov)
    # H=60 → confirmation starts inside outage day.
    assert out["surv_h60_status"] == STATUS_KNOWN_SOURCE_OUTAGE
    assert out["surv_h60_active"] is None


def test_survival_positives_monotonic_nonincreasing_by_h() -> None:
    T = 1_000_000.0
    # Activity at T+2h only → short H positive, long H negative.
    arr = _arr_from_trades([T - 1, T + 7200], [1.0, 1.1])
    cov = _cov(trade_start=T - 100, trade_end=T + 200_000)
    out = compute_activity_survival(arr, T=T, coverage=cov)
    flags = [
        1 if out[f"surv_h{h}_status"] == STATUS_VALID_POSITIVE else 0
        for h in SURVIVAL_HORIZONS_SEC
    ]
    assert flags == sorted(flags, reverse=True)
    assert out["surv_h30_status"] == STATUS_VALID_POSITIVE
    assert out["surv_h3600_status"] == STATUS_VALID_POSITIVE
    assert out["surv_h21600_status"] == STATUS_VALID_NEGATIVE


def test_survival_source_ends_before_h_censored() -> None:
    T = 1_000_000.0
    arr = _arr_from_trades([T - 1], [1.0])
    cov = _cov(trade_start=T - 100, trade_end=T + 10)  # ends before 30s
    out = compute_activity_survival(arr, T=T, coverage=cov)
    assert out["surv_h30_status"] in (
        STATUS_RIGHT_CENSORED,
        STATUS_SOURCE_BOUNDARY_UNAVAILABLE,
    )
    assert out["surv_h30_active"] is None


def test_persistence_null_when_uncovered_not_zero() -> None:
    T = 1_000_000.0
    arr = _arr_from_trades([T - 1], [1.0])
    cov = _cov(trade_start=T - 100, trade_end=T + 5)
    out = compute_persistence(arr, T=T, coverage=cov)
    assert out["act_h30_status"] in (
        STATUS_RIGHT_CENSORED,
        STATUS_SOURCE_BOUNDARY_UNAVAILABLE,
    )
    assert out["act_h30_trade_count"] is None
    assert out["act_h30_sol_buy_flow"] is None


def test_persistence_zero_valid_when_covered() -> None:
    T = 1_000_000.0
    arr = _arr_from_trades([T - 1], [1.0])
    cov = _cov(trade_start=T - 100, trade_end=T + 10_000)
    out = compute_persistence(arr, T=T, coverage=cov)
    assert out["act_h30_status"] == STATUS_VALID_NEGATIVE
    assert out["act_h30_trade_count"] == 0


def test_graduation_independent_of_token_trading() -> None:
    """No migration + proven migration coverage → valid negative even if no trades."""
    T = 1_000_000.0
    cov = _cov(
        trade_start=T - 10,
        trade_end=T + 10,
        mig_start=T - 1000,
        mig_end=T + 100_000,
        mig_supports_absence=True,
    )
    out = compute_graduation(T=T, migration_unix=None, coverage=cov)
    assert out["grad_h3600_status"] == STATUS_VALID_NEGATIVE
    assert out["grad_h3600_graduated"] is False


def test_graduation_censored_when_mig_coverage_unproven() -> None:
    """Conservative coverage: absence negatives censored without proven collection."""
    T = 1_000_000.0
    cov = _cov(mig_supports_absence=False)
    out = compute_graduation(T=T, migration_unix=None, coverage=cov)
    assert out["grad_h3600_status"] == STATUS_RIGHT_CENSORED
    assert out["grad_h3600_graduated"] is None


def test_graduation_later_migration_proves_shorter_horizon_negative() -> None:
    T = 1_000_000.0
    cov = _cov(mig_supports_absence=False)
    mig = T + 7200
    out = compute_graduation(T=T, migration_unix=mig, coverage=cov)
    assert out["grad_h3600_status"] == STATUS_VALID_NEGATIVE
    assert out["grad_h3600_graduated"] is False
    assert out["grad_h21600_status"] == STATUS_VALID_POSITIVE
    assert out["grad_h21600_graduated"] is True


def test_graduation_censored_when_mig_source_short() -> None:
    T = 1_000_000.0
    cov = _cov(
        trade_start=T - 1000,
        trade_end=T + 100_000,
        mig_start=T - 10,
        mig_end=T + 10,
        mig_supports_absence=True,
    )
    out = compute_graduation(T=T, migration_unix=None, coverage=cov)
    assert out["grad_h3600_status"] in (
        STATUS_RIGHT_CENSORED,
        STATUS_SOURCE_BOUNDARY_UNAVAILABLE,
    )
    assert out["grad_h3600_graduated"] is None


def test_future_events_strictly_after_t() -> None:
    t0 = 1_000_000.0
    times = [t0 - 10, t0, t0 + 1, t0 + 30]
    prices = [1.0, 1.0, 1.1, 1.2]
    arr = _arr_from_trades(times, prices)
    ref = reference_price_at_t(arr, t0)
    assert ref["price_target_available"]
    assert ref["reference_price_time"] == t0
    out = compute_return_barrier_mfe(arr, T=t0, ref=ref, last_path_t=times[-1])
    assert out["h30_future_mark_time"] > t0
    assert out["h30_future_mark_time"] == t0 + 30


def test_barrier_neither_not_downside() -> None:
    t0 = 1_000_000.0
    times = [t0 - 1, t0 + 10, t0 + 20, t0 + 30]
    prices = [1.0, 1.02, 1.03, 1.04]
    arr = _arr_from_trades(times, prices)
    ref = reference_price_at_t(arr, t0)
    out = compute_return_barrier_mfe(arr, T=t0, ref=ref, last_path_t=times[-1])
    assert out["h30_barrier_u10_l10_outcome"] == BARRIER_NEITHER


def test_barrier_upper_implies_mfe() -> None:
    t0 = 1_000_000.0
    times = [t0 - 1] + [t0 + i for i in range(1, 40)]
    prices = [1.0] + [1.0 + 0.01 * i for i in range(1, 40)]
    arr = _arr_from_trades(times, prices)
    ref = reference_price_at_t(arr, t0)
    out = compute_return_barrier_mfe(arr, T=t0, ref=ref, last_path_t=times[-1])
    assert out["h60_barrier_u10_l10_outcome"] == BARRIER_UPPER_FIRST
    assert out["h60_mfe"] + 1e-12 >= 0.10
    assert out["h60_barrier_u10_l10_consistency_ok"] is True


def test_explosion_detector_finds_sustained_transition() -> None:
    t0 = 1_780_000_000.0
    times: list[float] = []
    prices: list[float] = []
    wallets: list[str] = []
    is_buy: list[bool] = []
    sol: list[float] = []
    curve: list[float] = []
    for i in range(20):
        times.append(t0 + i * 10)
        prices.append(1.0)
        wallets.append(f"b{i % 3}")
        is_buy.append(True)
        sol.append(0.1)
        curve.append(1.0)
    onset = t0 + 200
    for i in range(40):
        times.append(onset + i * 1.5)
        prices.append(1.0 * (1.02**i))
        wallets.append(f"e{i}")
        is_buy.append(True)
        sol.append(2.0 if i < 20 else 1.5)
        curve.append(1.0 + i * 2.0)
    arr = _arr_from_trades(
        times, prices, is_buy=is_buy, wallets=wallets, sol=sol, curve=curve
    )
    events = detect_explosions_for_mint("m_expl", arr)
    primary = next(e for e in events if e["definition_id"] == PRIMARY_DEFINITION_ID)
    assert primary["explosion_exists"] is True
    assert abs(primary["onset_unix"] - onset) <= 20
    assert primary["explosion_magnitude"] >= 0.25


def test_explosion_imminence_censoring() -> None:
    primary = {
        "explosion_exists": True,
        "onset_unix": 1000.0,
        "peak_unix": 1100.0,
        "end_unix": 1200.0,
        "censoring_status": "observed",
        "peak_price": 2.0,
        "explosion_magnitude": 1.0,
        "onset_to_peak_sec": 100.0,
        "max_drawdown_after_peak": 0.3,
    }
    ref = {"price_target_available": True, "reference_price": 1.0}
    out = compute_explosion_obs(
        T=970.0,
        primary=primary,
        all_defs=[primary],
        coverage=_cov(),
        ref=ref,
    )
    assert out["explosion_within_30s"] is True
    assert out["explosion_within_10s"] is False
    assert out["time_to_explosion_sec"] == 30.0
    assert out["explosion_phase_target_type"] == "auxiliary_pseudolabel"


def _write_mini_inputs(
    world: Path, pump: Path, *, mint: str, t0: datetime, obs_offset_sec: int
) -> float:
    for rel in (
        "universe",
        "pit",
        "pit/global",
        "execution",
        "observations/observation_dataset_v1",
    ):
        (world / rel).mkdir(parents=True, exist_ok=True)
    (pump / "trades").mkdir(parents=True, exist_ok=True)

    times = [t0 + timedelta(seconds=s) for s in range(0, 50, 5)]
    times += [t0 + timedelta(seconds=s) for s in range(60, 120)]
    prices = [1.0] * 10 + [1.0 * (1.03**i) for i in range(60)]
    n = len(times)
    trades = pa.table(
        {
            "mint": [mint] * n,
            "user_wallet": [f"w{i}" for i in range(n)],
            "event_time": times,
            "is_buy": [True] * n,
            "sol_amount": [1.0] * n,
            "token_amount": [1e6] * n,
            "tx_signature": [f"sig{i}" for i in range(n)],
            "price_sol": prices,
            "v_sol_bonding_curve": [30.0 + i * 0.1 for i in range(n)],
            "v_tokens_bonding_curve": [1e9] * n,
            "curve_pct_depleted": [float(i) for i in range(n)],
        }
    )
    pq.write_table(trades, world / "pit" / "atomic_trades_subset.parquet")
    obs_t = (t0 + timedelta(seconds=obs_offset_sec)).timestamp()
    secs = list(range(int(obs_t) - 1000, int(obs_t) + 200_000))
    pq.write_table(
        pa.table({"second_unix": secs}),
        world / "pit" / "global" / "trade_second_buckets.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": pa.array([], pa.string()),
                "event_time_unix": pa.array([], pa.float64()),
            }
        ),
        world / "pit" / "global" / "migrations_timeline.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": pa.array([], pa.string()),
                "migrated_at": pa.array([], pa.timestamp("us", tz="UTC")),
            }
        ),
        pump / "migrations.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": pa.array([], pa.string()),
                "snapshot_time": pa.array([], pa.timestamp("us", tz="UTC")),
                "seconds_since_graduation": pa.array([], pa.float64()),
                "price_native": pa.array([], pa.float64()),
                "price_usd": pa.array([], pa.float64()),
                "incomplete_data": pa.array([], pa.bool_()),
                "pair_address": pa.array([], pa.string()),
                "liquidity_usd": pa.array([], pa.float64()),
            }
        ),
        pump / "postgard_snapshots.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": [mint],
                "observation_id": ["obs1"],
                "observation_time_t": [obs_t],
                "token_age_seconds": [float(obs_offset_sec)],
                "observation_trigger": ["fixed_age"],
                "trigger_value": [f"{obs_offset_sec}s"],
                "trigger_numeric": [obs_offset_sec],
                "source_freeze_version": ["world_model_freeze_v1"],
                "pit_engine_version": ["pit_v4"],
                "observation_dataset_version": ["observation_dataset_v1"],
                "launch_day": [t0.strftime("%Y-%m-%d")],
                "launch_regime": ["x"],
                "model_state_digest": ["deadbeef"],
                "ownership_level_at_t": ["none"],
                "holder_state_reconstructable_at_t": [False],
                "mask_event_history": [True],
                "mask_wallet_token_dynamic": [True],
                "mask_microstructure": [False],
                "mask_ownership_any": [False],
                "mask_global_market": [True],
            }
        ),
        world / "observations" / "observation_dataset_v1" / "observation_index_v1.parquet",
    )
    return obs_t


def test_mutate_future_changes_labels_not_inputs(tmp_path: Path) -> None:
    world = tmp_path / "world_model"
    pump = tmp_path / "pumpfun_memecoin_corpus"
    t0 = datetime(2026, 6, 10, 12, tzinfo=UTC)
    mint = "m_iso"
    obs_t = _write_mini_inputs(world, pump, mint=mint, t0=t0, obs_offset_sec=45)
    trades = pq.read_table(world / "pit" / "atomic_trades_subset.parquet")
    times = trades.column("event_time").to_pylist()
    prices = trades.column("price_sol").to_pylist()
    n = len(times)

    manifest = tmp_path / "manifests"
    audit1 = build_target_dataset_v3(
        world_root=world, manifest_dir=manifest, pump_root=pump
    )
    assert audit1["observations_processed"] == 1
    targets1 = pq.read_table(
        world / "targets" / "target_dataset_v3" / "observation_targets_v3.parquet"
    ).to_pylist()[0]
    ret1 = targets1.get("h60_simple_return")
    expl1 = targets1.get("explosion_within_60s")
    pre_t_prices = [
        p
        for ts, p in zip(times, prices, strict=True)
        if (ts.timestamp() if hasattr(ts, "timestamp") else float(ts)) <= obs_t
    ]
    digest1 = model_state_digest({"pre_t_prices": pre_t_prices, "mint": mint, "T": obs_t})

    prices2 = list(prices)
    v_sol2 = [30.0 + i * 0.1 for i in range(n)]
    v_tok2 = [1e9] * n
    for i, ts in enumerate(times):
        ts_u = ts.timestamp() if hasattr(ts, "timestamp") else float(ts)
        if ts_u > obs_t:
            prices2[i] = 1.0
            v_sol2[i] = 30.0
            v_tok2[i] = 1e9
    trades2 = trades.set_column(
        trades.schema.get_field_index("price_sol"), "price_sol", pa.array(prices2)
    )
    trades2 = trades2.set_column(
        trades2.schema.get_field_index("v_sol_bonding_curve"),
        "v_sol_bonding_curve",
        pa.array(v_sol2),
    )
    trades2 = trades2.set_column(
        trades2.schema.get_field_index("v_tokens_bonding_curve"),
        "v_tokens_bonding_curve",
        pa.array(v_tok2),
    )
    pq.write_table(trades2, world / "pit" / "atomic_trades_subset.parquet")
    build_target_dataset_v3(world_root=world, manifest_dir=manifest, pump_root=pump)
    targets2 = pq.read_table(
        world / "targets" / "target_dataset_v3" / "observation_targets_v3.parquet"
    ).to_pylist()[0]
    digest2 = model_state_digest({"pre_t_prices": pre_t_prices, "mint": mint, "T": obs_t})
    assert digest1 == digest2
    assert ret1 != targets2.get("h60_simple_return") or expl1 != targets2.get(
        "explosion_within_60s"
    )
    assert SYSTEM_PROGRAM_WALLET not in str(targets1)


def test_build_target_dataset_mini(tmp_path: Path) -> None:
    world = tmp_path / "world_model"
    pump = tmp_path / "pump"
    for rel in ("universe", "pit", "pit/global", "observations/observation_dataset_v1"):
        (world / rel).mkdir(parents=True)
    (pump / "trades").mkdir(parents=True)

    t0 = datetime(2026, 6, 8, 12, tzinfo=UTC)
    mint = "m1"
    rows_t = [t0 + timedelta(seconds=i * 5) for i in range(40)]
    rows_p = [1.0 * (1.01**i) for i in range(40)]
    pq.write_table(
        pa.table(
            {
                "mint": [mint] * 40,
                "user_wallet": [f"w{i % 5}" for i in range(40)],
                "event_time": rows_t,
                "is_buy": [i % 3 != 0 for i in range(40)],
                "sol_amount": [0.5] * 40,
                "token_amount": [1e6] * 40,
                "tx_signature": [f"s{i}" for i in range(40)],
                "price_sol": rows_p,
                "v_sol_bonding_curve": [30.0 + i for i in range(40)],
                "v_tokens_bonding_curve": [1e9] * 40,
                "curve_pct_depleted": [float(i) for i in range(40)],
            }
        ),
        world / "pit" / "atomic_trades_subset.parquet",
    )
    obs_t = (t0 + timedelta(seconds=50)).timestamp()
    pq.write_table(
        pa.table(
            {
                "second_unix": list(
                    range(int(obs_t) - 1000, int(obs_t) + 200_000)
                )
            }
        ),
        world / "pit" / "global" / "trade_second_buckets.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": pa.array([], pa.string()),
                "event_time_unix": pa.array([], pa.float64()),
            }
        ),
        world / "pit" / "global" / "migrations_timeline.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": pa.array([], pa.string()),
                "migrated_at": pa.array([], pa.timestamp("us", tz="UTC")),
            }
        ),
        pump / "migrations.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": pa.array([], pa.string()),
                "snapshot_time": pa.array([], pa.timestamp("us", tz="UTC")),
                "seconds_since_graduation": pa.array([], pa.float64()),
                "price_native": pa.array([], pa.float64()),
                "price_usd": pa.array([], pa.float64()),
                "incomplete_data": pa.array([], pa.bool_()),
                "pair_address": pa.array([], pa.string()),
                "liquidity_usd": pa.array([], pa.float64()),
            }
        ),
        pump / "postgard_snapshots.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": [mint],
                "observation_id": ["o1"],
                "observation_time_t": [obs_t],
                "token_age_seconds": [50.0],
                "observation_trigger": ["fixed_age"],
                "trigger_value": ["50s"],
                "trigger_numeric": [50],
                "source_freeze_version": ["world_model_freeze_v1"],
                "pit_engine_version": ["pit_v4"],
                "observation_dataset_version": ["observation_dataset_v1"],
                "launch_day": ["2026-06-08"],
                "launch_regime": ["early"],
                "model_state_digest": ["x"],
                "ownership_level_at_t": ["none"],
                "holder_state_reconstructable_at_t": [False],
                "mask_event_history": [True],
                "mask_wallet_token_dynamic": [True],
                "mask_microstructure": [False],
                "mask_ownership_any": [False],
                "mask_global_market": [True],
            }
        ),
        world / "observations" / "observation_dataset_v1" / "observation_index_v1.parquet",
    )
    manifest = tmp_path / "m"
    audit = build_target_dataset_v3(
        world_root=world, manifest_dir=manifest, pump_root=pump
    )
    assert audit["observations_processed"] == 1
    assert (manifest / "TARGET_CONTRACT_V3.json").exists()
    assert (manifest / "TARGET_DATASET_V3_AUDIT.json").exists()
    assert (manifest / "target_source_coverage_v2.json").exists()
    cov = (manifest / "target_source_coverage_v2.json").read_text()
    assert "coverage_basis" in cov
    assert "supports_absence_negatives" in cov
    assert audit.get("survival_positive_counts_monotonic_nonincreasing") is True
    tgt = pq.read_table(
        world / "targets" / "target_dataset_v3" / "observation_targets_v3.parquet"
    ).to_pylist()[0]
    assert tgt["price_target_available"] is True
    assert tgt["h30_status"] in (STATUS_VALID_POSITIVE, STATUS_VALID_NEGATIVE)
    assert math.isfinite(float(tgt["h30_simple_return"]))
    assert tgt.get("exec_1p0sol_status") in (
        "available",
        STATUS_RIGHT_CENSORED,
        STATUS_REFERENCE_PRICE_UNAVAILABLE,
    )
    if tgt.get("exec_1p0sol_status") == "available":
        assert tgt.get("exec_1p0sol_gross_return") is not None
    # Unproven migration coverage → absence graduation censored
    assert tgt.get("grad_h3600_status") == STATUS_RIGHT_CENSORED
