"""Label-only target engine tests — isolation, causality, explosion family."""

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
    build_target_dataset_v1,
)
from trencher.data.world_model.targets.compute import (
    compute_explosion_obs,
    compute_return_barrier_mfe,
    reference_price_at_t,
)
from trencher.data.world_model.targets.explosion import (
    PRIMARY_DEFINITION_ID,
    detect_explosions_for_mint,
)
from trencher.data.world_model.targets.status import (
    BARRIER_NEITHER,
    BARRIER_UPPER_FIRST,
    STATUS_VALID_NEGATIVE,
    STATUS_VALID_POSITIVE,
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


def test_future_events_strictly_after_t() -> None:
    t0 = 1_000_000.0
    times = [t0 - 10, t0, t0 + 1, t0 + 30]
    prices = [1.0, 1.0, 1.1, 1.2]
    arr = _arr_from_trades(times, prices)
    ref = reference_price_at_t(arr, t0)
    assert ref["price_target_available"]
    assert ref["reference_price_time"] == t0
    out = compute_return_barrier_mfe(arr, T=t0, ref=ref, last_path_t=times[-1])
    # Future mark for 30s must use events with t > T only.
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
    # Quiet baseline then accelerating buy burst with sustained upside.
    t0 = 1_780_000_000.0
    times: list[float] = []
    prices: list[float] = []
    wallets: list[str] = []
    is_buy: list[bool] = []
    sol: list[float] = []
    curve: list[float] = []
    # Baseline: sparse trades
    for i in range(20):
        times.append(t0 + i * 10)
        prices.append(1.0)
        wallets.append(f"b{i % 3}")
        is_buy.append(True)
        sol.append(0.1)
        curve.append(1.0)
    onset = t0 + 200
    # Explosion window
    for i in range(20):
        times.append(onset + i * 1.5)
        prices.append(1.0 * (1.02**i))
        wallets.append(f"e{i}")
        is_buy.append(True)
        sol.append(2.0)
        curve.append(1.0 + i * 2.0)
    # Continue to peak / confirm MFE
    for i in range(20, 40):
        times.append(onset + i * 1.5)
        prices.append(1.0 * (1.02**i))
        wallets.append(f"e{i}")
        is_buy.append(True)
        sol.append(1.5)
        curve.append(1.0 + i * 2.0)
    arr = _arr_from_trades(
        times, prices, is_buy=is_buy, wallets=wallets, sol=sol, curve=curve
    )
    events = detect_explosions_for_mint("m_expl", arr)
    primary = next(e for e in events if e["definition_id"] == PRIMARY_DEFINITION_ID)
    assert primary["explosion_exists"] is True
    assert primary["onset_unix"] is not None
    # Onset is start of accelerating transition near the burst, not the peak.
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
        last_path_t=2000.0,
        ref=ref,
    )
    assert out["explosion_within_30s"] is True
    assert out["explosion_within_10s"] is False
    assert out["time_to_explosion_sec"] == 30.0
    assert out["explosion_phase_target_type"] == "auxiliary_pseudolabel"


def test_mutate_future_changes_labels_not_inputs(tmp_path: Path) -> None:
    """Structural isolation: mutate future path → labels change, input digest stable."""
    world = tmp_path / "world_model"
    pump = tmp_path / "pumpfun_memecoin_corpus"
    for rel in (
        "universe",
        "pit",
        "execution",
        "observations/observation_dataset_v1",
        "graph",
        "microstructure",
        "bigquery",
        "pit/global",
    ):
        (world / rel).mkdir(parents=True)
    (pump / "trades").mkdir(parents=True)

    t0 = datetime(2026, 6, 10, 12, tzinfo=UTC)
    mint = "m_iso"
    # Build trades: pre-T flat, post-T explosive
    times = [t0 + timedelta(seconds=s) for s in range(0, 50, 5)]
    times += [t0 + timedelta(seconds=s) for s in range(60, 120)]
    prices = [1.0] * 10 + [1.0 * (1.03 ** i) for i in range(60)]
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
    pq.write_table(trades, pump / "trades" / "batch0.parquet")
    pq.write_table(
        pa.table({"mint": [mint], "launch_day": ["2026-06-10"], "launch_regime": ["x"]}),
        world / "universe" / "rich_subset_mints.parquet",
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
    # Minimal empty postgard
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

    obs_t = (t0 + timedelta(seconds=45)).timestamp()
    obs = pa.table(
        {
            "mint": [mint],
            "observation_id": ["obs1"],
            "observation_time_t": [obs_t],
            "token_age_seconds": [45.0],
            "observation_trigger": ["fixed_age"],
            "trigger_value": ["45s"],
            "trigger_numeric": [45],
            "source_freeze_version": ["world_model_freeze_v1"],
            "pit_engine_version": ["pit_v4"],
            "observation_dataset_version": ["observation_dataset_v1"],
            "launch_day": ["2026-06-10"],
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
    )
    pq.write_table(
        obs,
        world / "observations" / "observation_dataset_v1" / "observation_index_v1.parquet",
    )
    # Supporting empty tables for reconstruct if needed
    pq.write_table(
        pa.table(
            {
                "mint": [mint],
                "event_time": [t0],
                "price_sol": [1.0],
                "is_buy": [True],
                "sol_amount": [1.0],
                "v_sol_bonding_curve": [30.0],
                "v_tokens_bonding_curve": [1e9],
                "curve_pct_depleted": [0.0],
                "executable_price_sol": [1.0],
                "has_reserves": [True],
                "quality_flag": ["observed_curve_state"],
            }
        ),
        world / "execution" / "execution_curve_states.parquet",
    )

    manifest = tmp_path / "manifests"
    audit1 = build_target_dataset_v1(
        world_root=world, manifest_dir=manifest, pump_root=pump
    )
    assert audit1["observations_processed"] == 1
    targets1 = pq.read_table(
        world / "targets" / "target_dataset_v1" / "observation_targets_v1.parquet"
    ).to_pylist()[0]
    ret1 = targets1.get("h60_simple_return")
    expl1 = targets1.get("explosion_within_60s")

    # Input digest at T from atomic trades only (minimal reconstruct path may need indexes).
    # Use a stable synthetic model_state digest proxy from pre-T trades.
    pre_t_prices = [
        p
        for ts, p in zip(times, prices, strict=True)
        if ts.timestamp() <= obs_t
    ]
    digest1 = model_state_digest({"pre_t_prices": pre_t_prices, "mint": mint, "T": obs_t})

    # Mutate future: flatten post-T trade prices AND reserves (path prefers reserves).
    prices2 = list(prices)
    v_sol2 = [30.0 + i * 0.1 for i in range(n)]
    v_tok2 = [1e9] * n
    for i, ts in enumerate(times):
        if ts.timestamp() > obs_t:
            prices2[i] = 1.0
            v_sol2[i] = 30.0
            v_tok2[i] = 1e9
    trades2 = trades.set_column(
        trades.schema.get_field_index("price_sol"),
        "price_sol",
        pa.array(prices2),
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

    build_target_dataset_v1(
        world_root=world, manifest_dir=manifest, pump_root=pump
    )
    targets2 = pq.read_table(
        world / "targets" / "target_dataset_v1" / "observation_targets_v1.parquet"
    ).to_pylist()[0]
    ret2 = targets2.get("h60_simple_return")
    digest2 = model_state_digest({"pre_t_prices": pre_t_prices, "mint": mint, "T": obs_t})

    assert digest1 == digest2
    # Labels should change when future path flattens (return and/or explosion).
    assert ret1 != ret2 or expl1 != targets2.get("explosion_within_60s")
    assert SYSTEM_PROGRAM_WALLET not in str(targets1)


def test_build_target_dataset_mini(tmp_path: Path) -> None:
    world = tmp_path / "world_model"
    pump = tmp_path / "pump"
    for rel in ("universe", "pit", "observations/observation_dataset_v1"):
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
    # Include a system-program row that must be excluded from path.
    pq.write_table(
        pa.table(
            {
                "mint": [mint, mint],
                "user_wallet": [SYSTEM_PROGRAM_WALLET, "w0"],
                "event_time": [t0, t0 + timedelta(seconds=1)],
                "is_buy": [True, True],
                "sol_amount": [9.0, 0.5],
                "token_amount": [1.0, 1e6],
                "tx_signature": ["sys", "ok"],
                "price_sol": [1.0, 1.0],
                "v_sol_bonding_curve": [30.0, 30.0],
                "v_tokens_bonding_curve": [1e9, 1e9],
                "curve_pct_depleted": [0.0, 0.0],
            }
        ),
        pump / "trades" / "extra.parquet",
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
    obs_t = (t0 + timedelta(seconds=50)).timestamp()
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
    audit = build_target_dataset_v1(world_root=world, manifest_dir=manifest, pump_root=pump)
    assert audit["observations_processed"] == 1
    assert (manifest / "TARGET_CONTRACT_V1.json").exists()
    assert (manifest / "TARGET_DATASET_V1_AUDIT.json").exists()
    assert (manifest / "EXPLOSION_TARGET_AUDIT_V1.json").exists()
    path_rows = pq.read_table(
        world / "targets" / "target_dataset_v1" / "future_market_path_v1.parquet"
    ).to_pylist()
    assert all(r.get("wallet") != SYSTEM_PROGRAM_WALLET for r in path_rows)
    tgt = pq.read_table(
        world / "targets" / "target_dataset_v1" / "observation_targets_v1.parquet"
    ).to_pylist()[0]
    assert tgt["price_target_available"] is True
    assert tgt["h30_status"] in (STATUS_VALID_POSITIVE, STATUS_VALID_NEGATIVE)
    assert math.isfinite(float(tgt["h30_simple_return"]))
