"""Ownership completeness, PIT v4 leakage, observations, and freeze/audit tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.feasibility.concepts import SYSTEM_PROGRAM_WALLET
from trencher.data.world_model.freeze import FREEZE_ID, build_data_freeze
from trencher.data.world_model.observations import (
    build_observation_dataset_v1,
    select_observation_points,
)
from trencher.data.world_model.ownership import (
    build_fusion_coverage_truth,
    classify_ownership_at_t,
    classify_ownership_evidence,
    evaluate_holder_state_completeness,
    ledger_partition_covers_launch_to_t,
)
from trencher.data.world_model.pit import (
    assert_no_post_t_leakage,
    find_forbidden_future_fields,
    last_completed_second,
    materialize_pit_states_for_subset,
    model_state_digest,
    reconstruct_token_state_at_t,
)
from trencher.data.world_model.pit_audit import build_pit_v4_audit


def _write_mini_world(root: Path) -> Path:
    world = root / "world_model"
    for rel in (
        "universe",
        "execution",
        "graph",
        "microstructure",
        "bigquery",
        "global_context",
        "pit",
        "pit/global",
    ):
        (world / rel).mkdir(parents=True)

    t0 = datetime(2026, 6, 5, 12, tzinfo=UTC)
    subset = pa.table(
        {
            "mint": ["m_recon", "m_partial", "m_rpc", "m_proxy", "m_miss"],
            "launch_day": ["2026-06-05"] * 5,
            "launch_regime": ["early"] * 5,
            "graduated": [True, False, False, False, False],
            "observation_time_t": [t0] * 5,
            "detected_at": [t0] * 5,
        }
    )
    pq.write_table(subset, world / "universe" / "rich_subset_mints.parquet")
    pq.write_table(subset, world / "universe" / "universe_tokens.parquet")

    pq.write_table(
        pa.table(
            {
                "mint": ["m_recon", "m_partial", "m_rpc", "m_proxy"],
                "event_time": [t0, t0, t0, t0],
                "price_sol": [0.1, 0.2, 0.3, 0.4],
                "is_buy": [True, True, True, True],
                "sol_amount": [1.0, 1.0, 1.0, 1.0],
                "v_sol_bonding_curve": [30.0] * 4,
                "v_tokens_bonding_curve": [1e9] * 4,
                "curve_pct_depleted": [0.1] * 4,
            }
        ),
        world / "execution" / "execution_curve_states.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "src": ["w1", "w2", "w3", "w4"],
                "dst": ["m_recon", "m_partial", "m_rpc", "m_proxy"],
                "first_t": [t0] * 4,
                "last_t": [t0] * 4,
                "buys": [99, 99, 99, 99],
                "sells": [50, 50, 50, 50],
                "buy_sol": [100.0] * 4,
                "sell_sol": [40.0] * 4,
            }
        ),
        world / "graph" / "edges_wallet_token.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": ["m_recon", "m_partial", "m_rpc", "m_proxy"],
                "user_wallet": ["w1", "w2", "w3", "w4"],
                "event_time": [t0, t0, t0, t0],
                "is_buy": [True, True, True, True],
                "sol_amount": [1.0, 1.0, 1.0, 1.0],
            }
        ),
        world / "pit" / "atomic_trades_subset.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": ["m_recon", "m_partial", "m_rpc", "m_proxy"],
                "found": [True, True, True, True],
                "block_time": [t0.timestamp()] * 4,
                "tx_signature": ["s1", "s2", "s3", "s4"],
                "slot": [1, 2, 3, 4],
                "success": [True, True, True, True],
                "fee_lamports": [5000] * 4,
            }
        ),
        world / "microstructure" / "rpc_transactions.parquet",
    )

    recon_rows = [
        {
            "mint": "m_recon",
            "source": None,
            "destination": "holder0",
            "authority": "auth",
            "block_timestamp": t0,
            "value": 1_000_000.0,
            "tx_signature": "mint0",
            "transfer_type": "mintTo",
        }
    ]
    for i in range(25):
        recon_rows.append(
            {
                "mint": "m_recon",
                "source": "holder0",
                "destination": f"holder{i + 1}",
                "authority": "auth",
                "block_timestamp": t0,
                "value": 10.0,
                "tx_signature": f"tt{i}",
                "transfer_type": "spl-transfer",
            }
        )
    # Late transfers after launch+1h — must not upgrade early T.
    late = t0 + timedelta(hours=1)
    for i in range(30):
        recon_rows.append(
            {
                "mint": "m_recon",
                "source": "holder0",
                "destination": f"late{i}",
                "authority": "auth",
                "block_timestamp": late,
                "value": 10.0,
                "tx_signature": f"late{i}",
                "transfer_type": "spl-transfer",
            }
        )
    partial_rows = [
        {
            "mint": "m_partial",
            "source": "a",
            "destination": "b",
            "authority": None,
            "block_timestamp": t0,
            "value": 1.0,
            "tx_signature": "p1",
            "transfer_type": "spl-transfer",
        },
        {
            "mint": "m_partial",
            "source": "b",
            "destination": "c",
            "authority": None,
            "block_timestamp": t0,
            "value": 1.0,
            "tx_signature": "p2",
            "transfer_type": "spl-transfer",
        },
    ]
    pq.write_table(
        pa.Table.from_pylist(recon_rows + partial_rows),
        world / "bigquery" / "token_transfers_rich.parquet",
    )
    # Queried day partition covering launch→late transfers (same UTC day).
    pq.write_table(
        pa.Table.from_pylist(recon_rows + partial_rows),
        world / "bigquery" / "token_transfers_2026-06-05.parquet",
    )
    pq.write_table(
        pa.table({"mint": pa.array([], type=pa.string())}),
        world / "bigquery" / "token_transfers_unbiased_days.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": ["m_rpc"],
                "owner": ["ow1"],
                "delta_ui_amount": [1.0],
                "tx_signature": ["rpc1"],
                "block_time": [t0.timestamp()],
                "edge_type": ["rpc_token_balance_delta"],
            }
        ),
        world / "graph" / "edges_ownership_rpc_token_balances.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": ["m_recon"],
                "src": ["f1"],
                "dst": ["c1"],
                "lamports": [1],
                "block_time": [t0.timestamp()],
            }
        ),
        world / "graph" / "edges_funding_native_sol.parquet",
    )
    pq.write_table(
        pa.table({"day": ["2026-06-04"], "launches": [10], "trades": [100]}),
        world / "global_context" / "daily_pump_network_context.parquet",
    )
    # Prebuilt global index so tests skip full corpus scan.
    pq.write_table(
        pa.table(
            {
                "mint": ["m_recon", "other"],
                "event_time_unix": [t0.timestamp() - 30, t0.timestamp() - 5],
            }
        ),
        world / "pit" / "global" / "launches_timeline.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "second_unix": [int(t0.timestamp()) - 10, int(t0.timestamp())],
                "trades": [5, 3],
                "buys": [3, 2],
                "sells": [2, 1],
                "buy_sol": [1.0, 0.5],
                "sell_sol": [0.4, 0.2],
                "unique_mints": [2, 1],
                "unique_wallets": [3, 2],
            }
        ),
        world / "pit" / "global" / "trade_second_buckets.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": ["grad1"],
                "event_time_unix": [t0.timestamp() - 20],
            }
        ),
        world / "pit" / "global" / "migrations_timeline.parquet",
    )
    (world / "pit" / "global" / "global_context_index_manifest.json").write_text(
        '{"index_version":"global_context_v4","excluded_system_program_wallet_rows":0}',
        encoding="utf-8",
    )
    return world


def test_partition_coverage_required_for_reconstructable() -> None:
    detected = datetime(2026, 6, 5, 12, tzinfo=UTC)
    rows = []
    for i in range(25):
        rows.append(
            {
                "source": "holder0" if i else None,
                "destination": f"holder{i}",
                "value": 10.0 if i else 1_000_000.0,
                "tx_signature": f"tt{i}",
                "block_timestamp": detected,
                "transfer_type": "spl-transfer" if i else "mintTo",
            }
        )
    no_cov = evaluate_holder_state_completeness(
        transfer_rows=rows,
        detected_at=detected,
        queried_ledger_days=set(),
    )
    assert no_cov["holder_state_reconstructable"] is False
    assert "ledger_partition_covers_launch_to_t" in no_cov["failed_checks"]
    ok = evaluate_holder_state_completeness(
        transfer_rows=rows,
        detected_at=detected,
        queried_ledger_days={"2026-06-05"},
        observation_time_t=detected.timestamp(),
    )
    assert ok["holder_state_reconstructable"] is True
    cov = ledger_partition_covers_launch_to_t(
        queried_days={"2026-06-05"},
        detected_at=detected,
        observation_time_t=detected.timestamp() + 86400,
    )
    assert cov["covers"] is False
    assert "2026-06-06" in cov["missing_days"]


def test_last_completed_second_excludes_partial_second() -> None:
    assert last_completed_second(1000.200) == 999
    assert last_completed_second(1000.0) == 999
    assert last_completed_second(1000.999) == 999


def test_same_second_global_trade_bucket_not_leaked() -> None:
    """T=1000.200: trade@1000.100 valid historically; trade@1000.800 must not affect."""
    T = 1000.200
    indexes = {
        "own_level_full_history_audit": {},
        "detected_at_by_mint": {"m": 900.0},
        "curve_by_mint": {},
        "trades_by_mint": {},
        "micro_by_mint": {},
        "rpc_own_by_mint": {},
        "ledger_by_mint": {},
        "native_by_mint": {},
        "global_daily": [],
        "blocks_daily": [],
        "launch_times": [900.0],
        "migration_times": [],
        "queried_ledger_days": set(),
        "trade_seconds": [1000],
        "trade_bucket_by_sec": {
            1000: {
                "second_unix": 1000,
                "trades": 1,  # only future same-second trade if included
                "buys": 1,
                "sells": 0,
                "buy_sol": 99.0,
                "sell_sol": 0.0,
                "unique_mints": 0,
                "unique_wallets": 0,
            }
        },
    }
    # Past completed second with historically valid trade@1000.100 would live in
    # bucket 1000, but last_completed_second(T)=999 excludes that incomplete second.
    state = reconstruct_token_state_at_t(
        mint="m", observation_time_t=T, indexes=indexes
    )["model_state"]
    trailing = state["global_context"]["trailing_windows"]["trailing_60s"]["pump_trades"]
    assert trailing["trades"] == 0
    assert state["global_context"]["global_trade_buckets_last_completed_second"] == 999

    # Adding a completed-second bucket at 999 (trade @999.5) must be visible.
    indexes2 = {
        **indexes,
        "trade_seconds": [999, 1000],
        "trade_bucket_by_sec": {
            999: {
                "second_unix": 999,
                "trades": 1,
                "buys": 1,
                "sells": 0,
                "buy_sol": 1.0,
                "sell_sol": 0.0,
                "unique_mints": 0,
                "unique_wallets": 0,
            },
            1000: indexes["trade_bucket_by_sec"][1000],
        },
    }
    state2 = reconstruct_token_state_at_t(
        mint="m", observation_time_t=T, indexes=indexes2
    )["model_state"]
    trailing2 = state2["global_context"]["trailing_windows"]["trailing_60s"]["pump_trades"]
    assert trailing2["trades"] == 1
    # Mutating incomplete second 1000 must not change digest.
    d1 = model_state_digest(state2)
    indexes2["trade_bucket_by_sec"][1000] = {
        **indexes2["trade_bucket_by_sec"][1000],
        "trades": 500,
        "buy_sol": 1e9,
    }
    state3 = reconstruct_token_state_at_t(
        mint="m", observation_time_t=T, indexes=indexes2
    )["model_state"]
    assert model_state_digest(state3) == d1


def test_observation_points_causal_and_deduped() -> None:
    pts = select_observation_points(
        mint="m",
        launch_t=1000.0,
        trade_event_times=[1000.0, 1010.0, 1060.0, 2000.0],
    )
    triggers = {(p["observation_trigger"], p["trigger_value"]) for p in pts}
    assert ("fixed_age", "launch") in triggers
    assert ("fixed_age", "10s") in triggers
    assert ("fixed_age", "1h") in triggers
    assert ("fixed_age", "6h") in triggers
    # trades_1@1000 and trades_2@1010 coincide with fixed ages → deduped away
    assert ("event_count", "trades_1") not in triggers
    assert ("event_count", "trades_4") in triggers  # 4th trade @2000
    at_launch = [p for p in pts if p["observation_time_t"] == 1000.0]
    assert len(at_launch) == 1
    assert at_launch[0]["observation_trigger"] == "fixed_age"
    times = [p["observation_time_t"] for p in pts]
    assert len(times) == len({int(round(t * 1e6)) for t in times})


def test_build_observation_dataset_v1(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    classify_ownership_evidence(world)
    md = tmp_path / "manifests"
    md.mkdir()
    audit = build_observation_dataset_v1(
        world_root=world,
        manifest_dir=md,
        pump_root=tmp_path / "missing_pump",
        max_mints=5,
        leakage_sample_n=2,
    )
    assert audit["tokens_n"] == 5
    assert audit["observations_n"] > 0
    assert audit["leakage_tests"]["all_passed"] is True
    assert (md / "OBSERVATION_DATASET_V1_AUDIT.json").exists()
    idx = world / "observations" / "observation_dataset_v1" / "observation_index_v1.parquet"
    assert pq.read_table(idx).num_rows == audit["observations_n"]


def test_sparse_bq_is_partial_not_reconstructable() -> None:
    rows = [
        {
            "source": "a",
            "destination": "b",
            "value": 1.0,
            "tx_signature": "x",
            "block_timestamp": datetime(2026, 6, 5, 12, tzinfo=UTC),
            "transfer_type": "spl-transfer",
        }
    ]
    result = evaluate_holder_state_completeness(
        transfer_rows=rows,
        detected_at=datetime(2026, 6, 5, 12, tzinfo=UTC),
    )
    assert result["holder_state_reconstructable"] is False


def test_ownership_evidence_levels(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    out = classify_ownership_evidence(world)
    by_mint = {r["mint"]: r["ownership_evidence_level"] for r in out["rows"]}
    assert by_mint["m_recon"] == "holder_state_reconstructable"
    assert by_mint["m_partial"] == "transfer_ledger_observed_partial"
    assert by_mint["m_rpc"] == "rpc_balance_observed"


def test_ownership_at_t_does_not_use_future_ledger(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    # Full history reconstructable.
    full = classify_ownership_evidence(world)
    assert (
        {r["mint"]: r["ownership_evidence_level"] for r in full["rows"]}["m_recon"]
        == "holder_state_reconstructable"
    )
    transfers = pq.read_table(
        world / "bigquery" / "token_transfers_rich.parquet"
    ).to_pylist()
    recon_transfers = [r for r in transfers if r["mint"] == "m_recon"]
    t0 = datetime(2026, 6, 5, 12, tzinfo=UTC).timestamp()
    # At launch, only mint-time rows (26) — still reconstructable in this fixture.
    at_launch = classify_ownership_at_t(
        transfer_rows=recon_transfers,
        rpc_rows=[],
        trade_participation_before_t=True,
        detected_at=datetime(2026, 6, 5, 12, tzinfo=UTC),
        observation_time_t=t0,
    )
    assert at_launch["ownership_level_at_t"] in {
        "holder_state_reconstructable",
        "transfer_ledger_observed_partial",
    }
    # Before any transfers: missing even if full history is reconstructable.
    before = classify_ownership_at_t(
        transfer_rows=recon_transfers,
        rpc_rows=[],
        trade_participation_before_t=False,
        detected_at=datetime(2026, 6, 5, 12, tzinfo=UTC),
        observation_time_t=t0 - 10_000,
    )
    assert before["ownership_level_at_t"] == "missing"
    assert before["holder_state_reconstructable_at_t"] is False


def test_fusion_coverage_truth_does_not_inflate_sparse_ledger(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    md = tmp_path / "manifests"
    md.mkdir()
    result = build_fusion_coverage_truth(world, manifest_dir=md)
    counts = result["truth"]["counts"]
    assert counts["core_plus_holder_state_reconstructable"] == 1


def test_data_freeze_manifest(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    md = tmp_path / "manifests"
    freeze = build_data_freeze(world_root=world, manifest_dir=md)
    assert freeze["freeze_id"] == FREEZE_ID


def test_pit_model_state_excludes_future_and_full_history_level(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    classify_ownership_evidence(world)  # populate audit parquet
    t0 = datetime(2026, 6, 5, 12, tzinfo=UTC).timestamp()
    indexes = {
        "own_level_full_history_audit": {"m_recon": "holder_state_reconstructable"},
        "detected_at_by_mint": {"m_recon": datetime(2026, 6, 5, 12, tzinfo=UTC)},
        "curve_by_mint": {
            "m_recon": [
                {
                    "mint": "m_recon",
                    "event_time": datetime(2026, 6, 5, 11, tzinfo=UTC),
                    "price_sol": 0.1,
                    "is_buy": True,
                    "sol_amount": 1.0,
                }
            ]
        },
        "trades_by_mint": {
            "m_recon": [
                {
                    "mint": "m_recon",
                    "user_wallet": "w1",
                    "event_time": datetime(2026, 6, 5, 11, tzinfo=UTC),
                    "is_buy": True,
                    "sol_amount": 1.0,
                }
            ]
        },
        "micro_by_mint": {},
        "rpc_own_by_mint": {},
        "ledger_by_mint": {},
        "native_by_mint": {},
        "global_daily": [{"day": "2026-06-04", "launches": 1}],
        "blocks_daily": [],
        "launch_times": [t0 - 100],
        "trade_seconds": [int(t0) - 10],
        "trade_bucket_by_sec": {
            int(t0) - 10: {
                "second_unix": int(t0) - 10,
                "trades": 1,
                "buys": 1,
                "sells": 0,
                "buy_sol": 1.0,
                "sell_sol": 0.0,
                "unique_mints": 1,
                "unique_wallets": 1,
            }
        },
        "migration_times": [],
    }
    result = reconstruct_token_state_at_t(
        mint="m_recon", observation_time_t=t0, indexes=indexes
    )
    assert "model_state" in result and "provenance" in result
    state = result["model_state"]
    assert "ownership_evidence_level" not in state
    assert "future_labels_separate" not in state
    assert "generated_at" not in state
    assert "generated_at" in result["provenance"]
    assert state["ownership_level_at_t"] == "trade_derived_proxy"
    assert find_forbidden_future_fields(state) == []

    leak = assert_no_post_t_leakage(
        mint="m_recon", observation_time_t=t0, indexes=indexes
    )
    assert leak["passed"] is True
    assert leak["cases"]["post_t_trades_on_existing_wallet_edge"] is True
    assert leak["cases"]["model_state_has_no_forbidden_future_fields"] is True

    manifest = materialize_pit_states_for_subset(
        world_root=world, max_mints=5, pump_root=tmp_path / "missing_pump"
    )
    assert manifest["engine_version"] == "pit_v4"


def test_full_history_reconstructable_not_exposed_before_t(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    classify_ownership_evidence(world)
    from trencher.data.world_model.pit import load_pit_indexes

    indexes = load_pit_indexes(
        world, mint_filter={"m_recon"}, pump_root=tmp_path / "missing_pump"
    )
    assert (
        indexes["own_level_full_history_audit"]["m_recon"]
        == "holder_state_reconstructable"
    )
    early_t = datetime(2026, 6, 5, 12, tzinfo=UTC).timestamp() - 10_000
    state = reconstruct_token_state_at_t(
        mint="m_recon", observation_time_t=early_t, indexes=indexes
    )["model_state"]
    assert state["ownership_level_at_t"] == "missing"
    assert state["holder_state_reconstructable_at_t"] is False


def test_pit_v4_audit(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    md = tmp_path / "manifests"
    md.mkdir()
    audit = build_pit_v4_audit(
        world_root=world,
        manifest_dir=md,
        pump_root=tmp_path / "missing_pump",
        max_mints=5,
    )
    assert audit["audit_id"] == "pit_v4"
    assert audit["fusion_status"] == "MASKED"
    assert audit["future_labels_structurally_absent_from_model_state"] is True
    assert audit["leakage_tests"]["all_passed"] is True
    assert audit["leakage_tests"]["includes_same_second_future_trade_bucket"] is True
    assert (md / "PIT_V4_AUDIT.json").exists()


def test_system_program_wallet_constant_available() -> None:
    assert SYSTEM_PROGRAM_WALLET
    assert len(SYSTEM_PROGRAM_WALLET) > 20