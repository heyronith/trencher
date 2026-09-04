"""Ownership completeness, PIT v2 leakage, and freeze/audit tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.freeze import FREEZE_ID, build_data_freeze
from trencher.data.world_model.ownership import (
    build_fusion_coverage_truth,
    classify_ownership_evidence,
    evaluate_holder_state_completeness,
)
from trencher.data.world_model.pit import (
    assert_no_post_t_leakage,
    materialize_pit_states_for_subset,
    reconstruct_token_state_at_t,
)
from trencher.data.world_model.pit_audit import build_pit_v2_audit


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
    # Atomic trades for PIT (lifetime aggregates above must not be used).
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

    # Reconstructable ledger: dense, near launch, mintTo + many transfers.
    recon_rows = []
    recon_rows.append(
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
    )
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
    # Sparse partial ledger.
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
    return world


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
    assert "min_transfer_rows" in result["failed_checks"]


def test_ownership_evidence_levels(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    out = classify_ownership_evidence(world)
    by_mint = {r["mint"]: r["ownership_evidence_level"] for r in out["rows"]}
    assert by_mint["m_recon"] == "holder_state_reconstructable"
    assert by_mint["m_partial"] == "transfer_ledger_observed_partial"
    assert by_mint["m_rpc"] == "rpc_balance_observed"
    assert by_mint["m_proxy"] == "trade_derived_proxy"
    assert by_mint["m_miss"] == "missing"
    assert out["summary"]["counts_by_level"]["holder_state_reconstructable"] == 1
    assert out["summary"]["counts_by_level"]["transfer_ledger_observed_partial"] == 1


def test_fusion_coverage_truth_does_not_inflate_sparse_ledger(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    md = tmp_path / "manifests"
    md.mkdir()
    result = build_fusion_coverage_truth(world, manifest_dir=md)
    counts = result["truth"]["counts"]
    assert counts["core_event_graph_micro_exec_market"] == 4
    assert counts["core_plus_any_genuine_ownership"] == 3
    assert counts["core_plus_holder_state_reconstructable"] == 1
    assert counts["core_plus_reconstructable_ownership"] == 1


def test_data_freeze_manifest(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    md = tmp_path / "manifests"
    freeze = build_data_freeze(world_root=world, manifest_dir=md)
    assert freeze["freeze_id"] == FREEZE_ID
    assert freeze["fusion_coverage"]["core_plus_holder_state_reconstructable"] == 1


def test_pit_lifetime_wallet_aggregate_leakage(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    t0 = datetime(2026, 6, 5, 12, tzinfo=UTC).timestamp()
    indexes = {
        "own_level": {"m_recon": "holder_state_reconstructable"},
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
        "global_rows": [{"day": "2026-06-04", "launches": 1}],
        "launch_events": [],
        "future_labels_by_mint": {},
        "migrations_by_mint": {},
    }
    state = reconstruct_token_state_at_t(
        mint="m_recon", observation_time_t=t0, indexes=indexes
    )
    assert state["wallet_token_state"][0]["buys_before_t"] == 1
    assert state["quality_masks"]["lifetime_wallet_aggregates_excluded"] is True

    leak = assert_no_post_t_leakage(
        mint="m_recon", observation_time_t=t0, indexes=indexes
    )
    assert leak["passed"] is True
    assert leak["cases"]["post_t_trades_on_existing_wallet_edge"] is True
    assert leak["cases"]["post_t_ownership_transfers"] is True
    assert leak["cases"]["post_t_rpc_balance_observations"] is True
    assert leak["cases"]["post_t_transaction_metadata"] is True
    assert leak["cases"]["post_t_curve_observations"] is True
    assert leak["cases"]["post_t_global_activity"] is True
    assert leak["cases"]["post_t_graduation_migration_outcomes"] is True

    manifest = materialize_pit_states_for_subset(
        world_root=world, max_mints=5, pump_root=tmp_path / "missing_pump"
    )
    assert manifest["mints_materialized"] == 5
    assert manifest["engine_version"] == "pit_v2"
    assert manifest["labels"] == "not_generated"


def test_pit_v2_audit(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    md = tmp_path / "manifests"
    md.mkdir()
    audit = build_pit_v2_audit(
        world_root=world,
        manifest_dir=md,
        pump_root=tmp_path / "missing_pump",
        max_mints=5,
    )
    assert audit["truly_reconstructable_holder_state_count"] == 1
    assert audit["leakage_tests"]["all_passed"] is True
    assert (md / "PIT_V2_AUDIT.json").exists()
    assert (md / "PIT_V2_AUDIT.md").exists()
