"""Ownership evidence, fusion coverage truth, freeze, and PIT leakage tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.freeze import FREEZE_ID, build_data_freeze
from trencher.data.world_model.ownership import (
    build_fusion_coverage_truth,
    classify_ownership_evidence,
)
from trencher.data.world_model.pit import (
    assert_no_post_t_leakage,
    materialize_pit_states_for_subset,
    reconstruct_token_state_at_t,
)


def _write_mini_world(root: Path) -> Path:
    world = root / "world_model"
    (world / "universe").mkdir(parents=True)
    (world / "execution").mkdir(parents=True)
    (world / "graph").mkdir(parents=True)
    (world / "microstructure").mkdir(parents=True)
    (world / "bigquery").mkdir(parents=True)
    (world / "global_context").mkdir(parents=True)

    t0 = datetime(2026, 6, 5, 12, tzinfo=UTC)
    subset = pa.table(
        {
            "mint": ["m_ledger", "m_rpc", "m_proxy", "m_miss"],
            "launch_day": ["2026-06-05"] * 4,
            "launch_regime": ["early"] * 4,
            "graduated": [True, False, False, False],
            "observation_time_t": [t0, t0, t0, t0],
            "detected_at": [t0, t0, t0, t0],
        }
    )
    pq.write_table(subset, world / "universe" / "rich_subset_mints.parquet")
    pq.write_table(subset, world / "universe" / "universe_tokens.parquet")

    # Core modalities for ledger + rpc (+ proxy has graph only partially).
    pq.write_table(
        pa.table(
            {
                "mint": ["m_ledger", "m_rpc", "m_proxy"],
                "event_time": [t0, t0, t0],
                "price_sol": [0.1, 0.2, 0.3],
            }
        ),
        world / "execution" / "execution_curve_states.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "src": ["w1", "w2", "w3"],
                "dst": ["m_ledger", "m_rpc", "m_proxy"],
                "first_t": [t0, t0, t0],
                "last_t": [t0, t0, t0],
                "buys": [1, 1, 1],
                "sells": [0, 0, 0],
            }
        ),
        world / "graph" / "edges_wallet_token.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": ["m_ledger", "m_rpc", "m_proxy"],
                "found": [True, True, True],
                "block_time": [
                    t0.timestamp(),
                    t0.timestamp(),
                    t0.timestamp(),
                ],
                "tx_signature": ["s1", "s2", "s3"],
                "slot": [1, 2, 3],
                "success": [True, True, True],
                "fee_lamports": [5000, 5000, 5000],
            }
        ),
        world / "microstructure" / "rpc_transactions.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": ["m_ledger"],
                "source": ["a"],
                "destination": ["b"],
                "block_timestamp": [t0],
                "value": [1.0],
                "tx_signature": ["tt1"],
            }
        ),
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
                "mint": ["m_ledger"],
                "src": ["f1"],
                "dst": ["c1"],
                "lamports": [1],
            }
        ),
        world / "graph" / "edges_funding_native_sol.parquet",
    )
    pq.write_table(
        pa.table({"day": ["2026-06-04"], "launches": [10]}),
        world / "global_context" / "daily_pump_network_context.parquet",
    )
    return world


def test_ownership_evidence_levels(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    out = classify_ownership_evidence(world)
    by_mint = {r["mint"]: r["ownership_evidence_level"] for r in out["rows"]}
    assert by_mint["m_ledger"] == "ledger_full_or_reconstructable"
    assert by_mint["m_rpc"] == "rpc_balance_observed"
    assert by_mint["m_proxy"] == "trade_derived_proxy"
    assert by_mint["m_miss"] == "missing"
    assert out["summary"]["counts_by_level"]["ledger_full_or_reconstructable"] == 1
    assert out["summary"]["counts_by_level"]["rpc_balance_observed"] == 1


def test_fusion_coverage_truth_does_not_inflate_rpc(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    md = tmp_path / "manifests"
    md.mkdir()
    result = build_fusion_coverage_truth(world, manifest_dir=md)
    counts = result["truth"]["counts"]
    assert counts["core_event_graph_micro_exec_market"] == 3
    assert counts["core_plus_any_genuine_ownership"] == 2
    assert counts["core_plus_reconstructable_ownership"] == 1
    assert counts["with_true_native_sol_history"] == 1
    assert (md / "fusion_coverage_truth.json").exists()


def test_data_freeze_manifest(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    md = tmp_path / "manifests"
    freeze = build_data_freeze(world_root=world, manifest_dir=md)
    assert freeze["freeze_id"] == FREEZE_ID
    assert freeze["policy"]["append_only_raw"] is True
    assert freeze["fusion_coverage"]["core_plus_reconstructable_ownership"] == 1
    assert (md / "DATA_FREEZE.json").exists()
    assert (md / "DATA_FREEZE.md").exists()
    assert (md / f"{FREEZE_ID}.json").exists()


def test_pit_reconstruction_and_leakage(tmp_path: Path) -> None:
    world = _write_mini_world(tmp_path)
    t0 = datetime(2026, 6, 5, 12, tzinfo=UTC).timestamp()
    indexes = {
        "own_level": {"m_ledger": "ledger_full_or_reconstructable"},
        "curve_by_mint": {
            "m_ledger": [
                {
                    "mint": "m_ledger",
                    "event_time": datetime(2026, 6, 5, 11, tzinfo=UTC),
                    "price_sol": 0.1,
                }
            ]
        },
        "micro_by_mint": {
            "m_ledger": [
                {
                    "mint": "m_ledger",
                    "found": True,
                    "block_time": t0 - 10,
                    "tx_signature": "pre",
                    "slot": 1,
                    "success": True,
                    "fee_lamports": 1,
                }
            ]
        },
        "graph_by_mint": {
            "m_ledger": [
                {
                    "src": "w1",
                    "dst": "m_ledger",
                    "first_t": datetime(2026, 6, 5, 11, tzinfo=UTC),
                    "buys": 1,
                    "sells": 0,
                }
            ]
        },
        "rpc_own_by_mint": {},
        "ledger_by_mint": {
            "m_ledger": [
                {
                    "mint": "m_ledger",
                    "source": "a",
                    "destination": "b",
                    "block_timestamp": datetime(2026, 6, 5, 11, tzinfo=UTC),
                    "value": 1,
                    "tx_signature": "tt",
                }
            ]
        },
        "global_rows": [{"day": "2026-06-04", "launches": 1}],
    }
    state = reconstruct_token_state_at_t(
        mint="m_ledger", observation_time_t=t0, indexes=indexes
    )
    assert state["modality_masks"]["ownership_ledger"] is True
    assert state["token_curve_state_events_n"] == 1
    leak = assert_no_post_t_leakage(
        mint="m_ledger", observation_time_t=t0, indexes=indexes
    )
    assert leak["passed"] is True
    assert leak["state_unchanged"] is True

    manifest = materialize_pit_states_for_subset(world_root=world, max_mints=4)
    assert manifest["mints_materialized"] == 4
    assert (world / "pit" / "pit_states_sample.parquet").exists()
    assert manifest["labels"] == "not_generated"
