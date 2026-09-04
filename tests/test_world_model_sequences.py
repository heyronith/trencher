"""Sequence dataset stores, refs, windows, and causality tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.feasibility.concepts import SYSTEM_PROGRAM_WALLET
from trencher.data.world_model.observations import build_observation_dataset_v1
from trencher.data.world_model.ownership import classify_ownership_evidence
from trencher.data.world_model.pit import last_completed_second
from trencher.data.world_model.sequences import (
    WINDOW_PRESETS,
    apply_training_window,
    build_model_input_contract_v1,
    build_sequence_dataset_v1,
)


def test_model_input_contract_forbids_mint_embedding() -> None:
    contract = build_model_input_contract_v1()
    assert "mint" in contract["identifiers_not_structured_features"]
    assert "mint_as_categorical_embedding" in contract["structured_encoder_forbidden"]
    assert "lifetime_edges_wallet_token_aggregates" in contract[
        "structured_encoder_forbidden"
    ]
    assert "red_cohort_full_history_membership_at_earlier_T" in contract[
        "structured_encoder_forbidden"
    ]


def test_training_window_presets_causal() -> None:
    times = [100.0, 105.0, 110.0, 120.0, 150.0]
    t = 130.0
    end = 4
    lo, hi = apply_training_window(
        times=times,
        start_idx=0,
        end_idx=end,
        observation_time_t=t,
        preset={"kind": "last_n", "n": 2},
    )
    assert (lo, hi) == (2, 4)
    lo2, hi2 = apply_training_window(
        times=times,
        start_idx=0,
        end_idx=end,
        observation_time_t=t,
        preset={"kind": "last_seconds", "seconds": 15},
    )
    assert times[lo2] >= t - 15
    assert hi2 == end
    assert any(p["name"] == "last_2m" for p in WINDOW_PRESETS)


def test_sequence_dataset_v1_causal_refs(tmp_path: Path) -> None:
    from test_world_model_freeze_pit import _write_mini_world

    world = _write_mini_world(tmp_path)
    pump = tmp_path / "pumpfun_memecoin_corpus" / "trades"
    pump.mkdir(parents=True)
    t0 = datetime(2026, 6, 5, 12, tzinfo=UTC)
    pq.write_table(
        pa.table(
            {
                "mint": ["m_recon", "other_mint", "m_recon"],
                "user_wallet": ["w1", "w1", SYSTEM_PROGRAM_WALLET],
                "event_time": [t0, t0, t0],
                "is_buy": [True, True, True],
                "sol_amount": [1.0, 2.0, 9.0],
                "token_amount": [10.0, 20.0, 1.0],
                "tx_signature": ["s1", "s2", "s3"],
            }
        ),
        pump / "trades-00000.parquet",
    )
    classify_ownership_evidence(world)
    md = tmp_path / "manifests"
    md.mkdir()
    build_observation_dataset_v1(
        world_root=world,
        manifest_dir=md,
        pump_root=tmp_path / "pumpfun_memecoin_corpus",
        max_mints=5,
        leakage_sample_n=1,
    )
    audit = build_sequence_dataset_v1(
        world_root=world,
        manifest_dir=md,
        pump_root=tmp_path / "pumpfun_memecoin_corpus",
    )
    assert audit["audit_id"] == "sequence_dataset_v1"
    val = audit["causality_validation"]
    assert val["post_t_violations"] == 0
    assert val["ordering_violations"] == 0
    assert val["causal_boundary_ok"] is True
    assert val["stratified_sample"]["all_passed"] is True
    assert (md / "SEQUENCE_DATASET_V1_AUDIT.json").exists()
    assert (md / "MODEL_INPUT_CONTRACT_V1.json").exists()

    refs = pq.read_table(
        world
        / "sequences"
        / "sequence_dataset_v1"
        / "observation_sequence_refs_v1.parquet"
    )
    assert refs.num_rows == audit["observation_refs"]["rows"]
    events = pq.read_table(
        world / "sequences" / "sequence_dataset_v1" / "sequence_events_v1.parquet"
    )
    if events.num_rows:
        wallets = set(events.column("wallet").to_pylist())
        assert SYSTEM_PROGRAM_WALLET not in wallets

    for row in refs.to_pylist()[:5]:
        t = float(row["observation_time_t"])
        assert row["global_last_completed_second"] == last_completed_second(t)
