"""PIT / future-mutation guards for causal rich baseline features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trencher.baselines.feature_spec import rich_additive_feature_names
from trencher.baselines.rich_index import CausalRichSequenceIndex


def _write_entity_index(path: Path, entity: str, start: int, end: int) -> None:
    pq.write_table(
        pa.table({"entity": [entity], "start_idx": [start], "end_idx": [end]}),
        path,
    )


def test_rich_features_immune_to_post_T_mutations(tmp_path: Path) -> None:
    """Post-T wallet/graph/ownership rows must not change rich features at T."""
    seq = tmp_path / "sequences" / "sequence_dataset_v2"
    seq.mkdir(parents=True)
    mint = "MintAAA"
    wallet = "WalletAAA"
    oid = "obs_1"
    T = 1_000.0

    # Graph token: one pre-T edge (wallet observable), one post-T edge
    graph_rows = [
        {
            "mint": mint,
            "event_time_unix": T - 10,
            "order_key": 0,
            "edge_type": "buy",
            "src": wallet,
            "dst": mint,
            "wallet": wallet,
            "token_mint": mint,
            "sol_amount": 1.5,
            "token_amount": 10.0,
            "seq_idx": 0,
        },
        {
            "mint": mint,
            "event_time_unix": T + 50,
            "order_key": 1,
            "edge_type": "buy",
            "src": wallet,
            "dst": mint,
            "wallet": wallet,
            "token_mint": mint,
            "sol_amount": 99.0,
            "token_amount": 999.0,
            "seq_idx": 1,
        },
    ]
    pq.write_table(pa.Table.from_pylist(graph_rows), seq / "sequence_graph_token_v2.parquet")
    _write_entity_index(seq / "sequence_graph_token_v2_entity_index.parquet", mint, 0, 2)

    # Wallet prior: pre-T history + post-T mutation
    prior_rows = [
        {
            "wallet": wallet,
            "event_time_unix": T - 100,
            "order_key": 0,
            "edge_type": "buy",
            "src": wallet,
            "dst": "OtherMint",
            "token_mint": "OtherMint",
            "sol_amount": 2.0,
            "token_amount": 1.0,
            "seq_idx": 0,
        },
        {
            "wallet": wallet,
            "event_time_unix": T + 5,
            "order_key": 1,
            "edge_type": "sell",
            "src": wallet,
            "dst": "OtherMint",
            "token_mint": "OtherMint",
            "sol_amount": 50.0,
            "token_amount": 1.0,
            "seq_idx": 1,
        },
    ]
    pq.write_table(
        pa.Table.from_pylist(prior_rows), seq / "sequence_graph_wallet_prior_v2.parquet"
    )
    _write_entity_index(
        seq / "sequence_graph_wallet_prior_v2_entity_index.parquet", wallet, 0, 2
    )

    # Ownership: pre + post T; deltas only from real pre/post fields
    own_rows = [
        {
            "mint": mint,
            "event_time_unix": T - 20,
            "order_key": 0,
            "value": 3.0,
            "pre_ui_amount": 1.0,
            "post_ui_amount": 4.0,
            "seq_idx": 0,
        },
        {
            "mint": mint,
            "event_time_unix": T + 30,
            "order_key": 1,
            "value": 100.0,
            "pre_ui_amount": 4.0,
            "post_ui_amount": 104.0,
            "seq_idx": 1,
        },
    ]
    pq.write_table(pa.Table.from_pylist(own_rows), seq / "sequence_ownership_v2.parquet")
    _write_entity_index(seq / "sequence_ownership_v2_entity_index.parquet", mint, 0, 2)

    micro_rows = [
        {
            "mint": mint,
            "event_time_unix": T - 5,
            "order_key": 0,
            "fee_lamports": 5000.0,
            "compute_units_consumed": 100.0,
            "priority_fee_lamports": 1.0,
            "success": 1.0,
            "account_key_count": 8.0,
            "seq_idx": 0,
        },
        {
            "mint": mint,
            "event_time_unix": T + 1,
            "order_key": 1,
            "fee_lamports": 99999.0,
            "compute_units_consumed": 999.0,
            "priority_fee_lamports": 99.0,
            "success": 0.0,
            "account_key_count": 99.0,
            "seq_idx": 1,
        },
    ]
    pq.write_table(
        pa.Table.from_pylist(micro_rows), seq / "sequence_microstructure_v2.parquet"
    )
    _write_entity_index(
        seq / "sequence_microstructure_v2_entity_index.parquet", mint, 0, 2
    )

    refs = [
        {
            "observation_id": oid,
            "mint": mint,
            "observation_time_t": T,
            "mask_micro": True,
            "micro_start_idx": 0,
            "micro_end_idx": 2,
            "mask_ownership": True,
            "ownership_start_idx": 0,
            "ownership_end_idx": 2,
            "mask_graph_token": True,
            "graph_token_start_idx": 0,
            "graph_token_end_idx": 2,
        }
    ]
    pq.write_table(
        pa.Table.from_pylist(refs), seq / "observation_sequence_refs_v2.parquet"
    )

    idx = CausalRichSequenceIndex(tmp_path)
    feat_before = idx.features_for_observation(oid, T)

    # Mutate stores with additional far-future rows (and widen ref end idxs).
    graph_rows.append(
        {
            "mint": mint,
            "event_time_unix": T + 500,
            "order_key": 2,
            "edge_type": "sell",
            "src": wallet,
            "dst": mint,
            "wallet": wallet,
            "token_mint": mint,
            "sol_amount": 777.0,
            "token_amount": 1.0,
            "seq_idx": 2,
        }
    )
    pq.write_table(pa.Table.from_pylist(graph_rows), seq / "sequence_graph_token_v2.parquet")
    _write_entity_index(seq / "sequence_graph_token_v2_entity_index.parquet", mint, 0, 3)

    prior_rows.append(
        {
            "wallet": wallet,
            "event_time_unix": T + 900,
            "order_key": 2,
            "edge_type": "buy",
            "src": wallet,
            "dst": "FutureMint",
            "token_mint": "FutureMint",
            "sol_amount": 1000.0,
            "token_amount": 1.0,
            "seq_idx": 2,
        }
    )
    pq.write_table(
        pa.Table.from_pylist(prior_rows), seq / "sequence_graph_wallet_prior_v2.parquet"
    )
    _write_entity_index(
        seq / "sequence_graph_wallet_prior_v2_entity_index.parquet", wallet, 0, 3
    )

    own_rows.append(
        {
            "mint": mint,
            "event_time_unix": T + 800,
            "order_key": 2,
            "value": 500.0,
            "pre_ui_amount": 104.0,
            "post_ui_amount": 604.0,
            "seq_idx": 2,
        }
    )
    pq.write_table(pa.Table.from_pylist(own_rows), seq / "sequence_ownership_v2.parquet")
    _write_entity_index(seq / "sequence_ownership_v2_entity_index.parquet", mint, 0, 3)

    micro_rows.append(
        {
            "mint": mint,
            "event_time_unix": T + 700,
            "order_key": 2,
            "fee_lamports": 1.0,
            "compute_units_consumed": 1.0,
            "priority_fee_lamports": 1.0,
            "success": 1.0,
            "account_key_count": 1.0,
            "seq_idx": 2,
        }
    )
    pq.write_table(
        pa.Table.from_pylist(micro_rows), seq / "sequence_microstructure_v2.parquet"
    )
    _write_entity_index(
        seq / "sequence_microstructure_v2_entity_index.parquet", mint, 0, 3
    )

    refs[0].update(
        {
            "micro_end_idx": 3,
            "ownership_end_idx": 3,
            "graph_token_end_idx": 3,
        }
    )
    pq.write_table(
        pa.Table.from_pylist(refs), seq / "observation_sequence_refs_v2.parquet"
    )

    # New index instance so caches reload mutated stores
    idx2 = CausalRichSequenceIndex(tmp_path)
    feat_after = idx2.features_for_observation(oid, T)

    for name in rich_additive_feature_names():
        assert feat_before[name] == feat_after[name], name

    # Sanity: causal aggregates exclude the post-T 99 / 50 / 100 spikes
    assert feat_before["graph_sol_sum"] == 1.5
    assert feat_before["wallet_prior_sol_sum"] == 2.0
    assert feat_before["own_value_sum"] == 3.0
    assert feat_before["own_delta_sum"] == 3.0  # 4-1
    assert feat_before["micro_n"] == 1.0
    assert feat_before["mask_wallet_prior"] == 1.0


def test_ownership_does_not_synthesize_pre_zero() -> None:
    from trencher.baselines.features import rich_features_from_arrays

    # value present but pre/post missing → deltas stay 0 (not invented as value-0)
    feat = rich_features_from_arrays(
        ownership={
            "value": np.array([7.0]),
            "pre_ui_amount": np.array([np.nan]),
            "post_ui_amount": np.array([np.nan]),
            "delta_ui": np.array([np.nan]),
        }
    )
    assert feat["own_value_sum"] == 7.0
    assert feat["own_delta_sum"] == 0.0
    assert feat["own_delta_abs_mean"] == 0.0
