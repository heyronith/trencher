"""Tests for training loaders / train-only transforms / forbid-list."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.training.collate import collate_examples, pad_sequences
from trencher.data.world_model.training.forbid import (
    FORBIDDEN_FEATURE_KEYS,
    assert_no_forbidden,
    strip_forbidden,
)
from trencher.data.world_model.training.transforms import (
    apply_clip_then_zscore,
    apply_zscore,
    fit_numeric_stats,
    fit_sequence_transforms_from_unique_rows,
    load_transforms,
    persist_transforms,
)


def test_strip_forbidden_identities() -> None:
    feats = {
        "mint": "abc",
        "creator": "xyz",
        "wallet": "w",
        "token_age_seconds": 1.0,
        "base_split": "train",
    }
    clean = strip_forbidden(feats)
    assert "mint" not in clean
    assert "creator" not in clean
    assert "wallet" not in clean
    assert "base_split" not in clean
    assert clean["token_age_seconds"] == 1.0
    assert_no_forbidden(clean)


def test_missing_targets_not_zero_filled() -> None:
    batch = collate_examples(
        [
            {
                "features": {"token_age_seconds": 1.0},
                "targets": {"returns_h60": None, "returns_h60_status": "right_censored"},
                "meta": {},
            },
            {
                "features": {"token_age_seconds": 2.0},
                "targets": {"returns_h60": 0.1, "returns_h60_status": "valid_positive"},
                "meta": {},
            },
        ]
    )
    arr = batch["targets"]["returns_h60"]
    assert np.isnan(arr[0])
    assert abs(float(arr[1]) - 0.1) < 1e-6
    assert batch["targets"]["returns_h60_is_missing"][0]


def test_pad_sequences_masks() -> None:
    a = np.ones((2, 3), dtype=np.float32)
    b = np.ones((5, 3), dtype=np.float32)
    out, mask = pad_sequences([a, b])
    assert out.shape == (2, 5, 3)
    assert mask[0].sum() == 2
    assert mask[1].sum() == 5


def test_train_only_transform_ignores_val_distribution(
    tmp_path: Path,
) -> None:
    # Unique sequence rows for train mints only
    train_rows = [
        {"mint": "m1", "price": 1.0},
        {"mint": "m1", "price": 2.0},
        {"mint": "m2", "price": 3.0},
    ]
    val_rows = [{"mint": "m3", "price": 1000.0} for _ in range(100)]
    path = tmp_path / "seq.parquet"
    pq.write_table(pa.Table.from_pylist(train_rows + val_rows), path)
    fit = fit_sequence_transforms_from_unique_rows(
        sequence_parts=[path],
        train_mints={"m1", "m2"},
        numeric_columns=["price"],
    )
    persist_transforms(tmp_path / "t.json", fit)
    loaded = load_transforms(tmp_path / "t.json")
    mean = loaded["columns"]["price"]["mean"]
    # Mean of 1,2,3 = 2 — must not include 1000s from val mint
    assert abs(mean - 2.0) < 1e-6

    # Mutate val distribution and re-fit should still ignore val mints
    val_rows2 = [{"mint": "m3", "price": -999.0} for _ in range(100)]
    pq.write_table(pa.Table.from_pylist(train_rows + val_rows2), path)
    fit2 = fit_sequence_transforms_from_unique_rows(
        sequence_parts=[path],
        train_mints={"m1", "m2"},
        numeric_columns=["price"],
    )
    assert abs(fit2["columns"]["price"]["mean"] - mean) < 1e-9


def test_forbidden_set_covers_required_identities() -> None:
    for k in ("mint", "creator", "wallet", "base_split", "target_dataset_version"):
        assert k in FORBIDDEN_FEATURE_KEYS


def test_zscore_uses_frozen_stats() -> None:
    stats = fit_numeric_stats(np.array([0.0, 2.0, 4.0]))
    out = apply_clip_then_zscore(np.array([2.0]), stats)
    assert abs(float(out[0])) < 1e-6
    # raw zscore helper still centered at mean of clipped train distribution
    out2 = apply_zscore(np.array([stats["mean"]]), stats)
    assert abs(float(out2[0])) < 1e-6


def test_event_encoding_preserves_buy_sell() -> None:
    import pyarrow as pa

    from trencher.data.world_model.training.schemas import EVENT_CHANNELS
    from trencher.data.world_model.training.sequences_batch import encode_events_table

    rows = [
        {
            "event_time_unix": 100.0,
            "is_buy": True,
            "side": "buy",
            "sol_amount": 1.0,
            "token_amount": 10.0,
            "price_sol": 0.1,
            "v_sol": 30.0,
            "v_tok": 1e9,
            "curve_pct_depleted": 0.1,
            "has_reserves": True,
            "has_finite_price": True,
            "source_quality_flag": 1.0,
        },
        {
            "event_time_unix": 110.0,
            "is_buy": False,
            "side": "sell",
            "sol_amount": 2.0,
            "token_amount": 20.0,
            "price_sol": 0.2,
            "v_sol": 28.0,
            "v_tok": 1.1e9,
            "curve_pct_depleted": 0.2,
            "has_reserves": True,
            "has_finite_price": True,
            "source_quality_flag": 1.0,
        },
    ]
    table = pa.Table.from_pylist(rows)
    arr = encode_events_table(table, 0, 2, transforms=None)
    assert arr.shape == (2, len(EVENT_CHANNELS))
    is_buy_i = EVENT_CHANNELS.index("is_buy")
    side_i = EVENT_CHANNELS.index("side_id")
    sol_i = EVENT_CHANNELS.index("sol_amount")
    assert arr[0, is_buy_i] == 1.0
    assert arr[1, is_buy_i] == 0.0
    assert arr[0, side_i] == 1.0
    assert arr[1, side_i] == 2.0
    assert arr[0, sol_i] == 1.0
    assert arr[1, sol_i] == 2.0


def test_deterministic_shard_seed() -> None:
    from trencher.data.world_model.training.dataset import deterministic_shard_seed

    a = deterministic_shard_seed(0, "part-00000.parquet")
    b = deterministic_shard_seed(0, "part-00000.parquet")
    c = deterministic_shard_seed(1, "part-00000.parquet")
    assert a == b and a != c


def test_micro_encoding_not_presence_only() -> None:
    import pyarrow as pa

    from trencher.data.world_model.training.schemas import MICRO_CHANNELS
    from trencher.data.world_model.training.sequences_batch import encode_micro_table

    rows = [
        {
            "event_time_unix": 1.0,
            "success": True,
            "fee_lamports": 5000.0,
            "compute_units_consumed": 100000.0,
            "priority_fee_lamports": 100.0,
            "account_key_count": 8.0,
            "balance_change_count": 2.0,
            "log_message_count": 3.0,
            "inner_instruction_groups": 1.0,
            "availability_priority_fee": True,
            "source_quality_flag": 1.0,
        }
    ]
    arr = encode_micro_table(pa.Table.from_pylist(rows), 0, 1, transforms=None)
    assert arr.shape == (1, len(MICRO_CHANNELS))
    assert arr[0, MICRO_CHANNELS.index("fee_lamports")] == 5000.0
    assert arr[0, MICRO_CHANNELS.index("success")] == 1.0


def test_clip_then_zscore_matches_manual() -> None:
    raw = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    stats = fit_numeric_stats(raw)
    x = np.array([50.0])
    got = apply_clip_then_zscore(x, stats)
    clipped = np.clip(x, stats["p01"], stats["p99"])
    expect = (clipped - stats["mean"]) / stats["std"]
    assert abs(float(got[0]) - float(expect[0])) < 1e-12
