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
    out = apply_zscore(np.array([2.0]), stats)
    assert abs(float(out[0])) < 1e-6
