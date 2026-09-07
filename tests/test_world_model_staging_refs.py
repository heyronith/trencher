"""Regression: staging must require broad observation sequence refs for Stage A."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trencher.data.world_model.broad.sequences import (
    CREATOR_STORE,
    CURVE_STORE,
    EVENTS_STORE,
)
from trencher.data.world_model.training.sequences_batch import BroadSequenceReader
from trencher.world_model.staging import (
    assert_staging_manifest_complete,
    discover_frozen_training_manifest,
    verify_staging_manifest,
    write_staging_manifest,
)

WORLD = Path("data/external/world_model")


def _have_freeze() -> bool:
    return (
        (WORLD / "broad" / "broad_observation_sequence_refs_v1").exists()
        and (WORLD / "broad" / EVENTS_STORE).exists()
    )


def test_discover_includes_broad_sequence_refs():
    """FROZEN_STORE_GLOBS must enumerate broad observation sequence refs."""
    from trencher.world_model.staging import FROZEN_STORE_GLOBS

    semantics = {s for s, _, _ in FROZEN_STORE_GLOBS}
    assert "broad_sequence_refs" in semantics
    assert "broad_seq_creator" in semantics
    refs_globs = [g for s, g, _ in FROZEN_STORE_GLOBS if s == "broad_sequence_refs"]
    assert any("broad_observation_sequence_refs_v1" in g for g in refs_globs)


def test_missing_broad_refs_fail_closed(tmp_path: Path):
    """Assignments+events without refs must not produce a complete Stage-A manifest."""
    world = tmp_path / "world_model"
    # Minimal stubs for other required globs so only refs/creator are the focus.
    (world / "splits" / "split_v1").mkdir(parents=True)
    (world / "splits" / "split_v1" / "cutoffs.json").write_text("{}")
    (world / "splits" / "split_v1" / "walkforward_folds.json").write_text("[]")
    assign = world / "splits" / "split_v1" / "observation_assignments_broad"
    assign.mkdir(parents=True)
    pq.write_table(pa.table({"observation_id": ["o1"]}), assign / "part-00000.parquet")
    # Rich sequence stores required by current globs — empty placeholders where needed
    for rel in (
        "sequences/sequence_dataset_v2/observation_sequence_refs_v2.parquet",
        "sequences/sequence_dataset_v2/sequence_events_v2.parquet",
        "sequences/sequence_dataset_v2/sequence_curve_v2.parquet",
    ):
        p = world / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"x": [1]}), p)
    bt = world / "broad" / "broad_targets_v3" / "observation_targets"
    bt.mkdir(parents=True)
    pq.write_table(pa.table({"observation_id": ["o1"]}), bt / "part-00000.parquet")
    # Events + curve present (the production blind spot), refs absent
    for store in (EVENTS_STORE, CURVE_STORE):
        part = world / "broad" / store / "00"
        part.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "event_time_unix": [1.0, 2.0],
                    "is_buy": [True, False],
                    "side": ["buy", "sell"],
                    "sol_amount": [0.1, 0.2],
                    "token_amount": [1.0, 2.0],
                    "price_sol": [1.0, 1.1],
                    "v_sol": [10.0, 11.0],
                    "v_tok": [100.0, 110.0],
                    "curve_pct_depleted": [0.1, 0.2],
                    "source_quality": [1.0, 1.0],
                }
            ),
            part / "part.parquet",
        )
        (world / "broad" / store / "manifest.json").write_text("{}\n")

    man = discover_frozen_training_manifest(world)
    missing = {a.semantic_version for a in man.artifacts if a.key.endswith(":MISSING")}
    assert "broad_sequence_refs" in missing
    with pytest.raises(FileNotFoundError, match="broad_sequence_refs"):
        assert_staging_manifest_complete(man)

    out = write_staging_manifest(man, tmp_path / "man.json")
    # Point verify root at empty world so MISSING glob paths fail closed
    raw = json.loads(out.read_text())
    raw["root"] = str(world)
    out.write_text(json.dumps(raw))
    result = verify_staging_manifest(out, fail_closed=True)
    assert result["ok"] is False
    assert any("broad_observation_sequence_refs_v1" in e for e in result["errors"])


def test_broad_reader_empty_without_refs_nonempty_with_refs(tmp_path: Path):
    """Exact production failure shape: assignments+events exist, refs missing → empty events.

    With refs restored, BroadSequenceReader must return non-empty event windows.
    """
    broad = tmp_path / "broad"
    part = "ab"
    n = 8
    times = np.arange(1000.0, 1000.0 + n, dtype=np.float64)
    events_dir = broad / EVENTS_STORE / part
    events_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "event_time_unix": times,
                "is_buy": [True] * n,
                "side": ["buy"] * n,
                "sol_amount": [0.1] * n,
                "token_amount": [10.0] * n,
                "price_sol": [1.0] * n,
                "v_sol": [50.0] * n,
                "v_tok": [500.0] * n,
                "curve_pct_depleted": [0.05] * n,
                "source_quality": [1.0] * n,
            }
        ),
        events_dir / "part.parquet",
    )
    for store in (CURVE_STORE, CREATOR_STORE):
        d = broad / store / part
        d.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "event_time_unix": times[:2],
                    "price_sol": [1.0, 1.1],
                    "v_sol": [50.0, 51.0],
                    "v_tok": [500.0, 510.0],
                    "curve_pct_depleted": [0.05, 0.06],
                    "source_quality": [1.0, 1.0],
                    "prior_rank": [0.1, 0.2],
                    "prior_is_mayhem_mode": [0.0, 0.0],
                }
            ),
            d / "part.parquet",
        )

    assignment = {
        "observation_id": "obs_test_1",
        "mint_partition": part,
        "observation_time_t": float(times[-1] + 0.5),
    }

    reader = BroadSequenceReader(broad)
    empty = reader.load_for_assignment(assignment)
    assert empty["events"].shape[0] == 0  # refs absent → silent empty (the bug)

    refs_dir = broad / "broad_observation_sequence_refs_v1" / part
    refs_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "observation_id": ["obs_test_1"],
                "events_start_idx": [0],
                "events_end_idx": [n],
                "events_partition": [part],
                "curve_start_idx": [0],
                "curve_end_idx": [2],
                "curve_partition": [part],
                "creator_start_idx": [0],
                "creator_end_idx": [2],
                "creator_partition": [part],
            }
        ),
        refs_dir / "part.parquet",
    )
    # Clear LRU so the new refs file is visible
    reader = BroadSequenceReader(broad)
    filled = reader.load_for_assignment(assignment)
    assert filled["events"].shape[0] > 0
    assert filled["events"].shape[1] == 13


@pytest.mark.skipif(not _have_freeze(), reason="frozen world_model data not present")
def test_discover_real_freeze_includes_refs_paths():
    man = discover_frozen_training_manifest(WORLD)
    paths = [a.path for a in man.artifacts]
    assert any(p.startswith("broad/broad_observation_sequence_refs_v1/") for p in paths)
    assert any(p.startswith("broad/broad_sequence_creator_v1/") for p in paths)
    assert_staging_manifest_complete(man)
    assert not any(a.key.endswith(":MISSING") for a in man.artifacts if a.required)
