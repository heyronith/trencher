"""Rich→broad causal bridge: mint+T refs, not cross-population OID identity."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _write_entity_and_events(
    store: Path,
    *,
    mint: str,
    partition: str,
    times: list[float],
) -> None:
    from trencher.data.world_model.broad.sequences import ENTITY_INDEX_NAME

    part = store / partition
    part.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "mint": mint,
            "event_time_unix": float(t),
            "is_buy": True,
            "sol_amount": 1.0,
            "token_amount": 1.0,
            "price_sol": 1.0,
            "v_sol": 10.0,
            "v_tok": 100.0,
            "curve_pct_depleted": 0.1,
        }
        for t in times
    ]
    pq.write_table(pa.Table.from_pylist(rows), part / "part.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "mint": mint,
                    "partition": partition,
                    "start_idx": 0,
                    "end_idx": len(times),
                    "n_rows": len(times),
                }
            ]
        ),
        store / ENTITY_INDEX_NAME,
    )


def test_independent_oid_namespaces_still_reconstruct_causal_features(tmp_path: Path) -> None:
    """Two OID namespaces for the same (mint,T) must not require OID equality."""
    from trencher.baselines.rich_broad_bridge import build_rich_broad_sequence_refs_v1
    from trencher.data.world_model.observations import _observation_id

    mint = "MintABCDEF"
    T = 1_000.0
    oid_rich = _observation_id(
        mint=mint, observation_time_t=T, trigger="fixed_age", trigger_value="60"
    )
    oid_broad = _observation_id(
        mint=mint, observation_time_t=T + 0.001, trigger="fixed_age", trigger_value="60"
    )
    assert oid_rich != oid_broad

    world = tmp_path / "world"
    broad = world / "broad"
    from trencher.data.world_model.broad.sequences import (
        CREATOR_STORE,
        EVENTS_STORE,
        mint_partition,
    )

    mp = mint_partition(mint)
    _write_entity_and_events(
        broad / EVENTS_STORE, mint=mint, partition=mp, times=[900.0, 950.0, 999.0, 1001.0]
    )
    _write_entity_and_events(
        broad / CREATOR_STORE, mint=mint, partition=mp, times=[900.0, 980.0]
    )

    built = build_rich_broad_sequence_refs_v1(
        world_root=world,
        assignment_rows=[
            {
                "observation_id": oid_rich,
                "mint": mint,
                "observation_time_t": T,
                "mint_partition": mp,
                "detected_at_unix": 800.0,
            }
        ],
    )
    ref = built["by_mp"][mp][0]
    assert ref["observation_id"] == oid_rich
    assert ref["events_count"] == 3  # 900,950,999 <= 1000; not 1001
    assert ref["events_end_idx"] == 3
    assert abs(float(ref["observation_time_t"]) - T) < 1e-12


def test_post_T_mutation_does_not_change_causal_end_idx(tmp_path: Path) -> None:
    from trencher.baselines.feature_cache import _broad_features_for_row, _load_events_partition
    from trencher.baselines.feature_spec import broad_feature_names
    from trencher.baselines.rich_broad_bridge import (
        build_rich_broad_sequence_refs_v1,
        verify_no_future_mutation_affects_features,
    )
    from trencher.data.world_model.broad.sequences import EVENTS_STORE, mint_partition

    mint = "MintXYZ"
    T = 500.0
    world = tmp_path / "world"
    broad = world / "broad"
    mp = mint_partition(mint)
    times = [100.0, 200.0, 400.0]
    _write_entity_and_events(broad / EVENTS_STORE, mint=mint, partition=mp, times=times)
    # empty creator/curve indexes → unavailable
    (broad / "broad_sequence_creator_v1").mkdir(parents=True, exist_ok=True)

    # Minimal global context file so reader doesn't crash hard — stub if needed.
    # GlobalContextReader may require a parquet; monkeypatch features path via empty.
    class _G:
        def load_for_t(self, _t):  # noqa: ANN001
            return None

    row = {
        "observation_id": "rich_oid_1",
        "mint": mint,
        "observation_time_t": T,
        "mint_partition": mp,
        "token_age_seconds": 50.0,
        "detected_at_unix": 450.0,
    }
    built = build_rich_broad_sequence_refs_v1(world_root=world, assignment_rows=[row])
    ref = built["by_mp"][mp][0]
    assert ref["events_count"] == 3

    ev = _load_events_partition(broad / EVENTS_STORE / mp / "part.parquet")
    feat1 = _broad_features_for_row(
        row=row, ref=ref, ev=ev, cr=None, global_reader=_G()  # type: ignore[arg-type]
    )

    # Append a post-T event and rebuild entity end (simulating future mutation).
    new_times = times + [600.0, 700.0]
    _write_entity_and_events(
        broad / EVENTS_STORE, mint=mint, partition=mp, times=new_times
    )
    built2 = build_rich_broad_sequence_refs_v1(world_root=world, assignment_rows=[row])
    ref2 = built2["by_mp"][mp][0]
    assert ref2["events_count"] == 3  # still only <= T
    assert ref2["events_end_idx"] == 3
    ev2 = _load_events_partition(broad / EVENTS_STORE / mp / "part.parquet")
    feat2 = _broad_features_for_row(
        row=row, ref=ref2, ev=ev2, cr=None, global_reader=_G()  # type: ignore[arg-type]
    )
    verify_no_future_mutation_affects_features(
        feat_before=feat1,
        feat_after=feat2,
        feature_names=[n for n in broad_feature_names() if n in feat1],
    )


def test_index_assignments_derives_mint_partition_when_null(tmp_path: Path) -> None:
    from trencher.baselines.feature_cache import _index_assignments
    from trencher.baselines.progress import ProgressEmitter
    from trencher.data.world_model.broad.sequences import mint_partition

    assign = tmp_path / "observation_assignments_rich"
    assign.mkdir()
    mint = "MintNullMP"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "observation_id": "oid1",
                    "mint": mint,
                    "observation_time_t": 10.0,
                    "detected_at_unix": 1.0,
                    "token_age_seconds": 9.0,
                    "base_split": "train",
                    "creator": "c",
                    "mint_partition": None,
                    "in_rich_subset": True,
                    "outage_proximity_sec": None,
                    "observation_during_jul3_outage": False,
                }
            ]
        ),
        assign / "part-00000.parquet",
    )
    by_mp, skipped = _index_assignments(
        assign_dir=assign,
        q85=1e18,
        max_partitions=None,
        progress=ProgressEmitter("t"),
    )
    assert skipped == 0
    assert mint_partition(mint) in by_mp
    assert by_mp[mint_partition(mint)][0]["mint_partition"] == mint_partition(mint)
