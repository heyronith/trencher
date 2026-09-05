"""Broad event/curve supervision package tests (tiny fixtures only)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.feasibility.concepts import SYSTEM_PROGRAM_WALLET
from trencher.data.world_model.broad import build_broad_supervision_v1
from trencher.data.world_model.broad.graduation_censor import graduation_censor_decision
from trencher.data.world_model.broad.observations import OBS_REFS_STORE
from trencher.data.world_model.broad.population import POPULATION_VERSION
from trencher.data.world_model.broad.sequences import (
    EVENTS_STORE,
    load_entity_index,
    load_partition_times,
)
from trencher.data.world_model.targets.status import (
    STATUS_RIGHT_CENSORED,
    STATUS_VALID_NEGATIVE,
    STATUS_VALID_POSITIVE,
)
from trencher.data.world_model.targets.sufficiency import sufficiency_row


def _write_tiny_pump(root: Path) -> Path:
    pump = root / "pumpfun_memecoin_corpus"
    trades = pump / "trades"
    trades.mkdir(parents=True)
    t0 = datetime(2026, 6, 10, 12, tzinfo=UTC)
    # Two creators; creator A has a prior launch before m2.
    tokens = pa.table(
        {
            "mint": ["m_grad", "m1", "m2", "m_notrade"],
            "detected_at": [
                t0,
                t0 + timedelta(hours=1),
                t0 + timedelta(hours=2),
                t0 + timedelta(hours=3),
            ],
            "tracking_expires_at": [
                t0 + timedelta(hours=48),
                t0 + timedelta(hours=49),
                t0 + timedelta(hours=50),
                t0 + timedelta(hours=51),
            ],
            "creator": ["cA", "cA", "cA", "cB"],
            "graduated_at": [t0 + timedelta(hours=50), None, None, None],
            "trade_count": [3, 2, 2, 0],
            "is_mayhem_mode": [False, False, False, False],
            "name": ["g", "a", "b", "n"],
            "symbol": ["G", "A", "B", "N"],
        }
    )
    pq.write_table(tokens, pump / "tokens.parquet")
    # Migration evidence for m_grad only (metadata; must not bias membership).
    pq.write_table(
        pa.table(
            {
                "mint": ["m_grad"],
                "migrated_at": [t0 + timedelta(hours=50)],
                "pool_address": ["pool1"],
            }
        ),
        pump / "migrations.parquet",
    )
    # Trades: include System Program row that must be excluded.
    pq.write_table(
        pa.table(
            {
                "mint": ["m_grad", "m_grad", "m_grad", "m1", "m1", "m2", "m2"],
                "user_wallet": [
                    "w1",
                    SYSTEM_PROGRAM_WALLET,
                    "w2",
                    "w1",
                    "w3",
                    "w1",
                    "w4",
                ],
                "event_time": [
                    t0 + timedelta(seconds=1),
                    t0 + timedelta(seconds=2),
                    t0 + timedelta(seconds=3),
                    t0 + timedelta(hours=1, seconds=1),
                    t0 + timedelta(hours=1, seconds=5),
                    t0 + timedelta(hours=2, seconds=1),
                    t0 + timedelta(hours=2, seconds=8),
                ],
                "is_buy": [True, True, False, True, False, True, False],
                "sol_amount": [1.0, 9.0, 0.5, 1.0, 0.4, 1.2, 0.3],
                "token_amount": [100.0, 900.0, 40.0, 100.0, 30.0, 110.0, 20.0],
                "tx_signature": ["s1", "ssystem", "s2", "s3", "s4", "s5", "s6"],
                "price_sol": [0.01, 0.01, 0.012, 0.01, 0.011, 0.01, 0.009],
                "v_sol_bonding_curve": [30.0, 31.0, 30.5, 30.0, 29.8, 30.0, 29.5],
                "v_tokens_bonding_curve": [
                    1_000_000.0,
                    990_000.0,
                    995_000.0,
                    1_000_000.0,
                    1_005_000.0,
                    1_000_000.0,
                    1_010_000.0,
                ],
                "curve_pct_depleted": [0.1, 0.11, 0.105, 0.1, 0.12, 0.1, 0.13],
                "source": ["pump"] * 7,
            }
        ),
        trades / "trades-00000.parquet",
    )
    (pump / "KNOWN_ISSUES.md").write_text(
        "# Known issues\n\nNo mention of the expire field.\n", encoding="utf-8"
    )
    (pump / "README.md").write_text(
        "Collected continuously over a fixed 39-day window (June 5 – July 14, 2026).\n",
        encoding="utf-8",
    )
    return pump


def test_population_flags_outcome_independent(tmp_path: Path) -> None:
    pump = _write_tiny_pump(tmp_path)
    world = tmp_path / "world_model"
    world.mkdir()
    md = tmp_path / "manifests"
    summary = build_broad_supervision_v1(
        world_root=world,
        manifest_dir=md,
        pump_root=pump,
        phases=["population"],
    )
    assert "population" in summary["phases"]
    pop = pq.read_table(world / "broad" / f"{POPULATION_VERSION}.parquet").to_pylist()
    by = {r["mint"]: r for r in pop}
    # All launches present.
    assert set(by) == {"m_grad", "m1", "m2", "m_notrade"}
    for r in pop:
        assert r["launch_population"] is True
        assert r["graduation_event_population"] is True
    # event_curve excludes no-trade mint.
    assert by["m_notrade"]["event_curve_population"] is False
    assert by["m1"]["event_curve_population"] is True
    # Graduation metadata must not gate membership: non-grad still in graduation class.
    assert by["m1"]["graduation_event_population"] is True
    assert by["m_grad"]["has_migration_evidence"] is True
    assert by["m1"]["has_migration_evidence"] is False


def test_system_program_excluded_from_sequences(tmp_path: Path) -> None:
    pump = _write_tiny_pump(tmp_path)
    world = tmp_path / "world_model"
    world.mkdir()
    md = tmp_path / "manifests"
    build_broad_supervision_v1(
        world_root=world,
        manifest_dir=md,
        pump_root=pump,
        phases=["population", "sequences"],
    )
    events_root = world / "broad" / EVENTS_STORE
    idx = load_entity_index(events_root)
    assert "m_grad" in idx
    part = idx["m_grad"]["partition"]
    rows = pq.read_table(events_root / part / "part.parquet").to_pylist()
    wallets = {r.get("wallet") for r in rows if r.get("mint") == "m_grad"}
    assert SYSTEM_PROGRAM_WALLET not in wallets
    # Two non-system trades for m_grad remain.
    assert sum(1 for r in rows if r["mint"] == "m_grad") == 2


def test_observation_refs_causal(tmp_path: Path) -> None:
    pump = _write_tiny_pump(tmp_path)
    world = tmp_path / "world_model"
    world.mkdir()
    md = tmp_path / "manifests"
    build_broad_supervision_v1(
        world_root=world,
        manifest_dir=md,
        pump_root=pump,
        phases=["population", "sequences", "observations"],
    )
    refs_root = world / "broad" / OBS_REFS_STORE
    events_root = world / "broad" / EVENTS_STORE
    entity = load_entity_index(events_root)
    # Load all refs
    ref_rows: list[dict] = []
    for part_dir in refs_root.iterdir():
        if part_dir.is_dir() and (part_dir / "part.parquet").exists():
            ref_rows.extend(pq.read_table(part_dir / "part.parquet").to_pylist())
    assert ref_rows
    for ref in ref_rows:
        mint = ref["mint"]
        T = float(ref["observation_time_t"])
        meta = entity.get(mint)
        if not meta or int(ref.get("events_count") or 0) == 0:
            continue
        times = load_partition_times(events_root, str(meta["partition"]))
        lo, hi = int(ref["events_start_idx"]), int(ref["events_end_idx"])
        slice_times = times[lo:hi]
        assert all(t <= T for t in slice_times)


def test_sufficiency_positive_subseteq_valid() -> None:
    """Unit test on sufficiency module: positive tokens ⊆ valid tokens."""
    rows = [
        {
            "mint": "a",
            "h300_status": STATUS_VALID_POSITIVE,
            "h300_simple_return": 0.5,
            "h3600_barrier_u100_l30_outcome": "upper_first",
            "surv_h86400_status": STATUS_VALID_POSITIVE,
            "grad_h86400_status": STATUS_VALID_POSITIVE,
            "grad_h86400_graduated": True,
            "explosion_within_60s": True,
            "holder_state_reconstructable": False,
        },
        {
            "mint": "b",
            "h300_status": STATUS_RIGHT_CENSORED,
            "h300_simple_return": 1.5,  # must NOT count as positive
            "h3600_barrier_u100_l30_outcome": "censored",
            "surv_h86400_status": STATUS_RIGHT_CENSORED,
            "grad_h86400_status": STATUS_RIGHT_CENSORED,
            "grad_h86400_graduated": None,
            "explosion_within_60s": None,
            "explosion_within_60s_status": STATUS_RIGHT_CENSORED,
            "holder_state_reconstructable": False,
        },
        {
            "mint": "c",
            "h300_status": STATUS_VALID_NEGATIVE,
            "h300_simple_return": -0.2,
            "h3600_barrier_u100_l30_outcome": "neither",
            "surv_h86400_status": STATUS_VALID_NEGATIVE,
            "grad_h86400_status": STATUS_VALID_NEGATIVE,
            "grad_h86400_graduated": False,
            "explosion_within_60s": False,
            "holder_state_reconstructable": False,
        },
    ]
    from trencher.data.world_model.targets.sufficiency import compute_rich_sufficiency

    out = compute_rich_sufficiency(
        rows,
        primary_explosion_events=1,
        tokens_with_primary_explosion={"a"},
        pre_explosion_obs=1,
        explosion_obs_before_60s=1,
    )
    for row in out:
        assert (
            row["distinct_positive_tokens"]
            <= row["unique_tokens_with_valid_supervision"]
        )
    ret = next(r for r in out if r["head"] == "returns_h300")
    assert ret["distinct_positive_tokens"] == 1  # only mint a
    assert ret["unique_tokens_with_valid_supervision"] == 2  # a,c

    # Direct invariant guard on sufficiency_row
    try:
        sufficiency_row(
            head="bad",
            valid_observations=1,
            unique_tokens_with_valid_supervision=1,
            positive_observations=2,
            distinct_positive_tokens=2,
            independent_events=1,
            censored_observations=0,
            unavailable_observations=0,
            recommendation="x",
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_graduation_censor_decision_documents_conservative_retain(
    tmp_path: Path,
) -> None:
    pump = _write_tiny_pump(tmp_path)
    decision = graduation_censor_decision(pump)
    assert decision["retain_conservative_v3_graduation_treatment"] is True
    assert decision["tracking_expires_at_is_proven_graduation_tracking_end"] is False
    assert decision["policy"]["use_tracking_expires_at_as_coverage_end"] is False
    inv = decision["investigation"]
    assert inv["uniform_plus_48h"] is True
    assert inv["known_issues_mentions_tracking_expires_at"] is False
    assert inv["readme_claims_39_day_continuous_collection"] is True
    # Fixture has one grad after expire (grad at +50h, expire at +48h).
    assert inv["grads_after_tracking_expires"] == 1
