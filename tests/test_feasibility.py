"""Feasibility audit tests using tiny fixtures (no full corpus)."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from click.testing import CliRunner

from trencher.cli import main
from trencher.data.feasibility.assessment import label_feasibility, reconstruction_assessment
from trencher.data.feasibility.audit import run_feasibility_audit
from trencher.data.feasibility.concepts import SYSTEM_PROGRAM_WALLET
from trencher.data.feasibility.quality import audit_tokens_table, audit_trades_quality


def _write_fixture_corpus(root: Path) -> None:
    pump = root / "pumpfun_memecoin_corpus"
    trades = pump / "trades"
    trades.mkdir(parents=True)
    (pump / "KNOWN_ISSUES.md").write_text(
        "# Known Issues\n\n### 3.1 sol_amount\n### 3.3 System Program\n",
        encoding="utf-8",
    )

    tokens = pa.table(
        {
            "mint": ["m1", "m2", "m3"],
            "detected_at": [
                datetime(2026, 6, 5, tzinfo=UTC),
                datetime(2026, 6, 6, tzinfo=UTC),
                datetime(2026, 6, 7, tzinfo=UTC),
            ],
            "creator": ["c1", "c2", "c1"],
            "is_mayhem_mode": [True, False, False],
            "top10_pct_suspect": [False, True, None],
            "supply_bug_corrected": [True, False, False],
            "graduated_at": [datetime(2026, 6, 8, tzinfo=UTC), None, None],
            "is_training_ready": [True, True, False],
            "initial_top10_pct": [150.0, 50.0, 10.0],
            "initial_top10_pct_corrected": [75.0, 50.0, 10.0],
            "entry_price_20s_usd": [1.0, None, None],
            "entry_price_30s_usd": [1.1, None, None],
            "entry_price_1m_usd": [1.2, None, None],
        }
    )
    pq.write_table(tokens, pump / "tokens.parquet")

    trade_rows = pa.table(
        {
            "mint": ["m1", "m1", "m2", "m3"],
            "event_time": [
                datetime(2026, 6, 5, 1, tzinfo=UTC),
                datetime(2026, 6, 5, 2, tzinfo=UTC),
                datetime(2026, 6, 9, 1, tzinfo=UTC),
                datetime(2026, 6, 10, 1, tzinfo=UTC),
            ],
            "sol_amount": [1.0, 1000000.0, None, 1.0],
            "token_amount": [1.0, 1.0, 1.0, 1.0],
            "price_sol": [1.0, 1.0, None, 1.0],
            "user_wallet": ["w1", "w2", SYSTEM_PROGRAM_WALLET, "w3"],
            "curve_pct_depleted": [10.0, 110.0, 50.0, 20.0],
        }
    )
    pq.write_table(trade_rows, trades / "trades-00000.parquet")

    snaps = pa.table({"curve_pct_depleted_eob": [10.0, 105.0, 20.0]})
    pq.write_table(snaps, pump / "snapshots.parquet")

    migrations = pa.table(
        {
            "mint": ["m1", "mX"],
            "migrated_at": [
                datetime(2026, 6, 8, tzinfo=UTC),
                datetime(2026, 6, 8, tzinfo=UTC),
            ],
            "seconds_to_graduation": [100, 200],
            "pool_address": ["realpool", "synthetic_graduation_queue"],
        }
    )
    pq.write_table(migrations, pump / "migrations.parquet")

    wallet_stats = pa.table(
        {
            "wallet": ["w1"],
            "first_seen_at": [datetime(2026, 6, 5, tzinfo=UTC)],
            "last_seen_at": [datetime(2026, 6, 5, tzinfo=UTC)],
            "tokens_traded": [1],
            "graduated_tokens_traded": [1],
            "total_trades": [1],
            "total_buy_volume_sol": [1.0],
            "total_sell_volume_sol": [0.0],
        }
    )
    pq.write_table(wallet_stats, pump / "wallet_stats.parquet")

    postgard = pa.table(
        {
            "mint": ["m1"],
            "outcome_label": ["dead"],
            "rug_detected": [False],
            "computed_at": [datetime(2026, 6, 9, tzinfo=UTC)],
        }
    )
    pq.write_table(postgard, pump / "postgard_outcomes.parquet")
    pq.write_table(
        pa.table(
            {
                "mint": ["m1"],
                "snapshot_time": [datetime(2026, 6, 9, tzinfo=UTC)],
                "price_usd": [1.0],
            }
        ),
        pump / "postgard_snapshots.parquet",
    )

    red = root / "red_pump_2026"
    red.mkdir(parents=True)
    with gzip.open(red / "red_pump_2026_v1_launches.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"mint": "m1", "t": 1, "symbol": "A"}) + "\n")
        handle.write(json.dumps({"mint": "m2", "t": 2, "symbol": "B"}) + "\n")
        handle.write(json.dumps({"mint": "rp_only", "t": 3, "symbol": "C"}) + "\n")
    with gzip.open(red / "red_pump_2026_v1_outcomes.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            "graduated_at,mint,symbol,outcome,minutes_to_outcome,"
            "initial_market_cap_sol,final_market_cap_sol,note,"
            "created_at_chain_iso,minutes_to_outcome_chain,detection_lag_min\n"
        )
        handle.write("2026-06-08T00:00:00Z,m1,A,GRADUATED,10,30,40,,2026-06-05T00:00:00Z,10,0.1\n")
        handle.write("2026-06-08T00:00:00Z,m2,B,TIMEOUT,1440,30,20,,2026-06-06T00:00:00Z,1440,0.1\n")

    cohort_dir = root / "red_cohort_2026" / "extracted" / "RED-COHORT-2026-v1"
    cohort_dir.mkdir(parents=True)
    (cohort_dir / "sniper_cohorts.jsonl").write_text(
        json.dumps(
            {
                "cohort_id": "c1",
                "wallets": ["w1", "w9"],
                "mints_hit": [
                    {"mint": "m1", "n_wallets": 2},
                    {"mint": "m2", "n_wallets": 1},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_cli_audit_feasibility_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["data", "audit-feasibility", "--help"])
    assert result.exit_code == 0
    assert "audit-feasibility" in result.output


def test_audit_on_fixture_corpus(tmp_path: Path) -> None:
    external = tmp_path / "external"
    manifests = tmp_path / "manifests"
    _write_fixture_corpus(external)
    result = run_feasibility_audit(
        external_root=external,
        manifest_root=manifests,
        skip_heavy_trade_scan=False,
    )
    assert result.ok, result.errors
    assert (result.report_dir / "FEASIBILITY.md").exists()
    assert (result.report_dir / "verdict.json").exists()
    assert result.summary["verdict"]["can_build_research_grade_model"] is True
    assert result.summary["verdict"]["solarchive_supplementation_required"] is False
    assert result.summary["quality"]["tokens"]["rows"] == 3
    assert result.summary["quality"]["trades"]["system_program_trade_rows"] == 1
    assert result.summary["quality"]["postgard_outcomes"]["has_rug_label_value"] is False


def test_tokens_and_trades_quality_helpers(tmp_path: Path) -> None:
    external = tmp_path / "external"
    _write_fixture_corpus(external)
    tokens = audit_tokens_table(external / "pumpfun_memecoin_corpus" / "tokens.parquet")
    assert tokens["mayhem_tokens"] == 1
    assert tokens["top10_pct_suspect_true"] == 1
    trades = audit_trades_quality(
        external / "pumpfun_memecoin_corpus" / "trades"
    )
    assert trades["rows"] == 4
    assert trades["null_price_sol_rows"] == 1
    assert trades["curve_pct_depleted_out_of_0_100"] == 1


def test_assessment_documents_label_support() -> None:
    labels = label_feasibility()
    recon = reconstruction_assessment()
    assert labels["mfe_mae"]["supported"] == "full"
    assert recon["post_graduation_trajectory"]["supported"] == "partial"
