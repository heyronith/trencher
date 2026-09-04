"""World-model recovery tests — mocked; zero live bulk queries."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
from click.testing import CliRunner

from trencher.cli import main
from trencher.data.feasibility.concepts import SYSTEM_PROGRAM_WALLET
from trencher.data.world_model.acquire.gecko import GeckoClient, run_gecko_acquisition
from trencher.data.world_model.acquire.local_corpus import run_local_acquisition
from trencher.data.world_model.acquire.rpc import SolanaRpc, run_rpc_acquisition
from trencher.data.world_model.pipeline import run_world_model_pipeline
from trencher.data.world_model.probe import probe_solarchive, run_source_probe


def _write_mini_corpus(external: Path) -> None:
    pump = external / "pumpfun_memecoin_corpus"
    trades = pump / "trades"
    trades.mkdir(parents=True)
    tokens = pa.table(
        {
            "mint": ["m1", "m2", "m3", "m_outage"],
            "detected_at": [
                datetime(2026, 6, 5, tzinfo=UTC),
                datetime(2026, 6, 6, tzinfo=UTC),
                datetime(2026, 6, 7, tzinfo=UTC),
                datetime(2026, 7, 3, tzinfo=UTC),
            ],
            "creator": ["c1", "c2", "c1", "c3"],
            "graduated_at": [datetime(2026, 6, 8, tzinfo=UTC), None, None, None],
            "is_mayhem_mode": [False, False, True, False],
            "top10_pct_suspect": [False, False, False, False],
            "initial_top10_pct_corrected": [10.0, 20.0, 30.0, 40.0],
            "dev_buy_pct_corrected": [1.0, 2.0, 3.0, 4.0],
            "supply_bug_corrected": [False, False, True, False],
            "is_training_ready": [True, True, True, True],
            "initial_holder_count": [4, 8, 20, 5],
            "initial_buy_sol": [0.1, 0.5, 2.0, 0.2],
        }
    )
    pq.write_table(tokens, pump / "tokens.parquet")
    pq.write_table(
        pa.table(
            {
                "mint": ["m1", "mX"],
                "migrated_at": [
                    datetime(2026, 6, 8, tzinfo=UTC),
                    datetime(2026, 6, 9, tzinfo=UTC),
                ],
                "seconds_to_graduation": [10.0, 20.0],
                "pool_address": ["RealPool111", "synthetic_graduation_queue"],
            }
        ),
        pump / "migrations.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": ["m1", "m1", "m2", "m2"],
                "tx_signature": ["sigA", "sigB", "sigC", "sigD"],
                "event_time": [
                    datetime(2026, 6, 5, 1, tzinfo=UTC),
                    datetime(2026, 6, 5, 2, tzinfo=UTC),
                    datetime(2026, 6, 6, 1, tzinfo=UTC),
                    datetime(2026, 6, 6, 3, tzinfo=UTC),
                ],
                "user_wallet": ["w1", "w2", SYSTEM_PROGRAM_WALLET, "w3"],
                "is_buy": [True, False, True, False],
                "sol_amount": [1.0, 2.0, 3.0, 0.4],
                "token_amount": [10.0, 20.0, 30.0, 5.0],
                "price_sol": [0.1, 0.2, 0.3, 0.05],
                "v_sol_bonding_curve": [30.0, 31.0, 32.0, 33.0],
                "v_tokens_bonding_curve": [1e9, 9e8, 8e8, 7e8],
                "curve_pct_depleted": [1.0, 2.0, 3.0, 10.0],
            }
        ),
        trades / "trades-00000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "mint": ["m1"],
                "outcome_label": ["dead"],
                "rug_detected": [False],
            }
        ),
        pump / "postgard_outcomes.parquet",
    )
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
    cohort = (
        external / "red_cohort_2026" / "extracted" / "RED-COHORT-2026-v1"
    )
    cohort.mkdir(parents=True)
    (cohort / "sniper_cohorts.jsonl").write_text(
        json.dumps({"cohort_id": "c1", "wallets": ["w1"], "mints_hit": [{"mint": "m1"}]})
        + "\n",
        encoding="utf-8",
    )
    # Solarchive indexes with no 2026 coverage
    txs = external / "solarchive" / "txs"
    txs.mkdir(parents=True)
    (txs / "index.json").write_text(
        json.dumps(
            {
                "partitions": [
                    {
                        "partition": "2025-12-01",
                        "total_size_bytes": 1000,
                        "file_count": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (external / "solarchive" / "accounts").mkdir(parents=True)


def test_cli_probe_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["data", "probe-world-model-sources", "--help"])
    assert result.exit_code == 0


def test_probe_solarchive_no_2026(tmp_path: Path) -> None:
    external = tmp_path / "external"
    _write_mini_corpus(external)
    result = probe_solarchive(external / "solarchive")
    assert result["txs_coverage_complete"] is False
    assert result["decision"] == "skip_no_corpus_coverage"
    assert result["estimated_download_gib_if_present"] is not None


def test_probe_skip_live(tmp_path: Path) -> None:
    external = tmp_path / "external"
    manifests = tmp_path / "manifests"
    _write_mini_corpus(external)
    payload = run_source_probe(
        external_root=external, manifest_dir=manifests, skip_live=True
    )
    assert payload["bigquery"]["decision"] == "skipped_for_ci"
    assert payload["geckoterminal"]["decision"] == "skipped_for_ci"
    assert (manifests / "source_probe.json").exists()


def test_local_acquisition_and_pipeline_skip_live(tmp_path: Path) -> None:
    external = tmp_path / "external"
    manifests = tmp_path / "manifests"
    world = tmp_path / "world"
    _write_mini_corpus(external)
    result = run_world_model_pipeline(
        external_root=external,
        manifest_dir=manifests,
        world_root=world,
        skip_live=True,
        rpc_max_signatures=10,
        gecko_max_pools=1,
        max_trade_batches=2,
    )
    assert result.ok
    assert (manifests / "source_probe.json").exists()
    assert (manifests / "extraction_plan.json").exists()
    assert (manifests / "WORLD_MODEL_DATA.md").exists()
    md = (manifests / "WORLD_MODEL_DATA.md").read_text(encoding="utf-8")
    assert "Genuine wallet funding" not in md
    assert "native-SOL" in md or "True native-SOL" in md
    assert (world / "graph" / "edges_wallet_token.parquet").exists()
    local = run_local_acquisition(
        external_root=external,
        world_root=world / "again",
        rich_subset_max_mints=10,
        rpc_max_signatures=10,
        max_trade_batches=1,
    )
    assert local["layers"]["rich_subset"]["mint_count"] == 3
    assert local["layers"]["rich_subset"]["quality_flag"] == "outcome_independent_stratified"
    proof = local["layers"]["rich_subset"]["bias_proof"]
    assert proof["jul3_included"] is False
    assert proof["subset_non_graduated"] >= 1
    assert proof["selection_is_not_graduation_proxy"] is True
    sigs = pq.read_table(world / "again" / "microstructure" / "signature_targets.parquet")
    mints = set(sigs.column("mint").to_pylist())
    assert "m1" in mints and "m2" in mints


def test_gecko_and_rpc_mocked(tmp_path: Path) -> None:
    world = tmp_path / "world"
    micro = world / "microstructure"
    micro.mkdir(parents=True)
    pq.write_table(
        pa.table({"mint": ["m1"], "tx_signature": ["sigA"]}),
        micro / "signature_targets.parquet",
    )

    class FakeSession:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def get(self, url: str, params=None, timeout=60):
            resp = MagicMock()
            resp.status_code = 200
            if "ohlcv" in url:
                resp.json.return_value = {
                    "data": {
                        "attributes": {
                            "ohlcv_list": [[1780704000, 1, 2, 0.5, 1.5, 100]]
                        }
                    }
                }
            else:
                resp.json.return_value = {"data": {}}
            return resp

    gecko = run_gecko_acquisition(
        world_root=world,
        pools=["RealPool111"],
        max_pools=1,
        client=GeckoClient(session=FakeSession(), min_interval=0),
    )
    assert gecko["api_calls"] >= 1

    rpc_client = SolanaRpc(min_interval=0)

    def fake_tx(_method: str, _params: list):
        return {
            "slot": 1,
            "blockTime": 1,
            "meta": {
                "err": None,
                "fee": 5000,
                "computeUnitsConsumed": 100,
                "preBalances": [100, 50],
                "postBalances": [90, 60],
                "logMessages": ["ok"],
                "innerInstructions": [],
            },
            "transaction": {
                "message": {
                    "accountKeys": ["a", "b"],
                }
            },
        }

    def fake_call_batch(calls):
        return [fake_tx(method, params) for method, params in calls]

    with (
        patch.object(rpc_client, "call", side_effect=fake_tx),
        patch.object(rpc_client, "call_batch", side_effect=fake_call_batch),
    ):
        rpc = run_rpc_acquisition(
            world_root=world, max_signatures=5, rpc=rpc_client, skip_live=False
        )
    assert rpc["found"] == 1
    assert rpc["signature_match_rate"] == 1.0


def test_native_sol_parser_ignores_post_t_and_spl() -> None:
    from trencher.data.world_model.acquire.funding import parse_system_sol_transfers
    from trencher.data.world_model.constants import SYSTEM_PROGRAM_ID

    t = 1_000.0
    post = {
        "blockTime": 1_001,
        "slot": 9,
        "transaction": {
            "message": {
                "instructions": [
                    {
                        "program": "system",
                        "programId": SYSTEM_PROGRAM_ID,
                        "parsed": {
                            "type": "transfer",
                            "info": {
                                "source": "aaa",
                                "destination": "bbb",
                                "lamports": 5,
                            },
                        },
                    }
                ]
            }
        },
        "meta": {"innerInstructions": []},
    }
    assert parse_system_sol_transfers("sig", post, max_block_time=t) == []

    pre = dict(post)
    pre["blockTime"] = 999
    edges = parse_system_sol_transfers("sig", pre, max_block_time=t)
    assert len(edges) == 1
    assert edges[0]["edge_type"] == "system_sol_transfer"
    assert edges[0]["lamports"] == 5

    spl_only = {
        "blockTime": 999,
        "slot": 9,
        "transaction": {
            "message": {
                "instructions": [
                    {
                        "program": "spl-token",
                        "parsed": {
                            "type": "transfer",
                            "info": {
                                "source": "aaa",
                                "destination": "bbb",
                                "amount": "1",
                            },
                        },
                    }
                ]
            }
        },
        "meta": {"innerInstructions": []},
    }
    assert parse_system_sol_transfers("sig", spl_only, max_block_time=t) == []


def test_rpc_token_balance_ownership_edges() -> None:
    from trencher.data.world_model.acquire.rpc import (
        _parse_tx,
        extract_token_ownership_edges,
    )

    result = {
        "slot": 1,
        "blockTime": 100,
        "meta": {
            "err": None,
            "fee": 5000,
            "computeUnitsConsumed": 1,
            "preBalances": [1, 1],
            "postBalances": [1, 1],
            "preTokenBalances": [
                {
                    "accountIndex": 0,
                    "mint": "mintA",
                    "owner": "w1",
                    "uiTokenAmount": {"uiAmount": 0.0, "decimals": 6, "amount": "0"},
                }
            ],
            "postTokenBalances": [
                {
                    "accountIndex": 0,
                    "mint": "mintA",
                    "owner": "w1",
                    "uiTokenAmount": {
                        "uiAmount": 10.0,
                        "decimals": 6,
                        "amount": "10000000",
                    },
                }
            ],
            "logMessages": [],
            "innerInstructions": [],
        },
        "transaction": {"message": {"accountKeys": ["w1", "w2"]}},
    }
    parsed = _parse_tx("sigX", "mintA", result)
    assert parsed["token_balance_change_count"] == 1
    edges = extract_token_ownership_edges([parsed])
    assert len(edges) == 1
    assert edges[0]["edge_type"] == "rpc_token_balance_delta"
    assert edges[0]["owner"] == "w1"
    assert edges[0]["delta_ui_amount"] == 10.0

