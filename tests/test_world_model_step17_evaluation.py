"""Step-17 evaluation framework tests (CPU-only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trencher.world_model.config import tiny_smoke_config
from trencher.world_model.evaluation.ablations import scientific_ablation_configs
from trencher.world_model.evaluation.bootstrap import clustered_indices, paired_delta_bootstrap
from trencher.world_model.evaluation.gru_control import EventGRUControl, count_control_parameters
from trencher.world_model.evaluation.lineage import (
    stage_b_lineage,
    stage_c_lineage,
    verify_approved_stage_c_identity,
)
from trencher.world_model.evaluation.masking import (
    active_modalities_for_phase,
    mask_batch_modalities,
    rank_degradation,
)
from trencher.world_model.evaluation.metrics import barrier_metrics, binary_masked_metrics
from trencher.world_model.evaluation.modes import assert_step17_mode, provenance_q85_flag
from trencher.world_model.evaluation.paired import (
    align_by_observation_hash,
    population_compatibility_report,
)
from trencher.world_model.evaluation.preflight import build_preflight
from trencher.world_model.evaluation.provenance import sha256_json, write_sha256sums
from trencher.world_model.evaluation.report import (
    placeholder_report_payload,
    render_markdown_report,
)
from trencher.world_model.model import WorldModel
from trencher.world_model.split_guard import PermanentTestAccessError


@pytest.mark.parametrize("alias", ["q85_test", "test", "permanent_test", "q85", "holdout_test"])
def test_step17_refuses_test_aliases(alias: str) -> None:
    with pytest.raises(PermanentTestAccessError):
        assert_step17_mode(alias)


def test_permanent_val_mode_allowed() -> None:
    assert assert_step17_mode("permanent_val_eval") == "permanent_val_eval"


def test_provenance_q85_flag_abort_on_true() -> None:
    with pytest.raises(PermanentTestAccessError):
        provenance_q85_flag(accessed=True)
    assert provenance_q85_flag(accessed=False)["q85_test_accessed"] is False


def test_zero_support_and_zero_positives_non_identifiable() -> None:
    empty = binary_masked_metrics(np.array([]), np.array([]), np.array([]))
    assert empty["identifiable"] is False
    y = np.zeros(10, dtype=int)
    p = np.linspace(0.1, 0.9, 10)
    m = np.ones(10)
    zpos = binary_masked_metrics(y, p, m)
    assert zpos["identifiable"] is False
    assert zpos["non_identifiable_reason"] == "zero_positives"
    assert zpos["auroc"] is None
    assert zpos["ap"] is None


def test_paired_alignment_and_population_mismatch() -> None:
    a = np.array([1, 2, 3, 4], dtype=np.int64)
    b = np.array([3, 2, 9], dtype=np.int64)
    ia, ib = align_by_observation_hash(a, b)
    assert set(a[ia].tolist()) == {2, 3}
    assert b[ib].tolist() == a[ia].tolist()
    report = population_compatibility_report(
        population_a="broad", population_b="rich", n_a=100, n_b=10, n_paired=0
    )
    assert report["comparable"] is False
    assert "mismatch" in (report["mismatch_note"] or "").lower() or "differ" in (
        report["mismatch_note"] or ""
    ).lower()


def test_clustered_bootstrap_determinism() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=50)
    b = a + 0.1
    clusters = np.repeat(np.arange(10), 5)
    r1 = paired_delta_bootstrap(a, b, cluster_ids=clusters, n_boot=200, seed=123)
    r2 = paired_delta_bootstrap(a, b, cluster_ids=clusters, n_boot=200, seed=123)
    assert r1 == r2
    assert r1["n_clusters"] == 10
    assert r1["delta"] == pytest.approx(0.1, abs=1e-9)
    # Resampling by cluster changes composition vs iid
    idx = clustered_indices(clusters, np.random.default_rng(7))
    assert idx.size == 50


def test_modality_masking_zeros_availability() -> None:
    batch = {
        "wallet_available": torch.ones(4),
        "graph_available": torch.ones(4),
        "micro_available": torch.ones(4),
        "own_available": torch.ones(4),
    }
    masked = mask_batch_modalities(batch, omit=["wallet_prior", "graph"])
    assert float(masked["wallet_available"].sum()) == 0.0
    assert float(masked["graph_available"].sum()) == 0.0
    assert float(masked["micro_available"].sum()) == 4.0
    # original unchanged
    assert float(batch["wallet_available"].sum()) == 4.0


def test_ablated_modalities_unavailable_in_forward() -> None:
    cfg = tiny_smoke_config()
    model = WorldModel(cfg)
    model.omit_modalities = {"graph", "wallet_prior"}  # type: ignore[attr-defined]
    b = 2
    batch = {
        "event_x": torch.randn(b, 8, cfg.event.n_channels),
        "event_times": torch.linspace(0, 1, 8).unsqueeze(0).expand(b, -1),
        "event_mask": torch.ones(b, 8, dtype=torch.bool),
        "event_available": torch.ones(b),
        "curve_x": torch.randn(b, 4, cfg.curve.n_channels),
        "curve_times": torch.linspace(0, 1, 4).unsqueeze(0).expand(b, -1),
        "curve_mask": torch.ones(b, 4, dtype=torch.bool),
        "curve_available": torch.ones(b),
        "global_buckets": torch.randn(b, 4, cfg.global_temporal.n_bucket_channels),
        "global_times": torch.linspace(0, 1, 4).unsqueeze(0).expand(b, -1),
        "global_scalars": torch.randn(b, cfg.global_temporal.n_scalars),
        "global_mask": torch.ones(b, 4, dtype=torch.bool),
        "global_available": torch.ones(b),
        "micro_x": torch.randn(b, 4, cfg.microstructure.n_channels),
        "micro_times": torch.linspace(0, 1, 4).unsqueeze(0).expand(b, -1),
        "micro_mask": torch.ones(b, 4, dtype=torch.bool),
        "micro_available": torch.ones(b),
        "own_x": torch.randn(b, 4, cfg.ownership.n_channels),
        "own_times": torch.linspace(0, 1, 4).unsqueeze(0).expand(b, -1),
        "own_mask": torch.ones(b, 4, dtype=torch.bool),
        "own_available": torch.ones(b),
        "own_quality": torch.ones(b),
        "wallet_x": torch.randn(b, 4, cfg.wallet_prior.n_channels),
        "wallet_times": torch.linspace(0, 1, 4).unsqueeze(0).expand(b, -1),
        "wallet_mask": torch.ones(b, 4, dtype=torch.bool),
        "wallet_available": torch.ones(b),
        "graph_edge_feat": torch.randn(b, 4, cfg.graph.n_edge_channels),
        "graph_times": torch.linspace(0, 1, 4).unsqueeze(0).expand(b, -1),
        "graph_mask": torch.ones(b, 4, dtype=torch.bool),
        "graph_available": torch.ones(b),
    }
    active = active_modalities_for_phase(
        "rich",
        modalities_broad=cfg.modalities_broad,
        modalities_rich=cfg.modalities_rich,
        omit={"graph", "wallet_prior"},
    )
    vecs, avail, _ = model.encode_modalities(batch, active=active)
    assert "graph" not in vecs
    assert "wallet_prior" not in vecs
    assert "microstructure" in vecs
    out = model(batch, phase="rich")
    assert "explosion_logits" in out
    assert "graph" not in avail


def test_stage_lineage_hashes() -> None:
    b = stage_b_lineage()
    c = stage_c_lineage()
    assert b["checkpoint_sha256"].startswith("ccf27af7")
    assert c["checkpoint_sha256"].startswith("11bf593e")
    assert b["q85_test_accessed"] is False
    assert c["q85_test_accessed"] is False
    assert b["paired_ready_with_stage_c_rich"] is False
    verify_approved_stage_c_identity(
        checkpoint_sha256=c["checkpoint_sha256"],
        prediction_sha256=c["prediction_sha256"],
        metrics_sha256=c["metrics_sha256"],
        config_hash=c["config_hash"],
        runtime_git_sha=c["runtime_git_sha"],
    )
    with pytest.raises(RuntimeError):
        verify_approved_stage_c_identity(
            checkpoint_sha256="deadbeef",
            prediction_sha256=c["prediction_sha256"],
            metrics_sha256=c["metrics_sha256"],
            config_hash=c["config_hash"],
            runtime_git_sha=c["runtime_git_sha"],
        )


def test_artifact_sha_manifest(tmp_path: Path) -> None:
    f = tmp_path / "a.json"
    f.write_text(json.dumps({"q85_test_accessed": False}), encoding="utf-8")
    write_sha256sums(tmp_path, [f])
    assert (tmp_path / "SHA256SUMS").is_file()
    digest = sha256_json({"x": 1})
    assert len(digest) == 64


def test_report_generation() -> None:
    md = render_markdown_report(placeholder_report_payload())
    assert "Did Stage C improve over Stage B?" in md
    assert "q85" in md.lower()


def test_preflight_builds(tmp_path: Path) -> None:
    # Must run from repo root semantics
    repo = Path(__file__).resolve().parents[1]
    payload = build_preflight(repo_root=repo)
    assert payload["q85_test_accessed"] is False
    assert payload["stop_gate"]["do_not_provision_gpu"] is True
    assert payload["stage_c"]["checkpoint_sha256"].startswith("11bf593e")
    assert payload["stage_b"]["checkpoint_sha256"].startswith("ccf27af7")
    assert len(payload["q85_refusal_proof"]["aliases_tested"]) >= 3
    retrain = payload["proposed_ablation_gpu_runs"]
    assert len(retrain) == 4
    assert all(r["omit"] for r in retrain)
    full = [a for a in scientific_ablation_configs() if a["run_id"] == "stage_c_full"][0]
    assert full["retrain"] is False


def test_barrier_metrics_and_rank() -> None:
    y = np.array([0, 1, 2, 1, 0, 2])
    logits = np.eye(3)[y] * 5.0
    m = np.ones(6)
    out = barrier_metrics(y, logits, m)
    assert out["accuracy"] == 1.0
    ranked = rank_degradation(0.9, {"graph": 0.7, "wallet_prior": 0.85})
    assert ranked[0]["modality_or_config"] == "graph"


def test_gru_control_smaller_than_world_model() -> None:
    cfg = tiny_smoke_config()
    n_ctrl = count_control_parameters(cfg)
    n_wm = sum(p.numel() for p in WorldModel(cfg).parameters() if p.requires_grad)
    assert n_ctrl < n_wm
    model = EventGRUControl(cfg)
    b, t = 2, 8
    batch = {
        "event_x": torch.randn(b, t, cfg.event.n_channels),
        "event_mask": torch.ones(b, t),
        "curve_x": torch.randn(b, 4, cfg.curve.n_channels),
        "curve_mask": torch.ones(b, 4),
        "global_buckets": torch.randn(b, 4, cfg.global_temporal.n_bucket_channels),
        "global_scalars": torch.randn(b, cfg.global_temporal.n_scalars),
        "global_mask": torch.ones(b, 4),
    }
    out = model(batch)
    assert out["explosion_logits"].shape[0] == b
