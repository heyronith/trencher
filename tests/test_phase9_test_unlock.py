# ruff: noqa: E402
"""Phase-9 unlock guards: lock-before-TEST, window fn, no TEST without lock."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.review.phase8.features import FeatureBuildRefusal, ensure_broad_features
from scripts.review.phase8.stage_b import StageBRefusal, ensure_stage_b_preds
from scripts.review.phase8.windows import event_onset_window

from trencher.data.world_model.splits.q85_unlock import (
    clear_q85_test_unlock,
    q85_test_unlocked,
    set_q85_test_unlock,
)


@pytest.fixture(autouse=True)
def _clear_unlock():
    clear_q85_test_unlock()
    yield
    clear_q85_test_unlock()


def test_event_onset_window_val_is_q70_q85() -> None:
    lo, hi = event_onset_window("val")
    cut = json.loads(
        (
            REPO / "data/external/world_model/splits/split_v1/cutoffs.json"
        ).read_text(encoding="utf-8")
    )["quantiles_unix"]
    assert lo == pytest.approx(float(cut["q70"]))
    assert hi == pytest.approx(float(cut["q85"]))
    assert lo < hi


def test_event_onset_window_test_is_q85_to_max() -> None:
    lo, hi = event_onset_window("test")
    cut = json.loads(
        (
            REPO / "data/external/world_model/splits/split_v1/cutoffs.json"
        ).read_text(encoding="utf-8")
    )["quantiles_unix"]
    assert lo == pytest.approx(float(cut["q85"]))
    assert hi > lo
    assert hi > float(cut["q85"])


def test_test_feature_branch_refuses_without_lock() -> None:
    assert not q85_test_unlocked()
    with pytest.raises(FeatureBuildRefusal, match="TEST_ACCESSED.lock"):
        ensure_broad_features(split="test")


def test_test_stage_b_refuses_without_lock() -> None:
    assert not q85_test_unlocked()
    with pytest.raises(StageBRefusal, match="TEST_ACCESSED.lock"):
        ensure_stage_b_preds(
            split="test",
            checkpoint=REPO / "scratch/test_models_v2/stage_b_broad/latest.pt",
        )


def test_lock_written_before_any_test_file_open(tmp_path: Path) -> None:
    """simulate run_test_once order: lock → unlock → then TEST open allowed."""
    from scripts.review.phase7.score_test_once import (
        check_test_once_dir_empty,
        write_access_lock,
    )

    out_dir = tmp_path / "test_once"
    check_test_once_dir_empty(out_dir)
    opened: list[str] = []

    real_open = open

    def tracking_open(path, *args, **kwargs):  # noqa: ANN001
        p = str(path)
        if "TEST_ACCESSED.lock" not in p and "test" in p.lower():
            opened.append(p)
            if not (out_dir / "TEST_ACCESSED.lock").exists():
                raise AssertionError(f"TEST path opened before lock: {p}")
        return real_open(path, *args, **kwargs)

    lock = write_access_lock(out_dir, head_sha="deadbeef" * 5)
    assert lock.exists()
    set_q85_test_unlock(lock)
    assert q85_test_unlocked()

    fake_test = tmp_path / "fake_test_rows.parquet"
    with patch("builtins.open", tracking_open), open(
        fake_test, "w", encoding="utf-8"
    ) as f:
        f.write("x")
    assert any("fake_test" in p for p in opened)


def test_run_test_once_order_mocks_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """verify_prereg → empty → lock → unlock → pipeline; no TEST before lock."""
    import scripts.review.phase8.score_test_once as scorer

    order: list[str] = []
    out_dir = tmp_path / "out"

    def fake_verify(**kwargs):  # noqa: ANN003
        order.append("verify")
        return {
            "head_sha": "a" * 40,
            "preregistration_sha": "b" * 40,
            "protocol_match": "exact_prereg_blob",
        }

    def fake_empty(d):  # noqa: ANN001
        order.append("empty_check")
        assert not (Path(d) / "TEST_ACCESSED.lock").exists()

    real_write = scorer.write_access_lock

    def fake_write(d, *, head_sha: str):
        order.append("write_lock")
        return real_write(d, head_sha=head_sha)

    def fake_pipeline(**kwargs):  # noqa: ANN003
        order.append("pipeline")
        assert q85_test_unlocked()
        assert (out_dir / "TEST_ACCESSED.lock").exists()
        return {
            "split": "test",
            "q85_test_accessed": True,
            "by_head": {},
            "twox": {},
            "model_hashes": {},
            "runtime_sec": 0.1,
            "peak_rss_mb": 1.0,
            "pred_artifacts": {},
            "feature_meta": {},
            "stage_b_meta": {},
        }

    monkeypatch.setattr(scorer, "verify_preregistration", fake_verify)
    monkeypatch.setattr(scorer, "check_test_once_dir_empty", fake_empty)
    monkeypatch.setattr(scorer, "write_access_lock", fake_write)
    monkeypatch.setattr(scorer, "run_split_pipeline", fake_pipeline)
    monkeypatch.setattr(
        scorer,
        "build_test_once_report",
        lambda **k: ({"q85_test_accessed": True, **k["scored"]}, "# md\n"),
    )

    scorer.run_test_once(
        protocol=tmp_path / "proto.md",
        prereg_sha="b" * 40,
        out_dir=out_dir,
        split="test",
        allow_dirty=True,
        run_bootstrap=False,
    )
    assert order == ["verify", "empty_check", "write_lock", "pipeline"]
    assert (out_dir / "TEST_ACCESSED.lock").exists()


def test_test_branch_unreachable_via_features_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRENCHER_Q85_TEST_UNLOCK", raising=False)
    with pytest.raises(FeatureBuildRefusal):
        ensure_broad_features(split="test")
