"""Unit tests for Phase-7 score_test_once refusal gates (synthetic; no TEST IO)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_scorer_mod():
    path = REPO / "scripts" / "review" / "phase7" / "score_test_once.py"
    # Ensure repo root (and package imports used by the module) resolve.
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(
        "phase7_score_test_once", path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def scorer():
    return _load_scorer_mod()


def test_refuse_when_lock_exists(scorer, tmp_path: Path) -> None:
    lock = tmp_path / "TEST_ACCESSED.lock"
    lock.write_text("{}", encoding="utf-8")
    with pytest.raises(scorer.ScorerRefusal, match="TEST_ACCESSED.lock"):
        scorer.check_test_once_dir_empty(tmp_path)


def test_refuse_when_metrics_exist(scorer, tmp_path: Path) -> None:
    (tmp_path / "TEST_ONCE_REPORT.json").write_text("{}", encoding="utf-8")
    with pytest.raises(scorer.ScorerRefusal, match="metrics"):
        scorer.check_test_once_dir_empty(tmp_path)


def test_write_lock_then_second_write_refuses(scorer, tmp_path: Path) -> None:
    scorer.write_access_lock(tmp_path, head_sha="deadbeef")
    assert (tmp_path / "TEST_ACCESSED.lock").exists()
    with pytest.raises(scorer.ScorerRefusal, match="already exists"):
        scorer.write_access_lock(tmp_path, head_sha="deadbeef")


def test_refuse_dirty_tree(scorer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_git(args: list[str], *, cwd: Path = REPO, strip: bool = True) -> str:
        if args[:1] == ["rev-parse"]:
            return "abc123"
        if args[:2] == ["status", "--porcelain"]:
            return " M something.py"
        raise AssertionError(args)

    monkeypatch.setattr(scorer, "_git", fake_git)
    monkeypatch.setattr(scorer.subprocess, "call", lambda *a, **k: 0)
    proto = tmp_path / "PROTO.md"
    proto.write_text("hello", encoding="utf-8")
    with pytest.raises(scorer.ScorerRefusal, match="dirty"):
        scorer.verify_preregistration(
            protocol_path=proto,
            prereg_sha="abc123",
            repo=REPO,
            allow_dirty=False,
            protocol_git_path="manifests/world_model_data/evaluation/TEST_PROTOCOL_V2.md",
        )


def test_refuse_sha_not_ancestor(
    scorer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_git(args: list[str], *, cwd: Path = REPO, strip: bool = True) -> str:
        if args[:1] == ["rev-parse"]:
            return "abc123"
        if args[:2] == ["status", "--porcelain"]:
            return ""
        return ""

    monkeypatch.setattr(scorer, "_git", fake_git)
    monkeypatch.setattr(scorer.subprocess, "call", lambda *a, **k: 1)
    proto = tmp_path / "PROTO.md"
    proto.write_text("hello", encoding="utf-8")
    with pytest.raises(scorer.ScorerRefusal, match="not an ancestor"):
        scorer.verify_preregistration(
            protocol_path=proto,
            prereg_sha="0000000000000000000000000000000000000000",
            allow_dirty=False,
            protocol_git_path="manifests/world_model_data/evaluation/TEST_PROTOCOL_V2.md",
        )


def test_refuse_modified_protocol(
    scorer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proto = tmp_path / "PROTO.md"
    proto.write_text("working-tree-version\n", encoding="utf-8")

    def fake_git(args: list[str], *, cwd: Path = REPO, strip: bool = True) -> str:
        if args[:1] == ["rev-parse"]:
            return "abc123"
        if args[:2] == ["status", "--porcelain"]:
            return ""
        if args[:1] == ["show"]:
            return "blob-at-sha\n"
        return ""

    monkeypatch.setattr(scorer, "_git", fake_git)
    monkeypatch.setattr(scorer.subprocess, "call", lambda *a, **k: 0)
    with pytest.raises(scorer.ScorerRefusal, match="byte-identical"):
        scorer.verify_preregistration(
            protocol_path=proto,
            prereg_sha="abc123",
            allow_dirty=False,
            protocol_git_path="manifests/world_model_data/evaluation/TEST_PROTOCOL_V2.md",
        )


def test_git_available_for_repo() -> None:
    # Sanity: real git works in the repo (used by scorer gates).
    out = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    assert len(out) == 40
