"""Retry behavior tests."""

from __future__ import annotations

import pandas as pd
import pytest

from trencher.dune.client import DuneExecutionError, LiveDuneRunner, MockDuneRunner


def test_mock_runner_records_calls() -> None:
    runner = MockDuneRunner(responses={"SELECT 1": pd.DataFrame({"a": [1]})})
    result = runner.run_sql("SELECT 1")
    assert result.row_count == 1
    assert runner.calls[0]["sql"] == "SELECT 1"


def test_live_runner_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DUNE_API_KEY", raising=False)
    with pytest.raises(DuneExecutionError, match="DUNE_API_KEY"):
        LiveDuneRunner()


def test_live_runner_retries_on_transient(monkeypatch) -> None:
    calls = {"n": 0}

    class FakeResult:
        execution_id = "exec-1"
        result = type("R", (), {"rows": [{"x": 1}]})()

    class FlakyClient:
        def run_sql(self, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient")
            return FakeResult()

    monkeypatch.setenv("DUNE_API_KEY", "test-key")
    runner = LiveDuneRunner()
    runner._client = FlakyClient()
    result = runner.run_sql("SELECT 1")
    assert result.row_count == 1
    assert calls["n"] == 3
