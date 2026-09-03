"""Dune API client wrapper (mockable)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd
from dune_client.client import DuneClient
from dune_client.query import QueryBase
from dune_client.types import QueryParameter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class DuneExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueryExecutionResult:
    execution_id: str | None
    query_id: int | None
    row_count: int
    dataframe: pd.DataFrame


class DuneRunner(Protocol):
    def run_sql(
        self,
        sql: str,
        *,
        performance: str = "medium",
        query_id: int | None = None,
        parameters: dict[str, str] | None = None,
    ) -> QueryExecutionResult: ...


def _query_parameters(parameters: dict[str, str] | None) -> list[QueryParameter] | None:
    if not parameters:
        return None
    return [
        QueryParameter.text_type(name=key, value=str(value))
        for key, value in parameters.items()
    ]


class LiveDuneRunner:
    """Production Dune runner using dune-client."""

    def __init__(self, api_key: str | None = None, *, performance: str | None = None) -> None:
        key = api_key or os.environ.get("DUNE_API_KEY")
        if not key:
            raise DuneExecutionError("DUNE_API_KEY is not set")
        self._client = DuneClient(api_key=key)
        self._performance = performance or os.environ.get("TRENCHER_DUNE_PERFORMANCE", "medium")

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError, DuneExecutionError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def run_sql(
        self,
        sql: str,
        *,
        performance: str | None = None,
        query_id: int | None = None,
        parameters: dict[str, str] | None = None,
    ) -> QueryExecutionResult:
        perf = performance or self._performance
        try:
            if query_id is not None:
                query = QueryBase(
                    query_id=query_id,
                    params=_query_parameters(parameters),
                )
                response = self._client.run_query(query=query, performance=perf)
                rows = response.result.rows if response.result else []
                df = pd.DataFrame(rows)
                execution_id = response.execution_id
            else:
                response = self._client.run_sql(query_sql=sql, performance=perf)
                execution_id = response.execution_id
                rows = response.result.rows if response.result else []
                df = pd.DataFrame(rows)
            return QueryExecutionResult(
                execution_id=str(execution_id) if execution_id else None,
                query_id=query_id,
                row_count=len(df),
                dataframe=df,
            )
        except DuneExecutionError:
            raise
        except Exception as exc:
            if isinstance(exc, (ConnectionError, TimeoutError)):
                raise
            raise DuneExecutionError(str(exc)) from exc


class MockDuneRunner:
    """In-memory runner for tests."""

    def __init__(self, responses: dict[str, pd.DataFrame] | None = None) -> None:
        self.responses = responses or {}
        self.default_response: pd.DataFrame | None = None
        self.calls: list[dict[str, Any]] = []

    def run_sql(
        self,
        sql: str,
        *,
        performance: str = "medium",
        query_id: int | None = None,
        parameters: dict[str, str] | None = None,
    ) -> QueryExecutionResult:
        self.calls.append(
            {
                "sql": sql,
                "performance": performance,
                "query_id": query_id,
                "parameters": parameters,
            }
        )
        key = str(query_id) if query_id is not None else sql.strip()
        df = self.responses.get(key)
        if df is None:
            df = self.default_response
        if df is None and len(self.responses) == 1:
            df = next(iter(self.responses.values()))
        if df is None:
            df = pd.DataFrame()
        return QueryExecutionResult(
            execution_id="mock-exec-1",
            query_id=query_id,
            row_count=len(df),
            dataframe=df.copy(),
        )
