"""Low-cost live Dune catalog/query preflight."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trencher.dune.client import DuneExecutionError, DuneRunner
from trencher.dune.config import (
    OPTIONAL_COL_BLOCK,
    DatasetConfig,
    load_datasets_config,
    load_sql_template,
    render_sql,
    resolve_data_root,
    save_enabled_tables,
    strip_optional_blocks,
)
from trencher.dune.doctor import DoctorReport

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "pumpdotfun_solana.pump_call_create": (
        "call_block_time",
        "call_block_slot",
        "call_tx_id",
        "account_mint",
        "account_user",
        "creator",
        "call_tx_signer",
        "name",
        "symbol",
        "uri",
        "account_bondingCurve",
        "account_associatedBondingCurve",
    ),
    "pumpdotfun_solana.pump_call_create_v2": (
        "call_block_time",
        "account_mint",
        "account_user",
        "name",
        "symbol",
        "uri",
    ),
    "pumpdotfun_solana.pump_call_migrate": (
        "call_block_time",
        "call_tx_id",
        "account_mint",
    ),
    "pumpdotfun_solana.pump_evt_createevent": (
        "evt_block_time",
        "mint",
        "user",
        "name",
        "symbol",
        "uri",
        "bonding_curve",
    ),
    "pumpdotfun_solana.pump_evt_completeevent": (
        "evt_block_time",
        "mint",
        "user",
        "bonding_curve",
    ),
    "pumpdotfun_solana.pump_evt_completepumpammmigrationevent": (
        "evt_block_time",
        "mint",
        "user",
        "pool",
    ),
    "pumpdotfun_solana.pump_evt_tradeevent": (
        "evt_block_time",
        "mint",
        "user",
        "is_buy",
        "sol_amount",
        "token_amount",
    ),
    "pumpdotfun_solana.pump_amm_evt_buyevent": (
        "evt_block_time",
        "pool",
        "user",
        "base_amount_out",
        "quote_amount_in",
        "pool_base_token_reserves",
        "pool_quote_token_reserves",
    ),
    "pumpdotfun_solana.pump_amm_evt_sellevent": (
        "evt_block_time",
        "pool",
        "user",
        "base_amount_in",
        "quote_amount_out",
        "pool_base_token_reserves",
        "pool_quote_token_reserves",
    ),
    "pumpdotfun_solana.pump_amm_evt_createpoolevent": (
        "evt_block_time",
        "pool",
        "base_mint",
        "quote_mint",
        "lp_mint",
        "creator",
        "base_amount_in",
        "quote_amount_in",
        "pool_base_amount",
        "pool_quote_amount",
    ),
    "dex_solana.trades": (
        "block_time",
        "block_slot",
        "block_date",
        "block_month",
        "tx_id",
        "tx_index",
        "outer_instruction_index",
        "inner_instruction_index",
        "project",
        "trader_id",
        "token_bought_mint_address",
        "token_sold_mint_address",
        "token_bought_amount_raw",
        "token_sold_amount_raw",
        "trade_source",
        "project_program_id",
    ),
    "solana.instruction_calls": (
        "block_time",
        "block_slot",
        "block_date",
        "tx_id",
        "executing_account",
        "data",
        "account_arguments",
        "inner_instructions",
        "tx_success",
    ),
    "solana.transactions": (
        "block_time",
        "block_slot",
        "block_date",
        "id",
        "signature",
        "success",
        "instructions",
        "account_keys",
        "signatures",
        "signers",
        "version",
        "block_hash",
        "recent_block_hash",
        "address_table_lookups",
        "loaded_addresses",
        "fee",
        "compute_units_consumed",
    ),
    "tokens_solana.transfers": (
        "block_time",
        "block_slot",
        "block_date",
        "tx_id",
        "token_mint_address",
        "from_owner",
        "to_owner",
        "amount",
        "token_version",
        "action",
    ),
    "solana.blocks": (
        "time",
        "slot",
        "date",
        "hash",
        "height",
        "total_transactions",
        "successful_transactions",
        "failed_transactions",
        "total_non_vote_transactions",
        "successful_non_vote_transactions",
        "failed_non_vote_transactions",
        "parent_slot",
        "previous_block_hash",
    ),
}

OPTIONAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "pumpdotfun_solana.pump_call_create": ("account_tokenProgram",),
    "pumpdotfun_solana.pump_call_create_v2": ("creator", "account_tokenProgram"),
    "pumpdotfun_solana.pump_evt_createevent": ("creator", "token_program", "quote_mint"),
    "pumpdotfun_solana.pump_evt_completeevent": ("quote_mint",),
    "pumpdotfun_solana.pump_evt_completepumpammmigrationevent": ("quote_mint",),
    "pumpdotfun_solana.pump_evt_tradeevent": (
        "quote_mint",
        "quote_amount",
        "ix_name",
        "creator",
        "creator_fee",
        "virtual_quote_reserves",
        "real_quote_reserves",
    ),
    "pumpdotfun_solana.pump_amm_evt_buyevent": ("coin_creator", "coin_creator_fee", "ix_name"),
    "pumpdotfun_solana.pump_amm_evt_sellevent": ("coin_creator", "coin_creator_fee", "ix_name"),
    "pumpdotfun_solana.pump_amm_evt_createpoolevent": ("coin_creator",),
    "dex_solana.trades": (
        "token_bought_vault",
        "token_sold_vault",
        "project_main_id",
        "version",
        "version_name",
        "amount_usd",
        "fee_usd",
        "fee_tier",
        "token_pair",
    ),
    "solana.transactions": ("return_data", "return_data_program_id", "cost_units"),
}

VALIDATE_START_TS = "2024-01-14 00:00:00"
VALIDATE_END_TS = "2024-01-14 00:00:01"
_EXPLAIN_UNSUPPORTED = re.compile(
    r"mismatched input:?\s*'?EXPLAIN|EXPLAIN[^\n]{0,80}not supported|"
    r"Unknown statement|unexpected.*EXPLAIN",
    re.IGNORECASE,
)
_AS_ALIAS = re.compile(r"\bAS\s+[A-Za-z_][A-Za-z0-9_]*", re.IGNORECASE)
_SQL_STRING = re.compile(r"'(?:''|[^'])*'")
_SQL_LINE_COMMENT = re.compile(r"--[^\n]*")
_SQL_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


@dataclass
class PreflightReport:
    ok: bool
    data_root: str | None = None
    existing_tables: set[str] = field(default_factory=set)
    columns: dict[str, set[str]] = field(default_factory=dict)
    samples: list[dict[str, str]] = field(default_factory=list)
    sql_validations: list[dict[str, str]] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_ok(self, msg: str) -> None:
        self.checks.append(msg)

    def add_warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def to_doctor_report(self) -> DoctorReport:
        report = DoctorReport(ok=self.ok)
        report.checks = list(self.checks)
        report.warnings = list(self.warnings)
        report.errors = list(self.errors)
        return report


def _split_table(full_name: str) -> tuple[str, str]:
    schema, _, table = full_name.partition(".")
    return schema, table


def _all_dataset_tables(datasets: dict[str, DatasetConfig]) -> set[str]:
    tables: set[str] = set()
    for cfg in datasets.values():
        tables.update(cfg.source_tables)
    return tables


def _information_schema_sql(tables: set[str]) -> str:
    tuples = []
    for full in sorted(tables):
        schema, table = _split_table(full)
        tuples.append(f"('{schema}', '{table}')")
    joined = ",\n        ".join(tuples)
    return f"""
SELECT table_schema, table_name, column_name
FROM information_schema.columns
WHERE (table_schema, table_name) IN (
        {joined}
)
""".strip()


def _parse_catalog(df) -> dict[str, set[str]]:
    columns: dict[str, set[str]] = {}
    if df is None or df.empty:
        return columns
    for row in df.to_dict(orient="records"):
        schema = str(row.get("table_schema") or row.get("TABLE_SCHEMA") or "")
        table = str(row.get("table_name") or row.get("TABLE_NAME") or "")
        col = str(row.get("column_name") or row.get("COLUMN_NAME") or "")
        if not schema or not table:
            continue
        full = f"{schema}.{table}"
        columns.setdefault(full, set()).add(col)
    return columns


def _window_bounds(cfg: DatasetConfig, *, recent: bool) -> tuple[str, str] | None:
    if recent:
        end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        hours = 1 if cfg.preflight_cost == "high" else 24
        start = end - timedelta(hours=hours)
        return start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")
    if not cfg.coverage_start:
        return None
    start = datetime.fromisoformat(f"{cfg.coverage_start}T00:00:00")
    end = start + timedelta(days=1)
    return start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")


def _sample_sql(
    table: str,
    partition_col: str,
    start_ts: str,
    end_ts: str,
    *,
    window: str,
    aggregate: bool,
) -> str:
    if aggregate:
        return f"""
SELECT '{table}' AS source_table,
       CAST(MIN({partition_col}) AS varchar) AS min_time,
       CAST(MAX({partition_col}) AS varchar) AS max_time,
       CAST(COUNT(*) AS varchar) AS row_count,
       '{window}' AS window
FROM {table}
WHERE {partition_col} >= TIMESTAMP '{start_ts}'
  AND {partition_col} < TIMESTAMP '{end_ts}'
""".strip()
    return f"""
SELECT '{table}' AS source_table,
       CAST({partition_col} AS varchar) AS min_time,
       CAST({partition_col} AS varchar) AS max_time,
       CAST(1 AS varchar) AS row_count,
       '{window}' AS window
FROM {table}
WHERE {partition_col} >= TIMESTAMP '{start_ts}'
  AND {partition_col} < TIMESTAMP '{end_ts}'
LIMIT 1
""".strip()


def _strip_sql_noise(sql: str) -> str:
    sql = _SQL_STRING.sub("''", sql)
    sql = OPTIONAL_COL_BLOCK.sub(" ", sql)
    sql = _SQL_BLOCK_COMMENT.sub(" ", sql)
    sql = _SQL_LINE_COMMENT.sub(" ", sql)
    return _AS_ALIAS.sub(" ", sql)


def optional_col_guards(template: str) -> set[str]:
    return {match.group(1) for match in OPTIONAL_COL_BLOCK.finditer(template)}


def unguarded_optional_column_refs(template: str, tables: Iterable[str] | None = None) -> list[str]:
    """Optional catalog columns used in SQL without any ``optional-col`` guard.

    Identifier leftovers after stripping guards are ignored when the same
    ``schema.table.column`` is already guarded — those leftovers usually belong
    to a different table that shares the column name (e.g. required ``creator``).
    """
    guarded = optional_col_guards(template)
    body = _strip_sql_noise(template)
    found: list[str] = []
    table_iter: Iterable[str] = OPTIONAL_COLUMNS if tables is None else tables
    for table in table_iter:
        for col in OPTIONAL_COLUMNS.get(table, ()):
            key = f"{table}.{col}"
            if key in guarded:
                continue
            if re.search(rf"\b{re.escape(col)}\b", body):
                found.append(key)
    return found


def explain_validate_sql(sql: str) -> str:
    return "EXPLAIN (TYPE VALIDATE)\n" + sql.rstrip().rstrip(";")


def fallback_validate_sql(sql: str) -> str:
    inner = sql.rstrip().rstrip(";")
    return f"SELECT * FROM (\n{inner}\n) AS _trencher_validate WHERE 1 = 0"


def _explain_is_valid(df) -> bool:
    if df is None or df.empty:
        return False
    row = df.iloc[0].to_dict()
    for key, value in row.items():
        if str(key).lower() != "valid":
            continue
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "t"}
    return True


def _record_validation(report: PreflightReport, dataset: str, method: str, detail: str) -> None:
    report.sql_validations.append({"dataset": dataset, "method": method, "detail": detail})


def _validate_rendered_sql(
    runner: DuneRunner,
    report: PreflightReport,
    dataset: str,
    sql: str,
    *,
    use_explain: bool,
) -> bool:
    """Return whether EXPLAIN remains the preferred method."""
    if use_explain:
        wrapped = explain_validate_sql(sql)
        try:
            result = runner.run_sql(wrapped)
        except DuneExecutionError as exc:
            message = str(exc)
            if _EXPLAIN_UNSUPPORTED.search(message):
                report.add_warn(
                    f"{dataset}: EXPLAIN (TYPE VALIDATE) unsupported; "
                    "falling back to zero-row compile"
                )
                _validate_rendered_sql(runner, report, dataset, sql, use_explain=False)
                return False
            report.add_error(f"{dataset} EXPLAIN (TYPE VALIDATE) failed: {message}")
            _record_validation(report, dataset, "explain_validate", f"error: {message}")
            return True
        if not _explain_is_valid(result.dataframe):
            report.add_error(f"{dataset} EXPLAIN (TYPE VALIDATE) returned not valid")
            _record_validation(report, dataset, "explain_validate", "valid=false")
            return True
        report.add_ok(f"{dataset}: EXPLAIN (TYPE VALIDATE) ok")
        _record_validation(report, dataset, "explain_validate", "valid=true")
        return True

    wrapped = fallback_validate_sql(sql)
    try:
        runner.run_sql(wrapped)
    except DuneExecutionError as exc:
        report.add_error(f"{dataset} zero-row SQL validation failed: {exc}")
        _record_validation(report, dataset, "zero_row", f"error: {exc}")
        return False
    report.add_ok(f"{dataset}: zero-row SQL validation ok")
    _record_validation(report, dataset, "zero_row", "ok")
    return False


def run_preflight(
    runner: DuneRunner,
    *,
    config_path: Path | None = None,
    data_root: Path | None = None,
    include_samples: bool = True,
    include_sql_validation: bool = True,
) -> PreflightReport:
    report = PreflightReport(ok=True)
    root = resolve_data_root(data_root)
    report.data_root = str(root)
    datasets = load_datasets_config(config_path)
    tables = _all_dataset_tables(datasets)
    optional = {t for cfg in datasets.values() for t in cfg.optional_source_tables}

    catalog_sql = _information_schema_sql(tables)
    catalog = runner.run_sql(catalog_sql)
    report.columns = _parse_catalog(catalog.dataframe)
    report.existing_tables = set(report.columns)
    report.add_ok(
        f"information_schema: {len(report.existing_tables)} tables, "
        f"{sum(len(c) for c in report.columns.values())} columns"
    )

    for table in sorted(tables):
        is_optional = table in optional
        if table not in report.existing_tables:
            msg = f"Dune table missing: {table}"
            if is_optional:
                report.add_warn(msg)
            else:
                report.add_error(msg)
            continue
        report.add_ok(f"table present: {table}")
        have = {c.lower() for c in report.columns[table]}
        for col in REQUIRED_COLUMNS.get(table, ()):
            if col.lower() not in have:
                msg = f"{table} missing required column {col}"
                if is_optional:
                    report.add_warn(msg)
                else:
                    report.add_error(msg)
        for col in OPTIONAL_COLUMNS.get(table, ()):
            if col.lower() not in have:
                report.add_warn(f"{table} missing optional column {col}")

    save_enabled_tables(root, report.existing_tables, report.columns)
    report.add_ok(f"wrote enabled-table cache under {root}")

    use_explain = True
    for cfg in datasets.values():
        template = load_sql_template(cfg)
        remaining = strip_optional_blocks(template, report.existing_tables)
        present_tables = [t for t in cfg.source_tables if t in report.existing_tables]
        for ref in unguarded_optional_column_refs(remaining, present_tables):
            table, _, column = ref.rpartition(".")
            have = {c.lower() for c in report.columns.get(table, set())}
            if column.lower() not in have:
                report.add_error(
                    f"{cfg.name} SQL references unavailable column {ref} "
                    "without an optional-col guard"
                )

        if not include_sql_validation:
            continue
        missing_required = [
            table for table in cfg.required_source_tables if table not in report.existing_tables
        ]
        if missing_required:
            report.add_warn(
                f"{cfg.name}: skip SQL validation; missing required tables "
                + ", ".join(missing_required)
            )
            continue
        sql = render_sql(
            template,
            start_ts=VALIDATE_START_TS,
            end_ts=VALIDATE_END_TS,
            enabled_tables=report.existing_tables,
            enabled_columns=report.columns,
        )
        keep_explain = _validate_rendered_sql(
            runner, report, cfg.name, sql, use_explain=use_explain
        )
        if use_explain and not keep_explain:
            use_explain = False

    if include_samples:
        sample_parts: list[str] = []
        seen: set[tuple[str, str]] = set()
        for cfg in datasets.values():
            if not cfg.required_source_tables:
                continue
            table = cfg.required_source_tables[0]
            if table not in report.existing_tables:
                continue
            recent_bounds = _window_bounds(cfg, recent=True)
            if recent_bounds:
                key = (table, "recent")
                if key not in seen:
                    seen.add(key)
                    start_ts, end_ts = recent_bounds
                    sample_parts.append(
                        _sample_sql(
                            table,
                            cfg.partition_column,
                            start_ts,
                            end_ts,
                            window="recent",
                            aggregate=cfg.preflight_cost != "high",
                        )
                    )
            if cfg.preflight_cost == "low":
                early_bounds = _window_bounds(cfg, recent=False)
                if early_bounds:
                    key = (table, "coverage_start")
                    if key not in seen:
                        seen.add(key)
                        start_ts, end_ts = early_bounds
                        sample_parts.append(
                            _sample_sql(
                                table,
                                cfg.partition_column,
                                start_ts,
                                end_ts,
                                window="coverage_start",
                                aggregate=True,
                            )
                        )
        if sample_parts:
            union_sql = "\nUNION ALL\n".join(sample_parts)
            sample = runner.run_sql(union_sql)
            for row in sample.dataframe.to_dict(orient="records"):
                payload = {str(k): "" if v is None else str(v) for k, v in row.items()}
                report.samples.append(payload)
                report.add_ok(
                    f"sample {payload.get('source_table')} "
                    f"window={payload.get('window')} "
                    f"min={payload.get('min_time')} max={payload.get('max_time')} "
                    f"rows={payload.get('row_count')}"
                )

    return report


def format_preflight(report: PreflightReport) -> str:
    return json.dumps(
        {
            "ok": report.ok,
            "data_root": report.data_root,
            "existing_tables": sorted(report.existing_tables),
            "samples": report.samples,
            "sql_validations": report.sql_validations,
            "checks": report.checks,
            "warnings": report.warnings,
            "errors": report.errors,
        },
        indent=2,
    )
