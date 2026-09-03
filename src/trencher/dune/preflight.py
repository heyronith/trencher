"""Low-cost live Dune catalog/query preflight."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trencher.dune.client import DuneRunner
from trencher.dune.config import DatasetConfig, load_datasets_config, save_enabled_tables
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
        "tx_id",
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
        "total_transactions",
        "successful_transactions",
        "failed_transactions",
        "parent_slot",
    ),
}

OPTIONAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "pumpdotfun_solana.pump_call_create": ("account_tokenProgram",),
    "pumpdotfun_solana.pump_call_create_v2": ("creator", "account_tokenProgram"),
    "pumpdotfun_solana.pump_evt_createevent": ("creator", "token_program", "quote_mint"),
    "pumpdotfun_solana.pump_evt_completeevent": ("quote_mint",),
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


@dataclass
class PreflightReport:
    ok: bool
    existing_tables: set[str] = field(default_factory=set)
    columns: dict[str, set[str]] = field(default_factory=dict)
    samples: list[dict[str, str]] = field(default_factory=list)
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


def run_preflight(
    runner: DuneRunner,
    *,
    config_path: Path | None = None,
    data_root: Path | None = None,
    include_samples: bool = True,
) -> PreflightReport:
    report = PreflightReport(ok=True)
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

    if include_samples:
        sample_parts: list[str] = []
        seen: set[tuple[str, str]] = set()
        for cfg in datasets.values():
            if not cfg.required_source_tables:
                continue
            table = cfg.required_source_tables[0]
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

    if data_root is not None:
        save_enabled_tables(data_root, report.existing_tables)
        report.add_ok(f"wrote enabled-table cache under {data_root}")

    return report


def format_preflight(report: PreflightReport) -> str:
    return json.dumps(
        {
            "ok": report.ok,
            "existing_tables": sorted(report.existing_tables),
            "samples": report.samples,
            "checks": report.checks,
            "warnings": report.warnings,
            "errors": report.errors,
        },
        indent=2,
    )
