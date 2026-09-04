"""Ownership evidence levels and stratified fusion coverage (single source of truth)."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from trencher.data.world_model.io import ensure_dir, utc_now, write_json

OwnershipLevel = Literal[
    "holder_state_reconstructable",
    "transfer_ledger_observed_partial",
    "rpc_balance_observed",
    "trade_derived_proxy",
    "missing",
]

OWNERSHIP_LEVEL_ORDER = (
    "holder_state_reconstructable",
    "transfer_ledger_observed_partial",
    "rpc_balance_observed",
    "trade_derived_proxy",
    "missing",
)

GENUINE_OWNERSHIP_LEVELS = {
    "holder_state_reconstructable",
    "transfer_ledger_observed_partial",
    "rpc_balance_observed",
}

# Completeness thresholds for holder_state_reconstructable (documented).
HOLDER_STATE_MIN_TRANSFER_ROWS = 20
HOLDER_STATE_MIN_DISTINCT_TXS = 10
HOLDER_STATE_LAUNCH_WINDOW_SEC = 600
HOLDER_STATE_MIN_NONZERO_BALANCES = 3


def load_queried_ledger_days(world_root: Path) -> set[str]:
    """Calendar days with explicit Token Transfers day-partition extracts."""
    days: set[str] = set()
    bq = world_root / "bigquery"
    if not bq.exists():
        return days
    for path in bq.glob("token_transfers_20*.parquet"):
        # token_transfers_2026-06-05.parquet
        suffix = path.stem.removeprefix("token_transfers_")
        if len(suffix) == 10 and suffix[4] == "-" and suffix[7] == "-":
            days.add(suffix)
    return days


def calendar_days_inclusive(start_unix: float, end_unix: float) -> list[str]:
    """UTC calendar dates from start through end inclusive."""
    start = datetime.fromtimestamp(start_unix, tz=UTC).date()
    end = datetime.fromtimestamp(end_unix, tz=UTC).date()
    if end < start:
        return []
    out: list[str] = []
    cur = start
    while cur <= end:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def ledger_partition_covers_launch_to_t(
    *,
    queried_days: set[str],
    detected_at: Any,
    observation_time_t: float,
) -> dict[str, Any]:
    """True only when every UTC day from launch through T was queried."""
    launch = _to_unix(detected_at)
    if launch is None:
        return {
            "covers": False,
            "reason": "missing_detected_at",
            "required_days": [],
            "missing_days": [],
            "queried_days_n": len(queried_days),
        }
    required = calendar_days_inclusive(launch, observation_time_t)
    missing = [d for d in required if d not in queried_days]
    return {
        "covers": not missing and bool(required),
        "reason": None if not missing and required else "missing_queried_day_partitions",
        "required_days": required,
        "missing_days": missing,
        "queried_days_n": len(queried_days),
    }


def _to_unix(ts: Any) -> float | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return ts.timestamp()
    text = str(ts)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()
    except ValueError:
        return None


def _mint_set(path: Path, column: str = "mint") -> set[str]:
    if not path.exists() or not path.is_file():
        return set()
    try:
        if pq.ParquetFile(path).metadata.num_rows == 0:
            return set()
        return set(pq.read_table(path, columns=[column]).column(0).to_pylist()) - {None}
    except Exception:  # noqa: BLE001
        return set()


def holder_state_completeness_spec() -> dict[str, Any]:
    """Documented completeness test for holder_state_reconstructable."""
    return {
        "version": "ownership_completeness_v2",
        "required_all": [
            f"bq_token_transfer_rows >= {HOLDER_STATE_MIN_TRANSFER_ROWS}",
            f"distinct_tx_signatures >= {HOLDER_STATE_MIN_DISTINCT_TXS}",
            f"first_transfer_within_{HOLDER_STATE_LAUNCH_WINDOW_SEC}s_of_detected_at",
            f"nonzero_reconstructed_balances >= {HOLDER_STATE_MIN_NONZERO_BALANCES}",
            "mintTo_present OR bq_token_transfer_rows >= 50",
            "queried_ledger_day_partitions_cover_launch_through_T",
        ],
        "considers": [
            "coverage from mint/launch (detected_at) through observed transfers",
            "mintTo / spl-transfer / burn balance-changing events",
            "transaction/signature coverage",
            "signed balance reconstruction conservation across accounts",
            "optional agreement with Pump trade wallets and RPC balance observations",
            "explicit day-partition acquisition coverage launch→T when available",
        ],
        "explicitly_not_sufficient": [
            "bq_token_transfer_rows > 0 alone",
            "sparse day extracts without launch coverage",
            "row-density heuristics without partition coverage launch→T",
            "sampled RPC pre/postTokenBalances alone",
            "trade participation proxy",
        ],
    }


def evaluate_holder_state_completeness_at_t(
    *,
    transfer_rows: list[dict[str, Any]],
    detected_at: Any,
    observation_time_t: float,
    trade_wallets: set[str] | None = None,
    rpc_owners: set[str] | None = None,
    queried_ledger_days: set[str] | None = None,
) -> dict[str, Any]:
    """Completeness using only transfers with block_timestamp <= T."""
    eligible = []
    for row in transfer_rows:
        ts = _to_unix(row.get("block_timestamp"))
        if ts is not None and ts <= observation_time_t:
            eligible.append(row)
    result = evaluate_holder_state_completeness(
        transfer_rows=eligible,
        detected_at=detected_at,
        trade_wallets=trade_wallets,
        rpc_owners=rpc_owners,
        queried_ledger_days=queried_ledger_days,
        observation_time_t=observation_time_t,
    )
    result["observation_time_t"] = observation_time_t
    result["transfers_at_or_before_t"] = len(eligible)
    return result


def classify_ownership_at_t(
    *,
    transfer_rows: list[dict[str, Any]],
    rpc_rows: list[dict[str, Any]],
    trade_participation_before_t: bool,
    detected_at: Any,
    observation_time_t: float,
    queried_ledger_days: set[str] | None = None,
) -> dict[str, Any]:
    """Strongest ownership evidence knowable strictly from records at/before T."""
    transfers_at_t = []
    for row in transfer_rows:
        ts = _to_unix(row.get("block_timestamp"))
        if ts is not None and ts <= observation_time_t:
            transfers_at_t.append(row)

    rpc_at_t = []
    rpc_owners: set[str] = set()
    for row in rpc_rows:
        ts = row.get("block_time")
        if ts is None:
            continue
        if float(ts) <= observation_time_t:
            rpc_at_t.append(row)
            if row.get("owner"):
                rpc_owners.add(row["owner"])

    completeness = None
    if transfers_at_t:
        completeness = evaluate_holder_state_completeness_at_t(
            transfer_rows=transfers_at_t,
            detected_at=detected_at,
            observation_time_t=observation_time_t,
            rpc_owners=rpc_owners,
            queried_ledger_days=queried_ledger_days,
        )
        if completeness["holder_state_reconstructable"]:
            level: OwnershipLevel = "holder_state_reconstructable"
        else:
            level = "transfer_ledger_observed_partial"
    elif rpc_at_t:
        level = "rpc_balance_observed"
    elif trade_participation_before_t:
        level = "trade_derived_proxy"
    else:
        level = "missing"

    return {
        "ownership_level_at_t": level,
        "ownership_rows_observed_at_t": len(transfers_at_t),
        "ownership_rpc_observed_at_t": len(rpc_at_t),
        "holder_state_reconstructable_at_t": level == "holder_state_reconstructable",
        "completeness_at_t": completeness,
    }


def evaluate_holder_state_completeness(
    *,
    transfer_rows: list[dict[str, Any]],
    detected_at: Any,
    trade_wallets: set[str] | None = None,
    rpc_owners: set[str] | None = None,
    queried_ledger_days: set[str] | None = None,
    observation_time_t: float | None = None,
) -> dict[str, Any]:
    """Run the documented completeness test on BQ transfer rows for one mint."""
    n = len(transfer_rows)
    sigs = {r.get("tx_signature") for r in transfer_rows if r.get("tx_signature")}
    types = Counter(r.get("transfer_type") for r in transfer_rows)
    tss = [_to_unix(r.get("block_timestamp")) for r in transfer_rows]
    tss = [t for t in tss if t is not None]
    launch = _to_unix(detected_at)
    first_ts = min(tss) if tss else None
    last_ts = max(tss) if tss else None
    near_launch = (
        launch is not None
        and first_ts is not None
        and first_ts <= launch + HOLDER_STATE_LAUNCH_WINDOW_SEC
    )
    has_mint = int(types.get("mintTo") or 0) > 0
    has_burn = int(types.get("burn") or 0) > 0

    bal: dict[str, float] = defaultdict(float)
    for r in transfer_rows:
        v = float(r.get("value") or 0.0)
        src, dst = r.get("source"), r.get("destination")
        tt = r.get("transfer_type")
        if tt == "mintTo" and dst:
            bal[dst] += v
        elif tt == "burn" and src:
            bal[src] -= v
        else:
            if src:
                bal[src] -= v
            if dst:
                bal[dst] += v
    nonzero = sum(1 for b in bal.values() if abs(b) > 1e-12)

    ledger_wallets: set[str] = set()
    for r in transfer_rows:
        for key in ("source", "destination", "authority"):
            if r.get(key):
                ledger_wallets.add(r[key])
    trade_wallets = trade_wallets or set()
    rpc_owners = rpc_owners or set()
    trade_agree = (
        (len(trade_wallets & ledger_wallets) / len(trade_wallets))
        if trade_wallets
        else None
    )
    rpc_agree = (
        (len(rpc_owners & ledger_wallets) / len(rpc_owners)) if rpc_owners else None
    )

    mint_or_dense = has_mint or n >= 50
    as_of_t = observation_time_t if observation_time_t is not None else last_ts
    partition = (
        ledger_partition_covers_launch_to_t(
            queried_days=queried_ledger_days or set(),
            detected_at=detected_at,
            observation_time_t=float(as_of_t) if as_of_t is not None else 0.0,
        )
        if queried_ledger_days is not None and as_of_t is not None
        else {
            "covers": False,
            "reason": "queried_ledger_days_unknown",
            "required_days": [],
            "missing_days": [],
            "queried_days_n": 0,
        }
    )
    checks = {
        "min_transfer_rows": n >= HOLDER_STATE_MIN_TRANSFER_ROWS,
        "min_distinct_txs": len(sigs) >= HOLDER_STATE_MIN_DISTINCT_TXS,
        "launch_coverage": bool(near_launch),
        "min_nonzero_balances": nonzero >= HOLDER_STATE_MIN_NONZERO_BALANCES,
        "mint_or_dense_ledger": mint_or_dense,
        "ledger_partition_covers_launch_to_t": bool(partition.get("covers")),
    }
    reconstructable = all(checks.values())
    return {
        "holder_state_reconstructable": reconstructable,
        "checks": checks,
        "bq_token_transfer_rows": n,
        "distinct_tx_signatures": len(sigs),
        "transfer_type_counts": dict(types),
        "has_mint_to": has_mint,
        "has_burn": has_burn,
        "first_transfer_unix": first_ts,
        "last_transfer_unix": last_ts,
        "launch_unix": launch,
        "near_launch": bool(near_launch),
        "nonzero_reconstructed_balances": nonzero,
        "trade_wallet_agreement": trade_agree,
        "rpc_owner_agreement": rpc_agree,
        "partition_coverage": partition,
        "failed_checks": [k for k, ok in checks.items() if not ok],
    }


def classify_ownership_evidence(world_root: Path) -> dict[str, Any]:
    """Assign the strongest ownership evidence level per unbiased-subset mint."""
    subset_path = world_root / "universe" / "rich_subset_mints.parquet"
    subset = pq.read_table(subset_path)
    mints = subset.column("mint").to_pylist()
    S = set(mints)
    detected = {
        m: subset.column("detected_at")[i].as_py()
        for i, m in enumerate(mints)
        if "detected_at" in subset.column_names
    }

    transfers_by_mint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name in (
        "token_transfers_unbiased_days.parquet",
        "token_transfers_rich.parquet",
    ):
        path = world_root / "bigquery" / name
        if not path.exists() or pq.ParquetFile(path).metadata.num_rows == 0:
            continue
        for row in pq.read_table(path).to_pylist():
            mint = row.get("mint")
            if mint in S:
                transfers_by_mint[mint].append(row)

    accounts = _mint_set(world_root / "bigquery" / "accounts_rich.parquet") & S

    rpc_path = world_root / "graph" / "edges_ownership_rpc_token_balances.parquet"
    rpc_counts: dict[str, int] = defaultdict(int)
    rpc_owners: dict[str, set[str]] = defaultdict(set)
    rpc_sig_counts: dict[str, int] = {}
    if rpc_path.exists() and pq.ParquetFile(rpc_path).metadata.num_rows:
        for row in pq.read_table(rpc_path).to_pylist():
            mint = row.get("mint")
            if mint not in S:
                continue
            rpc_counts[mint] += 1
            if row.get("owner"):
                rpc_owners[mint].add(row["owner"])
        by_mint_sigs: dict[str, set[str]] = defaultdict(set)
        t = pq.read_table(rpc_path, columns=["mint", "tx_signature"])
        for mint, sig in zip(
            t.column("mint").to_pylist(), t.column("tx_signature").to_pylist(), strict=True
        ):
            if mint in S and sig:
                by_mint_sigs[mint].add(sig)
        rpc_sig_counts = {m: len(v) for m, v in by_mint_sigs.items()}

    trade_wallets: dict[str, set[str]] = defaultdict(set)
    st_path = world_root / "microstructure" / "signature_targets.parquet"
    if st_path.exists() and pq.ParquetFile(st_path).metadata.num_rows:
        st = pq.read_table(st_path, columns=["mint", "user_wallet"])
        for mint, wallet in zip(
            st.column("mint").to_pylist(),
            st.column("user_wallet").to_pylist(),
            strict=True,
        ):
            if mint in S and wallet:
                trade_wallets[mint].add(wallet)

    trade_proxy = _mint_set(world_root / "graph" / "edges_wallet_token.parquet", "dst") & S
    atomic = world_root / "pit" / "atomic_trades_subset.parquet"
    if atomic.exists() and pq.ParquetFile(atomic).metadata.num_rows:
        trade_proxy |= _mint_set(atomic, "mint")

    queried_ledger_days = load_queried_ledger_days(world_root)

    rows = []
    for i, mint in enumerate(mints):
        transfers = transfers_by_mint.get(mint, [])
        bq_n = len(transfers)
        rpc_n = int(rpc_counts.get(mint, 0))
        rpc_sigs = int(rpc_sig_counts.get(mint, 0))
        has_accounts = mint in accounts
        completeness: dict[str, Any] | None = None

        if bq_n > 0:
            completeness = evaluate_holder_state_completeness(
                transfer_rows=transfers,
                detected_at=detected.get(mint),
                trade_wallets=trade_wallets.get(mint),
                rpc_owners=rpc_owners.get(mint),
                queried_ledger_days=queried_ledger_days,
            )
            if completeness["holder_state_reconstructable"]:
                level: OwnershipLevel = "holder_state_reconstructable"
                source = "bigquery.Token Transfers"
                quality = "holder_state_completeness_passed"
            else:
                level = "transfer_ledger_observed_partial"
                source = "bigquery.Token Transfers"
                quality = (
                    "sparse_or_incomplete_ledger:"
                    + ",".join(completeness.get("failed_checks") or [])
                )
        elif rpc_n > 0:
            level = "rpc_balance_observed"
            source = "solana_rpc.getTransaction.meta.pre_post_token_balances"
            quality = (
                "single_sampled_tx"
                if rpc_sigs <= 1
                else "multi_sampled_tx_not_full_ledger"
            )
        elif mint in trade_proxy:
            level = "trade_derived_proxy"
            source = "pumpfun.trades"
            quality = "trade_participation_only"
        else:
            level = "missing"
            source = None
            quality = "missing"

        rows.append(
            {
                "mint": mint,
                "launch_day": subset.column("launch_day")[i].as_py()
                if "launch_day" in subset.column_names
                else None,
                "launch_regime": subset.column("launch_regime")[i].as_py()
                if "launch_regime" in subset.column_names
                else None,
                "graduated": bool(subset.column("graduated")[i].as_py())
                if "graduated" in subset.column_names
                else None,
                "ownership_evidence_level": level,
                "ownership_source": source,
                "ownership_quality": quality,
                "bq_token_transfer_rows": bq_n,
                "has_account_snapshots": has_accounts,
                "rpc_token_balance_edges": rpc_n,
                "rpc_distinct_tx_signatures": rpc_sigs,
                "has_trade_participation": mint in trade_proxy,
                "holder_state_reconstructable": level == "holder_state_reconstructable",
                "availability_mask_holder_reconstructable": (
                    level == "holder_state_reconstructable"
                ),
                "availability_mask_transfer_ledger_partial": level
                in {
                    "holder_state_reconstructable",
                    "transfer_ledger_observed_partial",
                },
                "availability_mask_rpc_balance": level in GENUINE_OWNERSHIP_LEVELS,
                "availability_mask_any_genuine": level in GENUINE_OWNERSHIP_LEVELS,
                "availability_mask_trade_proxy": level != "missing",
                "completeness_failed_checks": (
                    ",".join(completeness.get("failed_checks") or [])
                    if completeness
                    else None
                ),
                "completeness_nonzero_balances": (
                    completeness.get("nonzero_reconstructed_balances")
                    if completeness
                    else None
                ),
                "completeness_trade_wallet_agreement": (
                    completeness.get("trade_wallet_agreement") if completeness else None
                ),
            }
        )

    out_dir = ensure_dir(world_root / "universe")
    path = out_dir / "ownership_evidence.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    counts = Counter(r["ownership_evidence_level"] for r in rows)
    summary = {
        "generated_at": utc_now(),
        "path": str(path),
        "subset_n": len(rows),
        "definition_version": "ownership_evidence_v3",
        "queried_ledger_days_n": len(queried_ledger_days),
        "counts_by_level": {k: counts.get(k, 0) for k in OWNERSHIP_LEVEL_ORDER},
        "completeness_spec": holder_state_completeness_spec(),
        "notes": [
            "holder_state_reconstructable requires passing the documented "
            "completeness test — not merely bq_token_transfer_rows > 0.",
            "holder_state_reconstructable also requires queried Token Transfers "
            "day partitions covering launch through last observed transfer.",
            "transfer_ledger_observed_partial is genuine BQ transfer evidence that "
            "is sparse or fails launch/conservation/coverage checks.",
            "rpc_balance_observed is genuine on-chain meta for sampled txs only.",
            "trade_derived_proxy is participation only and must not be treated as "
            "genuine ownership reconstruction.",
        ],
    }
    write_json(out_dir / "ownership_evidence_summary.json", summary)
    return {"rows": rows, "summary": summary, "path": path}


def build_fusion_coverage_truth(
    world_root: Path, *, manifest_dir: Path | None = None
) -> dict[str, Any]:
    """Deterministic stratified fusion coverage — single source of truth."""
    own = classify_ownership_evidence(world_root)
    own_by_mint = {r["mint"]: r for r in own["rows"]}

    subset = pq.read_table(world_root / "universe" / "rich_subset_mints.parquet")
    mints = subset.column("mint").to_pylist()
    S = set(mints)

    event = _mint_set(world_root / "execution" / "execution_curve_states.parquet") & S
    graph = _mint_set(world_root / "graph" / "edges_wallet_token.parquet", "dst") & S
    graph |= _mint_set(world_root / "graph" / "edges_creator_token.parquet", "dst") & S
    atomic = world_root / "pit" / "atomic_trades_subset.parquet"
    if atomic.exists():
        graph |= _mint_set(atomic, "mint")
    exec_m = event
    market_ok = True

    rt_path = world_root / "microstructure" / "rpc_transactions.parquet"
    micro: set[str] = set()
    if rt_path.exists() and pq.ParquetFile(rt_path).metadata.num_rows:
        rt = pq.read_table(rt_path, columns=["mint", "found"])
        mcol = rt.column("mint").to_pylist()
        fcol = rt.column("found").to_pylist()
        micro = {m for m, f in zip(mcol, fcol, strict=True) if m and f} & S

    native_path = world_root / "graph" / "edges_funding_native_sol.parquet"
    native_mints = _mint_set(native_path) & S

    rows = []
    for i, mint in enumerate(mints):
        o = own_by_mint[mint]
        has_event = mint in event
        has_graph = mint in graph
        has_micro = mint in micro
        has_exec = mint in exec_m
        has_market = market_ok
        core = has_event and has_graph and has_micro and has_exec and has_market
        any_own = bool(o["availability_mask_any_genuine"])
        holder = bool(o["availability_mask_holder_reconstructable"])
        rows.append(
            {
                "mint": mint,
                "launch_day": subset.column("launch_day")[i].as_py()
                if "launch_day" in subset.column_names
                else None,
                "launch_regime": subset.column("launch_regime")[i].as_py()
                if "launch_regime" in subset.column_names
                else None,
                "graduated": bool(subset.column("graduated")[i].as_py())
                if "graduated" in subset.column_names
                else None,
                "has_event_curve": has_event,
                "has_graph": has_graph,
                "has_microstructure": has_micro,
                "has_execution_state": has_exec,
                "has_market_context": has_market,
                "has_native_sol_history": mint in native_mints,
                "ownership_evidence_level": o["ownership_evidence_level"],
                "ownership_quality": o["ownership_quality"],
                "core_modalities_without_ownership": core,
                "fusion_any_genuine_ownership": core and any_own,
                "fusion_reconstructable_ownership": core and holder,
                "fusion_holder_state_reconstructable": core and holder,
                "availability_combo": "+".join(
                    k
                    for k, ok in [
                        ("event", has_event),
                        ("graph", has_graph),
                        ("micro", has_micro),
                        ("exec", has_exec),
                        ("market", has_market),
                        ("own_" + o["ownership_evidence_level"], True),
                        ("native_sol", mint in native_mints),
                    ]
                    if ok
                ),
            }
        )

    out_dir = ensure_dir(world_root / "universe")
    path = out_dir / "fusion_coverage_truth.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)

    def _count(flag: str) -> int:
        return sum(1 for r in rows if r[flag])

    def _grad_split(flag: str) -> dict[str, int]:
        sel = [r for r in rows if r[flag]]
        return {
            "n": len(sel),
            "graduated": sum(1 for r in sel if r.get("graduated")),
            "non_graduated": sum(1 for r in sel if not r.get("graduated")),
        }

    combos = Counter(r["availability_combo"] for r in rows)
    truth = {
        "generated_at": utc_now(),
        "source_of_truth": str(path),
        "definition_version": "fusion_coverage_v3_holder_completeness",
        "prior_disagreement_note": (
            "v2 counted any BQ transfer row as reconstructable (~287). "
            "v3 requires documented holder-state completeness; sparse ledgers are "
            "transfer_ledger_observed_partial."
        ),
        "subset_n": len(rows),
        "counts": {
            "core_event_graph_micro_exec_market": _count(
                "core_modalities_without_ownership"
            ),
            "core_plus_any_genuine_ownership": _count("fusion_any_genuine_ownership"),
            "core_plus_reconstructable_ownership": _count(
                "fusion_reconstructable_ownership"
            ),
            "core_plus_holder_state_reconstructable": _count(
                "fusion_holder_state_reconstructable"
            ),
            "with_true_native_sol_history": _count("has_native_sol_history"),
            "core_plus_native_sol": sum(
                1
                for r in rows
                if r["core_modalities_without_ownership"] and r["has_native_sol_history"]
            ),
        },
        "composition": {
            "core": _grad_split("core_modalities_without_ownership"),
            "any_genuine_ownership": _grad_split("fusion_any_genuine_ownership"),
            "reconstructable_ownership": _grad_split(
                "fusion_reconstructable_ownership"
            ),
        },
        "ownership_evidence_counts": own["summary"]["counts_by_level"],
        "availability_combo_top": combos.most_common(30),
        "fusion_readiness_policy": {
            "evaluation_mode": "modality_masked_experimental",
            "do_not_gate_on": "2000_complete_ownership_ledgers",
            "holder_state_reconstructable_usable_as": (
                "strong ownership modality when completeness test passes"
            ),
            "partial_modalities_preserved": list(OWNERSHIP_LEVEL_ORDER),
            "do_not_equate": (
                "rpc_balance_observed != transfer_ledger_observed_partial != "
                "holder_state_reconstructable"
            ),
        },
        "completeness_spec": holder_state_completeness_spec(),
    }
    write_json(out_dir / "fusion_coverage_truth.json", truth)
    md = manifest_dir or Path("manifests") / "world_model_data"
    write_json(md / "fusion_coverage_truth.json", truth)
    return {"rows": rows, "truth": truth, "ownership": own, "path": path}
