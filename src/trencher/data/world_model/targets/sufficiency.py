"""Supervision-sufficiency bookkeeping with explicit field semantics.

Does not alter target values — only how audits count them.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from trencher.data.world_model.targets.status import (
    BARRIER_CENSORED,
    BARRIER_LOWER_FIRST,
    BARRIER_NEITHER,
    BARRIER_UNAVAILABLE,
    BARRIER_UPPER_FIRST,
    STATUS_RIGHT_CENSORED,
    STATUS_SOURCE_BOUNDARY_UNAVAILABLE,
    STATUS_VALID_NEGATIVE,
    STATUS_VALID_POSITIVE,
)

VALID_LABEL = (STATUS_VALID_POSITIVE, STATUS_VALID_NEGATIVE)
CENSORED_LIKE = (STATUS_RIGHT_CENSORED, STATUS_SOURCE_BOUNDARY_UNAVAILABLE)


def _finite_num(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        try:
            return float(x) == float(x) and abs(float(x)) != float("inf")
        except (TypeError, ValueError):
            return False
    return False


def sufficiency_row(
    *,
    head: str,
    valid_observations: int,
    unique_tokens_with_valid_supervision: int,
    positive_observations: int,
    distinct_positive_tokens: int,
    independent_events: int,
    censored_observations: int,
    unavailable_observations: int,
    recommendation: str,
    notes: str = "",
    prevalence_among_valid_tokens: float | None = None,
) -> dict[str, Any]:
    """Canonical sufficiency record. Positive tokens ⊆ valid-supervision tokens."""
    if distinct_positive_tokens > unique_tokens_with_valid_supervision:
        raise ValueError(
            f"{head}: distinct_positive_tokens ({distinct_positive_tokens}) "
            f"> unique_tokens_with_valid_supervision "
            f"({unique_tokens_with_valid_supervision})"
        )
    prev_obs = (
        positive_observations / valid_observations if valid_observations else None
    )
    if prevalence_among_valid_tokens is None and unique_tokens_with_valid_supervision:
        prevalence_among_valid_tokens = (
            distinct_positive_tokens / unique_tokens_with_valid_supervision
        )
    return {
        "head": head,
        "valid_observations": valid_observations,
        "unique_tokens_with_valid_supervision": unique_tokens_with_valid_supervision,
        "positive_observations": positive_observations,
        "distinct_positive_tokens": distinct_positive_tokens,
        "independent_events": independent_events,
        "censored_observations": censored_observations,
        "unavailable_observations": unavailable_observations,
        "prevalence_among_valid_observations": prev_obs,
        "prevalence_among_valid_tokens": prevalence_among_valid_tokens,
        "recommendation": recommendation,
        "notes": notes,
        # Backward-compatible aliases used by older markdown writers.
        "observations": valid_observations,
        "independent_event_count": independent_events,
        "prevalence": prev_obs,
    }


def _status_sets(
    rows: Iterable[dict[str, Any]], status_key: str
) -> tuple[set[str], set[str], int, int, int, int]:
    valid_tokens: set[str] = set()
    pos_tokens: set[str] = set()
    n_valid = n_pos = n_cens = n_unavail = 0
    for r in rows:
        st = r.get(status_key)
        mint = r.get("mint")
        if st in VALID_LABEL:
            n_valid += 1
            if mint is not None:
                valid_tokens.add(mint)
            if st == STATUS_VALID_POSITIVE:
                n_pos += 1
                if mint is not None:
                    pos_tokens.add(mint)
        elif st in CENSORED_LIKE or st == "known_source_outage":
            n_cens += 1
        elif st is not None:
            n_unavail += 1
        else:
            n_unavail += 1
    return valid_tokens, pos_tokens, n_valid, n_pos, n_cens, n_unavail


def compute_rich_sufficiency(
    target_rows: list[dict[str, Any]],
    *,
    primary_explosion_events: int,
    tokens_with_primary_explosion: set[str],
    pre_explosion_obs: int,
    explosion_obs_before_60s: int,
    holder_tokens: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Corrected sufficiency rows for the rich target_dataset_v3 population."""
    n_total = len(target_rows)
    mints = {r["mint"] for r in target_rows if r.get("mint")}

    # --- returns_h300: gate positives on valid status only ---
    ret_valid_tokens: set[str] = set()
    ret_pos_tokens: set[str] = set()
    ret_valid = ret_pos = ret_cens = 0
    for r in target_rows:
        st = r.get("h300_status")
        mint = r.get("mint")
        if st in VALID_LABEL:
            ret_valid += 1
            if mint:
                ret_valid_tokens.add(mint)
            # Positive = profitable / non-negative terminal return under contract.
            if st == STATUS_VALID_POSITIVE:
                ret_pos += 1
                if mint:
                    ret_pos_tokens.add(mint)
        elif st in CENSORED_LIKE:
            ret_cens += 1
    ret_unavail = n_total - ret_valid - ret_cens

    # --- barrier u100/l30 @ 1h ---
    bar_valid_tokens: set[str] = set()
    bar_pos_tokens: set[str] = set()
    bar_valid = bar_pos = bar_cens = 0
    key_out = "h3600_barrier_u100_l30_outcome"
    for r in target_rows:
        out = r.get(key_out)
        mint = r.get("mint")
        if out in (BARRIER_UPPER_FIRST, BARRIER_LOWER_FIRST, BARRIER_NEITHER):
            bar_valid += 1
            if mint:
                bar_valid_tokens.add(mint)
            if out == BARRIER_UPPER_FIRST:
                bar_pos += 1
                if mint:
                    bar_pos_tokens.add(mint)
        elif out in ("censored", "unavailable", BARRIER_CENSORED, BARRIER_UNAVAILABLE):
            bar_cens += 1
    bar_unavail = n_total - bar_valid - bar_cens

    # --- survival 24h ---
    surv_valid_tokens, surv_pos_tokens, surv_valid, surv_pos, surv_cens, surv_unavail = (
        _status_sets(target_rows, "surv_h86400_status")
    )

    # --- graduation @ 24h (valid label status; positives = graduated) ---
    grad_valid_tokens: set[str] = set()
    grad_pos_tokens: set[str] = set()
    grad_valid = grad_pos = grad_cens = 0
    for r in target_rows:
        st = r.get("grad_h86400_status")
        mint = r.get("mint")
        if st in VALID_LABEL:
            grad_valid += 1
            if mint:
                grad_valid_tokens.add(mint)
            if r.get("grad_h86400_graduated") is True or st == STATUS_VALID_POSITIVE:
                grad_pos += 1
                if mint:
                    grad_pos_tokens.add(mint)
        elif st in CENSORED_LIKE or st == "known_source_outage":
            grad_cens += 1
    # Any-horizon graduated tokens that also have valid 24h status
    any_grad = {
        r["mint"]
        for r in target_rows
        if r.get("mint")
        and any(
            r.get(f"grad_h{h}_graduated") is True
            for h in (10, 30, 60, 120, 300, 600, 1800, 3600, 21600, 86400)
        )
    }
    # Independent graduation events = distinct graduated tokens in population
    # with at least one valid positive label at some horizon.
    grad_event_tokens = any_grad & {
        r["mint"]
        for r in target_rows
        if r.get("mint")
        and any(
            r.get(f"grad_h{h}_status") == STATUS_VALID_POSITIVE
            for h in (10, 30, 60, 120, 300, 600, 1800, 3600, 21600, 86400)
        )
    }
    # Ensure positive ⊆ valid for 24h head: use only 24h-valid positives
    grad_pos_tokens &= grad_valid_tokens
    grad_unavail = n_total - grad_valid - grad_cens

    # --- explosion ---
    expl_valid_tokens = set(tokens_with_primary_explosion)
    expl_pos_obs = sum(1 for r in target_rows if r.get("explosion_within_60s") is True)
    expl_pos_tokens = {
        r["mint"]
        for r in target_rows
        if r.get("explosion_within_60s") is True and r.get("mint")
    } & expl_valid_tokens
    expl_cens = sum(
        1
        for r in target_rows
        if r.get("explosion_within_60s") is None
        and r.get("explosion_within_60s_status") in CENSORED_LIKE
    )
    # Observations with definitive True/False imminence
    expl_valid_obs = sum(
        1 for r in target_rows if r.get("explosion_within_60s") is not None
    )
    expl_unavail = n_total - expl_valid_obs - expl_cens

    holder_tokens = holder_tokens or {
        r["mint"] for r in target_rows if r.get("holder_state_reconstructable") and r.get("mint")
    }
    holder_obs = sum(1 for r in target_rows if r.get("holder_state_reconstructable"))

    def _rec_ret(n_tok: int) -> str:
        if n_tok >= 1000:
            return "sufficient_on_rich_4k"
        if n_tok >= 200:
            return "marginal"
        return "requires_broader_population"

    def _rec_bar(n_pos: int) -> str:
        if n_pos >= 200:
            return "sufficient_on_rich_4k"
        if n_pos >= 50:
            return "marginal"
        return "requires_broader_population"

    def _rec_surv(n_tok: int) -> str:
        if n_tok >= 1000:
            return "sufficient_on_rich_4k"
        if n_tok >= 200:
            return "marginal"
        return "requires_broader_population"

    def _rec_grad(n_events: int) -> str:
        if n_events >= 200:
            return "sufficient_on_rich_4k"
        if n_events >= 50:
            return "marginal"
        return "requires_broader_population"

    def _rec_expl(n_events: int, obs60: int) -> str:
        if n_events >= 500 and obs60 >= 200:
            return "sufficient_on_rich_4k"
        if n_events >= 80 and obs60 >= 40:
            return "marginal"
        return "requires_broader_population"

    return [
        sufficiency_row(
            head="returns_h300",
            valid_observations=ret_valid,
            unique_tokens_with_valid_supervision=len(ret_valid_tokens),
            positive_observations=ret_pos,
            distinct_positive_tokens=len(ret_pos_tokens),
            independent_events=len(ret_valid_tokens),
            censored_observations=ret_cens,
            unavailable_observations=ret_unavail,
            recommendation=_rec_ret(len(ret_valid_tokens)),
            notes=(
                "Positives require h300_status=valid_positive; censored rows with "
                "partial numeric returns are excluded from positive counts."
            ),
        ),
        sufficiency_row(
            head="barrier_u100_l30_h3600",
            valid_observations=bar_valid,
            unique_tokens_with_valid_supervision=len(bar_valid_tokens),
            positive_observations=bar_pos,
            distinct_positive_tokens=len(bar_pos_tokens),
            independent_events=len(bar_pos_tokens),
            censored_observations=bar_cens,
            unavailable_observations=bar_unavail,
            recommendation=_rec_bar(len(bar_pos_tokens)),
            notes="Positive = upper_first among resolved barrier outcomes.",
        ),
        sufficiency_row(
            head="survival_24h",
            valid_observations=surv_valid,
            unique_tokens_with_valid_supervision=len(surv_valid_tokens),
            positive_observations=surv_pos,
            distinct_positive_tokens=len(surv_pos_tokens),
            independent_events=len(surv_valid_tokens),
            censored_observations=surv_cens,
            unavailable_observations=surv_unavail,
            recommendation=_rec_surv(len(surv_valid_tokens)),
        ),
        sufficiency_row(
            head="graduation",
            valid_observations=grad_valid,
            unique_tokens_with_valid_supervision=len(grad_valid_tokens),
            positive_observations=grad_pos,
            distinct_positive_tokens=len(grad_pos_tokens),
            independent_events=(
                len(grad_event_tokens) if grad_event_tokens else len(grad_pos_tokens)
            ),
            censored_observations=grad_cens,
            unavailable_observations=grad_unavail,
            recommendation=_rec_grad(
                len(grad_event_tokens) if grad_event_tokens else len(grad_pos_tokens)
            ),
            notes=(
                "Valid = grad_h86400_status in {valid_positive, valid_negative}. "
                "Absence negatives remain mostly censored under coverage v2. "
                f"any_horizon_graduated_tokens={len(any_grad)}; "
                f"population_mints={len(mints)}."
            ),
        ),
        sufficiency_row(
            head="explosion_onset_primary",
            valid_observations=expl_valid_obs if expl_valid_obs else pre_explosion_obs,
            unique_tokens_with_valid_supervision=len(expl_valid_tokens),
            positive_observations=expl_pos_obs,
            distinct_positive_tokens=len(expl_pos_tokens),
            independent_events=primary_explosion_events,
            censored_observations=expl_cens,
            unavailable_observations=max(0, expl_unavail),
            recommendation=_rec_expl(primary_explosion_events, explosion_obs_before_60s),
            notes=(
                "independent_events = primary detector events; "
                "distinct_positive_tokens ⊆ tokens_with_primary_explosion "
                "(imminence@60s True)."
            ),
            prevalence_among_valid_tokens=(
                primary_explosion_events / len(mints) if mints else None
            ),
        ),
        sufficiency_row(
            head="holder_state_targets",
            valid_observations=holder_obs,
            unique_tokens_with_valid_supervision=len(holder_tokens),
            positive_observations=holder_obs,
            distinct_positive_tokens=len(holder_tokens),
            independent_events=len(holder_tokens),
            censored_observations=0,
            unavailable_observations=n_total - holder_obs,
            recommendation="requires_broader_population",
            notes="Sparse ownership evidence on rich subset only.",
        ),
    ]


def aggregate_sufficiency_streaming(
    targets_root: Path,
    *,
    primary_explosion_events: int,
    tokens_with_primary_explosion: set[str],
    population: str = "broad_event_curve",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan partitioned observation_targets without loading all rows into RAM."""
    import pyarrow.dataset as ds

    path = Path(targets_root) / "observation_targets"
    if not path.exists():
        path = Path(targets_root)
    if not path.exists():
        return [], {"observations_scanned": 0}

    # Accumulators
    ret_valid_tok: set[str] = set()
    ret_pos_tok: set[str] = set()
    ret_valid = ret_pos = ret_cens = 0

    bar_valid_tok: set[str] = set()
    bar_pos_tok: set[str] = set()
    bar_valid = bar_pos = bar_cens = 0

    surv_valid_tok: set[str] = set()
    surv_pos_tok: set[str] = set()
    surv_valid = surv_pos = surv_cens = 0

    grad_valid_tok: set[str] = set()
    grad_pos_tok: set[str] = set()
    grad_valid = grad_pos = grad_cens = 0
    grad_event_tok: set[str] = set()

    expl_valid_obs = expl_pos_obs = expl_cens = 0
    expl_pos_tok: set[str] = set()
    pre_expl = 0
    obs_before = {str(h): 0 for h in (10, 30, 60, 120, 300)}

    holder_tok: set[str] = set()
    holder_obs = 0
    n_total = 0
    price_avail = 0

    cols = [
        "mint",
        "h300_status",
        "h3600_barrier_u100_l30_outcome",
        "surv_h86400_status",
        "grad_h86400_status",
        "grad_h86400_graduated",
        "explosion_within_60s",
        "explosion_within_60s_status",
        "explosion_already_started_at_t",
        "explosion_event_observed",
        "holder_state_reconstructable",
        "price_target_available",
        "explosion_within_10s",
        "explosion_within_30s",
        "explosion_within_120s",
        "explosion_within_300s",
    ]
    # Only request columns that exist
    dataset = ds.dataset(str(path), format="parquet")
    available = set(dataset.schema.names)
    use_cols = [c for c in cols if c in available]
    # graduation any-horizon positives
    grad_h_cols = [
        c
        for c in available
        if c.startswith("grad_h") and c.endswith("_status")
    ]

    for batch in dataset.scanner(
        columns=list(dict.fromkeys(use_cols + grad_h_cols)),
        batch_size=50_000,
    ).to_batches():
        names = batch.schema.names
        colmap = {n: batch.column(n).to_pylist() for n in names}
        n = batch.num_rows
        n_total += n
        for i in range(n):
            mint = colmap.get("mint", [None] * n)[i]
            if colmap.get("price_target_available", [False] * n)[i]:
                price_avail += 1

            st = colmap.get("h300_status", [None] * n)[i]
            if st in VALID_LABEL:
                ret_valid += 1
                if mint:
                    ret_valid_tok.add(mint)
                if st == STATUS_VALID_POSITIVE:
                    ret_pos += 1
                    if mint:
                        ret_pos_tok.add(mint)
            elif st in CENSORED_LIKE:
                ret_cens += 1

            out = colmap.get("h3600_barrier_u100_l30_outcome", [None] * n)[i]
            if out in (BARRIER_UPPER_FIRST, BARRIER_LOWER_FIRST, BARRIER_NEITHER):
                bar_valid += 1
                if mint:
                    bar_valid_tok.add(mint)
                if out == BARRIER_UPPER_FIRST:
                    bar_pos += 1
                    if mint:
                        bar_pos_tok.add(mint)
            elif out in ("censored", "unavailable", BARRIER_CENSORED, BARRIER_UNAVAILABLE):
                bar_cens += 1

            sst = colmap.get("surv_h86400_status", [None] * n)[i]
            if sst in VALID_LABEL:
                surv_valid += 1
                if mint:
                    surv_valid_tok.add(mint)
                if sst == STATUS_VALID_POSITIVE:
                    surv_pos += 1
                    if mint:
                        surv_pos_tok.add(mint)
            elif sst in CENSORED_LIKE or sst == "known_source_outage":
                surv_cens += 1

            gst = colmap.get("grad_h86400_status", [None] * n)[i]
            if gst in VALID_LABEL:
                grad_valid += 1
                if mint:
                    grad_valid_tok.add(mint)
                if (
                    colmap.get("grad_h86400_graduated", [None] * n)[i] is True
                    or gst == STATUS_VALID_POSITIVE
                ):
                    grad_pos += 1
                    if mint:
                        grad_pos_tok.add(mint)
            elif gst in CENSORED_LIKE or gst == "known_source_outage":
                grad_cens += 1
            for hc in grad_h_cols:
                if colmap[hc][i] == STATUS_VALID_POSITIVE and mint:
                    grad_event_tok.add(mint)

            ew = colmap.get("explosion_within_60s", [None] * n)[i]
            est = colmap.get("explosion_within_60s_status", [None] * n)[i]
            if ew is not None:
                expl_valid_obs += 1
                if ew is True:
                    expl_pos_obs += 1
                    if mint and mint in tokens_with_primary_explosion:
                        expl_pos_tok.add(mint)
            elif est in CENSORED_LIKE:
                expl_cens += 1
            if (
                colmap.get("explosion_already_started_at_t", [None] * n)[i] is False
                and colmap.get("explosion_event_observed", [None] * n)[i]
            ):
                pre_expl += 1
            for h, key in (
                (10, "explosion_within_10s"),
                (30, "explosion_within_30s"),
                (60, "explosion_within_60s"),
                (120, "explosion_within_120s"),
                (300, "explosion_within_300s"),
            ):
                if colmap.get(key, [None] * n)[i] is True:
                    obs_before[str(h)] += 1

            if colmap.get("holder_state_reconstructable", [False] * n)[i]:
                holder_obs += 1
                if mint:
                    holder_tok.add(mint)

    # Build sufficiency via temporary synthetic rows is awkward; assemble directly.
    def _rec_broad(head: str, n_events: int, n_tokens: int) -> str:
        if head in {"returns_h300", "survival_24h"}:
            if n_tokens >= 50_000:
                return "sufficient_on_broad"
            if n_tokens >= 5_000:
                return "marginal_on_broad"
            return "still_sparse"
        if head == "barrier_u100_l30_h3600":
            if n_events >= 2_000:
                return "sufficient_on_broad"
            if n_events >= 200:
                return "marginal_on_broad"
            return "still_sparse"
        if head == "graduation":
            if n_events >= 1_000:
                return "sufficient_on_broad"
            if n_events >= 100:
                return "marginal_on_broad"
            return "still_sparse"
        if head == "explosion_onset_primary":
            if n_events >= 5_000 and obs_before["60"] >= 2_000:
                return "sufficient_on_broad"
            if n_events >= 500 and obs_before["60"] >= 200:
                return "marginal_on_broad"
            return "still_sparse"
        if head == "holder_state_targets":
            return "still_sparse_ownership"
        return "unspecified"

    rows = [
        sufficiency_row(
            head="returns_h300",
            valid_observations=ret_valid,
            unique_tokens_with_valid_supervision=len(ret_valid_tok),
            positive_observations=ret_pos,
            distinct_positive_tokens=len(ret_pos_tok),
            independent_events=len(ret_valid_tok),
            censored_observations=ret_cens,
            unavailable_observations=n_total - ret_valid - ret_cens,
            recommendation=_rec_broad("returns_h300", len(ret_valid_tok), len(ret_valid_tok)),
        ),
        sufficiency_row(
            head="barrier_u100_l30_h3600",
            valid_observations=bar_valid,
            unique_tokens_with_valid_supervision=len(bar_valid_tok),
            positive_observations=bar_pos,
            distinct_positive_tokens=len(bar_pos_tok),
            independent_events=len(bar_pos_tok),
            censored_observations=bar_cens,
            unavailable_observations=n_total - bar_valid - bar_cens,
            recommendation=_rec_broad(
                "barrier_u100_l30_h3600", len(bar_pos_tok), len(bar_valid_tok)
            ),
        ),
        sufficiency_row(
            head="survival_24h",
            valid_observations=surv_valid,
            unique_tokens_with_valid_supervision=len(surv_valid_tok),
            positive_observations=surv_pos,
            distinct_positive_tokens=len(surv_pos_tok),
            independent_events=len(surv_valid_tok),
            censored_observations=surv_cens,
            unavailable_observations=n_total - surv_valid - surv_cens,
            recommendation=_rec_broad(
                "survival_24h", len(surv_valid_tok), len(surv_valid_tok)
            ),
        ),
        sufficiency_row(
            head="graduation",
            valid_observations=grad_valid,
            unique_tokens_with_valid_supervision=len(grad_valid_tok),
            positive_observations=grad_pos,
            distinct_positive_tokens=len(grad_pos_tok & grad_valid_tok),
            independent_events=len(grad_event_tok),
            censored_observations=grad_cens,
            unavailable_observations=n_total - grad_valid - grad_cens,
            recommendation=_rec_broad("graduation", len(grad_event_tok), len(grad_valid_tok)),
            notes="Conservative v3 absence censoring retained on broad.",
        ),
        sufficiency_row(
            head="explosion_onset_primary",
            valid_observations=expl_valid_obs,
            unique_tokens_with_valid_supervision=len(tokens_with_primary_explosion),
            positive_observations=expl_pos_obs,
            distinct_positive_tokens=len(expl_pos_tok),
            independent_events=primary_explosion_events,
            censored_observations=expl_cens,
            unavailable_observations=max(0, n_total - expl_valid_obs - expl_cens),
            recommendation=_rec_broad(
                "explosion_onset_primary",
                primary_explosion_events,
                len(tokens_with_primary_explosion),
            ),
        ),
        sufficiency_row(
            head="holder_state_targets",
            valid_observations=holder_obs,
            unique_tokens_with_valid_supervision=len(holder_tok),
            positive_observations=holder_obs,
            distinct_positive_tokens=len(holder_tok),
            independent_events=len(holder_tok),
            censored_observations=0,
            unavailable_observations=n_total - holder_obs,
            recommendation=_rec_broad("holder_state_targets", len(holder_tok), len(holder_tok)),
        ),
    ]
    for row in rows:
        row["population"] = population

    extras = {
        "observations_scanned": n_total,
        "price_target_available_observations": price_avail,
        "explosion_imminence_positives_by_horizon": obs_before,
        "pre_explosion_observations": pre_expl,
    }
    return rows, extras


def compute_broad_sufficiency(
    target_rows: list[dict[str, Any]],
    *,
    primary_explosion_events: int,
    tokens_with_primary_explosion: set[str],
    pre_explosion_obs: int,
    explosion_obs_before_60s: int,
    holder_tokens: set[str] | None = None,
) -> list[dict[str, Any]]:
    """In-memory broad sufficiency (tests / small fixtures). Prefer streaming."""
    del pre_explosion_obs  # counted inside compute_rich when present
    rows = compute_rich_sufficiency(
        target_rows,
        primary_explosion_events=primary_explosion_events,
        tokens_with_primary_explosion=tokens_with_primary_explosion,
        pre_explosion_obs=0,
        explosion_obs_before_60s=explosion_obs_before_60s,
        holder_tokens=holder_tokens,
    )
    # Re-stamp recommendations with broad thresholds via a one-row stream fake:
    # keep independent counts from rich helper; override recommendation labels.
    for row in rows:
        head = str(row["head"])
        n_tok = int(row["unique_tokens_with_valid_supervision"])
        n_ev = int(row["independent_events"])
        if head in {"returns_h300", "survival_24h"}:
            row["recommendation"] = (
                "sufficient_on_broad"
                if n_tok >= 50_000
                else ("marginal_on_broad" if n_tok >= 5_000 else "still_sparse")
            )
        elif head == "barrier_u100_l30_h3600":
            row["recommendation"] = (
                "sufficient_on_broad"
                if n_ev >= 2_000
                else ("marginal_on_broad" if n_ev >= 200 else "still_sparse")
            )
        elif head == "graduation":
            row["recommendation"] = (
                "sufficient_on_broad"
                if n_ev >= 1_000
                else ("marginal_on_broad" if n_ev >= 100 else "still_sparse")
            )
        elif head == "explosion_onset_primary":
            if n_ev >= 5_000 and explosion_obs_before_60s >= 2_000:
                row["recommendation"] = "sufficient_on_broad"
            elif n_ev >= 500 and explosion_obs_before_60s >= 200:
                row["recommendation"] = "marginal_on_broad"
            else:
                row["recommendation"] = "still_sparse"
        elif head == "holder_state_targets":
            row["recommendation"] = "still_sparse_ownership"
        row["population"] = "broad_event_curve"
    return rows


def recompute_v3_sufficiency_revision(
    *,
    targets_path: Any,
    explosion_path: Any,
    prior_audit: dict[str, Any],
) -> dict[str, Any]:
    """Load frozen v3 targets and publish corrected sufficiency (no target rewrite)."""
    import pyarrow.parquet as pq

    from trencher.data.world_model.io import utc_now
    from trencher.data.world_model.targets.explosion import PRIMARY_DEFINITION_ID

    target_rows = pq.read_table(targets_path).to_pylist()
    expl_rows = []
    if explosion_path is not None and getattr(explosion_path, "exists", lambda: False)():
        expl_rows = pq.read_table(explosion_path).to_pylist()
    primary = [
        e
        for e in expl_rows
        if e.get("definition_id") == PRIMARY_DEFINITION_ID and e.get("explosion_exists")
    ]
    tokens_with = {e["mint"] for e in primary if e.get("mint")}
    prior_suff = prior_audit.get("supervision_sufficiency") or []
    prior_by_head = {s.get("head"): s for s in prior_suff}
    obs_before = (prior_audit.get("explosion_audit") or {}).get(
        "observations_before_onset_by_horizon"
    ) or {}
    pre_expl = (prior_audit.get("explosion_audit") or {}).get(
        "observations_pre_explosion"
    ) or 0

    corrected = compute_rich_sufficiency(
        target_rows,
        primary_explosion_events=len(primary),
        tokens_with_primary_explosion=tokens_with,
        pre_explosion_obs=int(pre_expl),
        explosion_obs_before_60s=int(obs_before.get("60") or 0),
    )
    diffs = []
    for row in corrected:
        old = prior_by_head.get(row["head"]) or {}
        diffs.append(
            {
                "head": row["head"],
                "prior_distinct_positive_tokens": old.get("distinct_positive_tokens"),
                "corrected_distinct_positive_tokens": row["distinct_positive_tokens"],
                "prior_unique_valid_tokens": old.get(
                    "unique_tokens_with_valid_supervision"
                ),
                "corrected_unique_valid_tokens": row[
                    "unique_tokens_with_valid_supervision"
                ],
                "prior_independent_events": old.get("independent_event_count"),
                "corrected_independent_events": row["independent_events"],
                "positive_subseteq_valid": (
                    row["distinct_positive_tokens"]
                    <= row["unique_tokens_with_valid_supervision"]
                ),
            }
        )
    return {
        "revision_id": "target_dataset_v3_sufficiency_revision_v1",
        "generated_at": utc_now(),
        "target_semantics_unchanged": True,
        "target_dataset_version": "target_dataset_v3",
        "field_contract": [
            "valid_observations",
            "unique_tokens_with_valid_supervision",
            "positive_observations",
            "distinct_positive_tokens",
            "independent_events",
            "censored_observations",
            "unavailable_observations",
            "prevalence_among_valid_observations",
            "prevalence_among_valid_tokens",
        ],
        "invariant": "distinct_positive_tokens ⊆ unique_tokens_with_valid_supervision",
        "supervision_sufficiency": corrected,
        "diffs_vs_prior_audit": diffs,
        "notes": (
            "Corrects returns_h300 (and analogous heads) so censored observations "
            "with partial numeric returns cannot enter positive-token counts."
        ),
    }
