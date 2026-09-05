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
EXPLOSION_IMMINENCE_HORIZONS_SEC = (10, 30, 60, 120, 300)
NOT_RECORDED = "not_recorded"


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


def _infer_imminence_status(flag: Any, status: Any) -> Any:
    """Prefer explicit status; if missing, infer from boolean imminence flag."""
    if status is not None:
        return status
    if flag is True:
        return STATUS_VALID_POSITIVE
    if flag is False:
        return STATUS_VALID_NEGATIVE
    return None


def explosion_imminence_counts(
    rows: Iterable[dict[str, Any]],
    *,
    horizon_sec: int,
) -> dict[str, Any]:
    """Status-gated imminence bookkeeping for one horizon.

    ``unique_tokens_with_valid_supervision`` = distinct mints with
    ``explosion_within_<H>s_status in {valid_positive, valid_negative}``
    (includes valid-negative / no-explosion tokens).
    """
    status_key = f"explosion_within_{horizon_sec}s_status"
    flag_key = f"explosion_within_{horizon_sec}s"
    valid_tokens: set[str] = set()
    pos_tokens: set[str] = set()
    n_valid = n_pos = n_cens = n_unavail = 0
    for r in rows:
        flag = r.get(flag_key)
        st = _infer_imminence_status(flag, r.get(status_key))
        mint = r.get("mint")
        if st in VALID_LABEL:
            n_valid += 1
            if mint is not None:
                valid_tokens.add(str(mint))
            if st == STATUS_VALID_POSITIVE and flag is True:
                n_pos += 1
                if mint is not None:
                    pos_tokens.add(str(mint))
        elif st in CENSORED_LIKE or st == "known_source_outage":
            n_cens += 1
        else:
            n_unavail += 1
    pos_tokens &= valid_tokens
    neg_tokens = valid_tokens - pos_tokens
    return {
        "horizon_sec": horizon_sec,
        "valid_tokens": valid_tokens,
        "pos_tokens": pos_tokens,
        "neg_tokens": neg_tokens,
        "valid_observations": n_valid,
        "positive_observations": n_pos,
        "negative_observations": n_valid - n_pos,
        "censored_observations": n_cens,
        "unavailable_observations": n_unavail,
    }


def graduation_time_to_event_counts(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Survival-style time-to-graduation supervision.

    Right-censored rows are legitimate survival supervision (included in the
    usable denominator), unlike binary within-H labels.
    """
    usable_tokens: set[str] = set()
    event_tokens: set[str] = set()
    n_usable = n_event = n_cens = n_resolved_neg = n_unavail = 0
    for r in rows:
        st = r.get("time_to_graduation_status")
        mint = r.get("mint")
        if st in VALID_LABEL or st in CENSORED_LIKE or st == "known_source_outage":
            n_usable += 1
            if mint is not None:
                usable_tokens.add(str(mint))
            if st == STATUS_VALID_POSITIVE:
                n_event += 1
                if mint is not None:
                    event_tokens.add(str(mint))
            elif st in CENSORED_LIKE or st == "known_source_outage":
                n_cens += 1
            elif st == STATUS_VALID_NEGATIVE:
                n_resolved_neg += 1
        else:
            n_unavail += 1
    event_tokens &= usable_tokens
    return {
        "usable_tokens": usable_tokens,
        "event_tokens": event_tokens,
        "usable_observations": n_usable,
        "event_observations": n_event,
        "censored_observations": n_cens,
        "resolved_negative_observations": n_resolved_neg,
        "unavailable_observations": n_unavail,
    }


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

    # --- graduation: split time-to-event vs binary within-H ---
    grad_bin_valid_tokens: set[str] = set()
    grad_bin_pos_tokens: set[str] = set()
    grad_bin_valid = grad_bin_pos = grad_bin_cens = 0
    for r in target_rows:
        st = r.get("grad_h86400_status")
        mint = r.get("mint")
        if st in VALID_LABEL:
            grad_bin_valid += 1
            if mint:
                grad_bin_valid_tokens.add(mint)
            if r.get("grad_h86400_graduated") is True or st == STATUS_VALID_POSITIVE:
                grad_bin_pos += 1
                if mint:
                    grad_bin_pos_tokens.add(mint)
        elif st in CENSORED_LIKE or st == "known_source_outage":
            grad_bin_cens += 1
    grad_bin_pos_tokens &= grad_bin_valid_tokens
    grad_bin_unavail = n_total - grad_bin_valid - grad_bin_cens

    tte = graduation_time_to_event_counts(target_rows)

    # --- explosion imminence (status-gated; all horizons) ---
    expl_by_h = {
        h: explosion_imminence_counts(target_rows, horizon_sec=h)
        for h in EXPLOSION_IMMINENCE_HORIZONS_SEC
    }
    expl60 = expl_by_h[60]

    holder_tokens = holder_tokens or {
        r["mint"]
        for r in target_rows
        if r.get("holder_state_reconstructable") and r.get("mint")
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

    def _rec_grad_tte(n_events: int) -> str:
        # Observed graduation events as survival events (~4k broad may suffice).
        if n_events >= 200:
            return "sufficient_on_rich_4k"
        if n_events >= 50:
            return "marginal"
        return "requires_broader_population"

    def _rec_grad_binary(n_valid_tok: int, n_pos_tok: int) -> str:
        # Only resolved valid+/valid- horizon labels; heavy censoring ≠ negatives.
        if n_valid_tok >= 1000 and n_pos_tok >= 50:
            return "sufficient_on_rich_4k"
        if n_valid_tok >= 100:
            return "marginal"
        return "requires_broader_population"

    def _rec_expl(n_events: int, obs60: int) -> str:
        if n_events >= 500 and obs60 >= 200:
            return "sufficient_on_rich_4k"
        if n_events >= 80 and obs60 >= 40:
            return "marginal"
        return "requires_broader_population"

    out_rows: list[dict[str, Any]] = [
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
            head="graduation_time_to_event",
            valid_observations=tte["usable_observations"],
            unique_tokens_with_valid_supervision=len(tte["usable_tokens"]),
            positive_observations=tte["event_observations"],
            distinct_positive_tokens=len(tte["event_tokens"]),
            independent_events=len(tte["event_tokens"]),
            censored_observations=tte["censored_observations"],
            unavailable_observations=tte["unavailable_observations"],
            recommendation=_rec_grad_tte(len(tte["event_tokens"])),
            notes=(
                "Survival-style time_to_graduation_status. Right-censored "
                "observations are legitimate supervision and remain in the "
                "usable denominator. independent_events = distinct mints with "
                "observed future graduation. "
                f"population_mints={len(mints)}."
            ),
        ),
        sufficiency_row(
            head="graduated_within_h86400_binary",
            valid_observations=grad_bin_valid,
            unique_tokens_with_valid_supervision=len(grad_bin_valid_tokens),
            positive_observations=grad_bin_pos,
            distinct_positive_tokens=len(grad_bin_pos_tokens),
            independent_events=len(grad_bin_pos_tokens),
            censored_observations=grad_bin_cens,
            unavailable_observations=grad_bin_unavail,
            recommendation=_rec_grad_binary(
                len(grad_bin_valid_tokens), len(grad_bin_pos_tokens)
            ),
            notes=(
                "Binary within-24h graduation only. Valid = "
                "grad_h86400_status in {valid_positive, valid_negative}. "
                "Heavily censored absence rows are NOT ordinary binary negatives "
                "and are excluded from the valid denominator."
            ),
        ),
    ]

    for h, stats in expl_by_h.items():
        out_rows.append(
            sufficiency_row(
                head=f"explosion_within_{h}s",
                valid_observations=stats["valid_observations"],
                unique_tokens_with_valid_supervision=len(stats["valid_tokens"]),
                positive_observations=stats["positive_observations"],
                distinct_positive_tokens=len(stats["pos_tokens"]),
                independent_events=primary_explosion_events,
                censored_observations=stats["censored_observations"],
                unavailable_observations=stats["unavailable_observations"],
                recommendation=(
                    _rec_expl(
                        primary_explosion_events,
                        stats["positive_observations"]
                        if h == 60
                        else explosion_obs_before_60s,
                    )
                    if h == 60
                    else _rec_expl(
                        primary_explosion_events, stats["positive_observations"]
                    )
                ),
                notes=(
                    f"Imminence@{h}s. valid = status in "
                    "{valid_positive, valid_negative} (includes no-explosion "
                    "valid_negatives). positives are status-gated "
                    "valid_positive with flag True. independent_events = "
                    "distinct primary explosion detector events. "
                    "prevalence_among_valid_tokens uses valid-imminence tokens, "
                    "not event-positive tokens only."
                ),
            )
        )

    # Compatibility alias: primary head == 60s imminence bookkeeping.
    out_rows.append(
        sufficiency_row(
            head="explosion_onset_primary",
            valid_observations=expl60["valid_observations"],
            unique_tokens_with_valid_supervision=len(expl60["valid_tokens"]),
            positive_observations=expl60["positive_observations"],
            distinct_positive_tokens=len(expl60["pos_tokens"]),
            independent_events=primary_explosion_events,
            censored_observations=expl60["censored_observations"],
            unavailable_observations=expl60["unavailable_observations"],
            recommendation=_rec_expl(
                primary_explosion_events,
                expl60["positive_observations"] or explosion_obs_before_60s,
            ),
            notes=(
                "Alias of explosion_within_60s. "
                f"pre_explosion_obs_arg={pre_explosion_obs}; "
                "tokens_with_primary_explosion is NOT the valid denominator."
            ),
        )
    )
    out_rows.append(
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
        )
    )
    return out_rows


def aggregate_sufficiency_streaming(
    targets_root: Path,
    *,
    primary_explosion_events: int,
    tokens_with_primary_explosion: set[str],
    population: str = "broad_event_curve",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan partitioned observation_targets without loading all rows into RAM."""
    import pyarrow.dataset as ds

    del tokens_with_primary_explosion  # not used as valid denominator

    path = Path(targets_root) / "observation_targets"
    if not path.exists():
        path = Path(targets_root)
    if not path.exists():
        return [], {"observations_scanned": 0}

    ret_valid_tok: set[str] = set()
    ret_pos_tok: set[str] = set()
    ret_valid = ret_pos = ret_cens = 0

    bar_valid_tok: set[str] = set()
    bar_pos_tok: set[str] = set()
    bar_valid = bar_pos = bar_cens = 0

    surv_valid_tok: set[str] = set()
    surv_pos_tok: set[str] = set()
    surv_valid = surv_pos = surv_cens = 0

    grad_bin_valid_tok: set[str] = set()
    grad_bin_pos_tok: set[str] = set()
    grad_bin_valid = grad_bin_pos = grad_bin_cens = 0

    tte_usable_tok: set[str] = set()
    tte_event_tok: set[str] = set()
    tte_usable = tte_event = tte_cens = 0

    expl_valid_tok: dict[int, set[str]] = {
        h: set() for h in EXPLOSION_IMMINENCE_HORIZONS_SEC
    }
    expl_pos_tok: dict[int, set[str]] = {
        h: set() for h in EXPLOSION_IMMINENCE_HORIZONS_SEC
    }
    expl_neg_tok: dict[int, set[str]] = {
        h: set() for h in EXPLOSION_IMMINENCE_HORIZONS_SEC
    }
    expl_valid_obs = {h: 0 for h in EXPLOSION_IMMINENCE_HORIZONS_SEC}
    expl_pos_obs = {h: 0 for h in EXPLOSION_IMMINENCE_HORIZONS_SEC}
    expl_neg_obs = {h: 0 for h in EXPLOSION_IMMINENCE_HORIZONS_SEC}
    expl_cens = {h: 0 for h in EXPLOSION_IMMINENCE_HORIZONS_SEC}
    expl_unavail = {h: 0 for h in EXPLOSION_IMMINENCE_HORIZONS_SEC}

    pre_expl = 0
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
        "time_to_graduation_status",
        "explosion_already_started_at_t",
        "explosion_event_observed",
        "holder_state_reconstructable",
        "price_target_available",
    ]
    for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
        cols.append(f"explosion_within_{h}s")
        cols.append(f"explosion_within_{h}s_status")

    dataset = ds.dataset(str(path), format="parquet")
    available = set(dataset.schema.names)
    use_cols = [c for c in cols if c in available]

    for batch in dataset.scanner(columns=use_cols, batch_size=50_000).to_batches():
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
            elif out in (
                "censored",
                "unavailable",
                BARRIER_CENSORED,
                BARRIER_UNAVAILABLE,
            ):
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
                grad_bin_valid += 1
                if mint:
                    grad_bin_valid_tok.add(mint)
                if (
                    colmap.get("grad_h86400_graduated", [None] * n)[i] is True
                    or gst == STATUS_VALID_POSITIVE
                ):
                    grad_bin_pos += 1
                    if mint:
                        grad_bin_pos_tok.add(mint)
            elif gst in CENSORED_LIKE or gst == "known_source_outage":
                grad_bin_cens += 1

            tte_st = colmap.get("time_to_graduation_status", [None] * n)[i]
            if (
                tte_st in VALID_LABEL
                or tte_st in CENSORED_LIKE
                or tte_st == "known_source_outage"
            ):
                tte_usable += 1
                if mint:
                    tte_usable_tok.add(mint)
                if tte_st == STATUS_VALID_POSITIVE:
                    tte_event += 1
                    if mint:
                        tte_event_tok.add(mint)
                elif tte_st in CENSORED_LIKE or tte_st == "known_source_outage":
                    tte_cens += 1

            for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
                flag = colmap.get(f"explosion_within_{h}s", [None] * n)[i]
                est = _infer_imminence_status(
                    flag, colmap.get(f"explosion_within_{h}s_status", [None] * n)[i]
                )
                if est in VALID_LABEL:
                    expl_valid_obs[h] += 1
                    if mint:
                        expl_valid_tok[h].add(mint)
                    if est == STATUS_VALID_POSITIVE and flag is True:
                        expl_pos_obs[h] += 1
                        if mint:
                            expl_pos_tok[h].add(mint)
                    elif est == STATUS_VALID_NEGATIVE:
                        expl_neg_obs[h] += 1
                        if mint:
                            expl_neg_tok[h].add(mint)
                elif est in CENSORED_LIKE or est == "known_source_outage":
                    expl_cens[h] += 1
                else:
                    expl_unavail[h] += 1

            if (
                colmap.get("explosion_already_started_at_t", [None] * n)[i] is False
                and colmap.get("explosion_event_observed", [None] * n)[i]
            ):
                pre_expl += 1

            if colmap.get("holder_state_reconstructable", [False] * n)[i]:
                holder_obs += 1
                if mint:
                    holder_tok.add(mint)

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
        if head == "graduation_time_to_event":
            if n_events >= 1_000:
                return "sufficient_on_broad"
            if n_events >= 100:
                return "marginal_on_broad"
            return "still_sparse"
        if head == "graduated_within_h86400_binary":
            if n_tokens >= 10_000 and n_events >= 1_000:
                return "sufficient_on_broad"
            if n_tokens >= 500:
                return "marginal_on_broad"
            return "still_sparse"
        if head.startswith("explosion_within_") or head == "explosion_onset_primary":
            pos60 = expl_pos_obs[60]
            if n_events >= 5_000 and pos60 >= 2_000:
                return "sufficient_on_broad"
            if n_events >= 500 and pos60 >= 200:
                return "marginal_on_broad"
            return "still_sparse"
        if head == "holder_state_targets":
            return "still_sparse_ownership"
        return "unspecified"

    rows: list[dict[str, Any]] = [
        sufficiency_row(
            head="returns_h300",
            valid_observations=ret_valid,
            unique_tokens_with_valid_supervision=len(ret_valid_tok),
            positive_observations=ret_pos,
            distinct_positive_tokens=len(ret_pos_tok),
            independent_events=len(ret_valid_tok),
            censored_observations=ret_cens,
            unavailable_observations=n_total - ret_valid - ret_cens,
            recommendation=_rec_broad(
                "returns_h300", len(ret_valid_tok), len(ret_valid_tok)
            ),
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
            head="graduation_time_to_event",
            valid_observations=tte_usable,
            unique_tokens_with_valid_supervision=len(tte_usable_tok),
            positive_observations=tte_event,
            distinct_positive_tokens=len(tte_event_tok),
            independent_events=len(tte_event_tok),
            censored_observations=tte_cens,
            unavailable_observations=n_total - tte_usable,
            recommendation=_rec_broad(
                "graduation_time_to_event",
                len(tte_event_tok),
                len(tte_usable_tok),
            ),
            notes=(
                "Survival-style time_to_graduation_status. Right-censored "
                "observations are legitimate supervision in the usable "
                "denominator. ~4k observed graduation events may be sufficient."
            ),
        ),
        sufficiency_row(
            head="graduated_within_h86400_binary",
            valid_observations=grad_bin_valid,
            unique_tokens_with_valid_supervision=len(grad_bin_valid_tok),
            positive_observations=grad_bin_pos,
            distinct_positive_tokens=len(grad_bin_pos_tok & grad_bin_valid_tok),
            independent_events=len(grad_bin_pos_tok & grad_bin_valid_tok),
            censored_observations=grad_bin_cens,
            unavailable_observations=n_total - grad_bin_valid - grad_bin_cens,
            recommendation=_rec_broad(
                "graduated_within_h86400_binary",
                len(grad_bin_pos_tok & grad_bin_valid_tok),
                len(grad_bin_valid_tok),
            ),
            notes=(
                "Binary within-24h only. Censored absence is not an ordinary "
                "binary negative; excluded from valid denominator."
            ),
        ),
    ]

    for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
        pos_tok = expl_pos_tok[h] & expl_valid_tok[h]
        rows.append(
            sufficiency_row(
                head=f"explosion_within_{h}s",
                valid_observations=expl_valid_obs[h],
                unique_tokens_with_valid_supervision=len(expl_valid_tok[h]),
                positive_observations=expl_pos_obs[h],
                distinct_positive_tokens=len(pos_tok),
                independent_events=primary_explosion_events,
                censored_observations=expl_cens[h],
                unavailable_observations=max(
                    0, n_total - expl_valid_obs[h] - expl_cens[h]
                ),
                recommendation=_rec_broad(
                    f"explosion_within_{h}s",
                    primary_explosion_events,
                    len(expl_valid_tok[h]),
                ),
                notes=(
                    f"Imminence@{h}s status-gated. Valid denominator includes "
                    "valid_negative (no-explosion) tokens."
                ),
            )
        )

    pos60 = expl_pos_tok[60] & expl_valid_tok[60]
    rows.append(
        sufficiency_row(
            head="explosion_onset_primary",
            valid_observations=expl_valid_obs[60],
            unique_tokens_with_valid_supervision=len(expl_valid_tok[60]),
            positive_observations=expl_pos_obs[60],
            distinct_positive_tokens=len(pos60),
            independent_events=primary_explosion_events,
            censored_observations=expl_cens[60],
            unavailable_observations=max(
                0, n_total - expl_valid_obs[60] - expl_cens[60]
            ),
            recommendation=_rec_broad(
                "explosion_onset_primary",
                primary_explosion_events,
                len(expl_valid_tok[60]),
            ),
            notes="Alias of explosion_within_60s (status-gated imminence).",
        )
    )
    rows.append(
        sufficiency_row(
            head="holder_state_targets",
            valid_observations=holder_obs,
            unique_tokens_with_valid_supervision=len(holder_tok),
            positive_observations=holder_obs,
            distinct_positive_tokens=len(holder_tok),
            independent_events=len(holder_tok),
            censored_observations=0,
            unavailable_observations=n_total - holder_obs,
            recommendation=_rec_broad(
                "holder_state_targets", len(holder_tok), len(holder_tok)
            ),
        )
    )
    for row in rows:
        row["population"] = population

    extras = {
        "observations_scanned": n_total,
        "price_target_available_observations": price_avail,
        "explosion_imminence_positives_by_horizon": {
            str(h): expl_pos_obs[h] for h in EXPLOSION_IMMINENCE_HORIZONS_SEC
        },
        "explosion_imminence_valid_tokens_by_horizon": {
            str(h): len(expl_valid_tok[h]) for h in EXPLOSION_IMMINENCE_HORIZONS_SEC
        },
        "explosion_imminence_by_horizon": {
            str(h): {
                "valid_observations": expl_valid_obs[h],
                "valid_negative_observations": expl_neg_obs[h],
                "valid_positive_observations": expl_pos_obs[h],
                "unique_tokens_with_valid_supervision": len(expl_valid_tok[h]),
                "distinct_positive_tokens": len(expl_pos_tok[h] & expl_valid_tok[h]),
                "distinct_valid_negative_tokens": len(
                    expl_neg_tok[h] & expl_valid_tok[h]
                ),
                "censored_observations": expl_cens[h],
                "unavailable_observations": expl_unavail[h],
                "prevalence_among_valid_observations": (
                    expl_pos_obs[h] / expl_valid_obs[h] if expl_valid_obs[h] else None
                ),
                "prevalence_among_valid_tokens": (
                    (len(expl_pos_tok[h] & expl_valid_tok[h]) / len(expl_valid_tok[h]))
                    if expl_valid_tok[h]
                    else None
                ),
                "independent_primary_explosion_events": primary_explosion_events,
            }
            for h in EXPLOSION_IMMINENCE_HORIZONS_SEC
        },
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
        elif head == "graduation_time_to_event":
            row["recommendation"] = (
                "sufficient_on_broad"
                if n_ev >= 1_000
                else ("marginal_on_broad" if n_ev >= 100 else "still_sparse")
            )
        elif head == "graduated_within_h86400_binary":
            row["recommendation"] = (
                "sufficient_on_broad"
                if n_tok >= 10_000 and n_ev >= 1_000
                else ("marginal_on_broad" if n_tok >= 500 else "still_sparse")
            )
        elif head.startswith("explosion_within_") or head == "explosion_onset_primary":
            pos60 = int(
                next(
                    (
                        r["positive_observations"]
                        for r in rows
                        if r["head"] == "explosion_within_60s"
                    ),
                    explosion_obs_before_60s,
                )
            )
            if n_ev >= 5_000 and pos60 >= 2_000:
                row["recommendation"] = "sufficient_on_broad"
            elif n_ev >= 500 and pos60 >= 200:
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
            "Corrects returns_h300 positives to require valid status; corrects "
            "explosion imminence so valid tokens include status-gated "
            "valid_negatives (not merely eventual explosion tokens); splits "
            "graduation into time-to-event vs binary within-H sufficiency."
        ),
    }
