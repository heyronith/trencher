"""Load frozen target rows + split eligibility + TTE overlays.

Never zero-fill unavailable/censored values. Never use ``a or b`` for numerics
where 0.0 is a valid target. Censored TTE retains duration with event_observed=0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

from trencher.data.world_model.splits.tte_overlay import (
    apply_tte_overlays_for_row,
    normalize_overlay_row,
)
from trencher.data.world_model.targets.status import (
    BARRIER_PAIRS,
    CORPUS_END_UNIX,
    EXPLOSION_IMMINENCE_HORIZONS_SEC,
    HORIZONS_SEC,
    POSITION_SIZES_SOL,
    STATUS_RIGHT_CENSORED,
    SURVIVAL_HORIZONS_SEC,
)

TARGET_VALUE_COLS = [
    "observation_id",
    "mint",
    "time_to_graduation_sec",
    "time_to_graduation_status",
    "graduated_at_or_before_t",
    "graduated",
    "time_to_explosion_sec",
    "explosion_event_observed",
    "explosion_already_started_at_t",
    "explosion_right_censored",
    "remaining_upside_to_explosion_peak",
    "explosion_peak_future_return",
    "time_to_inactivity_sec",
    "time_to_inactivity_status",
    "ownership_target_available",
    "ownership_target_status",
    "holder_count_change",
    "churn_persistence",
    "holder_state_reconstructable",
]


def _first_defined(*vals: Any) -> Any:
    """Return the first value that is not None (0.0 / False are valid)."""
    for v in vals:
        if v is not None:
            return v
    return None


def _horizon_cols() -> list[str]:
    cols: list[str] = []
    for h in HORIZONS_SEC:
        cols.extend(
            [
                f"h{h}_status",
                f"h{h}_simple_return",
                f"h{h}_log_return",
                f"h{h}_mfe",
                f"h{h}_mae",
                f"h{h}_log_mfe",
                f"h{h}_log_mae",
                f"grad_h{h}_status",
                f"grad_h{h}_graduated",
                f"act_h{h}_status",
                f"act_h{h}_trade_count",
                f"act_h{h}_activity_rate",
                f"act_h{h}_activity_acceleration",
                f"h{h}_future_mark",
                f"h{h}_future_mark_time",
                f"h{h}_future_path_start_idx",
                f"h{h}_future_path_end_idx",
            ]
        )
        for _, _, bid in BARRIER_PAIRS:
            cols.extend(
                [
                    f"h{h}_barrier_{bid}_outcome",
                    f"h{h}_barrier_{bid}_time_to_upper",
                    f"h{h}_barrier_{bid}_time_to_lower",
                    f"h{h}_barrier_{bid}_time_to_first",
                    f"h{h}_barrier_{bid}_first_type",
                ]
            )
    for h in SURVIVAL_HORIZONS_SEC:
        cols.extend([f"surv_h{h}_status", f"surv_h{h}_active"])
    for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
        cols.extend([f"explosion_within_{h}s", f"explosion_within_{h}s_status"])
    for sol in POSITION_SIZES_SOL:
        tag = str(sol).replace(".", "p")
        cols.extend(
            [
                f"exec_{tag}sol_available",
                f"exec_{tag}sol_status",
                f"exec_{tag}sol_gross_return",
                f"exec_{tag}sol_net_return",
                f"exec_{tag}sol_execution_target_available",
                f"exec_{tag}sol_slippage",
                f"exec_{tag}sol_price_impact",
            ]
        )
    return cols


class TargetStore:
    """Lazy/batched target lookup by observation_id (Parquet dataset filter)."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = [Path(p) for p in paths]
        self._cache: dict[str, dict[str, Any]] = {}
        self._dataset = ds.dataset([str(p) for p in self.paths], format="parquet")
        schema_names = set(self._dataset.schema.names)
        self._cols = [c for c in TARGET_VALUE_COLS + _horizon_cols() if c in schema_names]

    def prefetch(self, observation_ids: list[str]) -> None:
        missing = [oid for oid in observation_ids if oid not in self._cache]
        if not missing:
            return
        # Large chunks: one Arrow filter per mini-batch / shard window
        chunk_size = max(512, min(len(missing), 8192))
        for i in range(0, len(missing), chunk_size):
            chunk = missing[i : i + chunk_size]
            table = self._dataset.to_table(
                columns=self._cols,
                filter=pc.field("observation_id").isin(chunk),
            )
            for row in table.to_pylist():
                oid = row.get("observation_id")
                if oid:
                    self._cache[str(oid)] = row
        # Bound cache growth for long epochs
        if len(self._cache) > 200_000:
            # drop arbitrary oldest half
            keys = list(self._cache.keys())
            for k in keys[: len(keys) // 2]:
                del self._cache[k]

    def get(self, observation_id: str) -> dict[str, Any] | None:
        oid = str(observation_id)
        if oid not in self._cache:
            self.prefetch([oid])
        return self._cache.get(oid)

    def put_many(self, rows: dict[str, dict[str, Any]]) -> None:
        self._cache.update(rows)


class TTEOverlayStore:
    def __init__(self, paths: list[Path]) -> None:
        self.paths = list(paths)
        self._cache: dict[str, dict[str, Any]] = {}
        files: list[str] = []
        for path in self.paths:
            p = Path(path)
            if p.is_dir():
                files.extend(str(f) for f in sorted(p.glob("part-*.parquet")))
            elif p.exists():
                files.append(str(p))
        self._dataset = ds.dataset(files, format="parquet") if files else None

    def prefetch(self, observation_ids: list[str]) -> None:
        if self._dataset is None:
            return
        missing = [oid for oid in observation_ids if oid not in self._cache]
        if not missing:
            return
        for i in range(0, len(missing), 512):
            chunk = missing[i : i + 512]
            table = self._dataset.to_table(
                filter=pc.field("observation_id").isin(chunk),
            )
            for row in table.to_pylist():
                oid = row.get("observation_id")
                if oid:
                    self._cache[str(oid)] = row

    def get(self, observation_id: str) -> dict[str, Any] | None:
        oid = str(observation_id)
        if oid not in self._cache:
            self.prefetch([oid])
        return self._cache.get(oid)

    def put_many(self, rows: dict[str, dict[str, Any]]) -> None:
        self._cache.update(rows)


def _pack_head(
    *,
    value: Any,
    frozen_status: Any,
    eligible: bool,
    effective_status: Any = None,
    split_boundary_censored: bool = False,
    censor_mask: bool | None = None,
) -> dict[str, Any]:
    missing = value is None or (isinstance(value, float) and not np.isfinite(value))
    if isinstance(value, (bool, np.bool_)):
        missing = False
    status = effective_status if effective_status is not None else frozen_status
    censored = (
        bool(censor_mask)
        if censor_mask is not None
        else (str(status) in {STATUS_RIGHT_CENSORED, "split_boundary_purged"})
    )
    return {
        "value": None if missing else value,
        "frozen_status": frozen_status,
        "split_eligible": bool(eligible),
        "effective_status": status if eligible else "split_boundary_purged",
        "split_boundary_censored": bool(split_boundary_censored),
        "is_missing": bool(missing) or not eligible,
        "is_censored": bool(censored) or (not eligible),
        "value_or_nan": np.nan if (missing or not eligible) else value,
    }


def _pack_tte(
    *,
    overlay: dict[str, Any],
    frozen_status: Any,
    frozen_tte: Any,
    eligible: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """TTE pack: always retain duration when present, even if censored."""
    duration = overlay.get("duration_sec")
    event_obs = bool(overlay.get("event_observed"))
    censored = bool(overlay.get("censored"))
    at_risk = bool(overlay.get("at_risk"))
    missing = duration is None or not eligible or not at_risk
    out = {
        "value": None if missing else duration,
        "duration_sec": duration,  # retained even when censored / event=0
        "event_observed": event_obs,
        "censored": censored,
        "at_risk": at_risk,
        "effective_status": overlay.get("effective_status"),
        "effective_event_observed": event_obs,
        "effective_tte_sec": duration,
        "effective_censoring_status": overlay.get("effective_status"),
        "administrative_censor_time": overlay.get("administrative_censor_time"),
        "split_boundary_censored": bool(overlay.get("split_boundary_censored")),
        "frozen_status": frozen_status,
        "frozen_tte_sec": frozen_tte,
        "split_eligible": bool(eligible),
        "is_missing": bool(missing),
        "is_censored": censored,
        "value_or_nan": np.nan if missing else duration,
    }
    if extra:
        out.update(extra)
    return out


def build_target_payload(
    assignment: dict[str, Any],
    target_row: dict[str, Any] | None,
    tte_overlay: dict[str, Any] | None,
    *,
    q70: float,
    q85: float,
) -> dict[str, Any]:
    """Assemble multi-task target dict with status/eligibility masks."""
    out: dict[str, Any] = {}
    if target_row is None:
        out["available"] = False
        return out
    out["available"] = True

    # --- Returns / MFE / MAE ---
    for h in HORIZONS_SEC:
        elig = bool(assignment.get(f"eligible_returns_h{h}", False))
        st = target_row.get(f"h{h}_status")
        out[f"returns_h{h}"] = _pack_head(
            value=target_row.get(f"h{h}_simple_return"),
            frozen_status=st,
            eligible=elig,
            effective_status=st if elig else "split_boundary_purged",
        )
        out[f"log_returns_h{h}"] = _pack_head(
            value=target_row.get(f"h{h}_log_return"),
            frozen_status=st,
            eligible=elig,
            effective_status=st if elig else "split_boundary_purged",
        )
        mfe_elig = bool(assignment.get(f"eligible_mfe_mae_h{h}", elig))
        out[f"mfe_h{h}"] = _pack_head(
            value=target_row.get(f"h{h}_mfe"),
            frozen_status=st,
            eligible=mfe_elig,
            effective_status=st if mfe_elig else "split_boundary_purged",
        )
        out[f"mae_h{h}"] = _pack_head(
            value=target_row.get(f"h{h}_mae"),
            frozen_status=st,
            eligible=mfe_elig,
            effective_status=st if mfe_elig else "split_boundary_purged",
        )

        # Competing barriers
        bar_elig = bool(assignment.get(f"eligible_barriers_h{h}", elig))
        for _, _, bid in BARRIER_PAIRS:
            out[f"barrier_{bid}_h{h}"] = _pack_head(
                value=target_row.get(f"h{h}_barrier_{bid}_outcome"),
                frozen_status=target_row.get(f"h{h}_barrier_{bid}_outcome"),
                eligible=bar_elig,
            )

        # Activity / persistence
        act_st = target_row.get(f"act_h{h}_status", st)
        out[f"persistence_trade_count_h{h}"] = _pack_head(
            value=target_row.get(f"act_h{h}_trade_count"),
            frozen_status=act_st,
            eligible=elig,
        )
        out[f"persistence_activity_rate_h{h}"] = _pack_head(
            value=target_row.get(f"act_h{h}_activity_rate"),
            frozen_status=act_st,
            eligible=elig,
        )
        out[f"persistence_activity_accel_h{h}"] = _pack_head(
            value=target_row.get(f"act_h{h}_activity_acceleration"),
            frozen_status=act_st,
            eligible=elig,
        )

        # Future-path refs (indices only — not outcome values)
        out[f"future_path_ref_h{h}"] = {
            "value": {
                "mark": target_row.get(f"h{h}_future_mark"),
                "mark_time": target_row.get(f"h{h}_future_mark_time"),
                "start_idx": target_row.get(f"h{h}_future_path_start_idx"),
                "end_idx": target_row.get(f"h{h}_future_path_end_idx"),
            },
            "frozen_status": st,
            "split_eligible": elig,
            "effective_status": st if elig else "split_boundary_purged",
            "is_missing": target_row.get(f"h{h}_future_mark") is None,
            "is_censored": not elig,
        }

        # Graduation binary @ horizon
        out[f"graduation_binary_h{h}"] = _pack_head(
            value=target_row.get(f"grad_h{h}_graduated"),
            frozen_status=target_row.get(f"grad_h{h}_status"),
            eligible=bool(assignment.get(f"eligible_graduation_h{h}", elig)),
        )

    # --- Survival / inactivity ---
    for h in SURVIVAL_HORIZONS_SEC:
        elig = bool(assignment.get(f"eligible_survival_h{h}", False))
        st = target_row.get(f"surv_h{h}_status")
        out[f"survival_h{h}"] = _pack_head(
            value=target_row.get(f"surv_h{h}_active"),
            frozen_status=st,
            eligible=elig,
            effective_status=st if elig else "split_boundary_purged",
        )
    out["time_to_inactivity"] = _pack_head(
        value=target_row.get("time_to_inactivity_sec"),
        frozen_status=target_row.get("time_to_inactivity_status"),
        eligible=bool(assignment.get("eligible_survival_h1800", True)),
    )

    # --- Explosion imminence ---
    for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
        elig = bool(assignment.get(f"eligible_explosion_imminence_h{h}", False))
        st = target_row.get(f"explosion_within_{h}s_status")
        out[f"explosion_imminence_h{h}"] = _pack_head(
            value=target_row.get(f"explosion_within_{h}s"),
            frozen_status=st,
            eligible=elig,
            effective_status=st if elig else "split_boundary_purged",
        )

    # --- TTE overlays ---
    if tte_overlay is None:
        tte_overlay = apply_tte_overlays_for_row(
            target_row,
            observation_time_t=float(assignment["observation_time_t"]),
            base_split=str(assignment["base_split"]),
            q70=q70,
            q85=q85,
            corpus_end=CORPUS_END_UNIX,
        )

    grad = normalize_overlay_row(tte_overlay, "graduation_tte")
    # Prefer live recompute if already-occurred flag present on frozen row
    if target_row.get("graduated_at_or_before_t") and grad.get("at_risk", True):
        from trencher.data.world_model.splits.tte_overlay import (
            admin_boundary_unix,
            apply_graduation_tte_overlay,
        )

        live = apply_graduation_tte_overlay(
            observation_time_t=float(assignment["observation_time_t"]),
            split_end=admin_boundary_unix(
                str(assignment["base_split"]), q70=q70, q85=q85
            ),
            time_to_graduation_sec=target_row.get("time_to_graduation_sec"),
            time_to_graduation_status=target_row.get("time_to_graduation_status"),
            graduated_at_or_before_t=True,
        )
        grad = normalize_overlay_row(live, "graduation_tte")

    out["graduation_tte"] = _pack_tte(
        overlay=grad,
        frozen_status=target_row.get("time_to_graduation_status"),
        frozen_tte=target_row.get("time_to_graduation_sec"),
        eligible=bool(assignment.get("eligible_graduation_tte", True)),
    )

    expl = normalize_overlay_row(tte_overlay, "explosion_tte")
    if target_row.get("explosion_already_started_at_t") and expl.get("at_risk", True):
        from trencher.data.world_model.splits.tte_overlay import (
            admin_boundary_unix,
            apply_explosion_tte_overlay,
        )

        live = apply_explosion_tte_overlay(
            observation_time_t=float(assignment["observation_time_t"]),
            split_end=admin_boundary_unix(
                str(assignment["base_split"]), q70=q70, q85=q85
            ),
            time_to_explosion_sec=target_row.get("time_to_explosion_sec"),
            explosion_event_observed=target_row.get("explosion_event_observed"),
            explosion_already_started_at_t=True,
            explosion_right_censored=target_row.get("explosion_right_censored"),
        )
        expl = normalize_overlay_row(live, "explosion_tte")

    out["explosion_tte"] = _pack_tte(
        overlay=expl,
        frozen_status=None,
        frozen_tte=target_row.get("time_to_explosion_sec"),
        eligible=bool(assignment.get("eligible_explosion_tte", True)),
        extra={
            "frozen_event_observed": bool(target_row.get("explosion_event_observed")),
        },
    )

    # Magnitude: 0.0 is valid — never use `a or b`
    mag = _first_defined(
        target_row.get("remaining_upside_to_explosion_peak"),
        target_row.get("explosion_peak_future_return"),
    )
    out["explosion_magnitude"] = _pack_head(
        value=mag,
        frozen_status="valid_positive"
        if target_row.get("explosion_event_observed")
        else STATUS_RIGHT_CENSORED,
        eligible=bool(assignment.get("eligible_explosion_imminence_h60", True)),
    )

    # Ownership (sparse)
    own_avail = bool(target_row.get("ownership_target_available"))
    out["ownership"] = _pack_head(
        value=target_row.get("holder_count_change"),
        frozen_status=target_row.get("ownership_target_status"),
        eligible=own_avail,
    )
    out["ownership_churn_persistence"] = _pack_head(
        value=target_row.get("churn_persistence"),
        frozen_status=target_row.get("ownership_target_status"),
        eligible=own_avail,
    )

    # Execution primitives
    for sol in POSITION_SIZES_SOL:
        tag = str(sol).replace(".", "p")
        avail = target_row.get(f"exec_{tag}sol_execution_target_available")
        if avail is None:
            avail = target_row.get(f"exec_{tag}sol_available")
        elig = bool(avail) if avail is not None else False
        st = target_row.get(f"exec_{tag}sol_status")
        out[f"execution_gross_{tag}sol"] = _pack_head(
            value=target_row.get(f"exec_{tag}sol_gross_return"),
            frozen_status=st,
            eligible=elig,
        )
        out[f"execution_net_{tag}sol"] = _pack_head(
            value=target_row.get(f"exec_{tag}sol_net_return"),
            frozen_status=st,
            eligible=elig,
        )
        out[f"execution_slippage_{tag}sol"] = _pack_head(
            value=target_row.get(f"exec_{tag}sol_slippage"),
            frozen_status=st,
            eligible=elig,
        )

    out["families_present"] = sorted(
        {
            "returns",
            "barriers",
            "mfe_mae",
            "survival",
            "graduation_binary",
            "graduation_tte",
            "explosion_imminence",
            "explosion_tte",
            "explosion_magnitude",
            "persistence",
            "ownership",
            "execution",
            "future_path_refs",
        }
    )
    return out
