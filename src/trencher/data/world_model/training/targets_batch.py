"""Load frozen target rows + split eligibility + TTE overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

from trencher.data.world_model.splits.tte_overlay import apply_tte_overlays_for_row
from trencher.data.world_model.targets.status import (
    CORPUS_END_UNIX,
    EXPLOSION_IMMINENCE_HORIZONS_SEC,
    HORIZONS_SEC,
    STATUS_RIGHT_CENSORED,
)

TARGET_VALUE_COLS = [
    "observation_id",
    "mint",
    "time_to_graduation_sec",
    "time_to_graduation_status",
    "graduated_at_or_before_t",
    "time_to_explosion_sec",
    "explosion_event_observed",
    "explosion_already_started_at_t",
    "explosion_right_censored",
    "remaining_upside_to_explosion_peak",
    "explosion_peak_future_return",
]


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
                f"grad_h{h}_status",
                f"grad_h{h}_graduated",
            ]
        )
        if h in (30, 60, 120, 300, 600, 1800, 3600, 21600, 86400):
            cols.extend([f"surv_h{h}_status", f"surv_h{h}_active"])
    for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
        cols.extend([f"explosion_within_{h}s", f"explosion_within_{h}s_status"])
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
        # Chunk to keep filter size reasonable
        for i in range(0, len(missing), 512):
            chunk = missing[i : i + 512]
            table = self._dataset.to_table(
                columns=self._cols,
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


def _pack_head(
    *,
    value: Any,
    frozen_status: Any,
    eligible: bool,
    effective_status: Any = None,
    split_boundary_censored: bool = False,
) -> dict[str, Any]:
    missing = value is None or (isinstance(value, float) and not np.isfinite(value))
    return {
        "value": None if missing else value,
        "frozen_status": frozen_status,
        "split_eligible": bool(eligible),
        "effective_status": effective_status if effective_status is not None else frozen_status,
        "split_boundary_censored": bool(split_boundary_censored),
        "is_missing": bool(missing),
        "value_or_nan": np.nan if missing else value,
    }


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

    for h in (10, 30, 60, 300, 3600, 86400):
        elig = bool(assignment.get(f"eligible_returns_h{h}", False))
        st = target_row.get(f"h{h}_status")
        out[f"returns_h{h}"] = _pack_head(
            value=target_row.get(f"h{h}_simple_return"),
            frozen_status=st,
            eligible=elig,
            effective_status=st if elig else "split_boundary_purged",
        )
        out[f"mfe_h{h}"] = _pack_head(
            value=target_row.get(f"h{h}_mfe"),
            frozen_status=st,
            eligible=bool(assignment.get(f"eligible_mfe_mae_h{h}", elig)),
            effective_status=st if elig else "split_boundary_purged",
        )
        out[f"mae_h{h}"] = _pack_head(
            value=target_row.get(f"h{h}_mae"),
            frozen_status=st,
            eligible=bool(assignment.get(f"eligible_mfe_mae_h{h}", elig)),
            effective_status=st if elig else "split_boundary_purged",
        )

    for h in (30, 60, 300, 3600, 86400):
        elig = bool(assignment.get(f"eligible_survival_h{h}", False))
        st = target_row.get(f"surv_h{h}_status")
        out[f"survival_h{h}"] = _pack_head(
            value=target_row.get(f"surv_h{h}_active"),
            frozen_status=st,
            eligible=elig,
            effective_status=st if elig else "split_boundary_purged",
        )

    for h in EXPLOSION_IMMINENCE_HORIZONS_SEC:
        elig = bool(assignment.get(f"eligible_explosion_imminence_h{h}", False))
        st = target_row.get(f"explosion_within_{h}s_status")
        out[f"explosion_imminence_h{h}"] = _pack_head(
            value=target_row.get(f"explosion_within_{h}s"),
            frozen_status=st,
            eligible=elig,
            effective_status=st if elig else "split_boundary_purged",
        )

    if tte_overlay is None:
        tte_overlay = apply_tte_overlays_for_row(
            target_row,
            observation_time_t=float(assignment["observation_time_t"]),
            base_split=str(assignment["base_split"]),
            q70=q70,
            q85=q85,
            corpus_end=CORPUS_END_UNIX,
        )

    grad_censored = bool(tte_overlay.get("graduation_tte_split_boundary_censored"))
    grad_obs = bool(tte_overlay.get("graduation_tte_effective_event_observed")) and (
        not grad_censored
    )
    out["graduation_tte"] = {
        "value": tte_overlay.get("graduation_tte_effective_tte_sec") if grad_obs else None,
        "effective_event_observed": grad_obs,
        "effective_tte_sec": tte_overlay.get("graduation_tte_effective_tte_sec")
        if grad_obs
        else None,
        "effective_censoring_status": tte_overlay.get(
            "graduation_tte_effective_censoring_status"
        ),
        "administrative_censor_time": tte_overlay.get(
            "graduation_tte_administrative_censor_time"
        ),
        "split_boundary_censored": grad_censored,
        "frozen_status": target_row.get("time_to_graduation_status"),
        "frozen_tte_sec": target_row.get("time_to_graduation_sec"),
        "split_eligible": bool(assignment.get("eligible_graduation_tte", True)),
        "is_missing": not grad_obs,
    }

    expl_censored = bool(tte_overlay.get("explosion_tte_split_boundary_censored"))
    expl_obs = bool(tte_overlay.get("explosion_tte_effective_event_observed")) and (
        not expl_censored
    )
    out["explosion_tte"] = {
        "value": tte_overlay.get("explosion_tte_effective_tte_sec") if expl_obs else None,
        "effective_event_observed": expl_obs,
        "effective_tte_sec": tte_overlay.get("explosion_tte_effective_tte_sec")
        if expl_obs
        else None,
        "effective_censoring_status": tte_overlay.get(
            "explosion_tte_effective_censoring_status"
        ),
        "administrative_censor_time": tte_overlay.get(
            "explosion_tte_administrative_censor_time"
        ),
        "split_boundary_censored": expl_censored,
        "frozen_tte_sec": target_row.get("time_to_explosion_sec"),
        "frozen_event_observed": bool(target_row.get("explosion_event_observed")),
        "split_eligible": bool(assignment.get("eligible_explosion_tte", True)),
        "is_missing": not expl_obs,
    }

    out["explosion_magnitude"] = _pack_head(
        value=target_row.get("remaining_upside_to_explosion_peak")
        or target_row.get("explosion_peak_future_return"),
        frozen_status="valid_positive"
        if target_row.get("explosion_event_observed")
        else STATUS_RIGHT_CENSORED,
        eligible=bool(assignment.get("eligible_explosion_imminence_h60", True)),
    )
    return out
