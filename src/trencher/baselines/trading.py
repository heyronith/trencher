"""Trading-usefulness helpers: preserve legitimate zeros; no fabricated net fees."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_float_keep_zero(value: Any) -> float:
    """Convert to float; keep 0.0; map None/non-finite to NaN.

    Never use ``value or np.nan`` — that incorrectly drops legitimate zeros.
    """
    if value is None:
        return float("nan")
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def trading_arrays_from_targets(
    targets: list[dict[str, Any]],
    *,
    gross_keys: tuple[str, ...] = ("h300_simple_return", "exec_1p0sol_gross_return"),
    entry_impact_keys: tuple[str, ...] = ("exec_1p0sol_entry_impact_adjusted_return",),
    curve_impact_keys: tuple[str, ...] = ("exec_1p0sol_curve_impact_adjusted_return",),
    mfe_keys: tuple[str, ...] = ("h300_mfe",),
    mae_keys: tuple[str, ...] = ("h300_mae",),
) -> dict[str, np.ndarray]:
    def _pick(row: dict[str, Any], keys: tuple[str, ...]) -> float:
        for k in keys:
            if k in row and row[k] is not None:
                return as_float_keep_zero(row[k])
        return float("nan")

    n = len(targets)
    gross = np.empty(n, dtype=np.float64)
    entry = np.empty(n, dtype=np.float64)
    curve = np.empty(n, dtype=np.float64)
    mfe = np.empty(n, dtype=np.float64)
    mae = np.empty(n, dtype=np.float64)
    for i, t in enumerate(targets):
        gross[i] = _pick(t, gross_keys)
        entry[i] = _pick(t, entry_impact_keys)
        curve[i] = _pick(t, curve_impact_keys)
        mfe[i] = _pick(t, mfe_keys)
        mae[i] = _pick(t, mae_keys)
    return {
        "gross_return": gross,
        "entry_impact_adjusted_return": entry,
        "curve_impact_adjusted_return": curve,
        "mfe": mfe,
        "mae": mae,
        "net_return_unavailable": True,
    }
