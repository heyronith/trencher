"""Helpers for Phase-3 look-ahead leak audit."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scripts.review.lead_intake_lib import binary_auc_ap

Q85 = {"q85_test_accessed": False}
REPO = Path(__file__).resolve().parents[3]
WORLD = REPO / "data" / "external" / "world_model"
SPLIT = WORLD / "splits" / "split_v1"
OUT = REPO / "manifests" / "world_model_data" / "evaluation" / "phase3_leak_v1"
SCRATCH = REPO / "scratch" / "phase3_leak_v1"
STAGE_C_SHA = "11bf593e10e5dcc08915fc0fa2300ce988592f660b3608a5024c43121051d839"
WANT_SC_PRED = "2d97c92e3bf6a9b721355d9ea2537b7a54787594ba0ee659105643e8200def13"


def to_unix(t: Any) -> float | None:
    if t is None:
        return None
    if hasattr(t, "timestamp"):
        return float(t.timestamp())
    try:
        return float(t)
    except (TypeError, ValueError):
        return None


def load_signature_targets_by_mint(
    path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-mint selected signature times + extreme markers.

    Times come from ``sequence_microstructure_v2.event_time_unix`` (exact match to
    ``micro_n``). Size/curve fields come from ``signature_targets.parquet`` joined
    by ``tx_signature`` (datetime columns there can disagree with unix by TZ).
    """
    del path  # targets path fixed; micro store is the time authority
    st = pq.read_table(WORLD / "microstructure" / "signature_targets.parquet")
    sol_by_sig: dict[str, float] = {}
    curve_by_sig: dict[str, float] = {}
    for sig, sol, curve in zip(
        st["tx_signature"].to_pylist(),
        st["sol_amount"].to_pylist(),
        st["curve_pct_depleted"].to_pylist(),
        strict=True,
    ):
        if not sig:
            continue
        sol_by_sig[str(sig)] = float(sol) if sol is not None else float("nan")
        curve_by_sig[str(sig)] = float(curve) if curve is not None else float("nan")

    idx = pq.read_table(
        WORLD / "sequences/sequence_dataset_v2/sequence_microstructure_v2_entity_index.parquet"
    )
    micro = pq.read_table(
        WORLD / "sequences/sequence_dataset_v2/sequence_microstructure_v2.parquet",
        columns=["event_time_unix", "tx_signature"],
    )
    times_all = micro["event_time_unix"].to_numpy().astype(np.float64)
    sigs_all = micro["tx_signature"].to_pylist()

    out: dict[str, dict[str, Any]] = {}
    for mint, lo, hi in zip(
        idx["entity"].to_pylist(),
        idx["start_idx"].to_pylist(),
        idx["end_idx"].to_pylist(),
        strict=True,
    ):
        lo_i, hi_i = int(lo), int(hi)
        rows = []
        for j in range(lo_i, hi_i):
            sig = sigs_all[j]
            rows.append(
                {
                    "sig": sig,
                    "t": float(times_all[j]),
                    "sol": sol_by_sig.get(str(sig), float("nan")),
                    "curve": curve_by_sig.get(str(sig), float("nan")),
                }
            )
        rows = sorted(rows, key=lambda r: (r["t"], r["sig"] or ""))
        times = np.asarray([r["t"] for r in rows], dtype=np.float64)
        sols = np.asarray([r["sol"] for r in rows], dtype=np.float64)
        curves = np.asarray([r["curve"] for r in rows], dtype=np.float64)
        last_i = int(np.argmax(times)) if times.size else 0
        finite_sol = np.isfinite(sols)
        largest_i = int(np.nanargmax(sols)) if finite_sol.any() else last_i
        finite_c = np.isfinite(curves)
        max_curve_i = int(np.nanargmax(curves)) if finite_c.any() else last_i
        out[str(mint)] = {
            "n_sel_total": len(rows),
            "times": times,
            "last_trade_t": float(times[last_i]) if times.size else float("nan"),
            "largest_trade_t": float(times[largest_i]) if times.size else float("nan"),
            "max_curve_trade_t": float(times[max_curve_i]) if times.size else float("nan"),
        }
    return out


def load_event_times_by_mint() -> dict[str, np.ndarray]:
    """All sequence_events_v2 times per mint (frozen event store)."""
    idx = pq.read_table(
        WORLD / "sequences/sequence_dataset_v2/sequence_events_v2_entity_index.parquet"
    )
    ev = pq.read_table(
        WORLD / "sequences/sequence_dataset_v2/sequence_events_v2.parquet",
        columns=["event_time_unix"],
    )
    times_all = ev["event_time_unix"].to_numpy().astype(np.float64)
    out: dict[str, np.ndarray] = {}
    for mint, lo, hi in zip(
        idx["entity"].to_pylist(),
        idx["start_idx"].to_pylist(),
        idx["end_idx"].to_pylist(),
        strict=True,
    ):
        out[str(mint)] = np.sort(times_all[int(lo) : int(hi)])
    return out


def n_before(times: np.ndarray, T: float) -> int:
    """Count events with time <= T (matches CausalRichSequenceIndex side='right')."""
    if times.size == 0:
        return 0
    return int(np.searchsorted(times, T, side="right"))


def decile_labels(x: np.ndarray) -> np.ndarray:
    """1..10 deciles; NaN stays -1."""
    out = np.full(x.shape, -1, dtype=int)
    m = np.isfinite(x)
    if m.sum() == 0:
        return out
    # rank-based deciles
    ranks = np.argsort(np.argsort(x[m]))
    # 0..n-1 -> 1..10
    n = int(m.sum())
    dec = np.minimum(10, (ranks * 10) // max(n, 1) + 1)
    out[m] = dec.astype(int)
    return out


def rate_by_group(y: np.ndarray, group: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for g in sorted(set(int(v) for v in group if int(v) >= 0)):
        m = group == g
        if not m.any():
            continue
        yy = y[m]
        out[str(g)] = {
            "n": int(m.sum()),
            "n_pos": int(yy.sum()),
            "rate": float(yy.mean()) if yy.size else None,
        }
    return out


def rate_by_bool(y: np.ndarray, flag: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for val, name in ((False, "false"), (True, "true")):
        m = flag == val
        yy = y[m]
        out[name] = {
            "n": int(m.sum()),
            "n_pos": int(yy.sum()),
            "rate": float(yy.mean()) if yy.size else None,
        }
    return out


def single_feature_auroc(y: np.ndarray, score: np.ndarray, elig: np.ndarray) -> dict[str, Any]:
    yy = y[elig].astype(int)
    ss = score[elig]
    if yy.size == 0 or len(np.unique(yy)) < 2:
        return {
            "n": int(yy.size),
            "n_pos": int(yy.sum()) if yy.size else 0,
            "roc_auc": None,
            "pr_auc": None,
        }
    met = binary_auc_ap(yy, ss)
    return {
        "n": int(met["n"]),
        "n_pos": int(yy.sum()),
        "roc_auc": float(met["roc_auc"]),
        "pr_auc": float(met["pr_auc"]),
    }


def iso_now() -> str:
    return datetime.now(UTC).isoformat()
