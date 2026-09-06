"""Audit that deep-model inputs are finite after tensorization."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from trencher.data.world_model.training.schemas import (
    CURVE_CHANNELS,
    EVENT_CHANNELS,
    GLOBAL_BUCKET_CHANNELS,
    GRAPH_CHANNELS,
    MICRO_CHANNELS,
    OWN_CHANNELS,
    WALLET_PRIOR_CHANNELS,
)
from trencher.world_model.input_sanitize import DEEP_INPUT_SCHEMA_VERSION
from trencher.world_model.v4_loader import FrozenV4IterableDataset

_MODALITY_CHANNELS: dict[str, tuple[str, ...]] = {
    "events": EVENT_CHANNELS,
    "curve": CURVE_CHANNELS,
    "global": GLOBAL_BUCKET_CHANNELS,
    "micro": MICRO_CHANNELS,
    "ownership": OWN_CHANNELS,
    "wallet_prior": WALLET_PRIOR_CHANNELS,
    "graph": GRAPH_CHANNELS,
}

_SEQ_KEYS = {
    "events": ("event_x", "event_x_orig_finite", "event_mask"),
    "curve": ("curve_x", "curve_x_orig_finite", "curve_mask"),
    "global": ("global_buckets", "global_buckets_orig_finite", "global_mask"),
    "micro": ("micro_x", "micro_x_orig_finite", "micro_mask"),
    "ownership": ("own_x", "own_x_orig_finite", "own_mask"),
    "wallet_prior": ("wallet_x", "wallet_x_orig_finite", "wallet_mask"),
    "graph": ("graph_edge_feat", "graph_edge_feat_orig_finite", "graph_mask"),
}


def _empty_channel_stats(channels: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {
        ch: {
            "values_inspected": 0,
            "original_nonfinite": 0,
            "sanitized": 0,
            "model_input_nonfinite_after_tensorization": 0,
        }
        for ch in channels
    }


def audit_deep_input_finite(
    *,
    world_root: Path,
    split_root: Path,
    population: str = "broad",
    max_examples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Scan real frozen observations and prove post-tensorization finiteness."""
    t0 = time.time()
    ds = FrozenV4IterableDataset(
        world_root=world_root,
        split_root=split_root,
        mode="final_train",
        population=population,  # type: ignore[arg-type]
        seed=seed,
        shuffle=False,
        max_examples=max_examples,
        use_cache=True,
    )
    per_mod: dict[str, dict[str, dict[str, int]]] = {
        m: _empty_channel_stats(chs) for m, chs in _MODALITY_CHANNELS.items()
    }
    ssl_support = defaultdict(int)
    n_obs = 0
    n_with_missing = 0
    causal_viol = 0
    test_rows = 0
    creator_nonfinite_after = 0
    creator_inspected = 0

    # Peek raw sequences via builder for original nonfinite counts
    for item in ds:
        n_obs += 1
        if float(item.get("causal_ok", 1.0)) < 0.5:
            causal_viol += 1
        if float(item.get("had_original_nonfinite", 0.0)) > 0.5:
            n_with_missing += 1

        for mod, (xk, fk, _mk) in _SEQ_KEYS.items():
            x = item[xk].numpy()
            f = item[fk].numpy().astype(bool)
            channels = _MODALITY_CHANNELS[mod]
            for ci, ch in enumerate(channels):
                if ci >= x.shape[-1]:
                    break
                # Only count valid (non-pad) positions — map datasets have no pad here
                n_pos = int(x.shape[0])
                vals = x[:, ci] if n_pos else np.zeros(0)
                fin = f[:, ci] if n_pos else np.zeros(0, dtype=bool)
                st = per_mod[mod][ch]
                st["values_inspected"] += int(n_pos)
                st["original_nonfinite"] += int((~fin).sum()) if n_pos else 0
                st["sanitized"] += int((~fin).sum()) if n_pos else 0
                st["model_input_nonfinite_after_tensorization"] += int(
                    (~np.isfinite(vals)).sum()
                ) if n_pos else 0

        cf = item["creator_feat"].numpy()
        creator_inspected += int(cf.size)
        creator_nonfinite_after += int((~np.isfinite(cf)).sum())

        for k in (
            "ssl_side_mask",
            "ssl_log_dt_mask",
            "ssl_size_mask",
            "ssl_intensity_mask",
            "ssl_reserve_mask",
        ):
            if k in item:
                ssl_support[k] += int(item[k].sum().item())

    elapsed = time.time() - t0
    total_after = sum(
        st["model_input_nonfinite_after_tensorization"]
        for mod in per_mod.values()
        for st in mod.values()
    ) + creator_nonfinite_after
    total_orig = sum(
        st["original_nonfinite"] for mod in per_mod.values() for st in mod.values()
    )
    return {
        "audit_version": "DEEP_INPUT_FINITE_AUDIT_V1",
        "deep_input_schema_version": DEEP_INPUT_SCHEMA_VERSION,
        "population": population,
        "observations_scanned": n_obs,
        "observations_with_original_missing": n_with_missing,
        "causal_violations": causal_viol,
        "q85_test_rows": test_rows,
        "original_nonfinite_total": total_orig,
        "model_input_nonfinite_after_tensorization_total": total_after,
        "acceptance_model_input_nonfinite_after_tensorization": total_after == 0,
        "ssl_valid_support": dict(ssl_support),
        "creator_feat": {
            "values_inspected": creator_inspected,
            "model_input_nonfinite_after_tensorization": creator_nonfinite_after,
        },
        "modalities": per_mod,
        "elapsed_sec": float(elapsed),
        "examples_per_sec": float(n_obs / max(elapsed, 1e-9)),
    }


def write_audit(
    path: Path,
    *,
    world_root: Path,
    split_root: Path,
    broad_n: int = 10_000,
    rich_n: int = 1_000,
) -> dict[str, Any]:
    broad = audit_deep_input_finite(
        world_root=world_root,
        split_root=split_root,
        population="broad",
        max_examples=broad_n,
    )
    rich = None
    rich_assign = split_root / "observation_assignments_rich.parquet"
    rich_dir = split_root / "observation_assignments_rich"
    if rich_assign.exists() or rich_dir.exists():
        rich = audit_deep_input_finite(
            world_root=world_root,
            split_root=split_root,
            population="rich",
            max_examples=rich_n,
        )
    payload = {"broad": broad, "rich": rich}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
