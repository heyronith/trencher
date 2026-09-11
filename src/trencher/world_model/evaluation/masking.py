"""Inference-time modality masking diagnostics (not scientific ablations)."""

from __future__ import annotations

from typing import Any

import torch

from trencher.world_model.evaluation import INFERENCE_MASK_DIAGNOSTIC_LABEL

RICH_DIAGNOSTIC_MODALITIES = (
    "microstructure",
    "ownership",
    "wallet_prior",
    "graph",
)

_AVAIL_KEYS = {
    "microstructure": "micro_available",
    "ownership": "own_available",
    "wallet_prior": "wallet_available",
    "graph": "graph_available",
}


def mask_batch_modalities(
    batch: dict[str, Any],
    *,
    omit: set[str] | frozenset[str] | list[str],
) -> dict[str, Any]:
    """Return a shallow-copied batch with omitted modalities marked unavailable.

    Encoders still run, but fusion availability is zero — labeled diagnostic only.
    """
    omit_set = {str(x) for x in omit}
    out = dict(batch)
    for mod in omit_set:
        key = _AVAIL_KEYS.get(mod)
        if key is None or key not in out:
            continue
        avail = out[key]
        if torch.is_tensor(avail):
            out[key] = torch.zeros_like(avail)
        else:
            out[key] = 0.0
    return out


def active_modalities_for_phase(
    phase: str,
    *,
    modalities_broad: tuple[str, ...] | list[str],
    modalities_rich: tuple[str, ...] | list[str],
    omit: set[str] | frozenset[str] | list[str] | None = None,
) -> set[str]:
    """Compute the active modality set for training or true ablations."""
    if phase == "ssl":
        active = {"events"}
    elif phase == "broad":
        active = set(modalities_broad)
    else:
        active = set(modalities_rich)
    if omit:
        active -= {str(x) for x in omit}
    return active


def diagnostic_configs() -> list[dict[str, Any]]:
    """Prespecified inference-time mask diagnostics."""
    configs = [
        {
            "run_id": "diagnostic_full",
            "label": INFERENCE_MASK_DIAGNOSTIC_LABEL,
            "omit": [],
            "note": "baseline inference (no mask)",
        }
    ]
    for mod in RICH_DIAGNOSTIC_MODALITIES:
        configs.append(
            {
                "run_id": f"diagnostic_no_{mod}",
                "label": INFERENCE_MASK_DIAGNOSTIC_LABEL,
                "omit": [mod],
                "note": f"zero availability for {mod} at inference only",
            }
        )
    configs.append(
        {
            "run_id": "diagnostic_rich_all_off",
            "label": INFERENCE_MASK_DIAGNOSTIC_LABEL,
            "omit": list(RICH_DIAGNOSTIC_MODALITIES),
            "note": "approximate Stage-C information removal at inference time",
        }
    )
    return configs


def rank_degradation(
    baseline_score: float | None,
    ablated_scores: dict[str, float | None],
    *,
    higher_is_better: bool = True,
) -> list[dict[str, Any]]:
    """Rank modalities by validation degradation vs baseline."""
    rows: list[dict[str, Any]] = []
    if baseline_score is None:
        return rows
    for name, score in ablated_scores.items():
        if score is None:
            delta = None
            degrad = None
        else:
            delta = float(score - baseline_score)
            if higher_is_better:
                degrad = float(baseline_score - score)
            else:
                degrad = float(score - baseline_score)
        rows.append(
            {
                "modality_or_config": name,
                "score": score,
                "delta_vs_baseline": delta,
                "degradation": degrad,
            }
        )
    rows.sort(key=lambda r: (-1e99 if r["degradation"] is None else r["degradation"]), reverse=True)
    return rows
