"""Production example construction from frozen stores (not synthetic)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from trencher.data.world_model.training.forbid import assert_no_forbidden, strip_forbidden
from trencher.data.world_model.training.sequences_batch import (
    BroadSequenceReader,
    RichSequenceReader,
    default_window_preset,
)
from trencher.data.world_model.training.targets_batch import (
    TargetStore,
    TTEOverlayStore,
    build_target_payload,
)
from trencher.data.world_model.training.transforms import load_transforms


class ProductionExampleBuilder:
    """Build real feature/target examples for broad or rich populations."""

    def __init__(
        self,
        *,
        world_root: Path,
        split_root: Path,
        population: str,
        window_preset: dict[str, Any] | None = None,
    ) -> None:
        self.world_root = Path(world_root)
        self.split_root = Path(split_root)
        self.population = population
        self.preset = window_preset or default_window_preset()
        cutoffs = json.loads((self.split_root / "cutoffs.json").read_text(encoding="utf-8"))
        self.q70 = float(cutoffs["quantiles_unix"]["q70"])
        self.q85 = float(cutoffs["quantiles_unix"]["q85"])
        tpath = self.split_root / "transforms_train_v1" / "sequence_numeric_stats.json"
        self.transforms = load_transforms(tpath) if tpath.exists() else None

        if population == "broad":
            self.seq = BroadSequenceReader(self.world_root / "broad")
            tparts = sorted(
                (
                    self.world_root
                    / "broad"
                    / "broad_targets_v3"
                    / "observation_targets"
                ).glob("part-*.parquet")
            )
            self.targets = TargetStore(tparts)
            self.tte = TTEOverlayStore(
                [self.split_root / "tte_overlays_broad"]
            )
        else:
            self.seq = RichSequenceReader(self.world_root)
            self.targets = TargetStore(
                [
                    self.world_root
                    / "targets"
                    / "target_dataset_v3"
                    / "observation_targets_v3.parquet"
                ]
            )
            self.tte = TTEOverlayStore(
                [self.split_root / "tte_overlays_rich.parquet"]
            )

    def build(self, assignment: dict[str, Any]) -> dict[str, Any]:
        seq = self.seq.load_for_assignment(
            assignment, preset=self.preset, transforms=self.transforms
        )
        tgt_row = self.targets.get(str(assignment["observation_id"]))
        tte = self.tte.get(str(assignment["observation_id"]))
        targets = build_target_payload(
            assignment, tgt_row, tte, q70=self.q70, q85=self.q85
        )

        age = float(assignment.get("token_age_seconds") or 0.0)
        # Relative cyclical-ish age features; no raw absolute unix as generic feature.
        structured = strip_forbidden(
            {
                "token_age_seconds": age,
                "token_age_log1p": float(np.log1p(max(age, 0.0))),
                "token_age_day_frac": float((age / 86400.0) % 1.0),
                **(seq.get("structured_state") or {}),
                **seq.get("modality_masks", {}),
            }
        )
        assert_no_forbidden(structured)

        seq_tensors = {
            k: v
            for k, v in seq.items()
            if k
            not in {
                "modality_masks",
                "window_meta",
                "truncation",
                "causal_ok",
                "structured_state",
            }
            and not str(k).endswith("_mask")
        }
        return {
            "features": structured,
            "sequences": seq_tensors,
            "sequence_masks": {
                "events": seq.get("events_mask"),
                "curve": seq.get("curve_mask"),
                "creator": seq.get("creator_mask"),
                "micro": seq.get("micro_mask"),
                "ownership": seq.get("ownership_mask"),
                "graph_token": seq.get("graph_token_mask"),
                "wallet_prior": seq.get("wallet_prior_mask"),
            },
            "modality_masks": seq.get("modality_masks") or {},
            "window_meta": seq.get("window_meta") or {},
            "truncation": seq.get("truncation") or {},
            "causal_ok": bool(seq.get("causal_ok", True)),
            "targets": targets,
            "meta": {
                "observation_id": assignment.get("observation_id"),
                "base_split": assignment.get("base_split"),
                "population": self.population,
            },
        }
