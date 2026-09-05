"""Production example construction from frozen stores (not synthetic)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from trencher.data.world_model.training.cache_build import JoinedTargetCache, cache_root
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

_TTE_PREFIXES = ("graduation_tte_", "explosion_tte_")


def _split_joined_row(
    joined: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if joined is None:
        return None, None
    tte = {
        k: v
        for k, v in joined.items()
        if k.startswith(_TTE_PREFIXES)
        or k in ("admin_boundary_unix", "tte_admin_boundary_unix", "tte_base_split")
    }
    if tte:
        tte["observation_id"] = joined.get("observation_id")
    return joined, tte or None


class ProductionExampleBuilder:
    """Build real feature/target examples for broad or rich populations."""

    def __init__(
        self,
        *,
        world_root: Path,
        split_root: Path,
        population: str,
        window_preset: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> None:
        self.world_root = Path(world_root)
        self.split_root = Path(split_root)
        self.population = population
        self.preset = window_preset or default_window_preset()
        self.use_cache = bool(use_cache)
        cutoffs = json.loads((self.split_root / "cutoffs.json").read_text(encoding="utf-8"))
        self.q70 = float(cutoffs["quantiles_unix"]["q70"])
        self.q85 = float(cutoffs["quantiles_unix"]["q85"])

        croot = cache_root(self.split_root)
        tpath = croot / "transforms_train_v1" / "sequence_numeric_stats.json"
        if not tpath.exists():
            tpath = self.split_root / "transforms_train_v1" / "sequence_numeric_stats.json"
        self.transforms = load_transforms(tpath) if tpath.exists() else None

        self.joined: JoinedTargetCache | None = None
        if self.use_cache and (croot / "manifest.json").exists():
            # Require at least one joined parquet before enabling
            if population == "broad":
                sample = next(
                    (croot / "broad" / "by_mint_partition").glob("*/joined.parquet"),
                    None,
                )
                if sample is not None:
                    self.joined = JoinedTargetCache(croot, population=population)
            else:
                if (croot / "rich" / "joined.parquet").exists():
                    self.joined = JoinedTargetCache(croot, population=population)

        if population == "broad":
            self.seq = BroadSequenceReader(self.world_root / "broad")
            self.seq.set_world_root(self.world_root)
            tparts = sorted(
                (
                    self.world_root
                    / "broad"
                    / "broad_targets_v3"
                    / "observation_targets"
                ).glob("part-*.parquet")
            )
            self.targets = TargetStore(tparts)
            self.tte = TTEOverlayStore([self.split_root / "tte_overlays_broad"])
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
            self.tte = TTEOverlayStore([self.split_root / "tte_overlays_rich.parquet"])

    def prefetch(self, assignments: list[dict[str, Any]]) -> None:
        if self.joined is not None:
            self.joined.prefetch(assignments)
            return
        oids = [str(r["observation_id"]) for r in assignments]
        self.targets.prefetch(oids)
        self.tte.prefetch(oids)

    def build(self, assignment: dict[str, Any]) -> dict[str, Any]:
        seq = self.seq.load_for_assignment(
            assignment, preset=self.preset, transforms=self.transforms
        )
        if self.joined is not None:
            joined = self.joined.get(assignment)
            tgt_row, tte = _split_joined_row(joined)
        else:
            tgt_row = self.targets.get(str(assignment["observation_id"]))
            tte = self.tte.get(str(assignment["observation_id"]))
        targets = build_target_payload(
            assignment, tgt_row, tte, q70=self.q70, q85=self.q85
        )

        age = float(assignment.get("token_age_seconds") or 0.0)
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
                "channel_schemas",
                "graph_src_idx",
                "graph_dst_idx",
                "wallet_node_idx",
            }
            and not str(k).endswith("_mask")
        }
        return {
            "features": structured,
            "sequences": seq_tensors,
            "graph_index": {
                "graph_src_idx": seq.get("graph_src_idx"),
                "graph_dst_idx": seq.get("graph_dst_idx"),
                "wallet_node_idx": seq.get("wallet_node_idx"),
            },
            "sequence_masks": {
                "events": seq.get("events_mask"),
                "curve": seq.get("curve_mask"),
                "creator": seq.get("creator_mask"),
                "micro": seq.get("micro_mask"),
                "ownership": seq.get("ownership_mask"),
                "graph_token": seq.get("graph_token_mask"),
                "wallet_prior": seq.get("wallet_prior_mask"),
                "global": seq.get("global_mask"),
            },
            "modality_masks": seq.get("modality_masks") or {},
            "window_meta": seq.get("window_meta") or {},
            "truncation": seq.get("truncation") or {},
            "channel_schemas": seq.get("channel_schemas") or {},
            "causal_ok": bool(seq.get("causal_ok", True)),
            "targets": targets,
            "meta": {
                "observation_id": assignment.get("observation_id"),
                "base_split": assignment.get("base_split"),
                "population": self.population,
                "used_training_cache": self.joined is not None,
            },
        }
