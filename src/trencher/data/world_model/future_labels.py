"""Future / label sources — intentionally separate from PIT reconstruction.

PIT reconstruction must never import this module. Labels, outcomes, and
post-T supervision targets live here for later observation-state/target work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def load_postgard_outcomes(paths: list[Path] | None = None) -> list[dict[str, Any]]:
    """Load post-graduation outcomes (labels only — not PIT inputs)."""
    candidates = paths or [
        Path("data/external/world_model/postgrad/postgard_outcomes.parquet"),
        Path("data/external/pumpfun_memecoin_corpus/postgard_outcomes.parquet"),
    ]
    for path in candidates:
        if path.exists():
            return pq.read_table(path).to_pylist()
    return []


def load_migration_outcomes(paths: list[Path] | None = None) -> list[dict[str, Any]]:
    """Load migration/graduation outcome rows (labels / future supervision)."""
    candidates = paths or [
        Path("data/external/world_model/universe/migrations.parquet"),
        Path("data/external/pumpfun_memecoin_corpus/migrations.parquet"),
    ]
    for path in candidates:
        if path.exists():
            return pq.read_table(path).to_pylist()
    return []
