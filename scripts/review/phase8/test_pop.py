"""TEST population size from split metadata only — no TEST feature/label reads."""

from __future__ import annotations

import json
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq
from scripts.review.phase8.common import Q85, SPLIT


def test_population_from_metadata() -> dict[str, Any]:
    """Counts + launch-time range from token_assignments / cutoffs / assign stats."""
    cutoffs = json.loads((SPLIT / "cutoffs.json").read_text(encoding="utf-8"))
    q85 = float(cutoffs["quantiles_unix"]["q85"])

    assign_stats = json.loads(
        (SPLIT / "observation_assignments_broad_stats.json").read_text(encoding="utf-8")
    )
    n_obs = int(assign_stats["observations_by_base_split"]["test"])

    tok = pq.read_table(
        SPLIT / "token_assignments.parquet",
        columns=["mint", "base_split", "detected_at_unix"],
    )
    test_mask = pc.equal(tok["base_split"], "test")
    test_tok = tok.filter(test_mask)
    n_mints = int(test_tok.num_rows)
    t_min = float(pc.min(test_tok["detected_at_unix"]).as_py())
    t_max = float(pc.max(test_tok["detected_at_unix"]).as_py())

    # Sanity: launch times should be >= q85
    assert t_min >= q85 - 1e-3, (t_min, q85)

    return {
        **Q85,
        "source": (
            "token_assignments.parquet + "
            "observation_assignments_broad_stats.json + cutoffs.json"
        ),
        "n_test_mints": n_mints,
        "n_test_observations": n_obs,
        "launch_time_range_unix": {"min": t_min, "max": t_max},
        "q85_unix": q85,
        "note": (
            "Metadata only — no TEST features, labels, or observation_targets rows read."
        ),
    }


def main() -> int:
    out = test_population_from_metadata()
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
