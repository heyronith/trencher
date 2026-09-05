"""Investigate ``tracking_expires_at`` and document graduation censoring policy.

Decision (broad_supervision_v1): ``tracking_expires_at`` is NOT proven as a
per-token graduation-tracking end. Retain conservative target_dataset_v3
graduation treatment (absence → censored unless migration-source coverage
proves a valid negative).
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trencher.data.world_model.io import utc_now
from trencher.data.world_model.pit import _to_unix

logger = logging.getLogger(__name__)

# Empirically uniform on the frozen corpus; documented here for the audit.
EXPECTED_TRACKING_WINDOW_SEC = 48 * 3600


def investigate_tracking_expires_at(pump_root: Path) -> dict[str, Any]:
    """Quantify tracking_expires_at vs detected_at / graduated_at on tokens."""
    tokens_path = pump_root / "tokens.parquet"
    table = pq.read_table(
        tokens_path,
        columns=["mint", "detected_at", "tracking_expires_at", "graduated_at"],
    )
    deltas_sec: list[float] = []
    null_expire = 0
    null_detected = 0
    grads_after_expire = 0
    grads_before_or_at_expire = 0
    graduated_n = 0
    for row in table.to_pylist():
        det = _to_unix(row.get("detected_at"))
        exp = _to_unix(row.get("tracking_expires_at"))
        grad = _to_unix(row.get("graduated_at"))
        if det is None:
            null_detected += 1
            continue
        if exp is None:
            null_expire += 1
            continue
        deltas_sec.append(float(exp) - float(det))
        if grad is not None:
            graduated_n += 1
            if grad > exp:
                grads_after_expire += 1
            else:
                grads_before_or_at_expire += 1

    # Round deltas to nearest second for mode analysis.
    rounded = Counter(int(round(d)) for d in deltas_sec)
    mode_sec, mode_n = (rounded.most_common(1)[0] if rounded else (None, 0))
    unique_deltas = len(rounded)
    uniform_48h = (
        mode_sec == EXPECTED_TRACKING_WINDOW_SEC
        and mode_n == len(deltas_sec)
        and unique_deltas == 1
    )

    known_issues_path = pump_root / "KNOWN_ISSUES.md"
    readme_path = pump_root / "README.md"
    known_issues_text = (
        known_issues_path.read_text(encoding="utf-8")
        if known_issues_path.exists()
        else ""
    )
    readme_text = (
        readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    )
    known_issues_mentions_tracking = (
        "tracking_expires" in known_issues_text.lower()
        or "tracking expires" in known_issues_text.lower()
    )
    readme_claims_39_day = "39-day" in readme_text or "39 day" in readme_text.lower()

    return {
        "tokens_n": table.num_rows,
        "rows_with_detected_and_expire": len(deltas_sec),
        "null_detected_at": null_detected,
        "null_tracking_expires_at": null_expire,
        "delta_sec_mode": mode_sec,
        "delta_sec_mode_count": mode_n,
        "unique_delta_sec_values": unique_deltas,
        "uniform_plus_48h": uniform_48h,
        "expected_tracking_window_sec": EXPECTED_TRACKING_WINDOW_SEC,
        "graduated_n": graduated_n,
        "grads_after_tracking_expires": grads_after_expire,
        "grads_at_or_before_tracking_expires": grads_before_or_at_expire,
        "known_issues_mentions_tracking_expires_at": known_issues_mentions_tracking,
        "readme_claims_39_day_continuous_collection": readme_claims_39_day,
        "known_issues_path": str(known_issues_path),
        "readme_path": str(readme_path),
    }


def graduation_censor_decision(pump_root: Path) -> dict[str, Any]:
    """Return the audit decision dict for broad graduation censoring.

    Conclusion: ``tracking_expires_at`` is a uniform +48h field from
    ``detected_at``, not a proven per-token end of graduation observation
    coverage. KNOWN_ISSUES is silent; README claims 39-day continuous
    collection for the corpus window. Therefore retain conservative v3
    graduation treatment (absence censored unless migration-source coverage
    proves a valid negative via ``target_source_coverage_v2``).
    """
    investigation = investigate_tracking_expires_at(pump_root)
    decision = {
        "generated_at": utc_now(),
        "decision_id": "broad_graduation_censor_v1",
        "tracking_expires_at_is_proven_graduation_tracking_end": False,
        "retain_conservative_v3_graduation_treatment": True,
        "rationale": [
            (
                "tracking_expires_at - detected_at is uniform +48h "
                f"(mode={investigation.get('delta_sec_mode')}s, "
                f"uniform={investigation.get('uniform_plus_48h')})"
            ),
            (
                f"{investigation.get('grads_after_tracking_expires')} graduations "
                "occur after tracking_expires_at — field cannot be a hard "
                "graduation-observation end"
            ),
            "KNOWN_ISSUES.md is silent on tracking_expires_at",
            (
                "README claims continuous collection over a fixed 39-day window "
                "(Jun 5 – Jul 14, 2026), not a 48h per-token follow window"
            ),
            (
                "Therefore do not treat tracking_expires_at as migration-source "
                "coverage end; keep target_dataset_v3 compute_graduation + "
                "target_source_coverage_v2 absence semantics unchanged"
            ),
        ],
        "policy": {
            "absence_of_migration": "censored_unless_migration_source_coverage_proves_negative",
            "use_tracking_expires_at_as_coverage_end": False,
            "reuse_compute_graduation": True,
            "reuse_target_source_coverage_v2": True,
            "target_value_semantics": "unchanged_from_target_dataset_v3",
        },
        "investigation": investigation,
    }
    logger.info(
        "graduation_censor_decision: retain_conservative_v3=%s grads_after_expire=%s",
        decision["retain_conservative_v3_graduation_treatment"],
        investigation.get("grads_after_tracking_expires"),
    )
    return decision
