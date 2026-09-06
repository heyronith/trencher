"""Family status publishing for incomplete baseline families."""

from __future__ import annotations

import json
from pathlib import Path

from trencher.baselines.audit_v2 import publish_baseline_artifacts_v2


def test_publish_marks_incomplete_catboost(tmp_path: Path) -> None:
    art = tmp_path / "artifacts"
    (art / "lightgbm").mkdir(parents=True)
    (art / "xgboost").mkdir(parents=True)
    (art / "catboost").mkdir(parents=True)
    (art / "survival").mkdir(parents=True)
    for fam, payload in [
        (
            "lightgbm",
            {
                "family": "lightgbm",
                "protocol_version": "v2",
                "family_status": "complete",
                "heads": {},
                "train_n": 10,
                "val_n": 2,
            },
        ),
        (
            "xgboost",
            {
                "family": "xgboost",
                "protocol_version": "v2",
                "family_status": "complete",
                "heads": {},
                "train_n": 10,
                "val_n": 2,
            },
        ),
        (
            "survival",
            {
                "family": "survival",
                "protocol_version": "v2",
                "family_status": "complete",
                "heads": {},
                "train_n": 10,
                "val_n": 2,
            },
        ),
    ]:
        (art / fam / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    (art / "catboost" / "FAMILY_STATUS.json").write_text(
        json.dumps(
            {
                "family": "catboost",
                "status": "incomplete_compute_cost_stop",
                "partial_artifacts_retained": True,
                "operator_reported_approx_cost_usd": 20,
            }
        ),
        encoding="utf-8",
    )
    (art / "catboost" / "results.json").write_text(
        json.dumps({"family": "catboost", "heads": {"explosion_60s": {}}, "train_n": 1}),
        encoding="utf-8",
    )
    out = publish_baseline_artifacts_v2(
        manifest_dir=tmp_path / "manifests",
        artifacts_root=art,
    )
    st = out["results"]["family_statuses"]
    assert st["lightgbm"]["status"] == "complete"
    assert st["xgboost"]["status"] == "complete"
    assert st["survival"]["status"] == "complete"
    assert st["catboost"]["status"] == "incomplete_compute_cost_stop"
    assert out["audit"]["acceptance"]["full_population"] is False
    assert "catboost" in out["audit"]["acceptance"]["incomplete_families"]
    cat = out["results"]["families"]["catboost"]
    assert cat["do_not_treat_as_full_population_complete"] is True
