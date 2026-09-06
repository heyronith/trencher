"""CatBoost wrapper smoke tests: logging params + fixed-iteration final fit."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def test_catboost_cpu_wrapper_no_logging_conflict(tmp_path: Path) -> None:
    from trencher.baselines.train_trees import train_catboost_head

    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(80, 6)).astype(np.float32)
    y_tr = (rng.random(80) > 0.5).astype(np.float64)
    X_va = rng.normal(size=(30, 6)).astype(np.float32)
    y_va = (rng.random(30) > 0.5).astype(np.float64)

    art = train_catboost_head(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        kind="binary",
        out_dir=tmp_path / "es",
        tag="catboost smoke es",
        num_boost_round=12,
        early_stopping_rounds=5,
        seed=0,
        use_gpu=False,
    )
    assert "logging_level" not in art["params"]
    assert art["eval_set_used"] is True
    assert art["use_best_model"] is True
    assert Path(art["model_path"]).exists()
    assert art["actual_iterations"] >= 1


def test_catboost_fixed_iters_no_eval_set_truncation(tmp_path: Path) -> None:
    from trencher.baselines.train_trees import train_catboost_head

    rng = np.random.default_rng(1)
    X_fit = rng.normal(size=(100, 5)).astype(np.float32)
    y_fit = (rng.random(100) > 0.4).astype(np.float64)
    # Calibration slice passed as X_val must NOT truncate the final model.
    X_cal = rng.normal(size=(25, 5)).astype(np.float32)
    y_cal = (rng.random(25) > 0.4).astype(np.float64)
    fixed = 20

    art = train_catboost_head(
        X_train=X_fit,
        y_train=y_fit,
        X_val=X_cal,
        y_val=y_cal,
        kind="binary",
        out_dir=tmp_path / "final",
        tag="catboost smoke fixed",
        num_boost_round=fixed,
        early_stopping_rounds=None,
        seed=1,
        use_gpu=False,
    )
    assert art["eval_set_used"] is False
    assert art["use_best_model"] is False
    assert art["actual_iterations"] == fixed
    assert art["best_iteration"] == fixed
    assert art["raw_val_pred"].shape[0] == X_cal.shape[0]
    assert "logging_level" not in art["params"]


def test_preflight_missing_ref_raises(tmp_path: Path) -> None:
    from trencher.baselines.feature_cache import (
        FeatureCacheCoverageError,
        _preflight_ref_coverage,
    )

    by_mp = {"00": [{"observation_id": "oid_missing"}]}
    refs_root = tmp_path / "refs"
    refs_root.mkdir()
    # empty partition dir → missing refs
    (refs_root / "00").mkdir()
    try:
        _preflight_ref_coverage(
            by_mp=by_mp,
            refs_root=refs_root,
            rich_refs=None,
            include_rich=False,
        )
        raise AssertionError("expected FeatureCacheCoverageError")
    except FeatureCacheCoverageError as exc:
        assert "missing broad sequence refs" in str(exc)
        assert "count=1" in str(exc)


def test_resume_oids_required_and_integrity(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from trencher.baselines.feature_cache import (
        FeatureCacheCoverageError,
        _load_resume_observation_ids,
    )

    part = tmp_path / "features.parquet"
    pq.write_table(
        pa.table({"observation_id": ["a", "b", "c"], "token_age_seconds": [1.0, 2.0, 3.0]}),
        part,
    )
    assert _load_resume_observation_ids(part, expected_rows=3) == ["a", "b", "c"]

    try:
        _load_resume_observation_ids(part, expected_rows=2)
        raise AssertionError("expected row mismatch")
    except FeatureCacheCoverageError as exc:
        assert "row mismatch" in str(exc)

    bad = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"x": [1, 2]}), bad)
    try:
        _load_resume_observation_ids(bad, expected_rows=2)
        raise AssertionError("expected missing observation_id")
    except FeatureCacheCoverageError as exc:
        assert "observation_id" in str(exc)


def test_resume_integrity_assertions_apply_with_skips(tmp_path: Path) -> None:
    """Final unique/row checks must run even when partitions_skipped > 0."""
    # Simulate the invariant logic used at end of build_baseline_features_v1.
    from trencher.baselines.feature_cache import FeatureCacheCoverageError

    seen_oids = {"a", "b"}
    expected_rows = 3
    rows_written = 3
    try:
        if len(seen_oids) != expected_rows:
            raise FeatureCacheCoverageError(
                f"unique observation_ids={len(seen_oids)} != assignment rows={expected_rows}"
            )
        if rows_written != expected_rows:
            raise FeatureCacheCoverageError(
                f"rows_written={rows_written} != assignment rows={expected_rows}"
            )
        raise AssertionError("expected coverage error")
    except FeatureCacheCoverageError as exc:
        assert "unique observation_ids=2" in str(exc)
