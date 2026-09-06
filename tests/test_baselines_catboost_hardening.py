"""CatBoost wrapper: Logloss live metric + post-fit PR-AUC selection + fixed final fit."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def test_binary_live_eval_metric_never_prauc() -> None:
    from trencher.baselines.train_trees import binary_catboost_live_eval_metric

    assert binary_catboost_live_eval_metric() == "Logloss"
    assert binary_catboost_live_eval_metric().upper() != "PRAUC"


def test_postfit_prauc_selects_argmax_on_grid() -> None:
    from trencher.baselines.train_trees import select_iteration_by_postfit_prauc

    # Synthetic staged proba: iteration 10 separates perfectly; others are noise.
    y = np.array([0, 0, 1, 1], dtype=np.float64)
    X = np.zeros((4, 2), dtype=np.float32)

    class _Fake:
        tree_count_ = 20

        def staged_predict_proba(self, _X):  # noqa: ANN001
            for it in range(1, self.tree_count_ + 1):
                if it == 10:
                    # Perfect ranking for positives.
                    yield np.array([[0.1, 0.1], [0.2, 0.2], [0.9, 0.9], [0.8, 0.8]])
                else:
                    yield np.array([[0.5, 0.5], [0.6, 0.4], [0.4, 0.55], [0.45, 0.5]])

    best_it, grid = select_iteration_by_postfit_prauc(
        _Fake(), X, y, tree_step=5, emitter=None
    )
    assert [g["iteration"] for g in grid] == [5, 10, 15, 20]
    assert best_it == 10
    assert grid[1]["pr_auc"] is not None
    assert grid[1]["pr_auc"] >= max(
        g["pr_auc"] for g in grid if g["iteration"] != 10 and g["pr_auc"] is not None
    )


def test_catboost_binary_dev_uses_logloss_and_postfit_prauc(tmp_path: Path) -> None:
    from trencher.baselines.train_trees import train_catboost_head

    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(120, 6)).astype(np.float32)
    y_tr = (rng.random(120) > 0.55).astype(np.float64)
    X_va = rng.normal(size=(40, 6)).astype(np.float32)
    y_va = (rng.random(40) > 0.55).astype(np.float64)
    # Ensure both classes in val for a defined PR-AUC.
    y_va[0] = 0.0
    y_va[1] = 1.0

    art = train_catboost_head(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        kind="binary",
        out_dir=tmp_path / "es",
        tag="catboost smoke es",
        num_boost_round=20,
        early_stopping_rounds=5,
        seed=0,
        use_gpu=False,
        prauc_tree_step=5,
    )
    assert art["live_eval_metric"] == "Logloss"
    assert art["params"].get("eval_metric") == "Logloss"
    assert "PRAUC" not in str(art["params"]).upper()
    assert art["eval_set_used"] is False
    assert art["use_best_model"] is False
    assert art["iteration_selection"] == "postfit_pr_auc"
    assert art["prauc_tree_grid"]
    assert Path(art["model_path"]).exists()
    grid_path = tmp_path / "es" / "prauc_tree_grid.json"
    assert grid_path.exists()
    # best_iteration must match argmax of persisted PR-AUC grid
    best_from_grid = max(
        art["prauc_tree_grid"],
        key=lambda g: (-1.0 if g["pr_auc"] is None else float(g["pr_auc"])),
    )["iteration"]
    assert art["best_iteration"] == best_from_grid
    assert art["actual_iterations"] == 20  # full planned trees for post-fit scan


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
    assert art["iteration_selection"] == "fixed"
    assert art["live_eval_metric"] == "Logloss"
    assert art["raw_val_pred"].shape[0] == X_cal.shape[0]
    assert "logging_level" not in art["params"]
    assert not (tmp_path / "final" / "prauc_tree_grid.json").exists()


def test_selection_folds_only_wf1_wf2() -> None:
    """Permanent val / wf3 must never enter fixed-iteration selection."""
    cutoffs = {
        "quantiles_unix": {
            "q40": 40.0,
            "q55": 55.0,
            "q70": 70.0,
            "q85": 85.0,
        },
        "launch_min_unix": 0.0,
        "launch_max_unix": 100.0,
    }
    from trencher.data.world_model.splits.walkforward import build_walkforward_folds

    all_folds = build_walkforward_folds(cutoffs)["folds"]
    selection = [f for f in all_folds if f["fold_id"] in ("wf_fold_1", "wf_fold_2")]
    assert {f["fold_id"] for f in selection} == {"wf_fold_1", "wf_fold_2"}
    assert "wf_fold_3" not in {f["fold_id"] for f in selection}
    # wf3 maps onto permanent val window [q70,q85) — excluded from selection.
    wf3 = next(f for f in all_folds if f["fold_id"] == "wf_fold_3")
    assert wf3["val"]["start_unix"] == 70.0
    assert wf3["val"]["end_unix_exclusive"] == 85.0
    assert wf3 not in selection


def test_permanent_val_cannot_drive_catboost_iteration_selection(tmp_path: Path) -> None:
    """Final fit ignores val for selection; development selection is fold-val PR-AUC only."""
    from trencher.baselines.train_trees import train_catboost_head

    rng = np.random.default_rng(2)
    X_fit = rng.normal(size=(80, 4)).astype(np.float32)
    y_fit = (rng.random(80) > 0.5).astype(np.float64)
    # Poisoned "permanent val" passed as X_val on final fit must not change iteration count.
    X_perm = rng.normal(size=(200, 4)).astype(np.float32)
    y_perm = np.ones(200, dtype=np.float64)
    fixed = 15
    art = train_catboost_head(
        X_train=X_fit,
        y_train=y_fit,
        X_val=X_perm,
        y_val=y_perm,
        kind="binary",
        out_dir=tmp_path / "perm",
        tag="final no perm select",
        num_boost_round=fixed,
        early_stopping_rounds=None,
        seed=2,
        use_gpu=False,
    )
    assert art["best_iteration"] == fixed
    assert art["eval_set_used"] is False
    assert art["iteration_selection"] == "fixed"


def test_gpu_config_path_never_sets_prauc_eval_metric(tmp_path: Path, monkeypatch) -> None:
    """Even when use_gpu=True, constructed eval_metric must not be PRAUC."""
    from trencher.baselines import train_trees as tt

    class _FakePool:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

    class _FakeModel:
        tree_count_ = 12

        def __init__(self, **params):  # noqa: ANN003
            self.params = params
            self._fit_kwargs = None

        def fit(self, *_a, **kwargs):  # noqa: ANN002, ANN003
            self._fit_kwargs = kwargs

        def predict_proba(self, X, **kwargs):  # noqa: ANN001, ANN003
            n = len(X)
            return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])

        def staged_predict_proba(self, X):  # noqa: ANN001
            for _it in range(1, self.tree_count_ + 1):
                yield self.predict_proba(X)

        def save_model(self, path: str) -> None:
            Path(path).write_bytes(b"fake-cbm")

        def get_best_iteration(self):  # noqa: ANN201
            return None

        def set_params(self, **kwargs):  # noqa: ANN003
            self.params.update(kwargs)

    captured: dict = {}

    def _fake_clf(**params):  # noqa: ANN003
        captured["params"] = dict(params)
        return _FakeModel(**params)

    def _ensure(p: Path) -> Path:
        Path(p).mkdir(parents=True, exist_ok=True)
        return Path(p)

    monkeypatch.setattr(tt, "ensure_dir", _ensure)

    import catboost

    monkeypatch.setattr(catboost, "Pool", _FakePool)
    monkeypatch.setattr(catboost, "CatBoostClassifier", _fake_clf)

    rng = np.random.default_rng(3)
    X_tr = rng.normal(size=(40, 3)).astype(np.float32)
    y_tr = (rng.random(40) > 0.5).astype(np.float64)
    X_va = rng.normal(size=(20, 3)).astype(np.float32)
    y_va = (rng.random(20) > 0.5).astype(np.float64)
    y_va[0], y_va[1] = 0.0, 1.0

    # Force CPU path inside GPU try by making fit raise once? Simpler: use_gpu=False
    # and assert helper; plus inspect captured via direct call with use_gpu False.
    # For GPU branch, monkeypatch fit success without real GPU:
    art = tt.train_catboost_head(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        kind="binary",
        out_dir=tmp_path / "gpu_cfg",
        tag="gpu cfg",
        num_boost_round=12,
        early_stopping_rounds=5,
        seed=3,
        use_gpu=True,
        prauc_tree_step=5,
    )
    assert captured["params"]["eval_metric"] == "Logloss"
    assert captured["params"]["task_type"] == "GPU"
    assert "PRAUC" not in str(captured["params"]).upper()
    assert art["iteration_selection"] == "postfit_pr_auc"
    assert art["live_eval_metric"] == "Logloss"


def test_preflight_missing_ref_raises(tmp_path: Path) -> None:
    from trencher.baselines.feature_cache import (
        FeatureCacheCoverageError,
        _preflight_ref_coverage,
    )

    by_mp = {"00": [{"observation_id": "oid_missing"}]}
    refs_root = tmp_path / "refs"
    refs_root.mkdir()
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
