"""Regression: world_model.evaluation.metrics._safe_auc must match sklearn with ties."""

from __future__ import annotations

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.metrics import roc_auc_score  # noqa: E402

from trencher.world_model.evaluation.metrics import _safe_auc  # noqa: E402


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_safe_auc_matches_sklearn_with_ties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 500
    y = rng.integers(0, 2, size=n)
    # Continuous scores plus many exact ties.
    s = rng.normal(size=n)
    s[::3] = s[0]
    s[1::5] = s[1]
    if y.sum() == 0 or y.sum() == n:
        y[0] = 0
        y[1] = 1
    got = _safe_auc(y, s)
    want = float(roc_auc_score(y, s))
    assert got is not None
    assert abs(got - want) < 1e-12


def test_safe_auc_none_when_single_class() -> None:
    y = np.zeros(20, dtype=int)
    s = np.linspace(0, 1, 20)
    assert _safe_auc(y, s) is None
