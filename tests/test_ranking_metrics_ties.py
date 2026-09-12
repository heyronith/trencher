import itertools

import numpy as np
import pytest

from trencher.baselines.metrics import _safe_auc, average_precision


def test_all_scores_tied_use_prevalence_ap_and_half_auc():
    y = np.array([1, 0, 1, 0], dtype=int)
    scores = np.full(4, 0.5)

    assert average_precision(y, scores) == pytest.approx(0.5)
    assert _safe_auc(y, scores) == pytest.approx(0.5)


def test_perfect_separation_with_within_class_ties():
    y = np.array([1, 1, 0, 0], dtype=int)
    scores = np.array([0.9, 0.9, 0.1, 0.1])

    assert average_precision(y, scores) == pytest.approx(1.0)
    assert _safe_auc(y, scores) == pytest.approx(1.0)


def test_known_no_tie_case():
    y = np.array([1, 0, 1], dtype=int)
    scores = np.array([0.9, 0.8, 0.7])

    assert average_precision(y, scores) == pytest.approx(5.0 / 6.0)
    assert _safe_auc(y, scores) == pytest.approx(0.5)


def test_ap_and_auc_are_permutation_invariant_with_ties():
    y = np.array([1, 0, 1, 0, 1], dtype=int)
    scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5])

    expected_ap = average_precision(y, scores)
    expected_auc = _safe_auc(y, scores)

    for perm in itertools.permutations(range(len(y))):
        idx = np.asarray(perm)

        assert average_precision(
            y[idx], scores[idx]
        ) == pytest.approx(expected_ap)

        assert _safe_auc(
            y[idx], scores[idx]
        ) == pytest.approx(expected_auc)


def test_non_identifiable_cases_remain_none():
    assert average_precision(
        np.array([0, 0]),
        np.array([0.8, 0.2]),
    ) is None

    assert _safe_auc(
        np.array([1, 1]),
        np.array([0.8, 0.2]),
    ) is None
