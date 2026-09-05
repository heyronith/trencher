"""Tests for baseline guards, features, metrics, and tiny extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trencher.baselines.calibration import PlattCalibrator
from trencher.baselines.feature_spec import all_feature_names, feature_contract_manifest
from trencher.baselines.features import event_features_from_arrays, vectorize_features
from trencher.baselines.metrics import binary_metrics, concordance_index
from trencher.baselines.progress import ProgressEmitter
from trencher.baselines.split_guard import PermanentTestAccessError, refuse_test_split
from trencher.baselines.targets import (
    TargetSpec,
    extract_binary_labels,
    extract_tte_labels,
)


def test_refuse_permanent_test() -> None:
    with pytest.raises(PermanentTestAccessError):
        refuse_test_split("test")
    refuse_test_split("train")
    refuse_test_split("val")


def test_no_identifier_feature_names() -> None:
    names = all_feature_names(include_rich=True)
    for bad in ("mint", "wallet", "creator", "observation_id", "base_split"):
        assert bad not in names
    man = feature_contract_manifest()
    assert man["no_post_T"] is True


def test_event_features_causal_and_masked() -> None:
    times = np.array([100.0, 105.0, 110.0, 200.0])
    is_buy = np.array([1.0, 0.0, 1.0, 1.0])
    sol = np.array([1.0, 2.0, 3.0, 9.0])
    price = np.array([1.0, 1.1, 1.2, 5.0])
    v_sol = np.array([30.0, 29.0, 28.0, 10.0])
    curve = np.array([0.1, 0.2, 0.3, 0.9])
    T = 115.0
    feat = event_features_from_arrays(
        times=times, is_buy=is_buy, sol=sol, price=price, v_sol=v_sol, curve=curve, T=T
    )
    assert feat["mask_events"] == 1.0
    assert feat["event_count_wfull"] == 3.0  # excludes post-T event at 200
    assert feat["latest_price"] == 1.2
    names, vec = vectorize_features(feat, include_rich=False)
    assert len(names) == len(vec)
    assert names[0] == "token_age_seconds" or "token_age_seconds" in names


def test_censored_not_negative_tte() -> None:
    rows = [
        {
            "eligible_explosion_tte": True,
            "explosion_tte_at_risk": True,
            "explosion_tte_duration_sec": 100.0,
            "explosion_tte_event_observed": False,
        },
        {
            "eligible_explosion_tte": True,
            "explosion_tte_at_risk": False,
            "explosion_tte_duration_sec": None,
            "explosion_tte_event_observed": False,
        },
    ]
    spec = TargetSpec(
        "explosion_tte",
        "tte",
        "tte",
        None,
        eligible_col="eligible_explosion_tte",
        duration_col="explosion_tte_duration_sec",
        event_col="explosion_tte_event_observed",
        at_risk_col="explosion_tte_at_risk",
    )
    dur, ev, ar, counts = extract_tte_labels(rows, spec)
    assert counts["at_risk"] == 1
    assert counts["censored"] == 1
    assert counts["events"] == 0
    assert ev[0] == 0.0
    assert ar[0]
    assert not ar[1]


def test_binary_excludes_censored_status() -> None:
    rows = [
        {
            "eligible_explosion_imminence_h60": True,
            "explosion_within_60s": True,
            "explosion_within_60s_status": "valid_positive",
        },
        {
            "eligible_explosion_imminence_h60": True,
            "explosion_within_60s": None,
            "explosion_within_60s_status": "right_censored",
        },
    ]
    spec = TargetSpec(
        "explosion_60s",
        "binary",
        "ignition",
        "explosion_within_60s",
        "explosion_within_60s_status",
        "eligible_explosion_imminence_h60",
    )
    y, elig, counts = extract_binary_labels(rows, spec)
    assert counts["eligible"] == 1
    assert elig[0] and not elig[1]
    assert y[0] == 1.0


def test_platt_and_metrics() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=500)
    y = (scores + rng.normal(scale=0.5, size=500) > 0).astype(float)
    cal = PlattCalibrator().fit(scores, y)
    p = cal.transform(scores)
    m = binary_metrics(y, p)
    assert m["n"] == 500
    assert m["pr_auc"] is not None
    assert 0 <= m["ece"] <= 1


def test_cindex_basic() -> None:
    d = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.array([1, 1, 0, 1])
    # higher risk for shorter times
    s = np.array([4.0, 3.0, 2.0, 1.0])
    c = concordance_index(d, e, s)
    assert c is not None and c > 0.5


def test_progress_emitter(tmp_path: Path) -> None:
    p = ProgressEmitter("LightGBM explosion_60s fold_1", status_path=tmp_path / "p.json")
    p.update(37, 100, message="iter")
    assert p.last == 37
    p.complete()
    assert p.last == 100
    assert (tmp_path / "p.json").exists()
