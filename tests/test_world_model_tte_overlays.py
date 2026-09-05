"""Acceptance tests for TTE admin censoring and production loaders."""

from __future__ import annotations

import numpy as np

from trencher.data.world_model.splits.cutoffs import SPLIT_TRAIN, SPLIT_VAL
from trencher.data.world_model.splits.tte_overlay import (
    STATUS_LEFT_TRUNCATED,
    apply_explosion_tte_overlay,
    apply_graduation_tte_overlay,
)
from trencher.data.world_model.targets.status import (
    STATUS_RIGHT_CENSORED,
    STATUS_VALID_POSITIVE,
)
from trencher.data.world_model.training.forbid import strip_forbidden
from trencher.data.world_model.training.targets_batch import build_target_payload
from trencher.data.world_model.training.transforms import (
    apply_clip_then_zscore,
    fit_numeric_stats,
)


def test_train_tte_hides_post_boundary_graduation() -> None:
    """Train obs before q70; event after q70 → training view must not expose event."""
    T = 1000.0
    q70 = 2000.0
    raw_tte = 1500.0
    frozen_status = STATUS_VALID_POSITIVE
    train_view = apply_graduation_tte_overlay(
        observation_time_t=T,
        split_end=q70,
        time_to_graduation_sec=raw_tte,
        time_to_graduation_status=frozen_status,
    )
    assert train_view["graduation_tte_split_boundary_censored"] is True
    assert train_view["graduation_tte_event_observed"] is False
    assert train_view["graduation_tte_censored"] is True
    assert train_view["graduation_tte_at_risk"] is True
    assert abs(train_view["graduation_tte_duration_sec"] - (q70 - T)) < 1e-9
    assert train_view["graduation_tte_effective_status"] == STATUS_RIGHT_CENSORED
    assert train_view["graduation_tte_effective_status"] != "valid_negative"

    val_view = apply_graduation_tte_overlay(
        observation_time_t=T,
        split_end=5000.0,
        time_to_graduation_sec=raw_tte,
        time_to_graduation_status=frozen_status,
    )
    assert val_view["graduation_tte_event_observed"] is True
    assert val_view["graduation_tte_split_boundary_censored"] is False
    assert abs(val_view["graduation_tte_duration_sec"] - raw_tte) < 1e-9


def test_train_tte_hides_post_boundary_explosion() -> None:
    T = 1000.0
    q70 = 2000.0
    overlay = apply_explosion_tte_overlay(
        observation_time_t=T,
        split_end=q70,
        time_to_explosion_sec=1500.0,
        explosion_event_observed=True,
        explosion_already_started_at_t=False,
        explosion_right_censored=False,
    )
    assert overlay["explosion_tte_split_boundary_censored"] is True
    assert overlay["explosion_tte_event_observed"] is False
    assert overlay["explosion_tte_censored"] is True
    assert abs(overlay["explosion_tte_duration_sec"] - 1000.0) < 1e-9


def test_event_exactly_at_boundary_is_censored() -> None:
    T = 1000.0
    end = 2000.0
    ov = apply_graduation_tte_overlay(
        observation_time_t=T,
        split_end=end,
        time_to_graduation_sec=1000.0,
        time_to_graduation_status=STATUS_VALID_POSITIVE,
    )
    assert ov["graduation_tte_split_boundary_censored"] is True
    assert ov["graduation_tte_event_observed"] is False
    assert ov["graduation_tte_censored"] is True


def test_already_started_event_not_at_risk() -> None:
    ov = apply_explosion_tte_overlay(
        observation_time_t=1000.0,
        split_end=2000.0,
        time_to_explosion_sec=None,
        explosion_event_observed=True,
        explosion_already_started_at_t=True,
        explosion_right_censored=False,
    )
    assert ov["explosion_tte_at_risk"] is False
    assert ov["explosion_tte_event_observed"] is False
    assert ov["explosion_tte_effective_status"] == STATUS_LEFT_TRUNCATED


def test_censored_tte_retains_duration() -> None:
    assignment = {
        "observation_id": "x",
        "observation_time_t": 1000.0,
        "base_split": SPLIT_TRAIN,
        "eligible_graduation_tte": True,
        "eligible_explosion_tte": True,
        "eligible_returns_h60": True,
    }
    target = {
        "observation_id": "x",
        "time_to_graduation_sec": 1500.0,
        "time_to_graduation_status": STATUS_VALID_POSITIVE,
        "time_to_explosion_sec": 1600.0,
        "explosion_event_observed": True,
        "explosion_already_started_at_t": False,
        "explosion_right_censored": False,
        "h60_status": "valid_negative",
        "h60_simple_return": 0.0,  # valid numeric zero
        "remaining_upside_to_explosion_peak": 0.0,
    }
    payload = build_target_payload(assignment, target, None, q70=2000.0, q85=3000.0)
    g = payload["graduation_tte"]
    assert g["event_observed"] is False
    assert g["censored"] is True
    assert g["duration_sec"] == 1000.0  # retained
    assert g["split_boundary_censored"] is True
    assert payload["returns_h60"]["value"] == 0.0
    assert payload["explosion_magnitude"]["value"] == 0.0
    for fam in (
        "returns",
        "barriers",
        "mfe_mae",
        "survival",
        "graduation_tte",
        "explosion_imminence",
        "explosion_tte",
        "persistence",
        "execution",
        "future_path_refs",
    ):
        assert fam in payload["families_present"]

    assignment_val = dict(assignment)
    assignment_val["base_split"] = SPLIT_VAL
    payload_val = build_target_payload(
        assignment_val, target, None, q70=2000.0, q85=5000.0
    )
    assert payload_val["graduation_tte"]["event_observed"] is True
    assert payload_val["graduation_tte"]["duration_sec"] == 1500.0


def test_clip_then_zscore_order() -> None:
    raw = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 100.0])
    stats = fit_numeric_stats(raw)
    out = apply_clip_then_zscore(np.array([100.0]), stats)
    clipped = min(100.0, stats["p99"])
    expected = (clipped - stats["mean"]) / stats["std"]
    assert abs(float(out[0]) - expected) < 1e-9
    # zscore-without-clip on raw outlier differs when p99 < 100
    wrong = (100.0 - stats["mean"]) / stats["std"]
    assert stats["p99"] < 100.0
    assert abs(float(out[0]) - wrong) > 1e-6


def test_identities_stripped_from_features() -> None:
    clean = strip_forbidden(
        {"mint": "m", "wallet": "w", "token_age_seconds": 1.0, "base_split": "train"}
    )
    assert "mint" not in clean and "wallet" not in clean and "base_split" not in clean
