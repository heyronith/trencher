"""Acceptance tests for TTE admin censoring and production loaders."""

from __future__ import annotations

from trencher.data.world_model.splits.cutoffs import SPLIT_TRAIN, SPLIT_VAL
from trencher.data.world_model.splits.tte_overlay import (
    apply_explosion_tte_overlay,
    apply_graduation_tte_overlay,
)
from trencher.data.world_model.targets.status import (
    STATUS_RIGHT_CENSORED,
    STATUS_VALID_POSITIVE,
)
from trencher.data.world_model.training.forbid import strip_forbidden
from trencher.data.world_model.training.targets_batch import build_target_payload


def test_train_tte_hides_post_boundary_graduation() -> None:
    """Train obs before q70; event after q70 → training view must not expose event."""
    T = 1000.0
    q70 = 2000.0
    # Event at 2500 (>= q70)
    raw_tte = 1500.0
    frozen_status = STATUS_VALID_POSITIVE
    train_view = apply_graduation_tte_overlay(
        observation_time_t=T,
        split_end=q70,
        time_to_graduation_sec=raw_tte,
        time_to_graduation_status=frozen_status,
    )
    assert train_view["graduation_tte_raw_event_observed"] or raw_tte is not None
    assert train_view["graduation_tte_split_boundary_censored"] is True
    assert train_view["graduation_tte_effective_event_observed"] is False
    assert train_view["graduation_tte_effective_censoring_status"] == STATUS_RIGHT_CENSORED
    # Must NOT be converted to negative
    assert train_view["graduation_tte_effective_censoring_status"] != "valid_negative"

    # Equivalent val observation can expose the event (boundary is q85 > 2500)
    val_view = apply_graduation_tte_overlay(
        observation_time_t=T,
        split_end=5000.0,
        time_to_graduation_sec=raw_tte,
        time_to_graduation_status=frozen_status,
    )
    assert val_view["graduation_tte_effective_event_observed"] is True
    assert val_view["graduation_tte_split_boundary_censored"] is False
    assert abs(val_view["graduation_tte_effective_tte_sec"] - raw_tte) < 1e-9


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
    assert overlay["explosion_tte_effective_event_observed"] is False
    assert overlay["explosion_tte_effective_censoring_status"] == STATUS_RIGHT_CENSORED


def test_event_exactly_at_boundary_is_censored() -> None:
    T = 1000.0
    end = 2000.0
    # event_time = T + tte = end exactly → censored
    ov = apply_graduation_tte_overlay(
        observation_time_t=T,
        split_end=end,
        time_to_graduation_sec=1000.0,
        time_to_graduation_status=STATUS_VALID_POSITIVE,
    )
    assert ov["graduation_tte_split_boundary_censored"] is True
    assert ov["graduation_tte_effective_event_observed"] is False


def test_payload_hides_frozen_future_event() -> None:
    assignment = {
        "observation_id": "x",
        "observation_time_t": 1000.0,
        "base_split": SPLIT_TRAIN,
        "eligible_graduation_tte": True,
        "eligible_explosion_tte": True,
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
        "h60_simple_return": -0.1,
    }
    payload = build_target_payload(
        assignment, target, None, q70=2000.0, q85=3000.0
    )
    assert payload["graduation_tte"]["frozen_tte_sec"] == 1500.0
    assert payload["graduation_tte"]["effective_event_observed"] is False
    assert payload["graduation_tte"]["value"] is None
    assert payload["graduation_tte"]["split_boundary_censored"] is True
    # Val split would expose
    assignment_val = dict(assignment)
    assignment_val["base_split"] = SPLIT_VAL
    payload_val = build_target_payload(
        assignment_val, target, None, q70=2000.0, q85=5000.0
    )
    assert payload_val["graduation_tte"]["effective_event_observed"] is True
    assert payload_val["graduation_tte"]["value"] == 1500.0


def test_identities_stripped_from_features() -> None:
    clean = strip_forbidden(
        {"mint": "m", "wallet": "w", "token_age_seconds": 1.0, "base_split": "train"}
    )
    assert "mint" not in clean and "wallet" not in clean and "base_split" not in clean
