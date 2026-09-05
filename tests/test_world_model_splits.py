"""Unit tests for chronological splits (strict purge, global folds, ties)."""

from __future__ import annotations

import numpy as np

from trencher.data.world_model.splits.cutoffs import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    assign_base_split,
    compute_launch_cutoffs,
)
from trencher.data.world_model.splits.purge import (
    bounded_window_eligible,
    compute_eligibility_flags,
    eligibility_column,
    survival_confirmation_eligible,
)
from trencher.data.world_model.splits.walkforward import (
    build_walkforward_folds,
    fold_split_for_launch,
)
from trencher.data.world_model.targets.status import SURVIVAL_NEGATIVE_CONFIRMATION_SEC


def test_cutoffs_deterministic_and_ordered() -> None:
    launches = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    c1 = compute_launch_cutoffs(launches)
    c2 = compute_launch_cutoffs(launches)
    assert c1["quantiles_unix"] == c2["quantiles_unix"]
    q = c1["quantiles_unix"]
    assert q["q40"] <= q["q55"] <= q["q70"] <= q["q85"]


def test_tie_tokens_same_side_of_boundary() -> None:
    # Many tokens share the exact q70 timestamp → all must land in the same split.
    launches = np.array([1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 8.0, 9.0, 10.0, 11.0])
    cut = compute_launch_cutoffs(launches)
    q70 = cut["quantiles_unix"]["q70"]
    q85 = cut["quantiles_unix"]["q85"]
    splits = {assign_base_split(5.0, q70=q70, q85=q85)}
    assert len(splits) == 1


def test_purge_strict_less_than() -> None:
    split_end = 1000.0
    T = 900.0
    H = 100.0
    # T+H == split_end → NOT eligible
    assert not bounded_window_eligible(T, H, split_end)
    # T+H < split_end → eligible
    assert bounded_window_eligible(T, H - 0.001, split_end)


def test_survival_confirmation_strict() -> None:
    split_end = 10_000.0
    T = 8000.0
    conf = SURVIVAL_NEGATIVE_CONFIRMATION_SEC
    # Exactly at boundary → ineligible
    assert not survival_confirmation_eligible(
        T, split_end - T - conf, split_end, confirmation_sec=conf
    )
    # Comfortably inside → eligible
    assert survival_confirmation_eligible(T, 100.0, split_end, confirmation_sec=conf)


def test_late_observation_all_heads_ineligible() -> None:
    flags = compute_eligibility_flags(
        1500.0, split_start=0.0, split_end=1000.0
    )
    assert flags["obs_in_split_interval"] is False
    assert flags[eligibility_column("returns", 10)] is False
    assert flags[eligibility_column("survival", 30)] is False


def test_head_specific_columns_exist() -> None:
    flags = compute_eligibility_flags(100.0, split_start=0.0, split_end=1_000_000.0)
    assert eligibility_column("returns", 86400) in flags
    assert eligibility_column("barriers", 300) in flags
    assert eligibility_column("mfe_mae", 60) in flags
    assert eligibility_column("explosion_imminence", 60) in flags
    assert eligibility_column("graduation_binary", 86400) in flags
    assert eligibility_column("survival", 86400) in flags
    assert eligibility_column("execution", 300) in flags
    assert eligibility_column("future_path", 3600) in flags
    assert eligibility_column("graduation_tte") in flags


def test_walkforward_global_quantiles_exclude_test() -> None:
    launches = np.linspace(0.0, 100.0, 101)
    cut = compute_launch_cutoffs(launches)
    wf = build_walkforward_folds(cut)
    assert len(wf["folds"]) == 3
    q = cut["quantiles_unix"]
    f1, f2, f3 = wf["folds"]
    assert f1["train"]["end_unix_exclusive"] == q["q40"]
    assert f1["val"]["end_unix_exclusive"] == q["q55"]
    assert f2["val"]["end_unix_exclusive"] == q["q70"]
    assert f3["val"]["end_unix_exclusive"] == q["q85"]
    # Permanent test launch
    t_test = q["q85"] + 1.0
    for fold in wf["folds"]:
        assert fold_split_for_launch(t_test, fold) is None
    assert assign_base_split(t_test, q70=q["q70"], q85=q["q85"]) == SPLIT_TEST
    assert assign_base_split(q["q40"] - 1, q70=q["q70"], q85=q["q85"]) == SPLIT_TRAIN
    assert assign_base_split(q["q70"], q70=q["q70"], q85=q["q85"]) == SPLIT_VAL
